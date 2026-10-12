#!/usr/bin/env python3
"""Expanded-pool dense retrieval with note-card document representations:
zero-shot evaluation, hard-negative mining and fine-tuning (paper, Table 2).

Despite the file name, this runner is used for both retrievers in the paper:
  - BERT bi-encoder:      --model_name bert-base-chinese --full_param
                          (mean pooling, raw queries, dot-product scores)
  - Qwen3-Embedding-0.6B: --model_name Qwen/Qwen3-Embedding-0.6B --qwen_official_style
                          (instruction-prefixed queries, last-token pooling,
                           left padding, L2-normalised embeddings, LoRA)

Queries are search_test[0:1000]; the corpus is the union of the exposed
candidates of search_test[0:6192] restricted to notes with a note card
(113,548 notes with the released cards). Validation uses search_train[4000:5000]
against the candidates of search_train[0:5000].

Stages:
  preflight  - verify coverage and corpus sizes
  zero_shot  - evaluate Qwen3-Embedding on expanded val/test retrieval
  mine       - mine zero-shot hard negatives for train queries
  train      - LoRA/full fine-tune with mined hard negatives and expanded eval
  aggregate  - collect metrics into summary tables
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qilin_notecard_biencoder import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_QILIN_DIR,
    DEFAULT_QRELS_PATH,
    DEFAULT_TEST_NOTECARD,
    DEFAULT_TRAIN_NOTECARD,
    POSITIVE_FIELDS,
    build_note_text,
    calculate_metrics,
    candidate_ids_from_row,
    collect_candidate_map,
    encode_texts,
    is_negative_detail,
    is_positive_detail,
    last_token_pool,
    load_notecards,
    load_notes_subset,
    load_qrels,
    load_search_slice,
    mean_token_pool,
    qrels_from_details,
    write_rankings,
)

DEFAULT_OUTPUT = DEFAULT_OUTPUT_ROOT / "dense_retrieval"
MODES = ("original", "summary", "title_summary", "combined")
METRIC_KEYS = ("MRR@10", "MAP@10", "Recall@10", "MRR@100", "MAP@100", "Recall@100")
DEFAULT_QWEN_QUERY_INSTRUCTION = "Given a Xiaohongshu search query, retrieve relevant notes that match the user's search intent"


def qwen_style(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "qwen_official_style", False))


def format_query(query: str, args: argparse.Namespace) -> str:
    """Raw query (BERT) or instruction-prefixed query (Qwen3-Embedding official style)."""
    if qwen_style(args):
        return f"Instruct: {args.qwen_query_instruction}\nQuery: {query}"
    return query


def pooling_mode(args: argparse.Namespace) -> str:
    return "last_token" if qwen_style(args) else "mean"


def tokenizer_padding_side(args: argparse.Namespace) -> str:
    return "left" if qwen_style(args) else "right"


@dataclass
class TrainGroup:
    qid: int
    query: str
    positive_ids: list[int]
    exposure_negative_ids: list[int]


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_cards(path: Path) -> dict[int, str]:
    return load_notecards(path)


def unique_ids(candidate_map: dict[int, list[int]]) -> set[int]:
    return {pid for ids in candidate_map.values() for pid in ids}


def covered_corpus_ids(candidate_map: dict[int, list[int]], cards: dict[int, str]) -> list[int]:
    return sorted(unique_ids(candidate_map) & set(cards))


def build_fixed_doc_texts(
    corpus_ids: list[int],
    notes: dict[int, dict[str, str]],
    cards: dict[int, str],
    mode: str,
    max_doc_chars: int,
    content_head_chars: int,
) -> dict[int, str]:
    out: dict[int, str] = {}
    mode_cards = {} if mode == "original" else cards
    missing_policy = "fallback" if mode == "original" else "error"
    for note_id in corpus_ids:
        text = build_note_text(note_id, notes, mode_cards, mode, missing_policy, max_doc_chars, content_head_chars)
        if text:
            out[note_id] = text
    return out


def topk_from_embeddings(
    query_emb: np.ndarray,
    doc_emb: np.ndarray,
    doc_ids: list[int],
    qid_start: int,
    top_k: int,
    block_size: int,
) -> dict[int, list[int]]:
    rankings: dict[int, list[int]] = {}
    k = min(top_k, len(doc_ids))
    for start in range(0, query_emb.shape[0], block_size):
        scores = query_emb[start : start + block_size] @ doc_emb.T
        if len(doc_ids) <= k:
            idx = np.argsort(-scores, axis=1)[:, :k]
        else:
            idx_part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            part_scores = np.take_along_axis(scores, idx_part, axis=1)
            idx = np.take_along_axis(idx_part, np.argsort(-part_scores, axis=1), axis=1)
        for offset, row_idx in enumerate(idx):
            rankings[qid_start + start + offset] = [doc_ids[int(i)] for i in row_idx]
    return rankings


def load_encoder_model(model_name: str, device: str, fp16: bool = False, bf16: bool = False, padding_side: str = "right"):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = padding_side
    dtype = None
    if device == "cuda":
        if bf16:
            dtype = torch.bfloat16
        elif fp16:
            dtype = torch.float16
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=dtype)
    model.to(device)
    return model, tokenizer


def autocast_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cuda":
        return {"enabled": False}
    if getattr(args, "bf16", False):
        import torch

        return {"enabled": True, "dtype": torch.bfloat16}
    return {"enabled": bool(args.fp16)}


def maybe_apply_lora(model, args: argparse.Namespace):
    if args.full_param:
        return model
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def dataset_context(args: argparse.Namespace, smoke: bool = False) -> dict[str, Any]:
    train_end = args.train_start + args.smoke_train_queries if smoke else args.train_end
    val_end = args.val_start + args.smoke_val_queries if smoke else args.val_end
    train_rows = load_search_slice(args.qilin_dir, "search_train", args.train_start, train_end).to_dict("records")
    val_rows = load_search_slice(args.qilin_dir, "search_train", args.val_start, val_end).to_dict("records")
    test_rows = load_search_slice(args.qilin_dir, "search_test", args.test_start, args.test_end).to_dict("records")
    corpus_train_end = args.train_corpus_start + args.smoke_corpus_rows if smoke else args.train_corpus_end
    train_corpus_rows = load_search_slice(args.qilin_dir, "search_train", args.train_corpus_start, corpus_train_end).to_dict("records")
    test_corpus_rows = load_search_slice(args.qilin_dir, "search_test", args.test_corpus_start, args.test_corpus_end).to_dict("records")

    train_cards = load_cards(args.train_notecard_path)
    test_cards = load_cards(args.test_notecard_path)
    train_map = collect_candidate_map(train_rows, "details", None, args.train_start)
    val_map = collect_candidate_map(val_rows, "details", None, args.val_start)
    train_corpus_map = collect_candidate_map(train_corpus_rows, "details", None, args.train_corpus_start)
    test_corpus_map = collect_candidate_map(test_corpus_rows, args.test_candidate_key, None, args.test_corpus_start)
    train_corpus_ids = covered_corpus_ids(train_corpus_map, train_cards)
    test_corpus_ids = covered_corpus_ids(test_corpus_map, test_cards)
    train_qrels = qrels_from_details(train_rows, args.train_start, POSITIVE_FIELDS)
    val_qrels = qrels_from_details(val_rows, args.val_start, POSITIVE_FIELDS)
    test_qrels = load_qrels(args.qrels_path, args.test_start, args.test_end)
    all_note_ids = set(train_corpus_ids) | set(test_corpus_ids) | unique_ids(train_map) | unique_ids(val_map)
    notes = load_notes_subset(args.qilin_dir, all_note_ids)
    return {
        "train_rows": train_rows,
        "val_rows": val_rows,
        "test_rows": test_rows,
        "train_cards": train_cards,
        "test_cards": test_cards,
        "train_map": train_map,
        "val_map": val_map,
        "train_corpus_map": train_corpus_map,
        "test_corpus_map": test_corpus_map,
        "train_corpus_ids": [pid for pid in train_corpus_ids if pid in notes],
        "test_corpus_ids": [pid for pid in test_corpus_ids if pid in notes],
        "train_qrels": train_qrels,
        "val_qrels": val_qrels,
        "test_qrels": test_qrels,
        "notes": notes,
    }


def qrels_coverage(qrels: dict[int, set[int]], corpus_ids: list[int]) -> dict[str, float]:
    corpus = set(corpus_ids)
    rels = {pid for ids in qrels.values() for pid in ids}
    return {
        "queries": float(len(qrels)),
        "unique_relevant_notes": float(len(rels)),
        "covered_relevant_notes": float(len(rels & corpus)),
        "missing_relevant_notes": float(len(rels - corpus)),
        "coverage": float(len(rels & corpus) / len(rels)) if rels else 0.0,
    }


def stage_preflight(args: argparse.Namespace) -> None:
    ctx = dataset_context(args, smoke=False)
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_queries": [args.train_start, args.train_end],
        "val_queries": [args.val_start, args.val_end],
        "test_queries": [args.test_start, args.test_end],
        "train_corpus_rows": [args.train_corpus_start, args.train_corpus_end],
        "test_corpus_rows": [args.test_corpus_start, args.test_corpus_end],
        "cards": {"train": len(ctx["train_cards"]), "test": len(ctx["test_cards"])},
        "coverage": {
            "train_details_unique": len(unique_ids(ctx["train_map"])),
            "train_details_covered": len(unique_ids(ctx["train_map"]) & set(ctx["train_cards"])),
            "val_details_unique": len(unique_ids(ctx["val_map"])),
            "val_details_covered": len(unique_ids(ctx["val_map"]) & set(ctx["train_cards"])),
            "train_val_corpus_unique": len(unique_ids(ctx["train_corpus_map"])),
            "train_val_corpus_covered": len(ctx["train_corpus_ids"]),
            "test_corpus_unique": len(unique_ids(ctx["test_corpus_map"])),
            "test_corpus_covered": len(ctx["test_corpus_ids"]),
            "test_corpus_missing": len(unique_ids(ctx["test_corpus_map"])) - len(ctx["test_corpus_ids"]),
        },
        "qrels_coverage": {
            "val_expanded": qrels_coverage(ctx["val_qrels"], ctx["train_corpus_ids"]),
            "test_expanded": qrels_coverage(ctx["test_qrels"], ctx["test_corpus_ids"]),
        },
    }
    dump_json(args.output_dir / "preflight.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def evaluate_expanded(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    qid_start: int,
    qrels: dict[int, set[int]],
    corpus_ids: list[int],
    notes: dict[int, dict[str, str]],
    cards: dict[int, str],
    mode: str,
    args: argparse.Namespace,
    out_dir: Path,
    split_name: str,
) -> dict[str, float]:
    doc_texts = build_fixed_doc_texts(corpus_ids, notes, cards, mode, args.max_doc_chars, args.content_head_chars)
    doc_ids = sorted(doc_texts)
    query_texts = [format_query(str(row.get("query") or ""), args) for row in rows]
    t0 = time.time()
    doc_emb = encode_texts(
        model,
        tokenizer,
        [doc_texts[pid] for pid in doc_ids],
        args.device,
        args.eval_batch_size,
        args.max_length,
        args.normalize_embeddings,
        args.fp16,
        pooling=pooling_mode(args),
    )
    query_emb = encode_texts(
        model,
        tokenizer,
        query_texts,
        args.device,
        args.eval_batch_size,
        args.max_length,
        args.normalize_embeddings,
        args.fp16,
        pooling=pooling_mode(args),
    )
    rankings = topk_from_embeddings(query_emb, doc_emb, doc_ids, qid_start, args.top_k, args.score_block_size)
    metrics = calculate_metrics(rankings, qrels, [10, args.top_k])
    metrics.update({"queries": float(len(rows)), "corpus_docs": float(len(doc_ids)), "eval_seconds": time.time() - t0})
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rankings(out_dir / f"{split_name}_rankings.csv", rankings)
    dump_json(out_dir / f"{split_name}_metrics.json", metrics)
    del doc_emb, query_emb
    gc.collect()
    return metrics


def stage_zero_shot(args: argparse.Namespace) -> None:
    ctx = dataset_context(args, smoke=args.smoke)
    model, tokenizer = load_encoder_model(args.model_name, args.device, args.fp16, args.bf16, tokenizer_padding_side(args))
    run_dir = args.output_dir / ("smoke_zero_shot" if args.smoke else "zero_shot")
    rows: list[dict[str, Any]] = []
    for split in args.eval_splits:
        for mode in args.eval_modes:
            if split == "val":
                metrics = evaluate_expanded(
                    model,
                    tokenizer,
                    ctx["val_rows"],
                    args.val_start,
                    ctx["val_qrels"],
                    ctx["train_corpus_ids"],
                    ctx["notes"],
                    ctx["train_cards"],
                    mode,
                    args,
                    run_dir / split / mode,
                    split,
                )
            elif split == "test":
                metrics = evaluate_expanded(
                    model,
                    tokenizer,
                    ctx["test_rows"],
                    args.test_start,
                    ctx["test_qrels"],
                    ctx["test_corpus_ids"],
                    ctx["notes"],
                    ctx["test_cards"],
                    mode,
                    args,
                    run_dir / split / mode,
                    split,
                )
            else:
                raise ValueError(split)
            row = {"stage": "zero_shot", "split": split, "mode": mode, **{k: metrics.get(k, 0.0) for k in METRIC_KEYS}, "valid_queries": metrics["_valid_queries"]}
            rows.append(row)
            dump_json(run_dir / "run_summary.json", {"rows": rows})
    del model, tokenizer
    gc.collect()


def mine_for_mode(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    ctx = dataset_context(args, smoke=args.smoke)
    model, tokenizer = load_encoder_model(args.model_name, args.device, args.fp16, args.bf16, tokenizer_padding_side(args))
    doc_texts = build_fixed_doc_texts(ctx["train_corpus_ids"], ctx["notes"], ctx["train_cards"], mode, args.max_doc_chars, args.content_head_chars)
    doc_ids = sorted(doc_texts)
    doc_emb = encode_texts(model, tokenizer, [doc_texts[pid] for pid in doc_ids], args.device, args.eval_batch_size, args.max_length, args.normalize_embeddings, args.fp16, pooling=pooling_mode(args))
    query_texts = [format_query(str(row.get("query") or ""), args) for row in ctx["train_rows"]]
    query_emb = encode_texts(model, tokenizer, query_texts, args.device, args.eval_batch_size, args.max_length, args.normalize_embeddings, args.fp16, pooling=pooling_mode(args))
    rankings = topk_from_embeddings(query_emb, doc_emb, doc_ids, args.train_start, args.mine_top_k, args.score_block_size)
    hard: dict[str, list[int]] = {}
    for qid, ranked in rankings.items():
        positives = ctx["train_qrels"].get(qid, set())
        hard[str(qid)] = [pid for pid in ranked if pid not in positives][: args.hard_negative_k]
    out = {
        "mode": mode,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query_count": len(ctx["train_rows"]),
        "corpus_docs": len(doc_ids),
        "mine_top_k": args.mine_top_k,
        "hard_negative_k": args.hard_negative_k,
        "hard_negatives": hard,
    }
    del model, tokenizer, doc_emb, query_emb
    gc.collect()
    return out


def stage_mine(args: argparse.Namespace) -> None:
    out_root = args.output_dir / ("smoke_hard_negatives" if args.smoke else "hard_negatives")
    for mode in args.mine_modes:
        path = out_root / f"{mode}.json"
        if path.exists() and not args.force:
            print(f"skip existing {path}", flush=True)
            continue
        payload = mine_for_mode(args, mode)
        dump_json(path, payload)
        print(f"wrote {path}", flush=True)


def build_train_groups(ctx: dict[str, Any], args: argparse.Namespace) -> list[TrainGroup]:
    groups: list[TrainGroup] = []
    for offset, row in enumerate(ctx["train_rows"]):
        qid = args.train_start + offset
        details = row.get("search_result_details_with_idx")
        if details is None:
            details = []
        positives: list[int] = []
        negatives: list[int] = []
        for detail in details:
            note_id = int(detail["note_idx"])
            if is_positive_detail(detail, POSITIVE_FIELDS):
                positives.append(note_id)
            elif is_negative_detail(detail, args.exposure_negative_policy, POSITIVE_FIELDS):
                negatives.append(note_id)
        if positives:
            groups.append(TrainGroup(qid, str(row.get("query") or ""), positives, negatives))
    return groups


def encode_batch(model, tokenizer, texts: list[str], args: argparse.Namespace):
    import torch

    encoded = tokenizer(texts, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
    encoded = {k: v.to(args.device) for k, v in encoded.items()}
    with torch.cuda.amp.autocast(**autocast_kwargs(args)):
        outputs = model(**encoded)
        pool = last_token_pool if pooling_mode(args) == "last_token" else mean_token_pool
        emb = pool(outputs.last_hidden_state, encoded["attention_mask"])
        if args.normalize_embeddings:
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb


def train_one_config(args: argparse.Namespace) -> None:
    import torch
    from torch.optim import AdamW
    from tqdm import tqdm

    ctx = dataset_context(args, smoke=args.smoke)
    mode = args.train_mode
    hard_path = args.hard_negative_path or args.output_dir / ("smoke_hard_negatives" if args.smoke else "hard_negatives") / f"{mode}.json"
    hard_payload = read_json(hard_path)
    hard_negs: dict[str, list[int]] = hard_payload["hard_negatives"]
    model, tokenizer = load_encoder_model(args.model_name, args.device, args.fp16, args.bf16, tokenizer_padding_side(args))
    model = maybe_apply_lora(model, args)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    groups = build_train_groups(ctx, args)
    train_doc_ids = ctx["train_corpus_ids"]
    valid_doc_ids: list[int] = []
    for note_id in train_doc_ids:
        text = build_note_text(note_id, ctx["notes"], {} if mode == "original" else ctx["train_cards"], mode, "fallback" if mode == "original" else "error", args.max_doc_chars, args.content_head_chars)
        if text:
            valid_doc_ids.append(note_id)
    rng = random.Random(args.seed)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and args.device == "cuda" and not args.bf16)
    loss_fn = torch.nn.CrossEntropyLoss()
    docs_per_query = 1 + args.negative_samples
    run_dir = args.output_dir / ("smoke_runs" if args.smoke else "runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "run_name": args.run_name,
        "train_mode": mode,
        "negative_strategy": args.negative_strategy,
        "lr": args.lr,
        "lora": not args.full_param,
        "train_groups": len(groups),
        "hard_negative_path": str(hard_path),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    dump_json(run_dir / "run_summary.json", {"status": "running", **stats})
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(groups)
        losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(range(0, len(groups), args.train_batch_size), desc=f"train {args.run_name} epoch {epoch}")
        for step_idx, start in enumerate(pbar, start=1):
            batch = groups[start : start + args.train_batch_size]
            queries = [format_query(g.query, args) for g in batch]
            docs: list[str] = []
            for g in batch:
                positive_id = rng.choice(g.positive_ids)
                hard_candidates = [pid for pid in hard_negs.get(str(g.qid), []) if pid != positive_id]
                exposure_candidates = [pid for pid in g.exposure_negative_ids if pid != positive_id]
                if args.negative_strategy == "hard_mined_plus_exposure":
                    pool = hard_candidates + [pid for pid in exposure_candidates if pid not in hard_candidates]
                else:
                    pool = hard_candidates
                if len(pool) < args.negative_samples:
                    fallback = [pid for pid in exposure_candidates + valid_doc_ids if pid != positive_id and pid not in pool]
                    need = args.negative_samples - len(pool)
                    pool = pool + (rng.sample(fallback, min(len(fallback), need)) if fallback else [])
                if len(pool) >= args.negative_samples:
                    neg_ids = rng.sample(pool, args.negative_samples)
                elif pool:
                    neg_ids = pool + rng.choices(pool, k=args.negative_samples - len(pool))
                else:
                    neg_ids = rng.sample([pid for pid in valid_doc_ids if pid != positive_id], args.negative_samples)
                for note_id in [positive_id] + neg_ids:
                    text = build_note_text(note_id, ctx["notes"], {} if mode == "original" else ctx["train_cards"], mode, "fallback" if mode == "original" else "error", args.max_doc_chars, args.content_head_chars)
                    if not text:
                        raise RuntimeError(f"empty text for note {note_id}")
                    docs.append(text)
            emb = encode_batch(model, tokenizer, queries + docs, args)
            q_emb = emb[: len(queries)]
            d_emb = emb[len(queries) :]
            with torch.cuda.amp.autocast(**autocast_kwargs(args)):
                scores = (q_emb @ d_emb.T) / args.temperature
                labels = torch.arange(len(queries), device=args.device, dtype=torch.long) * docs_per_query
                loss = loss_fn(scores, labels) / args.grad_accum_steps
            if args.fp16 and args.device == "cuda" and not args.bf16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if step_idx % args.grad_accum_steps == 0 or start + args.train_batch_size >= len(groups):
                if args.fp16 and args.device == "cuda" and not args.bf16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            value = float(loss.detach().cpu()) * args.grad_accum_steps
            losses.append(value)
            pbar.set_postfix({"loss": f"{value:.4f}"})
        stats[f"epoch_{epoch}_loss"] = float(sum(losses) / max(1, len(losses)))
        dump_json(run_dir / "run_summary.json", {"status": "trained", **stats})
    model.save_pretrained(run_dir / "model")
    tokenizer.save_pretrained(run_dir / "model")
    eval_rows: list[dict[str, Any]] = []
    for split in args.eval_splits:
        eval_modes = args.eval_modes if args.eval_all_modes else [mode]
        for eval_mode in eval_modes:
            if split == "val":
                metrics = evaluate_expanded(model, tokenizer, ctx["val_rows"], args.val_start, ctx["val_qrels"], ctx["train_corpus_ids"], ctx["notes"], ctx["train_cards"], eval_mode, args, run_dir / "eval" / split / eval_mode, split)
            elif split == "test":
                metrics = evaluate_expanded(model, tokenizer, ctx["test_rows"], args.test_start, ctx["test_qrels"], ctx["test_corpus_ids"], ctx["notes"], ctx["test_cards"], eval_mode, args, run_dir / "eval" / split / eval_mode, split)
            else:
                raise ValueError(split)
            eval_rows.append({"split": split, "eval_mode": eval_mode, **{k: metrics.get(k, 0.0) for k in METRIC_KEYS}, "valid_queries": metrics["_valid_queries"]})
            dump_json(run_dir / "run_summary.json", {"status": "evaluating", **stats, "eval": eval_rows})
    dump_json(run_dir / "run_summary.json", {"status": "completed", **stats, "eval": eval_rows})
    del model, tokenizer
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()


def stage_aggregate(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    zero_root = args.output_dir / "zero_shot"
    if zero_root.exists():
        for path in sorted(zero_root.glob("*/*/*_metrics.json")):
            metrics = read_json(path)
            split = path.name.replace("_metrics.json", "")
            mode = path.parent.name
            rows.append({"stage": "zero_shot", "run_name": "zero_shot", "split": split, "train_mode": "zero_shot", "eval_mode": mode, **{k: metrics.get(k, 0.0) for k in METRIC_KEYS}, "valid_queries": metrics.get("_valid_queries", 0.0)})
    for summary in sorted((args.output_dir / "runs").glob("*/run_summary.json")):
        payload = read_json(summary)
        for item in payload.get("eval", []):
            rows.append({
                "stage": "lora" if payload.get("lora", True) else "full_param",
                "run_name": payload.get("run_name", summary.parent.name),
                "split": item["split"],
                "train_mode": payload.get("train_mode", ""),
                "eval_mode": item["eval_mode"],
                "negative_strategy": payload.get("negative_strategy", ""),
                "lr": payload.get("lr", ""),
                **{k: item.get(k, 0.0) for k in METRIC_KEYS},
                "valid_queries": item.get("valid_queries", 0.0),
            })
    if rows:
        out = args.output_dir / "summary_table.csv"
        fields = sorted({k for row in rows for k in row})
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        dump_json(args.output_dir / "summary_table.json", {"rows": rows})
        val_rows = [r for r in rows if r.get("split") == "val"]
        val_rows.sort(key=lambda r: float(r.get("MRR@10", 0.0)), reverse=True)
        lines = [
            "# Expanded-Pool Dense Retrieval Summary (validation, sorted by MRR@10)",
            "",
            "| stage | run | train | eval | neg | lr | MRR@10 | MAP@10 | R@10 |",
            "|---|---|---|---|---|---:|---:|---:|---:|",
        ]
        for r in val_rows[:30]:
            lines.append(f"| {r.get('stage','')} | {r.get('run_name','')} | {r.get('train_mode','')} | {r.get('eval_mode','')} | {r.get('negative_strategy','')} | {r.get('lr','')} | {float(r.get('MRR@10',0.0)):.6f} | {float(r.get('MAP@10',0.0)):.6f} | {float(r.get('Recall@10',0.0)):.6f} |")
        (args.output_dir / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {out}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expanded-pool dense retrieval: zero-shot, hard-negative mining, fine-tuning.")
    parser.add_argument("--stage", choices=["preflight", "zero_shot", "mine", "train", "aggregate"], required=True)
    parser.add_argument("--qilin_dir", type=Path, default=DEFAULT_QILIN_DIR)
    parser.add_argument("--qrels_path", type=Path, default=DEFAULT_QRELS_PATH)
    parser.add_argument("--train_notecard_path", type=Path, default=DEFAULT_TRAIN_NOTECARD)
    parser.add_argument("--test_notecard_path", type=Path, default=DEFAULT_TEST_NOTECARD)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model_name", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train_start", type=int, default=0)
    parser.add_argument("--train_end", type=int, default=4000)
    parser.add_argument("--val_start", type=int, default=4000)
    parser.add_argument("--val_end", type=int, default=5000)
    parser.add_argument("--test_start", type=int, default=0)
    parser.add_argument("--test_end", type=int, default=1000)
    parser.add_argument("--train_corpus_start", type=int, default=0)
    parser.add_argument("--train_corpus_end", type=int, default=5000)
    parser.add_argument("--test_corpus_start", type=int, default=0)
    parser.add_argument("--test_corpus_end", type=int, default=6192)
    parser.add_argument("--test_candidate_key", default="search_results")
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--mine_top_k", type=int, default=200)
    parser.add_argument("--hard_negative_k", type=int, default=32)
    parser.add_argument("--max_doc_chars", type=int, default=0)
    parser.add_argument("--content_head_chars", type=int, default=300)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--score_block_size", type=int, default=16)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--normalize_embeddings", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke_train_queries", type=int, default=20)
    parser.add_argument("--smoke_val_queries", type=int, default=20)
    parser.add_argument("--smoke_corpus_rows", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--eval_splits", nargs="+", choices=["val", "test"], default=["val"])
    parser.add_argument("--eval_modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--mine_modes", nargs="+", choices=MODES, default=["original", "title_summary", "combined"])
    parser.add_argument("--train_mode", choices=MODES, default="title_summary")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--hard_negative_path", type=Path, default=None)
    parser.add_argument("--negative_strategy", choices=["hard_mined_only", "hard_mined_plus_exposure"], default="hard_mined_only")
    parser.add_argument("--exposure_negative_policy", choices=["non_positive", "click_zero", "click_not_one", "qilin_official_nonzero"], default="non_positive")
    parser.add_argument("--negative_samples", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--full_param", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--eval_all_modes", action="store_true")
    parser.add_argument(
        "--qwen_official_style",
        action="store_true",
        help="Qwen3-Embedding usage: instruction-prefixed queries, last-token pooling, left padding.",
    )
    parser.add_argument("--qwen_query_instruction", default=DEFAULT_QWEN_QUERY_INSTRUCTION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.run_name:
        suffix = "fullparam" if args.full_param else "lora"
        args.run_name = f"{args.train_mode}_{args.negative_strategy}_lr{args.lr:g}_{suffix}"
        if args.smoke:
            args.run_name = "smoke_" + args.run_name
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.stage == "preflight":
        stage_preflight(args)
    elif args.stage == "zero_shot":
        stage_zero_shot(args)
    elif args.stage == "mine":
        stage_mine(args)
    elif args.stage == "train":
        train_one_config(args)
    elif args.stage == "aggregate":
        stage_aggregate(args)
    else:
        raise ValueError(args.stage)


if __name__ == "__main__":
    main()
