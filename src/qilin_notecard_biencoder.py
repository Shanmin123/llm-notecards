#!/usr/bin/env python3
"""
Shared utilities for the note-card experiments, plus a Qilin-style
exposure-pool bi-encoder runner.

Every script in src/ imports the helpers defined here:
  - Qilin parquet loading (search_train / search_test / notes)
  - note-card loading and the document representations (build_note_text)
  - official qrels loading and MRR / MAP / Recall@k (calculate_metrics)

Representation names in the code vs. the paper:
  original -> Original, summary -> NoteCard, combined -> StructuredConcat.
  title_summary and the *_content_head modes are extra ablations that are not
  reported in the paper.

When run directly, this script keeps the Qilin baseline dense-retrieval shape:
  - one shared AutoModel encoder for queries and notes
  - mean token pooling
  - dot-product scoring inside a candidate pool
  - official search.test.qrels.csv metrics for search_test

Default split:
  - train: search_train[0:4000]
  - val:   search_train[4000:5000]
  - test:  search_test[0:1000]

Text modes:
  - original: note_title + note_content                        (paper: Original)
  - summary: NoteCard only                                      (paper: NoteCard)
  - combined: NoteCard + original note text                     (paper: StructuredConcat)
  - title_summary: note_title + NoteCard                        (extra ablation)
  - title_summary_content_head: note_title + NoteCard + a short note_content head
  - title_content_head: note_title + a short note_content head, no NoteCard

Note cards: data/train_notecard.json covers the candidates of
search_train[0:5000]; data/test_notecard.json covers search_test[0:6192].
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


POSITIVE_FIELDS = ("click", "collect", "like", "comment", "share")
TEXT_MODES = (
    "original",
    "summary",
    "title_summary",
    "title_summary_content_head",
    "title_content_head",
    "combined",
)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_QILIN_DIR = DEFAULT_DATA_DIR  # holds notes/, search_train/, search_test/
DEFAULT_QRELS_PATH = DEFAULT_DATA_DIR / "qrels" / "search.test.qrels.csv"
DEFAULT_TRAIN_NOTECARD = DEFAULT_DATA_DIR / "train_notecard.json"
DEFAULT_TEST_NOTECARD = DEFAULT_DATA_DIR / "test_notecard.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs"


def text_or_empty(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def load_notecards(path: Path | None) -> dict[int, str]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items() if str(v).strip()}


def load_search_slice(qilin_dir: Path, split_name: str, start: int, end: int) -> pd.DataFrame:
    path = qilin_dir / split_name / "train-00000-of-00001.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_parquet(path).iloc[start:end].reset_index(drop=True)


def parse_ranked_candidates(values: Any, top_k: int | None = None) -> list[int]:
    if values is None:
        return []
    values = list(values)
    if not values:
        return []

    first = values[0]
    if isinstance(first, (int, np.integer)):
        out = [int(x) for x in values]
        return out[:top_k] if top_k is not None else out

    pairs: list[tuple[int, float]] = []
    for item in values:
        if isinstance(item, (int, np.integer)):
            pairs.append((int(item), 0.0))
            continue
        try:
            pid = int(item[0])
            score = float(item[1]) if len(item) > 1 else 0.0
        except Exception:
            continue
        pairs.append((pid, score))
    pairs.sort(key=lambda x: x[1], reverse=True)
    if top_k is not None:
        pairs = pairs[:top_k]
    return [pid for pid, _ in pairs]


def candidate_ids_from_row(row: dict[str, Any], candidate_key: str, depth: int | None) -> list[int]:
    if candidate_key in {"details", "search_result_details_with_idx"}:
        details = row.get("search_result_details_with_idx")
        if details is None:
            details = []
        ids = [int(item["note_idx"]) for item in details]
        return ids[:depth] if depth is not None else ids
    return parse_ranked_candidates(row.get(candidate_key), depth)


def collect_candidate_map(
    rows: list[dict[str, Any]],
    candidate_key: str,
    depth: int | None,
    qid_start: int,
) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for local_qid, row in enumerate(rows):
        qid = qid_start + local_qid
        ids = candidate_ids_from_row(row, candidate_key, depth)
        seen: set[int] = set()
        deduped: list[int] = []
        for note_id in ids:
            if note_id not in seen:
                deduped.append(note_id)
                seen.add(note_id)
        out[qid] = deduped
    return out


def is_positive_detail(detail: dict[str, Any], positive_fields: Iterable[str]) -> bool:
    for field in positive_fields:
        try:
            if float(detail.get(field, 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def qrels_from_details(
    rows: list[dict[str, Any]],
    qid_start: int,
    positive_fields: Iterable[str] = POSITIVE_FIELDS,
) -> dict[int, set[int]]:
    qrels: dict[int, set[int]] = {}
    for local_qid, row in enumerate(rows):
        rels: set[int] = set()
        details = row.get("search_result_details_with_idx")
        if details is None:
            details = []
        for detail in details:
            if is_positive_detail(detail, positive_fields):
                rels.add(int(detail["note_idx"]))
        if rels:
            qrels[qid_start + local_qid] = rels
    return qrels


def load_qrels(path: Path, start: int, end: int) -> dict[int, set[int]]:
    qrels: dict[int, set[int]] = {}
    if not path.exists():
        return qrels
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = int(row["qid"])
            if start <= qid < end:
                qrels.setdefault(qid, set()).add(int(row["pid"]))
    return qrels


def calculate_metrics(
    sorted_results: dict[int, list[int]],
    qrels: dict[int, set[int]],
    k_list: list[int],
) -> dict[str, float]:
    max_k = max(k_list)
    agg = {k: {"mrr": 0.0, "map": 0.0, "recall": 0.0, "precision": 0.0} for k in k_list}
    valid = 0

    for qid, relevant in qrels.items():
        ranked = sorted_results.get(qid)
        if not ranked:
            continue
        valid += 1
        hits = [pid in relevant for pid in ranked[:max_k]]
        for k in k_list:
            hits_k = hits[:k]
            if any(hits_k):
                agg[k]["mrr"] += 1.0 / (hits_k.index(True) + 1)

            ap = 0.0
            correct = 0
            for rank, is_rel in enumerate(hits_k, start=1):
                if is_rel:
                    correct += 1
                    ap += correct / rank
            if correct:
                agg[k]["map"] += ap / min(len(relevant), k)
            agg[k]["recall"] += sum(hits_k) / len(relevant)
            agg[k]["precision"] += sum(hits_k) / k

    out: dict[str, float] = {"_valid_queries": float(valid)}
    for k in k_list:
        for name in ("MRR", "MAP", "Recall", "Precision"):
            out[f"{name}@{k}"] = 0.0
    if valid == 0:
        return out

    for k in k_list:
        out[f"MRR@{k}"] = agg[k]["mrr"] / valid
        out[f"MAP@{k}"] = agg[k]["map"] / valid
        out[f"Recall@{k}"] = agg[k]["recall"] / valid
        out[f"Precision@{k}"] = agg[k]["precision"] / valid
    return out


def coverage_stats(candidate_map: dict[int, list[int]], cards: dict[int, str]) -> dict[str, float]:
    candidate_ids = {pid for ids in candidate_map.values() for pid in ids}
    covered = candidate_ids & set(cards)
    total = len(candidate_ids)
    missing = total - len(covered)
    return {
        "unique_candidates": float(total),
        "notecard_covered": float(len(covered)),
        "notecard_missing": float(missing),
        "notecard_coverage": float(len(covered) / total) if total else 0.0,
    }


def load_notes_subset(qilin_dir: Path, note_ids: set[int]) -> dict[int, dict[str, str]]:
    notes_dir = qilin_dir / "notes"
    if not notes_dir.exists():
        raise FileNotFoundError(f"Missing notes dir: {notes_dir}")

    columns = ["note_idx", "note_title", "note_content"]
    selected: dict[int, dict[str, str]] = {}
    for path in sorted(notes_dir.glob("*.parquet")):
        df = pd.read_parquet(path, columns=columns)
        df = df[df["note_idx"].isin(note_ids)]
        for row in df.itertuples(index=False):
            note_idx = int(row.note_idx)
            selected[note_idx] = {
                "note_title": text_or_empty(row.note_title),
                "note_content": text_or_empty(row.note_content),
            }
        if len(selected) == len(note_ids):
            break
    return selected


def build_note_text(
    note_id: int,
    notes: dict[int, dict[str, str]],
    cards: dict[int, str],
    mode: str,
    missing_policy: str,
    max_doc_chars: int,
    content_head_chars: int = 300,
) -> str | None:
    note = notes.get(note_id)
    if note is None:
        return None

    title = text_or_empty(note.get("note_title"))
    content = text_or_empty(note.get("note_content"))
    content_head = content[:content_head_chars].strip() if content_head_chars > 0 else ""
    original = "\n".join(part for part in (title, content) if part).strip()
    if mode == "original":
        text = original
    elif mode == "title_content_head":
        text = "\n".join(
            part
            for part in (
                f"Title: {title}" if title else "",
                f"Content: {content_head}" if content_head else "",
            )
            if part
        )
    else:
        card = text_or_empty(cards.get(note_id))
        if not card:
            if missing_policy == "drop":
                return None
            if missing_policy == "error":
                raise KeyError(f"Missing notecard for note_idx={note_id}")
            text = original
        elif mode == "summary":
            text = card
        elif mode == "title_summary":
            text = "\n".join(
                part
                for part in (
                    f"Title: {title}" if title else "",
                    "[Notecard]",
                    card,
                )
                if part
            )
        elif mode == "title_summary_content_head":
            text = "\n".join(
                part
                for part in (
                    f"Title: {title}" if title else "",
                    "[Notecard]",
                    card,
                    "[Content Head]",
                    content_head,
                )
                if part
            )
        elif mode == "combined":
            text = "\n".join(
                part
                for part in (
                    "[Notecard]",
                    card,
                    "[Original Note]",
                    f"Title: {title}" if title else "",
                    f"Content: {content}" if content else "",
                )
                if part
            )
        else:
            raise ValueError(f"Unsupported text mode: {mode}")

    if max_doc_chars > 0:
        text = text[:max_doc_chars]
    return text or None


@dataclass
class TrainExample:
    query: str
    positive_id: int
    negative_ids: list[int]


@dataclass
class TrainGroup:
    query: str
    positive_ids: list[int]
    negative_ids: list[int]


def build_train_examples(
    rows: list[dict[str, Any]],
    notes: dict[int, dict[str, str]],
    cards: dict[int, str],
    mode: str,
    missing_policy: str,
    max_doc_chars: int,
    content_head_chars: int,
    negative_samples: int,
    positive_fields: Iterable[str],
    seed: int,
) -> list[TrainExample]:
    rng = random.Random(seed)
    examples: list[TrainExample] = []
    for row in rows:
        positives: list[int] = []
        negatives: list[int] = []
        details = row.get("search_result_details_with_idx")
        if details is None:
            details = []
        for detail in details:
            note_id = int(detail["note_idx"])
            text = build_note_text(note_id, notes, cards, mode, missing_policy, max_doc_chars, content_head_chars)
            if text is None:
                continue
            if is_positive_detail(detail, positive_fields):
                positives.append(note_id)
            else:
                negatives.append(note_id)

        if not positives or not negatives:
            continue
        rng.shuffle(negatives)
        for positive_id in positives:
            if len(negatives) >= negative_samples:
                sampled_negatives = rng.sample(negatives, negative_samples)
            else:
                sampled_negatives = [rng.choice(negatives) for _ in range(negative_samples)]
            examples.append(TrainExample(str(row.get("query") or ""), positive_id, sampled_negatives))
    rng.shuffle(examples)
    return examples


def click_value(detail: dict[str, Any]) -> float:
    try:
        return float(detail.get("click", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def is_negative_detail(
    detail: dict[str, Any],
    negative_policy: str,
    positive_fields: Iterable[str],
) -> bool:
    click = click_value(detail)
    if negative_policy == "click_zero":
        return click == 0
    if negative_policy == "click_not_one":
        return click != 1
    if negative_policy == "qilin_official_nonzero":
        return click != 0
    if negative_policy == "non_positive":
        return not is_positive_detail(detail, positive_fields)
    raise ValueError(f"Unsupported negative_policy: {negative_policy}")


def build_qilin_train_groups(
    rows: list[dict[str, Any]],
    notes: dict[int, dict[str, str]],
    cards: dict[int, str],
    mode: str,
    missing_policy: str,
    max_doc_chars: int,
    content_head_chars: int,
    negative_policy: str,
    positive_fields: Iterable[str],
) -> list[TrainGroup]:
    groups: list[TrainGroup] = []
    for row in rows:
        positives: list[int] = []
        negatives: list[int] = []
        details = row.get("search_result_details_with_idx")
        if details is None:
            details = []
        for detail in details:
            note_id = int(detail["note_idx"])
            text = build_note_text(note_id, notes, cards, mode, missing_policy, max_doc_chars, content_head_chars)
            if text is None:
                continue
            if click_value(detail) == 1:
                positives.append(note_id)
            if is_negative_detail(detail, negative_policy, positive_fields):
                negatives.append(note_id)

        if positives:
            groups.append(TrainGroup(str(row.get("query") or ""), positives, negatives))
    return groups


def mean_token_pool(last_hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_states * mask).sum(dim=1)
    lengths = attention_mask.sum(dim=1).clamp(min=1).unsqueeze(1)
    return summed / lengths


def last_token_pool(last_hidden_states, attention_mask):
    """Qwen3-Embedding pooling: hidden state of the last non-padding token."""
    import torch

    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def load_encoder(model_name: str, tokenizer_name: str | None, device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok_name = tokenizer_name or model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"

    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.to(torch.device(device))
    return model, tokenizer


def encode_texts(
    model,
    tokenizer,
    texts: list[str],
    device: str,
    batch_size: int,
    max_length: int,
    normalize: bool,
    fp16: bool = False,
    pooling: str = "mean",
) -> np.ndarray:
    import torch

    pool = last_token_pool if pooling == "last_token" else mean_token_pool
    model.eval()
    arrays: list[np.ndarray] = []
    use_amp = fp16 and device == "cuda"
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**encoded)
                emb = pool(outputs.last_hidden_state, encoded["attention_mask"])
            if normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            arrays.append(emb.detach().float().cpu().numpy().astype(np.float32))
    if not arrays:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(arrays, axis=0)


def create_optimizer(model, args: argparse.Namespace):
    from torch.optim import AdamW

    if args.optimizer == "adamw":
        return AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "lamb":
        try:
            import torch_optimizer as optim
        except ImportError as exc:
            raise ImportError("optimizer=lamb requires `pip install torch_optimizer`.") from exc
        return optim.Lamb(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def create_scheduler(optimizer, args: argparse.Namespace):
    if args.scheduler == "none":
        return None
    if args.scheduler == "linear":
        import torch

        return torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=args.linear_total_iters)
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def sample_with_fallback(
    rng: random.Random,
    primary: list[int],
    fallback: list[int],
    n: int,
    exclude: set[int],
) -> list[int]:
    selected: list[int] = []
    if len(primary) >= n:
        return rng.sample(primary, n)
    selected.extend(primary)
    candidates = [pid for pid in fallback if pid not in exclude and pid not in selected]
    need = n - len(selected)
    if need <= 0:
        return selected
    if len(candidates) >= need:
        selected.extend(rng.sample(candidates, need))
    elif candidates:
        selected.extend(rng.choices(candidates, k=need))
    elif primary:
        selected.extend(rng.choices(primary, k=need))
    return selected


def train_one_mode(
    model,
    tokenizer,
    train_examples: list[TrainExample],
    notes: dict[int, dict[str, str]],
    cards: dict[int, str],
    args: argparse.Namespace,
    mode: str,
) -> dict[str, float]:
    import torch
    from tqdm import tqdm

    if not train_examples:
        raise RuntimeError(f"No train examples for mode={mode}. Check notecard coverage and missing policy.")

    device = torch.device(args.device)
    use_amp = args.fp16 and args.device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args)
    loss_fn = torch.nn.CrossEntropyLoss()
    docs_per_query = 1 + args.negative_samples
    stats: dict[str, float] = {"train_examples": float(len(train_examples))}

    for epoch in range(1, args.epochs + 1):
        rng = random.Random(args.seed + epoch)
        rng.shuffle(train_examples)
        model.train()
        losses: list[float] = []
        pbar = tqdm(range(0, len(train_examples), args.train_batch_size), desc=f"train {mode} epoch {epoch}")
        for start in pbar:
            batch = train_examples[start : start + args.train_batch_size]
            queries = [item.query for item in batch]
            docs: list[str] = []
            for item in batch:
                ids = [item.positive_id] + item.negative_ids
                for note_id in ids:
                    text = build_note_text(
                        note_id,
                        notes,
                        cards,
                        mode,
                        args.missing_policy,
                        args.max_doc_chars,
                        args.content_head_chars,
                    )
                    if text is None:
                        raise RuntimeError(f"Unexpected missing text during training: {note_id}")
                    docs.append(text)

            merged_texts = queries + docs
            encoded = tokenizer(
                merged_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**encoded)
                emb = mean_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
                if args.normalize_embeddings:
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                query_emb = emb[: len(queries)]
                doc_emb = emb[len(queries) :]

                scores = torch.matmul(query_emb, doc_emb.transpose(0, 1)) / args.temperature
                labels = torch.arange(len(queries), device=device, dtype=torch.long) * docs_per_query
                loss = loss_fn(scores, labels)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

            losses.append(float(loss.detach().cpu()))
            pbar.set_postfix({"loss": f"{losses[-1]:.4f}"})

        stats[f"epoch_{epoch}_train_loss"] = float(sum(losses) / max(len(losses), 1))
    return stats


def train_one_mode_qilin(
    model,
    tokenizer,
    train_groups: list[TrainGroup],
    notes: dict[int, dict[str, str]],
    cards: dict[int, str],
    args: argparse.Namespace,
    mode: str,
) -> dict[str, float]:
    import torch
    from tqdm import tqdm

    if not train_groups:
        raise RuntimeError(f"No train groups for mode={mode}. Check clicked positives and notecard coverage.")

    device = torch.device(args.device)
    use_amp = args.fp16 and args.device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args)
    loss_fn = torch.nn.CrossEntropyLoss()
    docs_per_query = 1 + args.negative_samples
    fallback_note_ids = []
    for note_id in sorted(notes):
        try:
            text = build_note_text(
                note_id,
                notes,
                cards,
                mode,
                args.missing_policy,
                args.max_doc_chars,
                args.content_head_chars,
            )
        except KeyError:
            text = None
        if text is not None:
            fallback_note_ids.append(note_id)
    stats: dict[str, float] = {
        "train_groups": float(len(train_groups)),
        "fallback_note_ids": float(len(fallback_note_ids)),
        "training_style": "qilin",
        "negative_policy": args.negative_policy,
    }

    for epoch in range(1, args.epochs + 1):
        rng = random.Random(args.seed + epoch)
        groups = list(train_groups)
        rng.shuffle(groups)
        model.train()
        losses: list[float] = []
        pbar = tqdm(range(0, len(groups), args.train_batch_size), desc=f"train {mode} epoch {epoch} qilin")
        for start in pbar:
            batch = groups[start : start + args.train_batch_size]
            queries = [item.query for item in batch]
            docs: list[str] = []
            for item in batch:
                positive_id = rng.choice(item.positive_ids)
                negative_ids = sample_with_fallback(
                    rng,
                    item.negative_ids,
                    fallback_note_ids,
                    args.negative_samples,
                    exclude={positive_id},
                )
                ids = [positive_id] + negative_ids
                for note_id in ids:
                    text = build_note_text(
                        note_id,
                        notes,
                        cards,
                        mode,
                        args.missing_policy,
                        args.max_doc_chars,
                        args.content_head_chars,
                    )
                    if text is None:
                        raise RuntimeError(f"Unexpected missing text during Qilin-style training: {note_id}")
                    docs.append(text)

            merged_texts = queries + docs
            encoded = tokenizer(
                merged_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**encoded)
                emb = mean_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
                if args.normalize_embeddings:
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                query_emb = emb[: len(queries)]
                doc_emb = emb[len(queries) :]
                scores = torch.matmul(query_emb, doc_emb.transpose(0, 1)) / args.temperature
                labels = torch.arange(len(queries), device=device, dtype=torch.long) * docs_per_query
                loss = loss_fn(scores, labels)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

            losses.append(float(loss.detach().cpu()))
            pbar.set_postfix({"loss": f"{losses[-1]:.4f}"})

        stats[f"epoch_{epoch}_train_loss"] = float(sum(losses) / max(len(losses), 1))
    return stats


def evaluate_mode(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    qrels: dict[int, set[int]],
    candidate_map: dict[int, list[int]],
    notes: dict[int, dict[str, str]],
    cards: dict[int, str],
    args: argparse.Namespace,
    mode: str,
    split_label: str,
) -> tuple[dict[str, float], dict[int, list[int]]]:
    note_ids = sorted({pid for ids in candidate_map.values() for pid in ids})
    doc_ids: list[int] = []
    doc_texts: list[str] = []
    for note_id in note_ids:
        text = build_note_text(
            note_id,
            notes,
            cards,
            mode,
            args.missing_policy,
            args.max_doc_chars,
            args.content_head_chars,
        )
        if text is not None:
            doc_ids.append(note_id)
            doc_texts.append(text)

    doc_index = {pid: i for i, pid in enumerate(doc_ids)}
    queries = [str(row.get("query") or "") for row in rows]
    qids = sorted(candidate_map.keys())

    t0 = time.time()
    query_emb = encode_texts(
        model,
        tokenizer,
        queries,
        args.device,
        args.eval_batch_size,
        args.max_length,
        args.normalize_embeddings,
        args.fp16,
    )
    doc_emb = encode_texts(
        model,
        tokenizer,
        doc_texts,
        args.device,
        args.eval_batch_size,
        args.max_length,
        args.normalize_embeddings,
        args.fp16,
    )

    rankings: dict[int, list[int]] = {}
    for local_idx, qid in enumerate(qids):
        candidates = [pid for pid in candidate_map[qid] if pid in doc_index]
        if not candidates:
            rankings[qid] = []
            continue
        idx = np.asarray([doc_index[pid] for pid in candidates], dtype=np.int64)
        scores = query_emb[local_idx] @ doc_emb[idx].T
        order = np.argsort(-scores)[: args.top_k]
        rankings[qid] = [candidates[i] for i in order]

    metrics = calculate_metrics(rankings, qrels, [10, args.top_k])
    metrics.update(
        {
            "queries": float(len(rows)),
            "unique_docs_encoded": float(len(doc_ids)),
            "eval_seconds": float(time.time() - t0),
        }
    )
    metrics.update({f"{split_label}_{k}": v for k, v in coverage_stats(candidate_map, cards).items()})
    return metrics, rankings


def write_rankings(path: Path, rankings: dict[int, list[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "pid", "rank"])
        for qid in sorted(rankings):
            for rank, pid in enumerate(rankings[qid], start=1):
                writer.writerow([qid, pid, rank])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qilin baseline-style bi-encoder notecard experiments.")
    parser.add_argument("--qilin_dir", type=Path, default=DEFAULT_QILIN_DIR)
    parser.add_argument("--qrels_path", type=Path, default=DEFAULT_QRELS_PATH)
    parser.add_argument("--train_notecard_path", type=Path, default=DEFAULT_TRAIN_NOTECARD)
    parser.add_argument("--test_notecard_path", type=Path, default=DEFAULT_TEST_NOTECARD)
    parser.add_argument("--reuse_test_notecard_for_train", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "biencoder_exposure_pool")

    parser.add_argument("--train_start", type=int, default=0)
    parser.add_argument("--train_end", type=int, default=4000)
    parser.add_argument("--val_start", type=int, default=4000)
    parser.add_argument("--val_end", type=int, default=5000)
    parser.add_argument("--test_start", type=int, default=0)
    parser.add_argument("--test_end", type=int, default=1000)

    parser.add_argument("--candidate_key", default="search_results", help="search_results, dpr_results, bm25_results, or details")
    parser.add_argument("--candidate_depth", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=TEXT_MODES,
        default=["original", "title_summary", "combined"],
        help="Text modes. summary is pure NoteCard ablation; title_summary is the main NoteCard mode.",
    )
    parser.add_argument("--missing_policy", choices=["fallback", "drop", "error"], default="fallback")
    parser.add_argument("--max_doc_chars", type=int, default=0)
    parser.add_argument(
        "--content_head_chars",
        type=int,
        default=300,
        help="Characters of note_content to append for *_content_head modes.",
    )

    parser.add_argument("--model_name", default="google-bert/bert-base-chinese")
    parser.add_argument("--tokenizer_name", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--normalize_embeddings", action="store_true")
    parser.add_argument("--fp16", action="store_true", help="Use CUDA autocast + GradScaler for lower VRAM training/eval.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable model gradient checkpointing during training.")

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument(
        "--training_style",
        choices=["expanded", "qilin"],
        default="expanded",
        help="expanded uses one example per positive; qilin matches Qilin DPR's one-positive-per-query batch construction.",
    )
    parser.add_argument(
        "--negative_policy",
        choices=["non_positive", "click_zero", "click_not_one", "qilin_official_nonzero"],
        default="non_positive",
        help="Negative selection for training_style=qilin. qilin_official_nonzero reproduces the public DPR source line.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--negative_samples", type=int, default=6)
    parser.add_argument("--optimizer", choices=["adamw", "lamb"], default="adamw")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--scheduler", choices=["none", "linear"], default="none")
    parser.add_argument("--linear_total_iters", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positive_fields", nargs="+", default=list(POSITIVE_FIELDS))
    parser.add_argument("--eval_val", action="store_true")
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_df = load_search_slice(args.qilin_dir, "search_train", args.train_start, args.train_end)
    val_df = load_search_slice(args.qilin_dir, "search_train", args.val_start, args.val_end)
    test_df = load_search_slice(args.qilin_dir, "search_test", args.test_start, args.test_end)
    train_rows = train_df.to_dict("records")
    val_rows = val_df.to_dict("records")
    test_rows = test_df.to_dict("records")

    test_cards = load_notecards(args.test_notecard_path)
    train_cards = load_notecards(args.train_notecard_path)
    if args.reuse_test_notecard_for_train and not train_cards:
        train_cards = test_cards

    train_candidate_map = collect_candidate_map(train_rows, "details", None, args.train_start)
    val_candidate_map = collect_candidate_map(val_rows, "details", None, args.val_start)
    test_candidate_map = collect_candidate_map(test_rows, args.candidate_key, args.candidate_depth, args.test_start)

    train_qrels = qrels_from_details(train_rows, args.train_start, args.positive_fields)
    val_qrels = qrels_from_details(val_rows, args.val_start, args.positive_fields)
    test_qrels = load_qrels(args.qrels_path, args.test_start, args.test_end)
    if not test_qrels:
        test_qrels = qrels_from_details(test_rows, args.test_start, args.positive_fields)

    all_needed_ids = set()
    for cmap in (train_candidate_map, val_candidate_map, test_candidate_map):
        all_needed_ids.update(pid for ids in cmap.values() for pid in ids)
    notes = load_notes_subset(args.qilin_dir, all_needed_ids)

    base_summary: dict[str, Any] = {
        "qilin_dir": str(args.qilin_dir),
        "qrels_path": str(args.qrels_path),
        "train_split": [args.train_start, args.train_end],
        "val_split": [args.val_start, args.val_end],
        "test_split": [args.test_start, args.test_end],
        "candidate_key": args.candidate_key,
        "candidate_depth": args.candidate_depth,
        "model_name": args.model_name,
        "missing_policy": args.missing_policy,
        "training_style": args.training_style,
        "negative_policy": args.negative_policy,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "train_notecards": len(train_cards),
        "test_notecards": len(test_cards),
        "notes_loaded": len(notes),
        "coverage": {
            "train_details": coverage_stats(train_candidate_map, train_cards),
            "val_candidates": coverage_stats(val_candidate_map, train_cards),
            "test_candidates": coverage_stats(test_candidate_map, test_cards),
        },
    }

    all_results: dict[str, Any] = {"run": base_summary, "modes": {}}

    for mode in args.modes:
        print(f"\n=== Mode: {mode} ===", flush=True)
        model, tokenizer = load_encoder(args.model_name, args.tokenizer_name, args.device)
        if args.gradient_checkpointing and args.do_train and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        mode_dir = args.output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        mode_payload: dict[str, Any] = {}

        cards_for_train = train_cards if mode != "original" else {}
        cards_for_test = test_cards if mode != "original" else {}

        if args.do_train:
            if args.training_style == "qilin":
                groups = build_qilin_train_groups(
                    train_rows,
                    notes,
                    cards_for_train,
                    mode,
                    args.missing_policy,
                    args.max_doc_chars,
                    args.content_head_chars,
                    args.negative_policy,
                    args.positive_fields,
                )
                mode_payload["train"] = train_one_mode_qilin(model, tokenizer, groups, notes, cards_for_train, args, mode)
            else:
                examples = build_train_examples(
                    train_rows,
                    notes,
                    cards_for_train,
                    mode,
                    args.missing_policy,
                    args.max_doc_chars,
                    args.content_head_chars,
                    args.negative_samples,
                    args.positive_fields,
                    args.seed,
                )
                mode_payload["train"] = train_one_mode(model, tokenizer, examples, notes, cards_for_train, args, mode)
            model.save_pretrained(mode_dir / "model")
            tokenizer.save_pretrained(mode_dir / "model")

        if args.eval_val:
            val_metrics, val_rankings = evaluate_mode(
                model,
                tokenizer,
                val_rows,
                val_qrels,
                val_candidate_map,
                notes,
                cards_for_train,
                args,
                mode,
                "val",
            )
            mode_payload["val_metrics"] = val_metrics
            write_rankings(mode_dir / "val_rankings.csv", val_rankings)

        test_metrics, test_rankings = evaluate_mode(
            model,
            tokenizer,
            test_rows,
            test_qrels,
            test_candidate_map,
            notes,
            cards_for_test,
            args,
            mode,
            "test",
        )
        mode_payload["test_metrics"] = test_metrics
        write_rankings(mode_dir / "test_rankings.csv", test_rankings)

        with (mode_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(mode_payload, f, ensure_ascii=False, indent=2)
        all_results["modes"][mode] = mode_payload
        print(json.dumps(mode_payload, ensure_ascii=False, indent=2), flush=True)

        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    with (args.output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved summary: {args.output_dir / 'run_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
