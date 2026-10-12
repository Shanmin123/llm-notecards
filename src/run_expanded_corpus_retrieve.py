#!/usr/bin/env python3
"""Zero-shot expanded-pool retrieval for the note-card representations.

This runner keeps the query split fixed to search_test[0:1000], builds one
global covered-only corpus from search_test[0:6192].search_results (113,548
notes with the released cards), and retrieves the top-100 from that corpus for
every representation. In the paper it produces the zero-shot BERT bi-encoder
rows of Table 2 (`--models bert_zero`); `bm25` is kept as an optional lexical
reference that is not reported in the paper.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qilin_notecard_biencoder import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_QILIN_DIR,
    DEFAULT_QRELS_PATH,
    DEFAULT_TEST_NOTECARD,
    build_note_text,
    calculate_metrics,
    collect_candidate_map,
    encode_texts,
    load_encoder,
    load_notecards,
    load_notes_subset,
    load_qrels,
    load_search_slice,
    write_rankings,
)

DEFAULT_OUTPUT = DEFAULT_OUTPUT_ROOT / "expanded_retrieval_zero_shot"

MODES = ("original", "summary", "title_summary", "combined")
MODE_LABELS = {
    "original": "Original",
    "summary": "NoteCard-only",
    "title_summary": "Title+NoteCard",
    "combined": "StructuredConcat",
}
METRIC_KEYS = ("MRR@10", "MAP@10", "Recall@10", "MRR@100", "MAP@100", "Recall@100")


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    architecture: str
    model_family: str
    base_model: str
    checkpoint_path: str
    trained_representation: str
    eval_representation: str
    model_path: str | None
    batch_size: int
    max_length: int
    fp16: bool
    normalize_embeddings: bool


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def text_or_path(value: Any) -> Any:
    return str(value) if isinstance(value, Path) else value


def build_corpus_ids(
    rows: list[dict[str, Any]],
    qid_start: int,
    candidate_key: str,
    corpus_depth: int | None,
    cards: dict[int, str],
) -> tuple[list[int], dict[str, int]]:
    candidate_map = collect_candidate_map(rows, candidate_key, corpus_depth, qid_start)
    union_ids = {pid for ids in candidate_map.values() for pid in ids}
    covered = sorted(union_ids & set(cards))
    dropped = sorted(union_ids - set(cards))
    return covered, {
        "full_corpus_union": len(union_ids),
        "covered_only_corpus": len(covered),
        "dropped_missing_notecard_notes": len(dropped),
    }


def build_doc_texts(
    corpus_ids: list[int],
    notes: dict[int, dict[str, str]],
    cards: dict[int, str],
    mode: str,
    max_doc_chars: int,
    content_head_chars: int,
) -> dict[int, str]:
    doc_texts: dict[int, str] = {}
    mode_cards = {} if mode == "original" else cards
    for note_id in corpus_ids:
        text = build_note_text(
            note_id,
            notes,
            mode_cards,
            mode,
            "error" if mode != "original" else "fallback",
            max_doc_chars,
            content_head_chars,
        )
        if text:
            doc_texts[note_id] = text
    return doc_texts


def representation_stats(doc_texts_by_mode: dict[str, dict[int, str]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for mode, doc_texts in doc_texts_by_mode.items():
        lengths = sorted(len(text) for text in doc_texts.values())
        if not lengths:
            out[mode] = {"docs": 0.0}
            continue
        out[mode] = {
            "docs": float(len(lengths)),
            "avg_chars": float(sum(lengths) / len(lengths)),
            "p50_chars": float(lengths[len(lengths) // 2]),
            "p90_chars": float(lengths[min(len(lengths) - 1, int(len(lengths) * 0.9))]),
            "max_chars": float(lengths[-1]),
        }
    return out


def qrels_coverage(qrels: dict[int, set[int]], corpus_ids: set[int]) -> dict[str, float]:
    rel_ids = {pid for rels in qrels.values() for pid in rels}
    covered = rel_ids & corpus_ids
    return {
        "qrels_queries": float(len(qrels)),
        "qrels_unique_relevant_notes": float(len(rel_ids)),
        "qrels_relevant_notes_in_corpus": float(len(covered)),
        "qrels_relevant_notes_missing_from_corpus": float(len(rel_ids - corpus_ids)),
        "qrels_relevant_note_coverage": float(len(covered) / len(rel_ids)) if rel_ids else 0.0,
    }


def load_context(args: argparse.Namespace, query_end: int, corpus_end: int) -> dict[str, Any]:
    cards = load_notecards(args.test_notecard_path)
    query_rows = load_search_slice(args.qilin_dir, "search_test", args.query_start, query_end).to_dict("records")
    corpus_rows = load_search_slice(args.qilin_dir, "search_test", args.corpus_start, corpus_end).to_dict("records")
    corpus_ids, corpus_counts = build_corpus_ids(
        corpus_rows,
        args.corpus_start,
        args.candidate_key,
        args.corpus_depth,
        cards,
    )
    notes = load_notes_subset(args.qilin_dir, set(corpus_ids))
    missing_notes = sorted(set(corpus_ids) - set(notes))
    if missing_notes:
        corpus_ids = [pid for pid in corpus_ids if pid in notes]
    qrels = load_qrels(args.qrels_path, args.query_start, query_end)
    doc_texts_by_mode = {
        mode: build_doc_texts(corpus_ids, notes, cards, mode, args.max_doc_chars, args.content_head_chars)
        for mode in MODES
    }
    query_candidate_map = {args.query_start + i: corpus_ids for i in range(len(query_rows))}
    exposure_rows = load_search_slice(args.qilin_dir, "search_test", args.query_start, query_end).to_dict("records")
    exposure_map = collect_candidate_map(exposure_rows, args.candidate_key, args.top_k, args.query_start)
    exposure_metrics = calculate_metrics(exposure_map, qrels, [10, args.top_k])
    preflight = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "qilin_dir": str(args.qilin_dir),
        "qrels_path": str(args.qrels_path),
        "test_notecard_path": str(args.test_notecard_path),
        "test_notecard_card_count": len(cards),
        "query_range": [args.query_start, query_end],
        "corpus_range": [args.corpus_start, corpus_end],
        "candidate_key": args.candidate_key,
        "corpus_depth": args.corpus_depth,
        "top_k": args.top_k,
        "counts": {
            "queries": len(query_rows),
            "corpus_source_rows": len(corpus_rows),
            **corpus_counts,
            "covered_corpus_after_note_load": len(corpus_ids),
            "missing_note_rows": len(missing_notes),
        },
        "qrels_coverage": qrels_coverage(qrels, set(corpus_ids)),
        "representation_stats": representation_stats(doc_texts_by_mode),
        "exposure_pool_order_metrics_same_query_range": exposure_metrics,
        "args": {k: text_or_path(v) for k, v in vars(args).items()},
    }
    return {
        "cards": cards,
        "query_rows": query_rows,
        "corpus_ids": corpus_ids,
        "notes": notes,
        "qrels": qrels,
        "doc_texts_by_mode": doc_texts_by_mode,
        "query_candidate_map": query_candidate_map,
        "preflight": preflight,
    }


def run_bm25(query_rows: list[dict[str, Any]], doc_texts: dict[int, str], top_k: int, qid_start: int) -> dict[int, list[int]]:
    from rank_bm25 import BM25Okapi

    doc_ids = sorted(doc_texts)
    tokenized = [list(doc_texts[pid]) for pid in doc_ids]
    bm25 = BM25Okapi(tokenized)
    rankings: dict[int, list[int]] = {}
    for local_idx, row in enumerate(query_rows):
        scores = bm25.get_scores(list(str(row.get("query") or "")))
        if len(scores) <= top_k:
            order = np.argsort(-scores)
        else:
            top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
            order = top_idx[np.argsort(-scores[top_idx])]
        rankings[qid_start + local_idx] = [doc_ids[int(i)] for i in order[:top_k]]
    return rankings


def topk_from_embeddings(
    query_emb: np.ndarray,
    doc_emb: np.ndarray,
    doc_ids: list[int],
    qid_start: int,
    top_k: int,
    block_size: int,
) -> dict[int, list[int]]:
    rankings: dict[int, list[int]] = {}
    n_docs = doc_emb.shape[0]
    block_size = max(1, block_size)
    for q_start in range(0, query_emb.shape[0], block_size):
        q_block = query_emb[q_start : q_start + block_size]
        scores = q_block @ doc_emb.T
        k = min(top_k, n_docs)
        if n_docs <= k:
            idx = np.argsort(-scores, axis=1)[:, :k]
        else:
            idx_part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            part_scores = np.take_along_axis(scores, idx_part, axis=1)
            order = np.argsort(-part_scores, axis=1)
            idx = np.take_along_axis(idx_part, order, axis=1)
        for local_offset, row_idx in enumerate(idx):
            rankings[qid_start + q_start + local_offset] = [doc_ids[int(i)] for i in row_idx]
    return rankings


def run_encoder_retrieve(
    spec: RunSpec,
    query_rows: list[dict[str, Any]],
    doc_texts: dict[int, str],
    args: argparse.Namespace,
) -> dict[int, list[int]]:
    import torch

    assert spec.model_path is not None
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model, tokenizer = load_encoder(spec.model_path, None, device)
    doc_ids = sorted(doc_texts)
    doc_inputs = [doc_texts[pid] for pid in doc_ids]
    query_inputs = [str(row.get("query") or "") for row in query_rows]
    t0 = time.time()
    doc_emb = encode_texts(
        model,
        tokenizer,
        doc_inputs,
        device,
        spec.batch_size,
        spec.max_length,
        spec.normalize_embeddings,
        spec.fp16,
    )
    query_emb = encode_texts(
        model,
        tokenizer,
        query_inputs,
        device,
        spec.batch_size,
        spec.max_length,
        spec.normalize_embeddings,
        spec.fp16,
    )
    rankings = topk_from_embeddings(query_emb, doc_emb, doc_ids, args.query_start, args.top_k, args.score_block_size)
    spec_time = time.time() - t0
    del model, tokenizer, doc_emb, query_emb
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    rankings[-1] = [int(math.ceil(spec_time))]
    return rankings


def clean_rankings(rankings: dict[int, list[int]]) -> tuple[dict[int, list[int]], float]:
    elapsed = 0.0
    if -1 in rankings:
        marker = rankings.pop(-1)
        elapsed = float(marker[0]) if marker else 0.0
    return rankings, elapsed


def completed_row_from_disk(run_dir: Path) -> dict[str, Any] | None:
    metrics_path = run_dir / "metrics.json"
    rankings_path = run_dir / "rankings.csv"
    if not metrics_path.exists() or not rankings_path.exists():
        return None
    payload = read_json(metrics_path)
    row = payload.get("row", {})
    if row:
        row["status"] = "skipped_existing"
    return row


def save_success(
    spec: RunSpec,
    metrics: dict[str, float],
    rankings: dict[int, list[int]],
    run_dir: Path,
    started: float,
) -> dict[str, Any]:
    metrics.update(
        {
            "retrieve_seconds": float(time.time() - started),
        }
    )
    rankings_path = run_dir / "rankings.csv"
    metrics_path = run_dir / "metrics.json"
    status_path = run_dir / "status.json"
    write_rankings(rankings_path, rankings)
    row = {
        "status": "completed",
        "architecture": spec.architecture,
        "model_family": spec.model_family,
        "base_model": spec.base_model,
        "trained_representation": spec.trained_representation,
        "eval_representation": spec.eval_representation,
        "trained_representation_label": MODE_LABELS.get(spec.trained_representation, spec.trained_representation),
        "eval_representation_label": MODE_LABELS[spec.eval_representation],
        "checkpoint_path": spec.checkpoint_path,
        "training_note": (
            "trained on search_train[0:4000], retrieved over expanded test corpus"
            if spec.trained_representation != "zero_shot"
            else "zero-shot/no-training, retrieved over expanded test corpus"
        ),
        **{key: metrics.get(key, 0.0) for key in METRIC_KEYS},
        "valid_queries": metrics.get("_valid_queries", 0.0),
        "corpus_docs": metrics["corpus_docs"],
        "retrieve_seconds": metrics["retrieve_seconds"],
    }
    payload = {"row": row, "metrics": metrics, "spec": spec.__dict__}
    dump_json(metrics_path, payload)
    dump_json(
        status_path,
        {
            "status": "completed",
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "spec": spec.__dict__,
        },
    )
    return row


def save_failure(spec: RunSpec, run_dir: Path, error: Exception) -> dict[str, Any]:
    dump_json(
        run_dir / "status.json",
        {
            "status": "failed",
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "spec": spec.__dict__,
            "error": repr(error),
        },
    )
    return {
        "status": "failed",
        "architecture": spec.architecture,
        "model_family": spec.model_family,
        "base_model": spec.base_model,
        "trained_representation": spec.trained_representation,
        "eval_representation": spec.eval_representation,
        "checkpoint_path": spec.checkpoint_path,
        "error": repr(error),
    }


def scan_completed_rows(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("*/*/*/metrics.json")):
        try:
            payload = read_json(path)
        except Exception:
            continue
        row = payload.get("row")
        if isinstance(row, dict) and row:
            rows.append(dict(row))
    return rows


def merge_rows(existing: list[dict[str, Any]], current: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in existing + current:
        key = (
            str(row.get("model_family", "")),
            str(row.get("trained_representation", "")),
            str(row.get("eval_representation", "")),
        )
        if all(key):
            merged[key] = row
    return list(merged.values())


def build_run_specs(args: argparse.Namespace) -> tuple[list[RunSpec], list[dict[str, Any]]]:
    specs: list[RunSpec] = []
    blocked: list[dict[str, Any]] = []
    if "bm25" in args.models:
        for mode in MODES:
            specs.append(
                RunSpec(
                    run_id=f"bm25/zero_shot/{mode}",
                    architecture="Sparse lexical",
                    model_family="bm25",
                    base_model="rank_bm25.BM25Okapi",
                    checkpoint_path="none",
                    trained_representation="zero_shot",
                    eval_representation=mode,
                    model_path=None,
                    batch_size=0,
                    max_length=0,
                    fp16=False,
                    normalize_embeddings=False,
                )
            )
    if "bert_zero" in args.models:
        for mode in MODES:
            specs.append(
                RunSpec(
                    run_id=f"bert_base_chinese_zero_shot/zero_shot/{mode}",
                    architecture="Bi-encoder zero-shot",
                    model_family="bert_base_chinese_zero_shot",
                    base_model="google-bert/bert-base-chinese",
                    checkpoint_path="google-bert/bert-base-chinese",
                    trained_representation="zero_shot",
                    eval_representation=mode,
                    model_path="google-bert/bert-base-chinese",
                    batch_size=args.bert_batch_size,
                    max_length=args.bert_max_length,
                    fp16=args.fp16,
                    normalize_embeddings=args.normalize_embeddings,
                )
            )
    return specs, blocked


def run_one(
    spec: RunSpec,
    context: dict[str, Any],
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_dir = out_dir / spec.run_id
    status_path = run_dir / "status.json"
    if not args.force:
        existing = completed_row_from_disk(run_dir)
        if existing is not None:
            return existing

    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status = {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "spec": spec.__dict__,
    }
    dump_json(status_path, status)
    doc_texts = context["doc_texts_by_mode"][spec.eval_representation]
    try:
        if spec.model_path is None:
            rankings = run_bm25(context["query_rows"], doc_texts, args.top_k, args.query_start)
            encode_seconds = time.time() - started
        else:
            rankings, encode_seconds = clean_rankings(run_encoder_retrieve(spec, context["query_rows"], doc_texts, args))
        metrics = calculate_metrics(rankings, context["qrels"], [10, args.top_k])
        metrics.update(
            {
                "queries": float(len(context["query_rows"])),
                "corpus_docs": float(len(doc_texts)),
                "retrieve_seconds": float(time.time() - started),
                "encode_and_score_seconds": float(encode_seconds),
            }
        )
        return save_success(spec, metrics, rankings, run_dir, started)
    except Exception as exc:
        return save_failure(spec, run_dir, exc)


def run_encoder_group(
    specs: list[RunSpec],
    context: dict[str, Any],
    out_dir: Path,
    args: argparse.Namespace,
    progress_start: int,
    progress_total: int,
) -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    pending: list[RunSpec] = []
    for offset, spec in enumerate(specs):
        print(f"[{progress_start + offset}/{progress_total}] {spec.run_id}", flush=True)
        run_dir = out_dir / spec.run_id
        if not args.force:
            existing = completed_row_from_disk(run_dir)
            if existing is not None:
                rows.append(existing)
                continue
        run_dir.mkdir(parents=True, exist_ok=True)
        dump_json(
            run_dir / "status.json",
            {
                "status": "running",
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "spec": spec.__dict__,
            },
        )
        pending.append(spec)
    if not pending:
        return rows

    first = pending[0]
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = None
    tokenizer = None
    query_emb: np.ndarray | None = None
    try:
        assert first.model_path is not None
        model, tokenizer = load_encoder(first.model_path, None, device)
        query_inputs = [str(row.get("query") or "") for row in context["query_rows"]]
        query_emb = encode_texts(
            model,
            tokenizer,
            query_inputs,
            device,
            first.batch_size,
            first.max_length,
            first.normalize_embeddings,
            first.fp16,
        )
        for spec in pending:
            started = time.time()
            run_dir = out_dir / spec.run_id
            try:
                doc_texts = context["doc_texts_by_mode"][spec.eval_representation]
                doc_ids = sorted(doc_texts)
                doc_emb = encode_texts(
                    model,
                    tokenizer,
                    [doc_texts[pid] for pid in doc_ids],
                    device,
                    spec.batch_size,
                    spec.max_length,
                    spec.normalize_embeddings,
                    spec.fp16,
                )
                rankings = topk_from_embeddings(query_emb, doc_emb, doc_ids, args.query_start, args.top_k, args.score_block_size)
                metrics = calculate_metrics(rankings, context["qrels"], [10, args.top_k])
                metrics.update(
                    {
                        "queries": float(len(context["query_rows"])),
                        "corpus_docs": float(len(doc_texts)),
                        "encode_and_score_seconds": float(time.time() - started),
                    }
                )
                rows.append(save_success(spec, metrics, rankings, run_dir, started))
                del doc_emb
                gc.collect()
            except Exception as exc:
                rows.append(save_failure(spec, run_dir, exc))
    except Exception as exc:
        for spec in pending:
            rows.append(save_failure(spec, out_dir / spec.run_id, exc))
    finally:
        del model, tokenizer, query_emb
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return rows


def write_summary(out_dir: Path, rows: list[dict[str, Any]], blocked: list[dict[str, Any]], preflight: dict[str, Any]) -> None:
    rows = merge_rows(scan_completed_rows(out_dir), rows)
    completed = [row for row in rows if row.get("status") in {"completed", "skipped_existing"}]
    baseline_by_family: dict[str, dict[str, Any]] = {}
    same_ckpt_original: dict[tuple[str, str], dict[str, Any]] = {}
    for row in completed:
        if row["trained_representation"] in {"zero_shot", "original"} and row["eval_representation"] == "original":
            baseline_by_family.setdefault(row["model_family"], row)
        if row["eval_representation"] == "original":
            same_ckpt_original[(row["model_family"], row["trained_representation"])] = row
    for row in completed:
        family_base = baseline_by_family.get(row["model_family"])
        ckpt_base = same_ckpt_original.get((row["model_family"], row["trained_representation"]))
        for key in METRIC_KEYS:
            if family_base and key in row and key in family_base:
                row[f"delta_{key}_vs_family_original_original"] = row[key] - family_base[key]
            if ckpt_base and key in row and key in ckpt_base:
                row[f"delta_{key}_vs_same_checkpoint_original_eval"] = row[key] - ckpt_base[key]

    fields = [
        "status",
        "architecture",
        "model_family",
        "base_model",
        "trained_representation",
        "eval_representation",
        "trained_representation_label",
        "eval_representation_label",
        "MRR@10",
        "MAP@10",
        "Recall@10",
        "MRR@100",
        "MAP@100",
        "Recall@100",
        "delta_MRR@10_vs_family_original_original",
        "delta_MAP@10_vs_family_original_original",
        "delta_Recall@10_vs_family_original_original",
        "delta_MRR@100_vs_family_original_original",
        "delta_MAP@100_vs_family_original_original",
        "delta_Recall@100_vs_family_original_original",
        "delta_MRR@10_vs_same_checkpoint_original_eval",
        "delta_MAP@10_vs_same_checkpoint_original_eval",
        "delta_Recall@10_vs_same_checkpoint_original_eval",
        "delta_MRR@100_vs_same_checkpoint_original_eval",
        "delta_MAP@100_vs_same_checkpoint_original_eval",
        "delta_Recall@100_vs_same_checkpoint_original_eval",
        "valid_queries",
        "corpus_docs",
        "retrieve_seconds",
        "checkpoint_path",
        "training_note",
        "error",
    ]
    with (out_dir / "summary_table.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    dump_json(out_dir / "summary_table.json", {"rows": rows, "blocked": blocked})

    best_by_family: dict[str, dict[str, Any]] = {}
    for row in completed:
        current = best_by_family.get(row["model_family"])
        if current is None or row.get("MRR@10", -1.0) > current.get("MRR@10", -1.0):
            best_by_family[row["model_family"]] = row

    lines = [
        "# Expanded-Corpus Retrieve Report",
        "",
        f"Task: global top-{preflight['top_k']} retrieval for `search_test[{preflight['query_range'][0]}:{preflight['query_range'][1]}]` queries "
        f"over the covered-only union corpus from `search_test[{preflight['corpus_range'][0]}:{preflight['corpus_range'][1]}].search_results`.",
        "",
        "## Preflight",
        "",
        f"- Test NoteCards: {preflight['test_notecard_card_count']:,}",
        f"- Full corpus union: {preflight['counts']['full_corpus_union']:,}",
        f"- Covered-only corpus: {preflight['counts']['covered_only_corpus']:,}",
        f"- Dropped missing NoteCard notes: {preflight['counts']['dropped_missing_notecard_notes']:,}",
        f"- Qrels relevant-note coverage in corpus: {preflight['qrels_coverage']['qrels_relevant_notes_in_corpus']:.0f} / {preflight['qrels_coverage']['qrels_unique_relevant_notes']:.0f}",
        f"- Exposure-pool order MRR@10 on same query range: {preflight['exposure_pool_order_metrics_same_query_range'].get('MRR@10', 0.0):.6f}",
        "",
        "## Best Rows By Model Family",
        "",
        "| model_family | trained | eval | MRR@10 | MAP@10 | R@10 | MRR@100 | MAP@100 | R@100 | delta MRR@10 vs O/O |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family, row in sorted(best_by_family.items()):
        lines.append(
            f"| {family} | {row['trained_representation']} | {row['eval_representation']} | "
            f"{row.get('MRR@10', 0.0):.6f} | {row.get('MAP@10', 0.0):.6f} | {row.get('Recall@10', 0.0):.6f} | "
            f"{row.get('MRR@100', 0.0):.6f} | {row.get('MAP@100', 0.0):.6f} | {row.get('Recall@100', 0.0):.6f} | "
            f"{row.get('delta_MRR@10_vs_family_original_original', 0.0):.6f} |"
        )
    lines.extend(["", "## Blocked / Failed", ""])
    failed = [row for row in rows if row.get("status") == "failed"]
    if not blocked and not failed:
        lines.append("- None.")
    for item in blocked:
        lines.append(f"- BLOCKED {item.get('model_family')}: {item.get('trained_representation', '')} {item.get('reason', '')}")
    for row in failed:
        lines.append(f"- FAILED {row.get('model_family')} {row.get('trained_representation')}->{row.get('eval_representation')}: {row.get('error')}")
    (out_dir / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run expanded-corpus retrieve over Qilin test notecard corpus.")
    parser.add_argument("--qilin_dir", type=Path, default=DEFAULT_QILIN_DIR)
    parser.add_argument("--qrels_path", type=Path, default=DEFAULT_QRELS_PATH)
    parser.add_argument("--test_notecard_path", type=Path, default=DEFAULT_TEST_NOTECARD)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--query_start", type=int, default=0)
    parser.add_argument("--query_end", type=int, default=1000)
    parser.add_argument("--corpus_start", type=int, default=0)
    parser.add_argument("--corpus_end", type=int, default=6192)
    parser.add_argument("--candidate_key", default="search_results")
    parser.add_argument("--corpus_depth", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--max_doc_chars", type=int, default=0)
    parser.add_argument("--content_head_chars", type=int, default=300)
    parser.add_argument("--models", default="bert_zero", help="Comma-separated subset of: bert_zero, bm25")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bert_batch_size", type=int, default=32)
    parser.add_argument("--bert_max_length", type=int, default=256)
    parser.add_argument("--score_block_size", type=int, default=16)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--normalize_embeddings", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.models = [item.strip() for item in str(args.models).split(",") if item.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    context = load_context(args, args.query_end, args.corpus_end)
    specs, blocked = build_run_specs(args)
    context["preflight"]["blocked_checkpoints"] = blocked
    context["preflight"]["planned_runs"] = len(specs)
    dump_json(args.output_dir / "preflight.json", context["preflight"])

    rows: list[dict[str, Any]] = []
    index = 1
    while index <= len(specs):
        spec = specs[index - 1]
        if spec.model_path is None:
            print(f"[{index}/{len(specs)}] {spec.run_id}", flush=True)
            rows.append(run_one(spec, context, args.output_dir, args))
            index += 1
            write_summary(args.output_dir, rows, blocked, context["preflight"])
            continue

        group_key = (
            spec.model_path,
            spec.batch_size,
            spec.max_length,
            spec.fp16,
            spec.normalize_embeddings,
        )
        group: list[RunSpec] = []
        while index + len(group) <= len(specs):
            candidate = specs[index + len(group) - 1]
            candidate_key = (
                candidate.model_path,
                candidate.batch_size,
                candidate.max_length,
                candidate.fp16,
                candidate.normalize_embeddings,
            )
            if candidate.model_path is None or candidate_key != group_key:
                break
            group.append(candidate)
        rows.extend(run_encoder_group(group, context, args.output_dir, args, index, len(specs)))
        index += len(group)
        write_summary(args.output_dir, rows, blocked, context["preflight"])
    write_summary(args.output_dir, rows, blocked, context["preflight"])
    print(f"Saved summary: {args.output_dir / 'summary_table.csv'}", flush=True)
    print(f"Saved report: {args.output_dir / 'FINAL_REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
