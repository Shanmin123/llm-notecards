#!/usr/bin/env python3
"""
Evaluate saved Qilin-style CrossEncoder checkpoints across text views.

This is the CrossEncoder counterpart of the BERT_bi cross-eval matrix: the
checkpoint is fixed by training representation, then candidate note text is
swapped at inference time. It produces the train x inference grids of Table 3
for the MacBERT and RoBERTa cross-encoders.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_QILIN_DIR,
    DEFAULT_QRELS_PATH,
    TEXT_MODES,
    coverage_stats,
    load_notecards,
    load_notes_subset,
    load_qrels,
    load_search_slice,
    qrels_from_details,
)
from rerank_cross_encoder import (
    DEFAULT_COMPLETED_TEST_NOTECARD,
    DEFAULT_TRAIN_NOTECARD,
    CrossEncoderModel,
    candidate_ids_with_fallback,
    collect_candidate_map_with_fallback,
    dump_json,
    ensure_dir,
    evaluate_mode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-evaluate saved CrossEncoder checkpoints.")
    parser.add_argument("--qilin_dir", type=Path, default=DEFAULT_QILIN_DIR)
    parser.add_argument("--qrels_path", type=Path, default=DEFAULT_QRELS_PATH)
    parser.add_argument("--train_notecard_path", type=Path, default=DEFAULT_TRAIN_NOTECARD)
    parser.add_argument("--test_notecard_path", type=Path, default=DEFAULT_COMPLETED_TEST_NOTECARD)
    parser.add_argument("--checkpoint_root", type=Path, required=True)
    parser.add_argument("--trained_modes", nargs="+", choices=TEXT_MODES, required=True)
    parser.add_argument("--eval_modes", nargs="+", choices=TEXT_MODES, default=list(TEXT_MODES))
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--test_start", type=int, default=0)
    parser.add_argument("--test_end", type=int, default=1000)
    parser.add_argument("--candidate_key", type=str, default="search_results")
    parser.add_argument("--candidate_depth", type=int, default=1000)
    parser.add_argument("--missing_policy", choices=["error", "drop", "original"], default="error")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_doc_chars", type=int, default=6000)
    parser.add_argument("--content_head_chars", type=int, default=300)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def mode_cards(mode: str, train_cards: dict[int, str], test_cards: dict[int, str]) -> dict[int, str]:
    if mode in {"original", "title_content_head"}:
        return {}
    return test_cards


def diagnose_rankings(
    ranking_path: Path,
    rows: list[dict[str, Any]],
    qid_start: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    by_qid: dict[int, list[tuple[int, float]]] = {}
    unique_scores: set[float] = set()
    with ranking_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = int(row["qid"])
            pid = int(row["pid"])
            score = float(row["score"])
            unique_scores.add(round(score, 6))
            by_qid.setdefault(qid, []).append((pid, score))

    same_order = 0
    compared = 0
    for local_qid, row in enumerate(rows):
        qid = qid_start + local_qid
        ranked = [pid for pid, _ in by_qid.get(qid, [])]
        if not ranked:
            continue
        candidates = candidate_ids_with_fallback(row, args.candidate_key, args.candidate_depth)
        candidates = [pid for pid in candidates if pid in set(ranked)]
        compared += 1
        if ranked == candidates:
            same_order += 1

    return {
        "unique_scores_rounded_6": float(len(unique_scores)),
        "queries_compared_order": float(compared),
        "same_as_input_order_queries": float(same_order),
        "same_as_input_order_rate": float(same_order / compared) if compared else 0.0,
    }


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(output_dir)
    fields = [
        "trained_mode",
        "eval_mode",
        "MRR@10",
        "MAP@10",
        "Recall@10",
        "MRR@100",
        "MAP@100",
        "Recall@100",
        "delta_MAP@100_vs_same_trained_original_eval",
        "unique_scores_rounded_6",
        "same_as_input_order_rate",
        "valid_queries",
    ]
    baseline_by_trained = {
        row["trained_mode"]: row["MAP@100"]
        for row in rows
        if row["eval_mode"] == "original"
    }
    for row in rows:
        base = baseline_by_trained.get(row["trained_mode"])
        row["delta_MAP@100_vs_same_trained_original_eval"] = row["MAP@100"] - base if base is not None else ""

    with (output_dir / "summary_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    lines = [
        "# CrossEncoder Saved-Checkpoint Cross-Eval",
        "",
        "| trained | eval | MRR@10 | MAP@10 | MRR@100 | MAP@100 | delta vs original eval | unique scores | same input order |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: item["MAP@100"], reverse=True):
        lines.append(
            "| {trained_mode} | {eval_mode} | {mrr10:.6f} | {map10:.6f} | {mrr100:.6f} | {map100:.6f} | {delta:.6f} | {uniq:.0f} | {same:.3f} |".format(
                trained_mode=row["trained_mode"],
                eval_mode=row["eval_mode"],
                mrr10=row["MRR@10"],
                map10=row["MAP@10"],
                mrr100=row["MRR@100"],
                map100=row["MAP@100"],
                delta=row["delta_MAP@100_vs_same_trained_original_eval"],
                uniq=row["unique_scores_rounded_6"],
                same=row["same_as_input_order_rate"],
            )
        )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    test_df = load_search_slice(args.qilin_dir, "search_test", args.test_start, args.test_end)
    test_rows = test_df.to_dict(orient="records")
    train_cards = load_notecards(args.train_notecard_path)
    test_cards = load_notecards(args.test_notecard_path)
    test_candidate_map = collect_candidate_map_with_fallback(
        test_rows, args.candidate_key, args.candidate_depth, args.test_start
    )
    all_ids = set().union(*test_candidate_map.values())
    notes = load_notes_subset(args.qilin_dir, all_ids)
    test_qrels = load_qrels(args.qrels_path, args.test_start, args.test_end)
    if not test_qrels:
        test_qrels = qrels_from_details(test_rows, args.test_start)

    preflight = {
        "checkpoint_root": str(args.checkpoint_root),
        "trained_modes": args.trained_modes,
        "eval_modes": args.eval_modes,
        "candidate_key": args.candidate_key,
        "candidate_depth": args.candidate_depth,
        "test_rows": len(test_rows),
        "test_notecards": len(test_cards),
        "coverage": coverage_stats(test_candidate_map, test_cards),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    dump_json(args.output_dir / "preflight.json", preflight)

    summary_rows: list[dict[str, Any]] = []
    for trained_mode in args.trained_modes:
        model_path = args.checkpoint_root / trained_mode / "model"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing CrossEncoder checkpoint: {model_path}")
        model = CrossEncoderModel(str(model_path), None, args.device)
        for eval_mode in args.eval_modes:
            eval_dir = args.output_dir / f"trained_{trained_mode}" / eval_mode
            cards = mode_cards(eval_mode, train_cards, test_cards)
            metrics = evaluate_mode(
                model,
                test_rows,
                args.test_start,
                test_qrels,
                notes,
                cards,
                eval_mode,
                args,
                eval_dir,
                "test",
            )
            diag = diagnose_rankings(eval_dir / "test_rankings.csv", test_rows, args.test_start, args)
            row = {
                "trained_mode": trained_mode,
                "eval_mode": eval_mode,
                "MRR@10": metrics["MRR@10"],
                "MAP@10": metrics["MAP@10"],
                "Recall@10": metrics["Recall@10"],
                "MRR@100": metrics["MRR@100"],
                "MAP@100": metrics["MAP@100"],
                "Recall@100": metrics["Recall@100"],
                "valid_queries": metrics["_valid_queries"],
                **diag,
            }
            summary_rows.append(row)
            dump_json(eval_dir / "diagnostics.json", diag)
            write_summary(args.output_dir, summary_rows)

        if args.device == "cuda":
            import torch

            del model
            torch.cuda.empty_cache()

    dump_json(args.output_dir / "run_summary.json", {"preflight": preflight, "rows": summary_rows})
    write_summary(args.output_dir, summary_rows)


if __name__ == "__main__":
    main()
