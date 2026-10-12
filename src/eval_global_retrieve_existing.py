#!/usr/bin/env python3
"""Evaluate an existing AutoModel bi-encoder checkpoint on expanded retrieval.

This is a test-confirmation helper for the Qilin NoteCard experiments. It
does not train. It loads either a full AutoModel checkpoint or a PEFT LoRA
adapter checkpoint, evaluates selected text representations on val/test
expanded retrieval, and writes metrics/rankings/run_summary files.

In the paper it produces the test-split rows of the fine-tuned BERT bi-encoder
(the BERT training runs only evaluate on validation for model selection).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any

from qwen3_embedding_global_ft import (
    DEFAULT_OUTPUT,
    DEFAULT_QWEN_QUERY_INSTRUCTION,
    DEFAULT_QILIN_DIR,
    DEFAULT_QRELS_PATH,
    DEFAULT_TEST_NOTECARD,
    DEFAULT_TRAIN_NOTECARD,
    METRIC_KEYS,
    MODES,
    dataset_context,
    dump_json,
    evaluate_expanded,
    read_json,
)


def load_eval_model(args: argparse.Namespace):
    import torch
    from transformers import AutoModel, AutoTokenizer

    dtype = None
    if args.device == "cuda":
        if args.bf16:
            dtype = torch.bfloat16
        elif args.fp16:
            dtype = torch.float16

    if args.checkpoint_type == "lora":
        from peft import PeftModel

        tokenizer_source = args.checkpoint_path if (args.checkpoint_path / "tokenizer_config.json").exists() else args.model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        base = AutoModel.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=dtype)
        model = PeftModel.from_pretrained(base, args.checkpoint_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(args.checkpoint_path, trust_remote_code=True, torch_dtype=dtype)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "left" if args.qwen_official_style else "right"
    model.to(args.device)
    model.eval()
    return model, tokenizer


def aggregate_output(output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    zero_root = output_dir / "zero_shot"
    if zero_root.exists():
        for path in sorted(zero_root.glob("*/*/*_metrics.json")):
            metrics = read_json(path)
            split = path.name.replace("_metrics.json", "")
            mode = path.parent.name
            rows.append({
                "stage": "zero_shot",
                "run_name": "zero_shot",
                "split": split,
                "train_mode": "zero_shot",
                "eval_mode": mode,
                "negative_strategy": "",
                "lr": "",
                "checkpoint_type": "zero_shot",
                "checkpoint_path": "",
                **{k: metrics.get(k, 0.0) for k in METRIC_KEYS},
                "valid_queries": metrics.get("_valid_queries", 0.0),
            })
    runs_root = output_dir / "runs"
    if runs_root.exists():
        for summary in sorted(runs_root.glob("*/run_summary.json")):
            payload = read_json(summary)
            for item in payload.get("eval", []):
                rows.append({
                    "stage": payload.get("stage", "eval_existing"),
                    "run_name": payload.get("run_name", summary.parent.name),
                    "split": item["split"],
                    "train_mode": payload.get("train_mode", ""),
                    "eval_mode": item["eval_mode"],
                    "negative_strategy": payload.get("negative_strategy", ""),
                    "lr": payload.get("lr", ""),
                    "checkpoint_type": payload.get("checkpoint_type", ""),
                    "checkpoint_path": payload.get("checkpoint_path", ""),
                    **{k: item.get(k, 0.0) for k in METRIC_KEYS},
                    "valid_queries": item.get("valid_queries", 0.0),
                })
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    out = output_dir / "summary_table.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    dump_json(output_dir / "summary_table.json", {"rows": rows})

    test_rows = [r for r in rows if r.get("split") == "test"]
    test_rows.sort(key=lambda r: float(r.get("MRR@10", 0.0)), reverse=True)
    lines = [
        "# Test Confirmation Summary",
        "",
        "| stage | run | train | eval | MRR@10 | MAP@10 | R@10 | MRR@100 | MAP@100 | R@100 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in test_rows:
        lines.append(
            f"| {row.get('stage','')} | {row.get('run_name','')} | {row.get('train_mode','')} | "
            f"{row.get('eval_mode','')} | {float(row.get('MRR@10',0.0)):.6f} | "
            f"{float(row.get('MAP@10',0.0)):.6f} | {float(row.get('Recall@10',0.0)):.6f} | "
            f"{float(row.get('MRR@100',0.0)):.6f} | {float(row.get('MAP@100',0.0)):.6f} | "
            f"{float(row.get('Recall@100',0.0)):.6f} |"
        )
    (output_dir / "FINAL_TEST_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an existing checkpoint on expanded retrieval.")
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--checkpoint_type", choices=["full", "lora"], default="full")
    parser.add_argument("--model_name", default="bert-base-chinese")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--train_mode", choices=MODES, required=True)
    parser.add_argument("--negative_strategy", default="")
    parser.add_argument("--lr", default="")
    parser.add_argument("--qilin_dir", type=Path, default=DEFAULT_QILIN_DIR)
    parser.add_argument("--qrels_path", type=Path, default=DEFAULT_QRELS_PATH)
    parser.add_argument("--train_notecard_path", type=Path, default=DEFAULT_TRAIN_NOTECARD)
    parser.add_argument("--test_notecard_path", type=Path, default=DEFAULT_TEST_NOTECARD)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT / "test_confirmation")
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
    parser.add_argument("--max_doc_chars", type=int, default=0)
    parser.add_argument("--content_head_chars", type=int, default=300)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--score_block_size", type=int, default=128)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--normalize_embeddings", action="store_true")
    parser.add_argument("--eval_splits", nargs="+", choices=["val", "test"], default=["test"])
    parser.add_argument("--eval_modes", nargs="+", choices=MODES, default=list(MODES))
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
    ctx = dataset_context(args, smoke=False)
    model, tokenizer = load_eval_model(args)

    run_dir = args.output_dir / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "status": "running",
        "stage": "eval_existing",
        "run_name": args.run_name,
        "train_mode": args.train_mode,
        "negative_strategy": args.negative_strategy,
        "lr": args.lr,
        "checkpoint_type": args.checkpoint_type,
        "checkpoint_path": str(args.checkpoint_path),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    dump_json(run_dir / "run_summary.json", payload)

    eval_rows: list[dict[str, Any]] = []
    for split in args.eval_splits:
        for mode in args.eval_modes:
            if split == "val":
                metrics = evaluate_expanded(
                    model, tokenizer, ctx["val_rows"], args.val_start, ctx["val_qrels"],
                    ctx["train_corpus_ids"], ctx["notes"], ctx["train_cards"], mode,
                    args, run_dir / "eval" / split / mode, split,
                )
            else:
                metrics = evaluate_expanded(
                    model, tokenizer, ctx["test_rows"], args.test_start, ctx["test_qrels"],
                    ctx["test_corpus_ids"], ctx["notes"], ctx["test_cards"], mode,
                    args, run_dir / "eval" / split / mode, split,
                )
            eval_rows.append({
                "split": split,
                "eval_mode": mode,
                **{key: metrics.get(key, 0.0) for key in METRIC_KEYS},
                "valid_queries": metrics.get("_valid_queries", 0.0),
            })
            dump_json(run_dir / "run_summary.json", {**payload, "status": "evaluating", "eval": eval_rows})

    dump_json(run_dir / "run_summary.json", {**payload, "status": "completed", "eval": eval_rows})
    aggregate_output(args.output_dir)

    del model, tokenizer
    gc.collect()
    if args.device == "cuda":
        import torch

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
