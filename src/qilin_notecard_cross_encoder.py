#!/usr/bin/env python3
"""
Qilin-style CrossEncoder experiments with NoteCard text representations.

This runner mirrors the public Qilin search CrossEncoder shape:
  - bert-base-chinese AutoModel
  - CLS pooling + linear classifier
  - query [SEP] note_text pairs
  - one clicked positive plus N click-zero negatives per query
  - BCEWithLogitsLoss

Only the candidate note text is swapped across the NoteCard representations.
In the paper (Table 3) it trains the MacBERT and RoBERTa cross-encoders, one
checkpoint per training representation; see scripts/06_rerank_cross_encoders.sh.
The default batch sizes are RTX 4080 Laptop safe and intentionally lower than
the Qilin config batch size.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from qilin_notecard_biencoder import (
    DEFAULT_QILIN_DIR,
    DEFAULT_QRELS_PATH,
    DEFAULT_TEST_NOTECARD,
    DEFAULT_TRAIN_NOTECARD,
    POSITIVE_FIELDS,
    TEXT_MODES,
    build_note_text,
    calculate_metrics,
    candidate_ids_from_row,
    click_value,
    collect_candidate_map,
    coverage_stats,
    create_optimizer,
    create_scheduler,
    is_negative_detail,
    load_notecards,
    load_notes_subset,
    load_qrels,
    load_search_slice,
    qrels_from_details,
)


# The released test card file already contains the completed exposure-pool cards.
DEFAULT_COMPLETED_TEST_NOTECARD = DEFAULT_TEST_NOTECARD


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def collect_detail_ids(rows: list[dict[str, Any]]) -> set[int]:
    out: set[int] = set()
    for row in rows:
        details = row.get("search_result_details_with_idx")
        if details is None:
            details = []
        for detail in details:
            out.add(int(detail["note_idx"]))
    return out


def candidate_ids_with_fallback(row: dict[str, Any], candidate_key: str, depth: int | None) -> list[int]:
    if candidate_key in row and row.get(candidate_key) is not None:
        return candidate_ids_from_row(row, candidate_key, depth)
    return candidate_ids_from_row(row, "details", depth)


def collect_candidate_map_with_fallback(
    rows: list[dict[str, Any]],
    candidate_key: str,
    depth: int | None,
    qid_start: int,
) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for local_qid, row in enumerate(rows):
        qid = qid_start + local_qid
        ids = candidate_ids_with_fallback(row, candidate_key, depth)
        seen: set[int] = set()
        deduped: list[int] = []
        for note_id in ids:
            if note_id not in seen:
                deduped.append(note_id)
                seen.add(note_id)
        out[qid] = deduped
    return out


@dataclass
class CrossGroup:
    query: str
    positive_ids: list[int]
    negative_ids: list[int]


class CrossEncoderModel:
    def __init__(self, model_name_or_path: str, tokenizer_name_or_path: str | None, device: str):
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.device = torch.device(device)
        tok_name = tokenizer_name_or_path or model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = "right"
        self.tokenizer.padding_side = "right"

        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(model_name_or_path, config=config, trust_remote_code=True)
        self.classifier = torch.nn.Linear(config.hidden_size, 1)

        classifier_path = Path(model_name_or_path) / "classifier.pt"
        if classifier_path.exists():
            self.classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))

        self.encoder.to(self.device)
        self.classifier.to(self.device)

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.classifier.parameters()

    def train(self) -> None:
        self.encoder.train()
        self.classifier.train()

    def eval(self) -> None:
        self.encoder.eval()
        self.classifier.eval()

    def gradient_checkpointing_enable(self) -> None:
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

    def __call__(self, **inputs):
        outputs = self.encoder(**inputs)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

    def save_pretrained(self, path: Path) -> None:
        import torch

        ensure_dir(path)
        self.encoder.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        torch.save(self.classifier.state_dict(), path / "classifier.pt")


def build_cross_groups(
    rows: list[dict[str, Any]],
    notes: dict[int, dict[str, str]],
    cards: dict[int, str],
    mode: str,
    args: argparse.Namespace,
) -> list[CrossGroup]:
    groups: list[CrossGroup] = []
    for row in rows:
        positives: list[int] = []
        negatives: list[int] = []
        details = row.get("search_result_details_with_idx")
        if details is None:
            details = []
        for detail in details:
            note_id = int(detail["note_idx"])
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
                continue
            if click_value(detail) == 1:
                positives.append(note_id)
            if is_negative_detail(detail, args.negative_policy, POSITIVE_FIELDS):
                negatives.append(note_id)
        if positives:
            groups.append(CrossGroup(str(row.get("query") or ""), positives, negatives))
    return groups


def sample_negatives(
    rng: random.Random,
    negatives: list[int],
    fallback_ids: list[int],
    n: int,
    exclude: set[int],
) -> list[int]:
    if len(negatives) >= n:
        return rng.sample(negatives, k=n)
    sampled = list(negatives)
    need = n - len(sampled)
    fallback = [pid for pid in fallback_ids if pid not in exclude and pid not in sampled]
    if len(fallback) >= need:
        sampled.extend(rng.sample(fallback, k=need))
    elif fallback:
        sampled.extend(rng.choices(fallback, k=need))
    elif negatives:
        sampled.extend(rng.choices(negatives, k=need))
    return sampled


def make_pairs(
    groups: list[CrossGroup],
    batch_indices: list[int],
    notes: dict[int, dict[str, str]],
    cards: dict[int, str],
    mode: str,
    args: argparse.Namespace,
    rng: random.Random,
    fallback_ids: list[int],
) -> tuple[list[str], list[float]]:
    pairs: list[str] = []
    labels: list[float] = []
    for idx in batch_indices:
        group = groups[idx]
        positive_id = rng.choice(group.positive_ids)
        ids = [positive_id]
        ids.extend(
            sample_negatives(
                rng,
                group.negative_ids,
                fallback_ids,
                args.negative_samples,
                exclude=set(group.positive_ids),
            )
        )
        for pos, note_id in enumerate(ids):
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
                continue
            pairs.append(f"{group.query} [SEP] {text}")
            labels.append(1.0 if pos == 0 else 0.0)
    return pairs, labels


def train_one_mode(
    model: CrossEncoderModel,
    groups: list[CrossGroup],
    notes: dict[int, dict[str, str]],
    cards: dict[int, str],
    mode: str,
    fallback_ids: list[int],
    args: argparse.Namespace,
) -> dict[str, float]:
    import torch
    from tqdm import tqdm

    if not groups:
        raise RuntimeError(f"No train groups for mode={mode}.")

    use_amp = args.fp16 and args.device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    stats: dict[str, float] = {"train_groups": float(len(groups))}

    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = list(range(len(groups)))
        rng = random.Random(args.seed + epoch)
        rng.shuffle(indices)
        running_loss = 0.0
        steps = 0
        pbar = tqdm(
            range(0, len(indices), args.train_batch_size),
            desc=f"train {mode} epoch {epoch}",
        )
        for start in pbar:
            batch_indices = indices[start : start + args.train_batch_size]
            pairs, labels = make_pairs(groups, batch_indices, notes, cards, mode, args, rng, fallback_ids)
            if not pairs:
                continue
            encoded = model.tokenizer(
                pairs,
                truncation=True,
                padding=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
            y = torch.tensor(labels, dtype=torch.float32, device=model.device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(**encoded).view(-1)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            steps += 1
            running_loss += float(loss.detach().cpu())
            pbar.set_postfix(loss=running_loss / steps)

        stats[f"epoch_{epoch}_loss"] = running_loss / max(steps, 1)
        if args.device == "cuda":
            torch.cuda.empty_cache()
    return stats


def score_pairs(
    model: CrossEncoderModel,
    pairs: list[str],
    batch_size: int,
    max_length: int,
    fp16: bool,
) -> list[float]:
    import torch

    scores: list[float] = []
    use_amp = fp16 and str(model.device) == "cuda"
    model.eval()
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            encoded = model.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(**encoded).view(-1)
            scores.extend(logits.detach().float().cpu().tolist())
    return scores


def evaluate_mode(
    model: CrossEncoderModel,
    rows: list[dict[str, Any]],
    qid_start: int,
    qrels: dict[int, set[int]],
    notes: dict[int, dict[str, str]],
    cards: dict[int, str],
    mode: str,
    args: argparse.Namespace,
    output_dir: Path,
    split_name: str,
) -> dict[str, float]:
    from tqdm import tqdm

    ensure_dir(output_dir)
    sorted_results: dict[int, list[int]] = {}
    score_rows: list[tuple[int, int, float, int]] = []

    for local_qid, row in enumerate(tqdm(rows, desc=f"eval {split_name} {mode}")):
        qid = qid_start + local_qid
        candidate_ids = candidate_ids_with_fallback(row, args.candidate_key, args.candidate_depth)
        pairs: list[str] = []
        kept_ids: list[int] = []
        for note_id in candidate_ids:
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
                continue
            kept_ids.append(note_id)
            pairs.append(f"{str(row.get('query') or '')} [SEP] {text}")
        scores = score_pairs(model, pairs, args.eval_batch_size, args.max_length, args.fp16)
        ranked = sorted(zip(kept_ids, scores), key=lambda x: x[1], reverse=True)
        sorted_results[qid] = [pid for pid, _ in ranked]
        for rank, (pid, score) in enumerate(ranked, start=1):
            score_rows.append((qid, pid, float(score), rank))

    metrics = calculate_metrics(sorted_results, qrels, [10, 100])
    dump_json(output_dir / f"{split_name}_metrics.json", metrics)
    if split_name == "test":
        dump_json(output_dir / "metrics.json", metrics)

    rankings_path = output_dir / f"{split_name}_rankings.csv"
    with rankings_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "pid", "score", "rank"])
        writer.writerows(score_rows)
    if split_name == "test":
        legacy_path = output_dir / "test_rankings.csv"
        if legacy_path != rankings_path:
            with legacy_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["qid", "pid", "score", "rank"])
                writer.writerows(score_rows)
    return metrics


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(output_dir)
    csv_path = output_dir / "summary_table.csv"
    fieldnames = [
        "mode",
        "train_groups",
        "train_loss",
        "val_MRR@10",
        "val_MAP@10",
        "val_MRR@100",
        "val_MAP@100",
        "test_MRR@10",
        "test_MAP@10",
        "test_MRR@100",
        "test_MAP@100",
        "delta_test_MRR@10_vs_original",
        "delta_test_MAP@10_vs_original",
        "delta_test_MRR@100_vs_original",
        "delta_test_MAP@100_vs_original",
        "runtime_sec",
    ]
    original = next((row for row in rows if row["mode"] == "original"), None)
    for row in rows:
        if original:
            for metric in ("MRR@10", "MAP@10", "MRR@100", "MAP@100"):
                row[f"delta_test_{metric}_vs_original"] = row.get(f"test_{metric}", 0.0) - original.get(
                    f"test_{metric}", 0.0
                )
        else:
            for metric in ("MRR@10", "MAP@10", "MRR@100", "MAP@100"):
                row[f"delta_test_{metric}_vs_original"] = ""

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    report_path = output_dir / "final_report.md"
    lines = [
        "# Qilin-style CrossEncoder NoteCard ablation",
        "",
        "This is not a full Qilin baseline reproduction. It keeps the public Qilin CrossEncoder model/training shape and the local subset split, while replacing only candidate note text representation.",
        "",
        f"- Modes: {', '.join(row['mode'] for row in rows)}",
        "- Split: train=search_train[0:4000], val=search_train[4000:5000], test=search_test[0:1000]",
        "- Candidate pool: search_results",
        "",
        "| mode | test MRR@10 | test MAP@10 | test MRR@100 | test MAP@100 | delta MAP@100 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {mode} | {mrr10:.6f} | {map10:.6f} | {mrr100:.6f} | {map100:.6f} | {delta:.6f} |".format(
                mode=row["mode"],
                mrr10=row.get("test_MRR@10", 0.0),
                map10=row.get("test_MAP@10", 0.0),
                mrr100=row.get("test_MRR@100", 0.0),
                map100=row.get("test_MAP@100", 0.0),
                delta=row.get("delta_test_MAP@100_vs_original", 0.0),
            )
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mode_cards(mode: str, train_cards: dict[int, str], test_cards: dict[int, str], split: str) -> dict[int, str]:
    if mode in {"original", "title_content_head"}:
        return {}
    return train_cards if split in {"train", "val"} else test_cards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qilin_dir", type=Path, default=DEFAULT_QILIN_DIR)
    parser.add_argument("--qrels_path", type=Path, default=DEFAULT_QRELS_PATH)
    parser.add_argument("--train_notecard_path", type=Path, default=DEFAULT_TRAIN_NOTECARD)
    parser.add_argument("--test_notecard_path", type=Path, default=DEFAULT_COMPLETED_TEST_NOTECARD)
    parser.add_argument("--model_name_or_path", type=str, default="google-bert/bert-base-chinese")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--train_start", type=int, default=0)
    parser.add_argument("--train_end", type=int, default=4000)
    parser.add_argument("--val_start", type=int, default=4000)
    parser.add_argument("--val_end", type=int, default=5000)
    parser.add_argument("--test_start", type=int, default=0)
    parser.add_argument("--test_end", type=int, default=1000)
    parser.add_argument("--candidate_key", type=str, default="search_results")
    parser.add_argument("--candidate_depth", type=int, default=1000)
    parser.add_argument("--modes", nargs="+", choices=TEXT_MODES, default=list(TEXT_MODES))
    parser.add_argument("--missing_policy", choices=["error", "drop", "original"], default="error")
    parser.add_argument("--negative_policy", choices=["click_zero", "click_not_one", "qilin_official_nonzero", "non_positive"], default="click_zero")
    parser.add_argument("--negative_samples", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_doc_chars", type=int, default=6000)
    parser.add_argument("--content_head_chars", type=int, default=300)
    parser.add_argument("--optimizer", choices=["lamb", "adamw"], default="lamb")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--scheduler", choices=["linear", "none"], default="linear")
    parser.add_argument("--linear_total_iters", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--eval_val", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    random.seed(args.seed)

    train_df = load_search_slice(args.qilin_dir, "search_train", args.train_start, args.train_end)
    val_df = load_search_slice(args.qilin_dir, "search_train", args.val_start, args.val_end)
    test_df = load_search_slice(args.qilin_dir, "search_test", args.test_start, args.test_end)
    train_rows = train_df.to_dict(orient="records")
    val_rows = val_df.to_dict(orient="records")
    test_rows = test_df.to_dict(orient="records")

    train_cards = load_notecards(args.train_notecard_path)
    test_cards = load_notecards(args.test_notecard_path if args.test_notecard_path.exists() else DEFAULT_TEST_NOTECARD)

    train_candidate_map = collect_candidate_map_with_fallback(
        train_rows, args.candidate_key, args.candidate_depth, args.train_start
    )
    val_candidate_map = collect_candidate_map_with_fallback(
        val_rows, args.candidate_key, args.candidate_depth, args.val_start
    )
    test_candidate_map = collect_candidate_map_with_fallback(
        test_rows, args.candidate_key, args.candidate_depth, args.test_start
    )
    all_ids = set().union(*train_candidate_map.values(), *val_candidate_map.values(), *test_candidate_map.values())
    all_ids |= collect_detail_ids(train_rows)
    all_ids |= collect_detail_ids(val_rows)
    notes = load_notes_subset(args.qilin_dir, all_ids)

    train_note_ids = collect_detail_ids(train_rows)
    fallback_ids_by_mode: dict[str, list[int]] = {}
    for mode in args.modes:
        cards = mode_cards(mode, train_cards, test_cards, "train")
        fallback: list[int] = []
        for pid in sorted(train_note_ids):
            try:
                text = build_note_text(
                    pid,
                    notes,
                    cards,
                    mode,
                    args.missing_policy,
                    args.max_doc_chars,
                    args.content_head_chars,
                )
            except KeyError:
                continue
            if text is not None:
                fallback.append(pid)
        fallback_ids_by_mode[mode] = fallback

    preflight = {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "train_notecards": len(train_cards),
        "test_notecards": len(test_cards),
        "coverage": {
            "train": coverage_stats(train_candidate_map, train_cards),
            "val": coverage_stats(val_candidate_map, train_cards),
            "test": coverage_stats(test_candidate_map, test_cards),
        },
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    dump_json(args.output_dir / "preflight.json", preflight)

    test_qrels = load_qrels(args.qrels_path, args.test_start, args.test_end)
    if not test_qrels:
        test_qrels = qrels_from_details(test_rows, args.test_start)
    val_qrels = qrels_from_details(val_rows, args.val_start)

    summary_rows: list[dict[str, Any]] = []
    for mode in args.modes:
        mode_start = time.time()
        mode_dir = args.output_dir / mode
        ensure_dir(mode_dir)
        train_cards_for_mode = mode_cards(mode, train_cards, test_cards, "train")
        test_cards_for_mode = mode_cards(mode, train_cards, test_cards, "test")

        model = CrossEncoderModel(args.model_name_or_path, args.tokenizer_name_or_path, args.device)
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        groups = build_cross_groups(train_rows, notes, train_cards_for_mode, mode, args)
        train_stats = train_one_mode(
            model,
            groups,
            notes,
            train_cards_for_mode,
            mode,
            fallback_ids_by_mode[mode],
            args,
        )
        model.save_pretrained(mode_dir / "model")

        val_metrics: dict[str, float] = {}
        if args.eval_val:
            val_metrics = evaluate_mode(
                model,
                val_rows,
                args.val_start,
                val_qrels,
                notes,
                train_cards_for_mode,
                mode,
                args,
                mode_dir,
                "val",
            )
        test_metrics = evaluate_mode(
            model,
            test_rows,
            args.test_start,
            test_qrels,
            notes,
            test_cards_for_mode,
            mode,
            args,
            mode_dir,
            "test",
        )

        row: dict[str, Any] = {
            "mode": mode,
            "train_groups": train_stats.get("train_groups", 0.0),
            "train_loss": train_stats.get(f"epoch_{args.epochs}_loss", 0.0),
            "runtime_sec": time.time() - mode_start,
        }
        for key, value in val_metrics.items():
            row[f"val_{key}"] = value
        for key, value in test_metrics.items():
            row[f"test_{key}"] = value
        summary_rows.append(row)
        dump_json(mode_dir / "run_summary.json", {"train": train_stats, "val": val_metrics, "test": test_metrics})
        write_summary(args.output_dir, summary_rows)

        if args.device == "cuda":
            import torch

            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
