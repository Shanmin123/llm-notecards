#!/usr/bin/env python3
"""
Note-card generation quality audit (paper, Section 4).

Two reliability views over a set of generated cards:
  - format view:   a card is *format-broken* if it does not parse into the
                   required field labels;
  - field view:    a field is *valid* if it is present and has usable content,
                   i.e. it is not empty and not an explicit "none" marker
                   (无 / None / N/A ...). A card is *fully complete* if every
                   field is valid.

Cards written with English labels (Category / Keywords / Scene / Audience /
Entities / Gist) are mapped to the same six fields. A field that appears in no
card at all (e.g. a five-field schema without a scenario) is reported as n/a
and does not count towards the format check.

Usage (from the repository root):
    # audit the cards of the test exposure pool, search_test[0:1000]
    python notecard_generation/evaluate_card_quality.py --cards data/test_notecard.json \
        --split test --query_end 1000 --output outputs/notecard_quality/test_pool.json

    # audit every card in a file
    python notecard_generation/evaluate_card_quality.py --cards data/train_notecard.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from generate_notecards import DEFAULT_DATA_DIR, FIELDS, candidate_note_ids

FIELD_NAMES = {
    "类目": "Category",
    "关键词": "Keywords",
    "场景": "Scenario",
    "人群": "Audience",
    "实体": "Entities",
    "摘要": "Gist",
}
_ALIASES = {
    "类目": "类目", "category": "类目",
    "关键词": "关键词", "keywords": "关键词",
    "场景": "场景", "scene": "场景", "scenario": "场景",
    "人群": "人群", "audience": "人群",
    "实体": "实体", "entities": "实体",
    "摘要": "摘要", "gist": "摘要",
}
_RE_LINE = re.compile(
    r"^\s*(类目|关键词|场景|人群|实体|摘要|category|keywords|scene|scenario|audience|entities|gist)\s*[:：]\s*(.*)$",
    flags=re.IGNORECASE,
)
NONE_MARKERS = {"", "无", "none", "n/a", "na", "null", "暂无", "未提及", "不详", "-"}


def parse_fields(text: Any) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in str(text or "").splitlines():
        match = _RE_LINE.match(line)
        if match:
            field = _ALIASES[match.group(1).lower()]
            fields.setdefault(field, match.group(2).strip())
    return fields


def is_valid_value(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().strip("。.").strip().lower() not in NONE_MARKERS


def audit(cards: dict[str, str], note_ids: list[str]) -> dict[str, Any]:
    parsed = {nid: parse_fields(cards[nid]) for nid in note_ids}
    schema = [f for f in FIELDS if any(f in p for p in parsed.values())]
    n = len(parsed)
    broken = sum(any(f not in p for f in schema) for p in parsed.values())
    complete = sum(all(is_valid_value(p.get(f)) for f in schema) for p in parsed.values())
    per_field = {
        FIELD_NAMES[f]: (sum(is_valid_value(p.get(f)) for p in parsed.values()) / n if f in schema else None)
        for f in FIELDS
    }
    return {
        "cards_audited": n,
        "format_broken": broken,
        "format_broken_rate": broken / n if n else 0.0,
        "fully_complete_rate": complete / n if n else 0.0,
        "field_validity": per_field,
    }


def to_markdown(result: dict[str, Any]) -> str:
    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{100 * x:.2f}%"

    lines = [
        f"Cards audited: {result['cards_audited']:,}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Format-broken rate | {pct(result['format_broken_rate'])} |",
        f"| Fully complete rate | {pct(result['fully_complete_rate'])} |",
    ]
    lines += [f"| {name} | {pct(value)} |" for name, value in result["field_validity"].items()]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit the format and field validity of generated note cards.")
    p.add_argument("--cards", type=Path, required=True, help="Card JSON (note_idx -> card text).")
    p.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--split", choices=["train", "test"], default=None,
                   help="Restrict the audit to the candidates of a query slice; default audits every card.")
    p.add_argument("--query_start", type=int, default=0)
    p.add_argument("--query_end", type=int, default=None)
    p.add_argument("--rerank_depth", type=int, default=100, help="Candidates per query (0 = all).")
    p.add_argument("--output", type=Path, default=None, help="Optional JSON report path (a .md table is written next to it).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with args.cards.open("r", encoding="utf-8") as f:
        cards: dict[str, str] = json.load(f)

    if args.split is None:
        note_ids = sorted(cards, key=int)
        scope = "all cards in file"
    else:
        pool = candidate_note_ids(args.data_dir, args.split, args.query_start, args.query_end, args.rerank_depth)
        note_ids = [str(nid) for nid in pool if str(nid) in cards]
        scope = (f"candidates of search_{args.split}[{args.query_start}:{args.query_end}] "
                 f"with a card ({len(note_ids):,} of {len(pool):,})")

    result = {"cards_file": args.cards.as_posix(), "scope": scope, **audit(cards, note_ids)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()
    print(to_markdown(result))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        args.output.with_suffix(".md").write_text(f"Scope: {scope}\n\n{to_markdown(result)}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
