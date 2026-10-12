#!/usr/bin/env python3
"""
LLM-based note-card generation (paper, Section 3).

Each Qilin note (title + body) is denoised with regular expressions and then
compressed by an instruction-tuned LLM into a six-field structured note card:

    类目 (category) / 关键词 (keywords) / 场景 (search scenario) /
    人群 (target audience) / 实体 (named entities) / 摘要 (one-sentence gist)

Generation is training-free: a fixed prompt (field definitions, field-level
rules, five author-written exemplars) and greedy decoding. The prompt, the
denoising rules and the decoding settings are the ones used to produce the
released cards with Qwen/Qwen3-4B-Instruct-2507 (prompt by Xinlu Lin, from
summarize_notes.py in https://github.com/AzumaLin/qilin-note-rerank).

The run is resumable: cards that already exist in --output and pass the format
check are kept, missing or malformed cards are (re)generated, and the output is
saved atomically every --save_every batches.

Usage (from the repository root):
    # candidates of search_train[0:5000] (training + validation queries)
    python notecard_generation/generate_notecards.py --split train --query_end 5000 \
        --output data/train_notecard.json

    # exposed candidates of search_test[0:6192] (test pool + expanded corpus)
    python notecard_generation/generate_notecards.py --split test --query_end 6192 \
        --output data/test_notecard.json

    # the smaller generator of the reliability comparison
    python notecard_generation/generate_notecards.py --split test --query_end 1000 \
        --llm_model Qwen/Qwen2.5-3B-Instruct --output outputs/notecards_qwen2.5-3b/test_notecard.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"

# Card fields, in output order.
FIELDS = ("类目", "关键词", "场景", "人群", "实体", "摘要")
_RE_FIELD = re.compile(r"^\s*(类目|关键词|场景|人群|实体|摘要)\s*[:：]\s*(.*)$")

# ── Regex denoising ───────────────────────────────────────────────────────────
# Topic hashtags in Xiaohongshu markup: #话题文本[话题]#
_RE_HASHTAG = re.compile(r"#[^#\[\]]*\[话题\]#")
# Platform emoticon markers such as [笑哭R]
_RE_PLATFORM_MARKER = re.compile(r"\[[^\[\]]*R\]")
# Unicode emoji ranges
_RE_EMOJI = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended-A
    "\U00002600-\U000026FF"  # misc symbols
    "\U0000200D"             # zero width joiner
    "\U00002B50"             # star
    "\U000023F0-\U000023FA"  # misc technical
    "\U0000203C-\U00003299"  # enclosed CJK
    "]+",
    flags=re.UNICODE,
)
_RE_WHITESPACE = re.compile(r"\s+")


def regex_denoise(text: str) -> str:
    """Remove topic hashtags, platform markers, emoji and redundant whitespace."""
    text = _RE_HASHTAG.sub("", text)
    text = _RE_PLATFORM_MARKER.sub("", text)
    text = _RE_EMOJI.sub("", text)
    return _RE_WHITESPACE.sub(" ", text).strip()


def text_or_empty(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN
        return ""
    return str(value).strip()


# ── Prompt ────────────────────────────────────────────────────────────────────
def build_prompt(title: str, content: str) -> str:
    return (
        "你是搜索摘要助手。将小红书帖子总结成结构化卡片，用于搜索匹配。\n"
        "严格按照以下格式输出，不要输出任何其他内容：\n"
        "类目: <类目>\n"
        "关键词: <k1, k2, k3, k4, k5>\n"
        "场景: <用户在什么情况下会搜索这个，10字以内>\n"
        "人群: <目标人群>\n"
        "实体: <品牌、产品、人物、地点>\n"
        "摘要: <一句话，适合搜索匹配的简洁描述>\n\n"
        "规则：\n"
        "1. 关键词：混合类目词、属性词、用户实际会搜索的意图词，共5个\n"
        "2. 场景：描述用户搜索场景，10字以内（例：小个子找冬季显瘦穿搭）\n"
        "3. 实体：只提取标题、正文、标签中明确出现的品牌/产品/人物/地点，没有则写「无」\n"
        "4. 摘要：用简洁陈述句改写，去除网络用语和营销夸张词\n"
        "5. 标签（#...#）：提取地点标签、活动标签、品牌标签作为实体或关键词，忽略#日常分享# #笔记灵感#等无意义标签\n\n"
        "示例1：\n"
        "Title: 蕉内羽绒服穿搭显瘦又保暖\n"
        "Body: 3件都是蕉内氢气羽绒服502Cloud，实穿又保暖，不怕臃肿，短款小个子友好\n"
        "类目: 穿搭\n"
        "关键词: 羽绒服, 短款, 显瘦, 保暖, 小个子, 冬季穿搭, 蕉内, 不臃肿\n"
        "场景: 小个子找冬季显瘦保暖穿搭\n"
        "人群: 小个子女生, 冬季穿搭爱好者\n"
        "实体: 蕉内, 氢气羽绒服502Cloud\n"
        "摘要: 蕉内短款羽绒服穿搭，显瘦保暖，小个子友好\n\n"
        "示例2：\n"
        "Title: nan\n"
        "Body: 一分钟教会你国补和省补的区别 #买车那点事儿# #新手买车攻略# #新能源汽车补贴# #买车推荐#\n"
        "类目: 汽车\n"
        "关键词: 国补, 省补, 新能源补贴, 买车攻略, 购车补贴, 新手买车, 补贴区别, 新能源汽车\n"
        "场景: 买新能源车想了解国补省补怎么算\n"
        "人群: 准备购车的新手, 新能源车买家\n"
        "实体: 无\n"
        "摘要: 一分钟了解国补和省补的区别，新手买车新能源补贴攻略\n\n"
        "示例3：\n"
        "Title: 追星女行李托运保姆级教程‼️千万不要被扔啊\n"
        "Body: #追星女行李托运教程# #时代少年团五周年# #追星女必看# #时代少年团重庆五周年演唱会#\n"
        "类目: 演唱会\n"
        "关键词: 追星, 行李托运, 演唱会, 保姆级教程, 时代少年团, 重庆演唱会, 五周年, 行李不丢\n"
        "场景: 追星去演唱会想知道行李怎么托运\n"
        "人群: 追星女孩, 时代少年团粉丝\n"
        "实体: 时代少年团, 时代少年团重庆五周年演唱会\n"
        "摘要: 追星女演唱会行李托运保姆级教程，时代少年团重庆五周年演唱会适用\n\n"
        "示例4：\n"
        "Title: 在南京也可以吃到内蒙古奶疙瘩噜～\n"
        "Body: #南京国际博览中心# #南京中外商品博览会# #内蒙古奶疙瘩# #乌梨# #文玩葡萄# #糕点软软糯糯# #日常分享#\n"
        "类目: 美食\n"
        "关键词: 奶疙瘩, 内蒙古特产, 南京, 博览会, 特色食品, 南京美食, 乌梨, 糕点\n"
        "场景: 在南京想找内蒙古特产或博览会美食\n"
        "人群: 对内蒙古特产感兴趣的人, 南京本地人\n"
        "实体: 南京国际博览中心, 南京中外商品博览会\n"
        "摘要: 南京中外商品博览会可以品尝到内蒙古奶疙瘩、乌梨等特产糕点\n\n"
        "示例5：\n"
        "Title: nan\n"
        "Body: 我只是想同步个微信步数而已，不是让你记录我运动，立马把手环取消绑定 #华为手环9# #华为运动健康#\n"
        "类目: 数码\n"
        "关键词: 华为手环, 微信步数, 手环绑定, 运动记录, 取消绑定, 华为手环9, 步数同步, 隐私\n"
        "场景: 华为手环同步微信步数出现问题想解决\n"
        "人群: 华为手环用户, 关注隐私的用户\n"
        "实体: 华为手环9, 华为运动健康\n"
        "摘要: 华为手环9同步微信步数时会记录运动数据，可取消绑定关闭\n\n"
        f"Title: {title}\n"
        f"Body: {content}\n"
    )


# ── Card format check ─────────────────────────────────────────────────────────
def parse_card(text: Any) -> dict[str, str]:
    """Map each field label found in a card to its (stripped) value."""
    fields: dict[str, str] = {}
    for line in str(text or "").splitlines():
        match = _RE_FIELD.match(line)
        if match and match.group(1) not in fields:
            fields[match.group(1)] = match.group(2).strip()
    return fields


def is_valid_card(text: Any) -> bool:
    """A card is well-formed when all six field labels parse."""
    return all(field in parse_card(text) for field in FIELDS)


# ── Data ──────────────────────────────────────────────────────────────────────
def candidate_note_ids(
    data_dir: Path,
    split: str,
    query_start: int,
    query_end: int | None,
    depth: int | None,
) -> list[int]:
    """Union of the exposed candidates (search_results and result details) of a query slice."""
    import pandas as pd

    path = data_dir / f"search_{split}" / "train-00000-of-00001.parquet"
    df = pd.read_parquet(path).iloc[query_start:query_end]
    ids: set[int] = set()
    for column in ("search_results", "search_result_details_with_idx"):
        if column not in df.columns:
            continue
        for value in df[column]:
            if value is None:
                continue
            items = list(value)
            if depth:  # 0 / None = every candidate
                items = items[:depth]
            for item in items:
                if isinstance(item, dict):
                    ids.add(int(item["note_idx"]))
                elif hasattr(item, "__len__") and not isinstance(item, str):
                    ids.add(int(item[0]))
                else:
                    ids.add(int(item))
    return sorted(ids)


def load_notes(data_dir: Path, note_ids: set[int]) -> dict[int, tuple[str, str]]:
    """Denoised (title, content) for the requested notes, read shard by shard."""
    import pandas as pd

    notes: dict[int, tuple[str, str]] = {}
    for path in sorted((data_dir / "notes").glob("*.parquet")):
        df = pd.read_parquet(path, columns=["note_idx", "note_title", "note_content"])
        df = df[df["note_idx"].isin(note_ids)]
        for row in df.itertuples(index=False):
            notes[int(row.note_idx)] = (
                regex_denoise(text_or_empty(row.note_title)),
                regex_denoise(text_or_empty(row.note_content)),
            )
        if len(notes) == len(note_ids):
            break
    return notes


def load_json(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


# ── LLM ───────────────────────────────────────────────────────────────────────
def load_llm(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return tokenizer, model, device


def summarize_batch(prompts: list[str], tokenizer, model, device: str, max_new_tokens: int) -> list[str]:
    """Greedy decoding of one batch of chat-formatted prompts."""
    import torch

    texts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    return [tokenizer.decode(output[input_len:], skip_special_tokens=True).strip() for output in outputs]


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate structured note cards with an instruction-tuned LLM.")
    p.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR,
                   help="Qilin data folder with notes/, search_train/, search_test/.")
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--query_start", type=int, default=0)
    p.add_argument("--query_end", type=int, default=None, help="Exclusive end of the query slice (default: all).")
    p.add_argument("--rerank_depth", type=int, default=100, help="Candidates per query to cover (0 = all).")
    p.add_argument("--output", type=Path, default=None,
                   help="Card JSON (note_idx -> card). Default: data/{split}_notecard.json")
    p.add_argument("--llm_model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=350)
    p.add_argument("--save_every", type=int, default=50, help="Save after this many batches.")
    p.add_argument("--limit", type=int, default=0, help="Generate at most this many cards (0 = no limit).")
    p.add_argument("--dry_run", action="store_true", help="Only report how many cards would be generated.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.data_dir / f"{args.split}_notecard.json"
    report_path = output.with_name(output.stem + "_report.json")

    target_ids = candidate_note_ids(args.data_dir, args.split, args.query_start, args.query_end, args.rerank_depth)
    cards = load_json(output)
    todo = [nid for nid in target_ids if not is_valid_card(cards.get(str(nid)))]
    existing_valid = len(target_ids) - len(todo)
    if args.limit > 0:
        todo = todo[: args.limit]
    print(f"[notecard] split={args.split} queries=[{args.query_start}:{args.query_end}] "
          f"targets={len(target_ids):,} existing_valid={existing_valid:,} todo={len(todo):,}", flush=True)

    report: dict[str, Any] = {
        "split": args.split,
        "query_range": [args.query_start, args.query_end],
        "rerank_depth": args.rerank_depth,
        "llm_model": args.llm_model,
        "output": output.as_posix(),
        "target_notes": len(target_ids),
        "existing_valid_at_start": existing_valid,
        "todo_at_start": len(todo),
    }
    if todo and not args.dry_run:
        from tqdm import tqdm

        notes = load_notes(args.data_dir, set(todo))
        missing = [nid for nid in todo if nid not in notes]
        if missing:
            print(f"[notecard] warning: {len(missing)} target notes not found in notes/*.parquet", flush=True)
        todo = [nid for nid in todo if nid in notes]

        tokenizer, model, device = load_llm(args.llm_model)
        started = time.time()
        for batch_no, start in enumerate(tqdm(range(0, len(todo), args.batch_size), desc="Generating"), start=1):
            batch_ids = todo[start : start + args.batch_size]
            prompts = [build_prompt(*notes[nid]) for nid in batch_ids]
            for nid, card in zip(batch_ids, summarize_batch(prompts, tokenizer, model, device, args.max_new_tokens)):
                cards[str(nid)] = card
            if batch_no % args.save_every == 0:
                save_json(output, cards)
        save_json(output, cards)
        report["generation_seconds"] = round(time.time() - started, 1)

    valid = sum(is_valid_card(cards.get(str(nid))) for nid in target_ids)
    report.update({
        "target_valid": valid,
        "target_missing_or_malformed": len(target_ids) - valid,
        "target_coverage": valid / len(target_ids) if target_ids else 0.0,
        "cards_in_file": len(cards),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    if not args.dry_run:
        save_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
