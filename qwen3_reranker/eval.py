"""
Evaluate a fine-tuned Qwen3-Reranker-0.6B checkpoint on Qilin search_test[0:1000].

Author: Xinlu Lin. Adapted from https://github.com/AzumaLin/qilin-note-rerank
(paths now point to the repository-level data/ folder).

Scores candidates using yes/no logit difference at the last token position.

Usage (from the repository root):
    python qwen3_reranker/eval.py --ckpt_dir outputs/qwen3_reranker/run1_qwen3_ft_summary --infer_mode summary
    python qwen3_reranker/eval.py --ckpt_dir outputs/qwen3_reranker/run1_qwen3_ft_summary --infer_mode original
"""

import os
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')

os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

_SYSTEM = (
    'Judge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)
_INSTRUCT = (
    'Given a Chinese social media search query, '
    'retrieve relevant posts that match the search intent'
)


def get_config(ckpt_dir: str, infer_mode: str, data_dir: str = DATA_DIR,
               sample_start: int = 0, sample_end: int = 1000) -> dict:
    return {
        'infer_mode':   infer_mode,
        'model_path':   ckpt_dir,

        'notes_dir':    os.path.join(data_dir, 'notes'),
        'test_path':    os.path.join(data_dir, 'search_test', 'train-00000-of-00001.parquet'),
        'summary_path': os.path.join(data_dir, 'test_notecard.json'),
        'qrels_path':   os.path.join(data_dir, 'qrels', 'search.test.qrels.csv'),

        'sample_start': sample_start,
        'sample_end':   sample_end,
        'max_length':   384 if infer_mode == 'combined' else 256,
        'batch_size':   4,

        'output_dir': os.path.join(ckpt_dir, f'eval_infer{infer_mode}'),
    }


def format_input(tokenizer, query: str, document: str) -> str:
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": f"<Instruct>: {_INSTRUCT}\n<Query>: {query}\n<Document>: {document}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def calculate_metrics(sorted_results, qrels, k_list):
    max_k = max(k_list)
    agg   = {k: {"mrr": 0.0, "map_sum": 0.0, "recall": 0.0, "precision": 0.0} for k in k_list}
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
            ap, correct = 0.0, 0
            for i, is_rel in enumerate(hits_k, start=1):
                if is_rel:
                    correct += 1
                    ap += correct / i
            if correct:
                agg[k]["map_sum"] += ap / min(len(relevant), k)
            agg[k]["recall"]    += sum(hits_k) / len(relevant)
            agg[k]["precision"] += sum(hits_k) / k

    if valid == 0:
        return {f"{m}@{k}": 0.0 for k in k_list for m in ["MRR", "MAP", "Recall", "Precision"]}

    out = {"_valid_queries": valid}
    for k in k_list:
        out[f"MRR@{k}"]       = agg[k]["mrr"]      / valid
        out[f"MAP@{k}"]       = agg[k]["map_sum"]   / valid
        out[f"Recall@{k}"]    = agg[k]["recall"]    / valid
        out[f"Precision@{k}"] = agg[k]["precision"] / valid
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir',   required=True,
                        help='Path to fine-tuned checkpoint (e.g. outputs/qwen3_reranker/run1_qwen3_ft_summary)')
    parser.add_argument('--infer_mode', choices=['original', 'summary', 'combined'],
                        default='original')
    parser.add_argument('--data_dir',     type=str, default=DATA_DIR)
    parser.add_argument('--sample_start', type=int, default=0)
    parser.add_argument('--sample_end',   type=int, default=1000)
    args = parser.parse_args()

    config = get_config(args.ckpt_dir, args.infer_mode, args.data_dir,
                        args.sample_start, args.sample_end)
    os.makedirs(config['output_dir'], exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device:     {device}")
    print(f"Checkpoint: {config['model_path']}")
    print(f"infer_mode: {config['infer_mode']}")

    if not os.path.isdir(config['model_path']):
        print(f'\n[ERROR] checkpoint not found: {config["model_path"]}')
        raise SystemExit(1)

    print('\n加载 notes corpus...')
    notes_df = pd.concat([
        pd.read_parquet(os.path.join(config['notes_dir'], f))
        for f in sorted(os.listdir(config['notes_dir']))
        if f.endswith('.parquet')
    ], ignore_index=True)
    print(f'  {len(notes_df):,} 条')

    print('加载 search_test...')
    test_df = pd.read_parquet(config['test_path']).iloc[
        config['sample_start']:config['sample_end']
    ]
    print(f'  {len(test_df)} 条 queries')

    summaries: dict[int, str] = {}
    if config['infer_mode'] in ('summary', 'combined'):
        if os.path.exists(config['summary_path']):
            print('加载 summaries...')
            with open(config['summary_path'], 'r', encoding='utf-8') as f:
                raw = json.load(f)
            summaries = {int(k): str(v) for k, v in raw.items()}
            print(f'  {len(summaries):,} 条')
        else:
            print(f'  [警告] summaries 不存在，fallback 到原文: {config["summary_path"]}')

    def get_text(note_idx: int) -> str:
        row      = notes_df.iloc[note_idx]
        title    = str(row['note_title']   or '').strip()
        content  = str(row['note_content'] or '').strip()
        original = (title + '\n' + content).strip()
        card = summaries.get(note_idx, '')
        if config['infer_mode'] == 'summary':
            return card if card else original
        if config['infer_mode'] == 'combined':
            return (card + '\n' + original).strip() if card else original
        return original

    print('加载 qrels...')
    qrels_df = pd.read_csv(config['qrels_path'])
    qrels: dict[int, set[int]] = {}
    for _, r in qrels_df.iterrows():
        qid = int(r['qid'])
        if config['sample_start'] <= qid < config['sample_end']:
            qrels.setdefault(qid, set()).add(int(r['pid']))
    print(f'  有 qrels 的 queries: {len(qrels)}')

    print(f'\n加载模型: {config["model_path"]}')
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_path'], trust_remote_code=True, padding_side='left',
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id  = tokenizer.convert_tokens_to_ids("yes")

    model = AutoModelForCausalLM.from_pretrained(
        config['model_path'], dtype=torch.float16,
        trust_remote_code=True, local_files_only=True,
    ).to(device)
    model.eval()

    print('\n开始推理...')
    sorted_results: dict[int, list[int]] = {}

    for qid, row in enumerate(tqdm(test_df.itertuples(), total=len(test_df), desc='Scoring'),
                              start=config['sample_start']):
        query    = str(row.query)
        details  = row.search_result_details_with_idx
        note_ids = [int(d['note_idx']) for d in details]
        if not note_ids:
            continue

        all_scores = []
        for start in range(0, len(note_ids), config['batch_size']):
            batch_ids   = note_ids[start: start + config['batch_size']]
            batch_texts = [format_input(tokenizer, query, get_text(idx)) for idx in batch_ids]

            encoded = tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=config['max_length'], return_tensors='pt',
            ).to(device)

            with torch.no_grad():
                logits      = model(**encoded).logits
                last_logits = logits[:, -1, :].float()
                scores      = last_logits[:, token_true_id] - last_logits[:, token_false_id]

            all_scores.extend(scores.cpu().tolist())

        sorted_results[qid] = [nid for nid, _ in sorted(
            zip(note_ids, all_scores), key=lambda x: x[1], reverse=True
        )]

    results_path = os.path.join(config['output_dir'], 'rerank_results.csv')
    with open(results_path, 'w') as f:
        f.write('qid,pid,rank\n')
        for qid, pids in sorted_results.items():
            for rank, pid in enumerate(pids, start=1):
                f.write(f'{qid},{pid},{rank}\n')
    print(f'推理结果已保存: {results_path}')

    metrics = calculate_metrics(sorted_results, qrels, [10, 100])

    print('\n' + '=' * 55)
    print(f'Checkpoint: {os.path.basename(config["model_path"])}  infer={config["infer_mode"]}')
    print('=' * 55)
    for key, val in metrics.items():
        if not key.startswith('_'):
            print(f'  {key:<18} {val:.4f}')
    print(f'  valid queries:    {metrics.get("_valid_queries", 0)}')
    print('=' * 55)

    metrics_path = os.path.join(config['output_dir'], 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({'ckpt_dir': config['model_path'], 'infer_mode': config['infer_mode'],
                   'metrics': metrics}, f, ensure_ascii=False, indent=2)
    print(f'指标已保存: {metrics_path}')


if __name__ == '__main__':
    main()
