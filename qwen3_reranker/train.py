"""
Fine-tune Qwen3-Reranker-0.6B on Qilin search data using SFT.

Author: Xinlu Lin. Adapted from https://github.com/AzumaLin/qilin-note-rerank
(paths now point to the repository-level data/ and outputs/ folders).

Loss: cross-entropy on yes/no token at position -2.
Positive samples: any engagement signal (click/collect/like/comment/share > 0).

Modes:
  original -- note title + content
  summary  -- LLM-generated notecard (train_notecard.json, fallback to original)
  combined -- notecard + original text

Usage (from the repository root):
    python qwen3_reranker/train.py --mode original --prefix run1
    python qwen3_reranker/train.py --mode summary  --prefix run1
    python qwen3_reranker/train.py --mode combined --prefix run1
"""

import os
import json
import random
import argparse
import functools

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_linear_schedule_with_warmup

POSITIVE_FIELDS = ('click', 'collect', 'like', 'comment', 'share')

_SYSTEM = (
    'Judge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)
_INSTRUCT = (
    'Given a Chinese social media search query, '
    'retrieve relevant posts that match the search intent'
)


def get_config(mode: str, neg_per_pos: int, prefix: str, lr: float, data_dir: str = DATA_DIR) -> dict:
    return {
        'mode': mode,

        'model_name':   'Qwen/Qwen3-Reranker-0.6B',
        'notes_dir':    os.path.join(data_dir, 'notes'),
        'train_path':   os.path.join(data_dir, 'search_train', 'train-00000-of-00001.parquet'),
        'summary_path': os.path.join(data_dir, 'train_notecard.json'),

        'data_start':   0,
        'data_end':     5000,
        'train_size':   4000,
        'val_size':     1000,

        'max_length':   384 if mode == 'combined' else 256,
        'batch_size':   2,
        'epochs':       3,
        'lr':           lr,
        'warmup_ratio': 0.1,
        'neg_per_pos':  neg_per_pos,
        'seed':         42,

        'output_dir': os.path.join(PROJECT_ROOT, 'outputs', 'qwen3_reranker', f'{prefix}_qwen3_ft_{mode}'),
    }


def format_input(tokenizer, query: str, document: str) -> str:
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": f"<Instruct>: {_INSTRUCT}\n<Query>: {query}\n<Document>: {document}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


class RerankDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, tokenizer, max_length, token_true_id, token_false_id):
    texts    = [text for text, _ in batch]
    labels   = torch.tensor([token_true_id if pos else token_false_id for _, pos in batch],
                             dtype=torch.long)
    suffixes = ["yes" if pos else "no" for _, pos in batch]
    encoded  = tokenizer(
        [t + s for t, s in zip(texts, suffixes)],
        padding=True, truncation=True,
        max_length=max_length, return_tensors='pt',
    )
    return encoded, labels


def build_samples(rows, notes_df, summaries, mode, neg_per_pos, tokenizer):
    samples = []

    def get_text(note_idx):
        row      = notes_df.iloc[note_idx]
        title    = str(row['note_title']   or '').strip()
        content  = str(row['note_content'] or '').strip()
        original = (title + '\n' + content).strip()
        if mode == 'summary':
            return summaries.get(note_idx, original)
        if mode == 'combined':
            card = summaries.get(note_idx, '')
            return (card + '\n' + original).strip() if card else original
        return original

    for row in rows:
        query   = str(row['query'])
        details = row['search_result_details_with_idx']

        positives, negatives = [], []
        for d in details:
            note_idx = int(d['note_idx'])
            is_pos = any(float(d.get(f, 0) or 0) > 0 for f in POSITIVE_FIELDS)
            (positives if is_pos else negatives).append(note_idx)

        if not positives or not negatives:
            continue

        random.shuffle(negatives)
        for pos_idx in positives:
            samples.append((format_input(tokenizer, query, get_text(pos_idx)), True))
        for neg_idx in negatives[:len(positives) * neg_per_pos]:
            samples.append((format_input(tokenizer, query, get_text(neg_idx)), False))

    return samples


def run_epoch(model, dataloader, token_false_id, token_true_id,
              optimizer, scheduler, device, train=True, accum_steps=2):
    model.train() if train else model.eval()
    total_loss = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        pbar = tqdm(dataloader, desc='Train' if train else 'Val', leave=False)
        for step, (encoded, labels) in enumerate(pbar):
            encoded = {k: v.to(device) for k, v in encoded.items()}
            labels  = labels.to(device)

            hidden = model.model(**encoded).last_hidden_state
            pred   = model.lm_head(hidden[:, -2, :]).float()
            loss   = F.cross_entropy(pred, labels) / accum_steps

            if train:
                loss.backward()
                if (step + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            total_loss += loss.item() * accum_steps
            pbar.set_postfix({'loss': f'{loss.item() * accum_steps:.4f}'})

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',        choices=['original', 'summary', 'combined'], default='original')
    parser.add_argument('--neg_per_pos', type=int,   default=2)
    parser.add_argument('--prefix',      type=str,   default='run1')
    parser.add_argument('--lr',          type=float, default=1e-5)
    parser.add_argument('--data_dir',    type=str,   default=DATA_DIR)
    args = parser.parse_args()

    config = get_config(args.mode, args.neg_per_pos, args.prefix, args.lr, args.data_dir)
    os.makedirs(config['output_dir'], exist_ok=True)
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device:     {device}")
    print(f"Mode:       {config['mode']}")
    print(f"LR:         {config['lr']}")
    print(f"Output dir: {config['output_dir']}")

    print('\n加载 notes corpus...')
    notes_df = pd.concat([
        pd.read_parquet(os.path.join(config['notes_dir'], f))
        for f in sorted(os.listdir(config['notes_dir']))
        if f.endswith('.parquet')
    ], ignore_index=True)
    print(f'  {len(notes_df):,} 条')

    print(f'加载 search_train [{config["data_start"]}:{config["data_end"]}]...')
    all_df = pd.read_parquet(config['train_path']).iloc[
        config['data_start']:config['data_end']
    ]
    rows       = all_df.to_dict('records')
    train_rows = rows[:config['train_size']]
    val_rows   = rows[config['train_size']:]
    print(f'  训练: {len(train_rows)} 条 | 验证: {len(val_rows)} 条')

    summaries: dict[int, str] = {}
    if config['mode'] in ('summary', 'combined'):
        if os.path.exists(config['summary_path']):
            print('加载 summaries...')
            with open(config['summary_path'], 'r', encoding='utf-8') as f:
                raw = json.load(f)
            summaries = {int(k): str(v) for k, v in raw.items()}
            print(f'  {len(summaries):,} 条')
        else:
            print(f'  [警告] summaries 不存在，fallback 到原文: {config["summary_path"]}')

    print(f'\n加载模型: {config["model_name"]}')
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_name'], trust_remote_code=True, padding_side='left'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = 'left'

    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id  = tokenizer.convert_tokens_to_ids("yes")
    print(f'  yes={token_true_id}, no={token_false_id}')

    print('\n构建训练样本...')
    train_samples = build_samples(train_rows, notes_df, summaries, config['mode'],
                                  config['neg_per_pos'], tokenizer)
    val_samples   = build_samples(val_rows,   notes_df, summaries, config['mode'],
                                  config['neg_per_pos'], tokenizer)
    random.shuffle(train_samples)
    n_pos = sum(1 for _, p in train_samples if p)
    print(f'  训练: {len(train_samples):,}  (正:{n_pos} 负:{len(train_samples)-n_pos})')
    print(f'  验证: {len(val_samples):,}')

    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'], dtype=torch.float32, trust_remote_code=True,
    ).to(device)

    fn = functools.partial(collate_fn, tokenizer=tokenizer, max_length=config['max_length'],
                           token_true_id=token_true_id, token_false_id=token_false_id)
    train_loader = DataLoader(RerankDataset(train_samples), batch_size=config['batch_size'],
                              shuffle=True,  collate_fn=fn)
    val_loader   = DataLoader(RerankDataset(val_samples),   batch_size=config['batch_size'],
                              shuffle=False, collate_fn=fn)

    optimizer    = AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    total_steps  = len(train_loader) * config['epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f'\n开始训练 ({config["epochs"]} epochs)...\n')
    best_val_loss = float('inf')
    best_epoch    = 0

    for epoch in range(1, config['epochs'] + 1):
        train_loss = run_epoch(model, train_loader, token_false_id, token_true_id,
                               optimizer, scheduler, device, train=True)
        val_loss   = run_epoch(model, val_loader,   token_false_id, token_true_id,
                               optimizer, scheduler, device, train=False)

        print(f'Epoch {epoch}/{config["epochs"]}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            model.save_pretrained(config['output_dir'])
            tokenizer.save_pretrained(config['output_dir'])
            print(f'  ✓ 最佳模型已保存 (val_loss={best_val_loss:.4f})')

    print(f'\n训练完成。最佳 epoch: {best_epoch}，val_loss: {best_val_loss:.4f}')
    print(f'模型保存在: {config["output_dir"]}')


if __name__ == '__main__':
    main()
