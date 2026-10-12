#!/usr/bin/env bash
# Step 1 - LLM note-card generation (paper, Section 3).
#
# The released cards (data/README.md) were produced with exactly this setup.
# Regenerating them is expensive (about 10 s per card at batch size 1 on a
# 12 GB laptop GPU), so download them unless you need new cards. The script is
# resumable: re-running it only fills in missing or malformed cards.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"   # Qwen/Qwen2.5-3B-Instruct for the smaller generator
BATCH_SIZE="${BATCH_SIZE:-4}"

# Candidates of the training and validation queries, search_train[0:5000]
python notecard_generation/generate_notecards.py --split train --query_start 0 --query_end 5000 \
  --llm_model "$MODEL" --batch_size "$BATCH_SIZE" --output data/train_notecard.json

# Top-100 exposed candidates of search_test[0:6192] (test pool + expanded retrieval corpus)
python notecard_generation/generate_notecards.py --split test --query_start 0 --query_end 6192 \
  --llm_model "$MODEL" --batch_size "$BATCH_SIZE" --output data/test_notecard.json

# Every exposed candidate of the evaluation queries search_test[0:1000], so that
# the reranking pool has full card coverage
python notecard_generation/generate_notecards.py --split test --query_start 0 --query_end 1000 --rerank_depth 0 \
  --llm_model "$MODEL" --batch_size "$BATCH_SIZE" --output data/test_notecard.json
