#!/usr/bin/env bash
# Step 7 - Candidate-pool reranking with Qwen3-Reranker-0.6B (paper, Table 3).
#
# Supervised fine-tuning on the yes/no answer token with engagement positives
# (search_train[0:4000], validation search_train[4000:5000], best checkpoint by
# validation loss), then evaluation of each checkpoint on every inference
# representation over the exposed candidates of search_test[0:1000].
# Full-parameter fp32 training needs a GPU with at least 16 GB.
set -euo pipefail
cd "$(dirname "$0")/.."

PREFIX="${PREFIX:-run1}"
LR="${LR:-1e-5}"

for MODE in original summary combined; do
  python qwen3_reranker/train.py --mode "$MODE" --prefix "$PREFIX" --lr "$LR"
  for INFER in original summary combined; do
    python qwen3_reranker/eval.py --ckpt_dir "outputs/qwen3_reranker/${PREFIX}_qwen3_ft_${MODE}" --infer_mode "$INFER"
  done
done
