#!/usr/bin/env bash
# Step 3 - Expanded-pool retrieval with a zero-shot BERT bi-encoder (paper, Table 2).
#
# Queries search_test[0:1000]; corpus = exposed candidates of search_test[0:6192]
# that have a note card (113,548 notes). Mean pooling, dot-product scores.
# Runs on a 12 GB GPU in about 15 minutes.
set -euo pipefail
cd "$(dirname "$0")/.."

python src/run_expanded_corpus_retrieve.py --models bert_zero \
  --query_start 0 --query_end 1000 --corpus_start 0 --corpus_end 6192 \
  --bert_batch_size 32 --bert_max_length 256 --fp16 \
  --output_dir outputs/expanded_retrieval_zero_shot
