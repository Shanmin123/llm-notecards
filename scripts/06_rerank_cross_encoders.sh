#!/usr/bin/env bash
# Step 6 - Candidate-pool reranking with MacBERT / RoBERTa cross-encoders (paper, Table 3).
#
# One cross-encoder per training representation (CLS pooling + linear head,
# BCE loss, AdamW 2e-5, 1 epoch), then every checkpoint re-scores each query's
# exposed candidates (search_results of search_test[0:1000]) with every
# inference representation. title_summary is an extra ablation that is not
# reported in the paper; drop it from MODES to save time.
# Fits on a 12 GB GPU (fp16, gradient checkpointing).
set -euo pipefail
cd "$(dirname "$0")/.."

MODES=(original summary title_summary combined)
SPLIT=(--test_start 0 --test_end 1000 --candidate_key search_results --candidate_depth 1000 --missing_policy error)

for SPEC in "macbert hfl/chinese-macbert-base" "roberta hfl/chinese-roberta-wwm-ext"; do
  read -r NAME MODEL <<< "$SPEC"
  TRAIN_DIR="outputs/rerank_cross_encoder/$NAME/train"

  python src/rerank_cross_encoder.py --model_name_or_path "$MODEL" --modes "${MODES[@]}" \
    --train_start 0 --train_end 4000 --val_start 4000 --val_end 5000 --eval_val "${SPLIT[@]}" \
    --negative_policy qilin_official_nonzero --negative_samples 10 --epochs 1 \
    --train_batch_size 2 --eval_batch_size 8 --max_length 512 \
    --optimizer adamw --lr 2e-5 --weight_decay 0.01 --scheduler linear --linear_total_iters 10 \
    --fp16 --gradient_checkpointing --output_dir "$TRAIN_DIR"

  python src/rerank_cross_encoder_eval.py --checkpoint_root "$TRAIN_DIR" \
    --trained_modes "${MODES[@]}" --eval_modes "${MODES[@]}" "${SPLIT[@]}" \
    --eval_batch_size 8 --max_length 512 --fp16 \
    --output_dir "outputs/rerank_cross_encoder/$NAME/cross_eval"
done
