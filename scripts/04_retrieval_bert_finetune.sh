#!/usr/bin/env bash
# Step 4 - Expanded-pool retrieval with a fine-tuned BERT bi-encoder (paper, Table 2).
#
# Full-parameter fine-tuning of bert-base-chinese with in-batch + hard negatives,
# one model per training representation (Original / StructuredConcat), learning
# rate selected on validation (best: 5e-5), then a single test evaluation of the
# selected checkpoints on every inference representation.
# The paper runs used one 71 GB GPU slice (bf16, batch size 32).
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=outputs/bert_bi_retrieval
TRAIN_MODES=(original combined)

# (1) Hard negatives: top-ranked non-relevant notes from zero-shot Qwen3-Embedding-0.6B
#     (mean pooling, raw queries), one file per training representation.
#     The files used in the paper are in the release (bert_hard_negatives/);
#     copy them to $OUT/hard_negatives/ to skip this step.
for MODE in "${TRAIN_MODES[@]}"; do
  if [[ ! -f "$OUT/hard_negatives/$MODE.json" ]]; then
    python src/retrieval_train.py --stage mine --model_name Qwen/Qwen3-Embedding-0.6B \
      --mine_modes "$MODE" --eval_batch_size 4 --max_length 256 --fp16 --output_dir "$OUT"
  fi
done

# (2) Training; validation only (model selection)
COMMON=(
  --stage train --model_name bert-base-chinese --full_param --output_dir "$OUT"
  --train_batch_size 32 --grad_accum_steps 1 --negative_samples 15 --epochs 1
  --negative_strategy hard_mined_plus_exposure
  --eval_batch_size 128 --score_block_size 128 --max_length 256 --bf16 --gradient_checkpointing
  --eval_splits val --eval_all_modes
)
for LR in 2e-5 5e-5; do
  for MODE in "${TRAIN_MODES[@]}"; do
    python src/retrieval_train.py "${COMMON[@]}" --lr "$LR" --train_mode "$MODE" \
      --run_name "bert_${MODE}_lr${LR}"
  done
done
python src/retrieval_train.py --stage aggregate --output_dir "$OUT"

# (3) Test evaluation of the selected learning rate
for MODE in "${TRAIN_MODES[@]}"; do
  python src/retrieval_eval.py --checkpoint_type full --model_name bert-base-chinese \
    --checkpoint_path "$OUT/runs/bert_${MODE}_lr5e-5/model" --run_name "bert_${MODE}_lr5e-5_test" \
    --train_mode "$MODE" --negative_strategy hard_mined_plus_exposure --lr 5e-5 \
    --eval_splits test --eval_modes original summary combined \
    --eval_batch_size 128 --score_block_size 128 --max_length 256 --bf16 \
    --output_dir outputs/bert_bi_retrieval_test
done
