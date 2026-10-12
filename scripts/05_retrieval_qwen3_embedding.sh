#!/usr/bin/env bash
# Step 5 - Expanded-pool retrieval with Qwen3-Embedding-0.6B (paper, Table 2).
#
# Official Qwen3-Embedding usage (--qwen_official_style): instruction-prefixed
# queries, last-token pooling, left padding, L2-normalised embeddings.
# Zero-shot evaluation, hard-negative mining, then LoRA fine-tuning per training
# representation; the learning rate is selected on validation (best: 2e-6).
# The paper runs used one 71 GB GPU slice (bf16); zero-shot fits on 18 GB.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=outputs/qwen3_embedding_retrieval
TRAIN_MODES=(original combined)
QWEN=(
  --model_name Qwen/Qwen3-Embedding-0.6B --qwen_official_style --output_dir "$OUT"
  --score_block_size 64 --max_length 256 --bf16 --normalize_embeddings
)

# (1) Zero-shot, validation + test
python src/qwen3_embedding_global_ft.py --stage zero_shot "${QWEN[@]}" --eval_batch_size 4 \
  --eval_splits val test --eval_modes original summary combined

# (2) Hard negatives mined with the zero-shot model
python src/qwen3_embedding_global_ft.py --stage mine "${QWEN[@]}" --eval_batch_size 32 \
  --mine_modes "${TRAIN_MODES[@]}" --force

# (3) LoRA fine-tuning (r=16, alpha=32), validation + test
for LR in 2e-6 1e-6; do
  for MODE in "${TRAIN_MODES[@]}"; do
    python src/qwen3_embedding_global_ft.py --stage train "${QWEN[@]}" --eval_batch_size 32 \
      --train_batch_size 4 --grad_accum_steps 8 --negative_samples 8 --epochs 1 --gradient_checkpointing \
      --negative_strategy hard_mined_only --lr "$LR" --train_mode "$MODE" \
      --eval_splits val test --eval_all_modes --run_name "qwen_${MODE}_lr${LR}_lora"
  done
done
python src/qwen3_embedding_global_ft.py --stage aggregate --output_dir "$OUT"
