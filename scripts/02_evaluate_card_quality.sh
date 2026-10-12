#!/usr/bin/env bash
# Step 2 - Generation quality audit (paper, Section 4).
#
# Audits the first-pass cards of the test exposure pool (top-100 candidates of
# search_test[0:1000], 22,598 cards): format-broken rate, fully-complete rate
# and per-field validity.
set -euo pipefail
cd "$(dirname "$0")/.."

CARDS="${CARDS:-data/test_notecard_first_pass.json}"

python notecard_generation/evaluate_card_quality.py --cards "$CARDS" \
  --split test --query_start 0 --query_end 1000 \
  --output outputs/notecard_quality/test_pool.json
