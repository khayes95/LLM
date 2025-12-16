#!/usr/bin/env bash
set -euo pipefail

# load local secrets
source configs/env.local.sh

python -m uq_eval.cli \
  --model_backend chat_http \
  --model_name "${UQ_MODEL_NAME}" \
  --bench dummy_qa \
  --max_examples 3
