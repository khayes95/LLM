#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -f configs/env.local.sh ]; then
  cp configs/env.example.sh configs/env.local.sh
  echo "Created configs/env.local.sh. Please edit it (set UQ_API_KEY / UQ_BASE_URL / UQ_MODEL_NAME) then rerun." >&2
  exit 1
fi

source configs/env.local.sh

# Default to your real CSV path (you can override in env.local.sh)
export UQ_GPQA_DATA_PATH="${UQ_GPQA_DATA_PATH:-data/gpqa/gpqa_diamond.csv}"

# default small run to avoid cost; pass a number to run more
MAX_EXAMPLES="${1:-20}"

python -m uq_eval.cli \
  --model_backend chat_http \
  --model_name "${UQ_MODEL_NAME:-gpt-4o}" \
  --bench gpqa_diamond \
  --max_examples "${MAX_EXAMPLES}"
