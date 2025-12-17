#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# For team sharing: keep env.example.sh as a template; each person copies it to env.local.sh and fills in their own API key.
source "${ROOT}/configs/env.local.sh"

# When UQ_USE_SAMPLE=1, it will automatically read data/.../*.sample.* files.
: "${UQ_USE_SAMPLE:=0}"

python -m uq_eval.cli \
  --model_backend chat_http \
  --model_name "${UQ_MODEL_NAME:-gpt-4o}" \
  --bench simpleqa \
  --split test \
  --max_examples "${MAX_EXAMPLES:-50}"
