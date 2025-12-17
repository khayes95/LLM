#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/configs/env.local.sh"

: "${UQ_USE_SAMPLE:=1}"           # Start with the sample split to verify the pipeline (including the judge) works end-to-end.
: "${MAX_EXAMPLES:=10}"


export UQ_JUDGE_BASE_URL="${UQ_JUDGE_BASE_URL:-${UQ_BASE_URL:-}}"
export UQ_JUDGE_API_KEY="${UQ_JUDGE_API_KEY:-${UQ_API_KEY:-${KEY:-}}}"
export UQ_JUDGE_MODEL_NAME="${UQ_JUDGE_MODEL_NAME:-${UQ_MODEL_NAME:-gpt-4o}}"

## By default, only evaluate level:example rubrics; set to 1 to include cluster-level rubrics as well.
export UQ_HEALTHBENCH_INCLUDE_CLUSTER="${UQ_HEALTHBENCH_INCLUDE_CLUSTER:-0}"

python -m uq_eval.cli \
  --model_backend chat_http \
  --model_name "${UQ_MODEL_NAME:-gpt-4o}" \
  --bench healthbench_main \
  --split test \
  --max_examples "${MAX_EXAMPLES}"
