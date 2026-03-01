#!/bin/bash
# Run HLE multimodal evaluation with GPT-5-mini
# Uses only questions that explicitly reference images (~500)

set -e

python -m uq_eval.cli \
    --bench hle \
    --hle_with_images \
    --model_backend openai \
    --model_name gpt-5-mini \
    --max_examples 100 \
    --seed 42 \
    --timeout_s 300 \
    --max_output_tokens 16384

echo "HLE multimodal evaluation complete"
