#!/bin/bash
# Fill in missing GPT-5-mini samples
cd /scratch/khayes/LLM

echo "=== Running missing GPT-5-mini samples ==="

# TriviaQA: have 123, need 250 (127 more)
echo "TriviaQA: Running 250 samples..."
python -m uq_eval.cli \
    --bench triviaqa \
    --model_backend openai \
    --model_name gpt-5-mini \
    --max_examples 250 \
    --seed 42 \
    --out_dir runs/gpt5_mini_triviaqa_full \
    2>&1 | tee logs/gpt5_triviaqa_full.log

# MMVet: have 100, need 218 (all available)
echo "MMVet: Running all available samples..."
python -m uq_eval.cli \
    --bench mmvet \
    --model_backend openai \
    --model_name gpt-5-mini \
    --out_dir runs/gpt5_mini_mmvet_full \
    2>&1 | tee logs/gpt5_mmvet_full.log

echo "Done!"
