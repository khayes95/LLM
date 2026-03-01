#!/bin/bash
# Run 150 additional unique samples for each benchmark to reach 250 total
# Only includes benchmarks with GPT-5-mini accuracy between 15% and 85%

set -e

# Text benchmarks with 15-85% accuracy:
# gpqa (81%), livebench (74%), chembench (66%), bbeh (54%), omnimath (39%), hle (18%), simpleqa (16%)
# Excluded: babilong (93% - too easy)
TEXT_BENCHMARKS="gpqa livebench chembench bbeh omnimath hle simpleqa"

# VLM benchmarks with 15-85% accuracy:
# mmstar (80%), mmmu (68%), charxiv (63%), erqa (62%), vizwiz (58%), realworldqa (48%), mathverse (39%), mathvista (34%), mathvision (33%)
# Excluded: aokvqa (92%), vsr (87%), hallusionbench (81%) - too easy
VLM_BENCHMARKS="mmstar mmmu charxiv erqa vizwiz realworldqa mathverse mathvista mathvision"

# HLE multimodal (VLM version of HLE)
HLE_MULTIMODAL="hle_multimodal"

# Rubric-based benchmarks (need grading, accuracy TBD):
RUBRIC_BENCHMARKS="healthbench tutorbench prbench"

# Create exclude_ids directory
mkdir -p /scratch/khayes/LLM/data/exclude_ids

echo "=== Benchmarks included (15-85% accuracy) ==="
echo "Text: $TEXT_BENCHMARKS"
echo "VLM: $VLM_BENCHMARKS"
echo "Rubric (TBD): $RUBRIC_BENCHMARKS"
echo ""

echo "=== Extracting existing IDs from previous runs ==="

# Function to extract IDs from a benchmark's predictions
extract_ids() {
    local bench=$1
    local outfile="/scratch/khayes/LLM/data/exclude_ids/${bench}_ids.json"

    python3 << PYEOF
import json
import glob

ids = set()
pattern = "/scratch/khayes/LLM/runs/*_${bench}_gpt-5-mini/predictions.jsonl"
files = glob.glob(pattern)

for pred_file in files:
    try:
        with open(pred_file) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    # ID is at top level in our format
                    ex_id = obj.get('id')
                    if ex_id:
                        ids.add(str(ex_id))
    except Exception as e:
        print(f"Warning: {pred_file}: {e}")

with open('${outfile}', 'w') as f:
    json.dump(sorted(ids), f, indent=2)

print(f"${bench}: {len(ids)} existing IDs extracted")
PYEOF
}

# Extract IDs for all benchmarks
for bench in $TEXT_BENCHMARKS $VLM_BENCHMARKS $HLE_MULTIMODAL $RUBRIC_BENCHMARKS; do
    extract_ids "$bench"
done

echo ""
echo "=== Run commands for 150 additional samples ==="
echo "Copy and paste these commands to run (with your API key set):"
echo ""

# Generate the commands
echo "# Text benchmarks (15-85% accuracy):"
for bench in $TEXT_BENCHMARKS; do
    exclude_file="/scratch/khayes/LLM/data/exclude_ids/${bench}_ids.json"
    echo "python -m uq_eval.cli --bench $bench --model_backend openai --model_name gpt-5-mini --max_examples 150 --seed 43 --timeout_s 300 --max_output_tokens 16384 --exclude_ids $exclude_file &"
done

echo ""
echo "# VLM benchmarks (15-85% accuracy):"
for bench in $VLM_BENCHMARKS; do
    exclude_file="/scratch/khayes/LLM/data/exclude_ids/${bench}_ids.json"
    echo "python -m uq_eval.cli --bench $bench --model_backend openai --model_name gpt-5-mini --max_examples 150 --seed 43 --timeout_s 300 --max_output_tokens 16384 --exclude_ids $exclude_file &"
done

echo ""
echo "# HLE multimodal (VLM version - 250 total samples):"
exclude_file="/scratch/khayes/LLM/data/exclude_ids/${HLE_MULTIMODAL}_ids.json"
echo "python -m uq_eval.cli --bench $HLE_MULTIMODAL --model_backend openai --model_name gpt-5-mini --max_examples 250 --seed 42 --timeout_s 300 --max_output_tokens 16384 --exclude_ids $exclude_file &"

echo ""
echo "# Rubric-based benchmarks (need grading after):"
for bench in $RUBRIC_BENCHMARKS; do
    exclude_file="/scratch/khayes/LLM/data/exclude_ids/${bench}_ids.json"
    echo "python -m uq_eval.cli --bench $bench --model_backend openai --model_name gpt-5-mini --max_examples 150 --seed 43 --timeout_s 300 --max_output_tokens 16384 --exclude_ids $exclude_file &"
done

echo ""
echo "wait  # Wait for all jobs to complete"
echo ""
echo "# After completion, grade rubric benchmarks:"
echo "# ./scripts/run_grading.sh"
