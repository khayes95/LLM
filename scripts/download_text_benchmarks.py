#!/usr/bin/env python3
"""
Download and analyze text/reasoning benchmarks for UQ training.
Focus on benchmarks that work with open-source models.
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter

from datasets import load_dataset, get_dataset_config_names

OUTPUT_DIR = Path("/scratch/khayes/LLM/data/benchmark_catalog")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Text benchmarks to analyze
# Format: (hf_id, config, expected_gpt5_acc, notes, requires_execution)
TEXT_BENCHMARKS = [
    # Reasoning - Already have HLE, adding more
    ("cais/hle", None, "25-30%", "Humanity's Last Exam - text only", False),
    ("ScaleAI/EnigmaEval", None, "~19%", "Cryptic reasoning puzzles", False),

    # Coding - Require execution environment
    ("princeton-nlp/SWE-bench_Verified", None, "52-75%", "Software engineering - needs execution", True),
    ("livecodebench/code_generation_lite", None, "4-90%", "Live coding - needs execution", True),
    ("bigcode/bigcodebench", None, "~56%", "Code generation - needs execution", True),

    # Tool Use / Agents
    ("HuggingFaceH4/tau2-bench-data", None, "24-58%", "Tool use benchmark", False),
    ("Tevatron/browsecomp-plus", None, "50-69%", "Web browsing comprehension", False),
    ("gaia-benchmark/GAIA", "2023_all", "~30%", "General AI assistant tasks", False),

    # Long Context
    ("zai-org/LongBench-v2", None, "~63%", "Long context understanding", False),
    ("oolongbench/oolong-real", None, "47%", "Real-world long context", False),
    ("princeton-nlp/HELMET", None, "varies", "Long context eval", False),

    # Math
    ("KbsdJames/Omni-MATH", None, "~72%", "Competition math", False),

    # Other reasoning
    ("ScaleAI/MultiNRC", None, "~65%", "Multi-hop reading comprehension", False),

    # Function Calling
    ("gorilla-llm/Berkeley-Function-Calling-Leaderboard", None, "varies", "Function calling eval", False),

    # Visual (text-based versions)
    ("MMMU/MMMU", "Art", "~62%", "College multimodal - Art subset", False),
    ("dataartist/arc-agi", None, "~10%", "Abstract reasoning corpus", False),
]


def analyze_benchmark(hf_id: str, config: str = None, expected_acc: str = "", notes: str = "", requires_exec: bool = False):
    """Analyze a single benchmark."""
    result = {
        "hf_id": hf_id,
        "config": config,
        "expected_gpt5_accuracy": expected_acc,
        "notes": notes,
        "requires_execution": requires_exec,
        "status": "pending",
        "error": None,
        "metadata": {}
    }

    try:
        print(f"[INFO] Loading {hf_id} (config={config})...")

        # Get available configs
        try:
            configs = get_dataset_config_names(hf_id)
            result["metadata"]["available_configs"] = configs[:10]
        except:
            configs = [None]
            result["metadata"]["available_configs"] = ["default"]

        # Load dataset
        if config:
            ds = load_dataset(hf_id, config, trust_remote_code=True)
        else:
            ds = load_dataset(hf_id, trust_remote_code=True)

        splits = list(ds.keys())
        result["metadata"]["splits"] = splits

        # Find main split
        main_split = None
        for s in ["test", "validation", "val", "dev", "train"]:
            if s in splits:
                main_split = s
                break
        if main_split is None and splits:
            main_split = splits[0]

        if main_split:
            data = ds[main_split]
            result["metadata"]["main_split"] = main_split
            result["metadata"]["size"] = len(data)
            result["metadata"]["columns"] = list(data.features.keys())

            # Check for question/answer columns
            question_cols = []
            answer_cols = []
            for col in data.features.keys():
                col_lower = col.lower()
                if any(x in col_lower for x in ['question', 'query', 'prompt', 'input', 'problem']):
                    question_cols.append(col)
                if any(x in col_lower for x in ['answer', 'label', 'output', 'solution', 'target']):
                    answer_cols.append(col)

            result["metadata"]["question_columns"] = question_cols
            result["metadata"]["answer_columns"] = answer_cols

            # Check for images
            image_cols = []
            for col in data.features.keys():
                col_lower = col.lower()
                if 'image' in col_lower or 'img' in col_lower:
                    image_cols.append(col)
            result["metadata"]["image_columns"] = image_cols
            result["metadata"]["has_images"] = len(image_cols) > 0

            # Sample items
            try:
                sample = data.select(range(min(2, len(data))))
                sample_items = []
                for i in range(len(sample)):
                    item = {}
                    for col in data.features.keys():
                        val = sample[i][col]
                        if 'image' not in col.lower():
                            if isinstance(val, (str, int, float, bool)):
                                item[col] = str(val)[:300] + "..." if len(str(val)) > 300 else val
                            elif isinstance(val, list):
                                item[col] = val[:3] if len(val) > 3 else val
                    sample_items.append(item)
                result["metadata"]["sample_items"] = sample_items
            except:
                pass

        result["status"] = "success"
        print(f"[OK] {hf_id}: {result['metadata'].get('size', '?')} examples")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:200]
        print(f"[ERROR] {hf_id}: {str(e)[:100]}")

    return result


def main():
    print("=" * 60)
    print("Text Benchmark Downloader and Analyzer")
    print("=" * 60)

    all_results = []

    for hf_id, config, expected_acc, notes, requires_exec in TEXT_BENCHMARKS:
        result = analyze_benchmark(hf_id, config, expected_acc, notes, requires_exec)
        all_results.append(result)

    # Save results
    output_file = OUTPUT_DIR / "text_benchmarks_catalog.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    successful = [r for r in all_results if r["status"] == "success"]
    failed = [r for r in all_results if r["status"] == "error"]

    print(f"Total: {len(all_results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    print("\n=== Successful Downloads ===")
    for r in successful:
        meta = r["metadata"]
        size = meta.get("size", "?")
        has_img = "IMG" if meta.get("has_images") else "TXT"
        exec_req = "EXEC" if r["requires_execution"] else ""
        print(f"  {r['hf_id']:50} | {size:>6} | {has_img:4} | {r['expected_gpt5_accuracy']:>8} | {exec_req}")

    print("\n=== Failed Downloads ===")
    for r in failed:
        print(f"  {r['hf_id']}: {r['error'][:80]}")

    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    main()
