#!/usr/bin/env python3
"""
Fix sample matching between GPT-5-mini predictions and current benchmark IDs.

Some benchmarks have different IDs between when GPT-5-mini ran and now.
This script extracts the actual question content from GPT-5 predictions
and matches them to current benchmark examples by content.

Creates a new mapping file that can be used for exact sample matching
in the 235B evaluation.

Usage:
    python scripts/fix_sample_matching.py
"""

import json
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_question_hash(input_data: dict | str) -> str:
    """Extract a hash from the question content for matching."""
    if isinstance(input_data, str):
        text = input_data
    elif isinstance(input_data, dict):
        # Try common keys for question text
        text = input_data.get("question", "")
        if not text:
            text = input_data.get("query", "")
        if not text:
            text = str(input_data)
    else:
        text = str(input_data)

    # Normalize and hash
    text = text.strip().lower()
    return hashlib.md5(text.encode()).hexdigest()[:16]


def main():
    combined_dir = Path("/scratch/khayes/LLM/runs/gpt5_mini_combined")
    output_dir = Path("/scratch/khayes/LLM/runs/gpt5_mini_combined_matched")
    output_dir.mkdir(exist_ok=True)

    from uq_eval.registry import load_benchmark

    # Benchmarks to fix
    benchmarks = [
        "mathvista", "mathvision", "mathverse", "mmmu", "mmstar",
        "realworldqa", "vizwiz", "erqa", "hle_multimodal",
        "bbeh", "chembench", "gpqa", "livebench", "omnimath", "simpleqa", "hle"
    ]

    results = {}

    for bench_name in benchmarks:
        print(f"\n{'='*60}")
        print(f"Processing: {bench_name}")
        print("=" * 60)

        preds_file = combined_dir / bench_name / "predictions.jsonl"
        if not preds_file.exists():
            print(f"  No predictions file found")
            results[bench_name] = {"error": "No predictions file"}
            continue

        # Load GPT-5 predictions
        gpt5_preds = []
        with open(preds_file) as f:
            for line in f:
                gpt5_preds.append(json.loads(line))

        print(f"  Loaded {len(gpt5_preds)} GPT-5 predictions")

        # Build hash -> GPT-5 prediction mapping
        gpt5_by_hash = {}
        for pred in gpt5_preds:
            input_data = pred.get("input", {})
            h = extract_question_hash(input_data)
            gpt5_by_hash[h] = pred

        print(f"  Unique hashes: {len(gpt5_by_hash)}")

        # Load current benchmark examples
        try:
            benchmark = load_benchmark(bench_name)
            all_examples = list(benchmark.iter_examples("test"))
        except Exception as e:
            print(f"  Error loading benchmark: {e}")
            results[bench_name] = {"error": str(e)}
            continue

        print(f"  Loaded {len(all_examples)} benchmark examples")

        # Build hash -> benchmark example mapping
        bench_by_hash = {}
        for ex in all_examples:
            # Get the input text
            if hasattr(ex, 'input'):
                input_data = ex.input
            else:
                input_data = str(ex)

            if isinstance(input_data, dict):
                h = extract_question_hash(input_data)
            else:
                h = extract_question_hash(str(input_data))

            bench_by_hash[h] = ex

        # Find matches
        matched = []
        unmatched_gpt5 = []

        for h, pred in gpt5_by_hash.items():
            if h in bench_by_hash:
                ex = bench_by_hash[h]
                matched.append({
                    "gpt5_id": pred["id"],
                    "bench_id": ex.id,
                    "hash": h,
                })
            else:
                unmatched_gpt5.append(pred["id"])

        print(f"  Matched: {len(matched)}/{len(gpt5_preds)}")
        print(f"  Unmatched: {len(unmatched_gpt5)}")

        # Save mapping
        bench_out = output_dir / bench_name
        bench_out.mkdir(exist_ok=True)

        with open(bench_out / "id_mapping.json", "w") as f:
            json.dump(matched, f, indent=2)

        # Create new sampled_ids.json with benchmark IDs
        matched_bench_ids = [m["bench_id"] for m in matched]
        with open(bench_out / "sampled_ids.json", "w") as f:
            json.dump(matched_bench_ids, f)

        # Copy original predictions with updated IDs
        gpt5_id_to_bench_id = {m["gpt5_id"]: m["bench_id"] for m in matched}
        matched_preds = []
        for pred in gpt5_preds:
            if pred["id"] in gpt5_id_to_bench_id:
                new_pred = pred.copy()
                new_pred["original_id"] = pred["id"]
                new_pred["id"] = gpt5_id_to_bench_id[pred["id"]]
                matched_preds.append(new_pred)

        with open(bench_out / "predictions.jsonl", "w") as f:
            for pred in matched_preds:
                f.write(json.dumps(pred) + "\n")

        results[bench_name] = {
            "total_gpt5": len(gpt5_preds),
            "total_bench": len(all_examples),
            "matched": len(matched),
            "match_rate": len(matched) / len(gpt5_preds) if gpt5_preds else 0,
        }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Benchmark':<20} {'GPT-5':>8} {'Bench':>8} {'Matched':>8} {'Rate':>8}")
    print("-" * 60)

    for bench, r in sorted(results.items()):
        if "error" in r:
            print(f"{bench:<20} {'ERROR':>8}")
        else:
            rate = f"{r['match_rate']:.0%}"
            print(f"{bench:<20} {r['total_gpt5']:>8} {r['total_bench']:>8} {r['matched']:>8} {rate:>8}")

    print(f"\nMatched data saved to: {output_dir}")


if __name__ == "__main__":
    main()
