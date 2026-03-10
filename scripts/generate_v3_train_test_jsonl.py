#!/usr/bin/env python3
"""Generate v3 train/test JSONL files from the scored data + split_info.

The contamination check script needs train/test JSONL files with fields:
  id, benchmark, input, target, model_response, prediction, correct

Since the v3 split was done at the question level, we rebuild the train/test
files from the all-data scored JSONL plus the split_info.json.

Usage:
    python scripts/generate_v3_train_test_jsonl.py
"""
import json
import os
from pathlib import Path


def main():
    split_info_path = "uq_models/best_v3_qsplit/split_info.json"
    scored_dir = "data/use_cases/scored_v3_all"
    output_dir = "data/finetune"

    print("Loading split_info...")
    with open(split_info_path) as f:
        split_info = json.load(f)

    train_ids = set(split_info["train_ids"])
    test_ids = set(split_info["test_ids"])
    print(f"  Train IDs: {len(train_ids)}, Test IDs: {len(test_ids)}")

    # Load all scored data
    all_samples = []
    for target in ["gpt5mini", "gpt52", "qwen35"]:
        fpath = Path(scored_dir) / f"{target}_scored.jsonl"
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found")
            continue
        with open(fpath) as f:
            samples = [json.loads(line) for line in f]
        for s in samples:
            s["source_model"] = target
        all_samples.extend(samples)
        print(f"  {target}: {len(samples)} samples")

    print(f"  Total: {len(all_samples)} samples")

    # Split into train/test
    train_samples = []
    test_samples = []
    unmatched = 0
    for s in all_samples:
        sid = s.get("id", "")
        if sid in test_ids:
            test_samples.append(s)
        elif sid in train_ids:
            train_samples.append(s)
        else:
            unmatched += 1

    print(f"\n  Train: {len(train_samples)}, Test: {len(test_samples)}, Unmatched: {unmatched}")

    # Convert to the format expected by contamination check
    def convert(sample):
        return {
            "id": sample.get("id", ""),
            "benchmark": sample.get("benchmark", ""),
            "input": sample.get("question_preview", ""),
            "target": "",  # not available in scored data
            "model_response": sample.get("response_preview", ""),
            "prediction": "",
            "correct": sample.get("is_correct", False),
            "source_model": sample.get("source_model", ""),
        }

    train_out = Path(output_dir) / "train_v3.jsonl"
    test_out = Path(output_dir) / "test_v3.jsonl"

    with open(train_out, "w") as f:
        for s in train_samples:
            f.write(json.dumps(convert(s)) + "\n")

    with open(test_out, "w") as f:
        for s in test_samples:
            f.write(json.dumps(convert(s)) + "\n")

    print(f"\n  Wrote {train_out} ({len(train_samples)} samples)")
    print(f"  Wrote {test_out} ({len(test_samples)} samples)")


if __name__ == "__main__":
    main()
