#!/usr/bin/env python3
"""
Download and analyze VLM benchmarks from HuggingFace.
Extracts metadata: size, format, class balance, image requirements.
Optimized: skips large downloads, handles configs properly.
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset, get_dataset_config_names
from huggingface_hub import hf_hub_download, HfApi

# Output directory for benchmark metadata
OUTPUT_DIR = Path("/scratch/khayes/LLM/data/benchmark_catalog")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Priority VLM benchmarks to download (based on our catalog research)
# Format: (hf_id, config/subset, expected_gpt5_acc, notes)
# Updated with correct configs and smaller/faster datasets first
VLM_BENCHMARKS = [
    # Tier 1: High Priority - Low GPT accuracy, single image
    ("cambridgeltl/vsr_random", None, "~70%", "Spatial reasoning, true/false, binary"),
    ("AI4Math/MathVerse", "testmini", "~25%", "Visual math, 6 versions per problem"),
    ("MathLLMs/MathVision", None, "~24%", "Competition math with images"),
    ("AI4Math/MathVista", None, "~50%", "IQ tests, plots, figures"),  # default config
    ("lmms-lab/HallusionBench", None, "~31%", "Hallucination detection"),
    ("jonathan-roberts1/zerobench", None, "0%", "Designed to be unsolvable"),

    # Tier 2: Standard VLM benchmarks - Medium accuracy (smaller datasets first)
    ("Lin-Chen/MMStar", None, "~55%", "Vision-indispensable, 1.5K"),
    ("lmms-lab/RealWorldQA", None, "~70%", "Real-world spatial, 765"),
    ("lmms-lab/ai2d", None, "~85%", "Science diagrams, 3K"),
    ("HuggingFaceM4/ChartQA", None, "~75%", "Chart understanding, 2.5K test"),
    ("princeton-nlp/CharXiv", None, "~60%", "arXiv figure reasoning, 1.3K"),
    ("lmms-lab/MMVet", None, "~55%", "Integrated VL capabilities, 218 Q"),
    ("facebook/winoground", None, "~50%", "Compositional reasoning, 800"),
    ("TIGER-Lab/NLVR2", None, "~80%", "Visual reasoning, 2-image"),
    ("BLINK-Benchmark/BLINK", None, "~50%", "Multimodal benchmark"),
    ("HuggingFaceM4/A-OKVQA", None, "~60%", "Outside knowledge VQA"),
    ("lmms-lab/SEED-Bench", None, "~70%", "Comprehensive VLM eval"),
    ("echo840/OCRBench", None, "~70%", "OCR tasks"),
    ("lmms-lab/VizWiz-VQA", None, "~60%", "Blind user photos"),
    ("m-a-p/CMMMU", None, "~55%", "Chinese MMMU"),
    ("Hothan/OlympiadBench", None, "~40%", "Olympiad math+physics"),

    # Tier 3: MMMU variants (need specific configs)
    ("MMMU/MMMU", "Accounting", "~62%", "College-level multimodal - Accounting"),
    ("MMMU/MMMU", "Art", "~62%", "College-level multimodal - Art"),
    ("MMMU/MMMU", "Biology", "~62%", "College-level multimodal - Biology"),
    ("MMMU/MMMU", "Chemistry", "~62%", "College-level multimodal - Chemistry"),
    ("MMMU/MMMU", "Computer_Science", "~62%", "College-level multimodal - CS"),
    ("MMMU/MMMU", "Physics", "~62%", "College-level multimodal - Physics"),
    ("MMMU/MMMU", "Math", "~62%", "College-level multimodal - Math"),

    # Tier 4: ScienceQA (has images, but also text-only)
    ("derek-thomas/ScienceQA", None, "~85%", "K-12 science with images"),

    # Skip: TextVQA (7GB), DocVQA (large), raw COCO datasets
]


def analyze_benchmark(hf_id: str, config: str = None, expected_acc: str = "", notes: str = ""):
    """Download and analyze a single benchmark, extracting metadata."""
    result = {
        "hf_id": hf_id,
        "config": config,
        "expected_gpt5_accuracy": expected_acc,
        "notes": notes,
        "status": "pending",
        "error": None,
        "metadata": {}
    }

    try:
        print(f"[INFO] Loading {hf_id} (config={config})...")

        # Try to get available configs
        try:
            configs = get_dataset_config_names(hf_id)
            result["metadata"]["available_configs"] = configs[:10]  # Limit to first 10
        except:
            configs = [None]
            result["metadata"]["available_configs"] = ["default"]

        # Load the dataset with streaming to check size first
        if config:
            ds = load_dataset(hf_id, config, trust_remote_code=True)
        else:
            ds = load_dataset(hf_id, trust_remote_code=True)

        # Get split info
        splits = list(ds.keys())
        result["metadata"]["splits"] = splits

        # Analyze the main split (prefer test > validation > train)
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

            # Analyze image format
            image_cols = []
            for col in data.features.keys():
                col_lower = col.lower()
                if 'image' in col_lower or 'img' in col_lower or 'picture' in col_lower or 'photo' in col_lower:
                    image_cols.append(col)
                # Check feature type
                feat = data.features[col]
                feat_str = str(type(feat).__name__)
                if 'Image' in feat_str:
                    if col not in image_cols:
                        image_cols.append(col)

            result["metadata"]["image_columns"] = image_cols
            result["metadata"]["num_image_columns"] = len(image_cols)

            # Better multi-image detection: check if any column is a list of images or multiple distinct image cols
            # Note: some datasets have "image" and "decoded_image" which are the same image
            unique_image_cols = [c for c in image_cols if 'decoded' not in c.lower() and 'link' not in c.lower()]
            result["metadata"]["is_multi_image"] = len(unique_image_cols) > 1

            # Check for answer/label columns and class balance
            answer_cols = []
            for col in data.features.keys():
                col_lower = col.lower()
                if any(x in col_lower for x in ['answer', 'label', 'correct', 'target', 'class']):
                    answer_cols.append(col)

            result["metadata"]["answer_columns"] = answer_cols

            # Try to compute class balance for the first answer column
            if answer_cols:
                try:
                    sample_size = min(1000, len(data))
                    sample = data.select(range(sample_size))
                    answers = [str(x) for x in sample[answer_cols[0]]]
                    counter = Counter(answers)
                    total = sum(counter.values())

                    # If binary or small number of classes, show balance
                    if len(counter) <= 10:
                        balance = {k: round(v/total, 3) for k, v in counter.most_common(10)}
                        result["metadata"]["class_balance"] = balance
                        result["metadata"]["num_classes"] = len(counter)
                    else:
                        result["metadata"]["num_classes"] = len(counter)
                        result["metadata"]["class_balance"] = f"{len(counter)} unique values (sampled {sample_size})"
                except Exception as e:
                    result["metadata"]["class_balance_error"] = str(e)

            # Check question format
            question_cols = []
            for col in data.features.keys():
                col_lower = col.lower()
                if any(x in col_lower for x in ['question', 'query', 'prompt', 'text', 'instruction']):
                    question_cols.append(col)
            result["metadata"]["question_columns"] = question_cols

            # Check if MCQ format
            for col in data.features.keys():
                col_lower = col.lower()
                if any(x in col_lower for x in ['option', 'choice', 'candidate']):
                    result["metadata"]["is_mcq"] = True
                    break
            else:
                result["metadata"]["is_mcq"] = False

            # Sample a few examples to understand format (text only, skip images)
            try:
                sample = data.select(range(min(3, len(data))))
                sample_items = []
                for i in range(len(sample)):
                    item = {}
                    for col in data.features.keys():
                        val = sample[i][col]
                        if 'image' not in col.lower():
                            if isinstance(val, (str, int, float, bool)):
                                item[col] = val if len(str(val)) < 200 else str(val)[:200] + "..."
                            elif isinstance(val, list):
                                item[col] = val[:5] if len(val) > 5 else val
                    sample_items.append(item)
                result["metadata"]["sample_items"] = sample_items
            except:
                pass

        result["status"] = "success"
        print(f"[OK] {hf_id}: {result['metadata'].get('size', '?')} examples, {result['metadata'].get('num_image_columns', '?')} image cols, multi_image={result['metadata'].get('is_multi_image', '?')}")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"[ERROR] {hf_id}: {str(e)[:100]}")

    return result


def main():
    print("=" * 60)
    print("VLM Benchmark Downloader and Analyzer")
    print("=" * 60)

    # Load existing results to avoid re-downloading
    output_file = OUTPUT_DIR / "vlm_benchmarks_catalog.json"
    existing_results = []
    existing_ids = set()
    if output_file.exists():
        with open(output_file) as f:
            existing_results = json.load(f)
            for r in existing_results:
                key = f"{r['hf_id']}_{r.get('config', 'default')}"
                existing_ids.add(key)
        print(f"[INFO] Loaded {len(existing_results)} existing results")

    all_results = existing_results.copy()

    # Process benchmarks
    for hf_id, config, expected_acc, notes in VLM_BENCHMARKS:
        key = f"{hf_id}_{config or 'default'}"

        # Skip if already have successful result
        if key in existing_ids:
            existing = [r for r in existing_results if f"{r['hf_id']}_{r.get('config', 'default')}" == key]
            if existing and existing[0].get("status") == "success":
                print(f"[SKIP] {hf_id} (config={config}) - already downloaded")
                continue
            # Remove failed result to retry
            all_results = [r for r in all_results if f"{r['hf_id']}_{r.get('config', 'default')}" != key]

        result = analyze_benchmark(hf_id, config, expected_acc, notes)
        all_results.append(result)

        # Save intermediate results
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Generate summary
    summary = {
        "total_benchmarks": len(all_results),
        "successful": len([r for r in all_results if r["status"] == "success"]),
        "failed": len([r for r in all_results if r["status"] == "error"]),
        "single_image_benchmarks": [],
        "multi_image_benchmarks": [],
        "binary_classification": [],
        "mcq_benchmarks": [],
        "open_ended_benchmarks": [],
        "recommended_for_uq": [],
        "by_accuracy_tier": {
            "very_hard_0_30": [],
            "hard_30_50": [],
            "medium_50_70": [],
            "easy_70_plus": []
        }
    }

    for r in all_results:
        if r["status"] != "success":
            continue

        meta = r["metadata"]
        entry = {
            "hf_id": r["hf_id"],
            "config": r.get("config"),
            "size": meta.get("size", "?"),
            "expected_gpt5_acc": r["expected_gpt5_accuracy"],
            "notes": r["notes"],
            "is_mcq": meta.get("is_mcq", False),
            "num_classes": meta.get("num_classes", "?"),
            "class_balance": meta.get("class_balance", "?")
        }

        # Categorize by image format
        if meta.get("is_multi_image"):
            summary["multi_image_benchmarks"].append(entry)
        else:
            summary["single_image_benchmarks"].append(entry)

        if meta.get("is_mcq"):
            summary["mcq_benchmarks"].append(r["hf_id"])
        else:
            summary["open_ended_benchmarks"].append(r["hf_id"])

        num_classes = meta.get("num_classes", 0)
        if num_classes == 2:
            summary["binary_classification"].append(entry)

        # Categorize by accuracy
        try:
            acc_str = r["expected_gpt5_accuracy"].replace("~", "").replace("%", "")
            acc = float(acc_str)
            if acc < 30:
                summary["by_accuracy_tier"]["very_hard_0_30"].append(entry)
            elif acc < 50:
                summary["by_accuracy_tier"]["hard_30_50"].append(entry)
            elif acc < 70:
                summary["by_accuracy_tier"]["medium_50_70"].append(entry)
            else:
                summary["by_accuracy_tier"]["easy_70_plus"].append(entry)

            # Recommended if: single image, reasonable size, good accuracy range
            size = meta.get("size", 0)
            if (not meta.get("is_multi_image") and
                size >= 100 and
                20 <= acc <= 80):  # Sweet spot for UQ training
                summary["recommended_for_uq"].append(entry)
        except:
            pass

    summary_file = OUTPUT_DIR / "vlm_benchmarks_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total benchmarks analyzed: {summary['total_benchmarks']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Single-image benchmarks: {len(summary['single_image_benchmarks'])}")
    print(f"Multi-image benchmarks: {len(summary['multi_image_benchmarks'])}")
    print(f"Binary classification: {len(summary['binary_classification'])}")
    print(f"MCQ format: {len(summary['mcq_benchmarks'])}")
    print(f"\nBy GPT-5 accuracy tier:")
    print(f"  Very hard (0-30%): {len(summary['by_accuracy_tier']['very_hard_0_30'])}")
    print(f"  Hard (30-50%): {len(summary['by_accuracy_tier']['hard_30_50'])}")
    print(f"  Medium (50-70%): {len(summary['by_accuracy_tier']['medium_50_70'])}")
    print(f"  Easy (70%+): {len(summary['by_accuracy_tier']['easy_70_plus'])}")
    print(f"\nRecommended for UQ (20-80% acc, single image): {len(summary['recommended_for_uq'])}")
    print(f"\nOutput saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
