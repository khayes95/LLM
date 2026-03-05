#!/usr/bin/env python3
"""Convert FineGRAIN T2I data to UQ training format.

Loads two data sources:
  1. Human-labeled data from /scratch/khayes/diff/t2i-finegrain/metadata.csv
     - 5 models (flux, sd3.5_large, sd3.5_medium, sd3_m, sd3_xl), 750 each
     - human_labels: 1.0 = failure, 0.0 = compliant
  2. LLM judge-labeled data from finegrain_dev judged JSON files
     - 15 models, ~700-760 each
     - llm_evaluation_tailored.boolean: 1 = failure, 0 = compliant

Outputs JSONL files in UQ training format compatible with train_best_uq.py.

Usage:
    # Full conversion
    python scripts/finegrain_convert_data.py

    # Dry run (print stats, no file writes)
    python scripts/finegrain_convert_data.py --dry_run

    # Custom output dir
    python scripts/finegrain_convert_data.py --output_dir data/finegrain_uq/finetune_v2
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

# Human-labeled data
HUMAN_METADATA_CSV = Path("/scratch/khayes/diff/t2i-finegrain/metadata.csv")
HUMAN_IMAGE_DIR = Path("/scratch/khayes/diff/t2i-finegrain/images")
HUMAN_MODELS = ["flux", "sd3.5_large", "sd3.5_medium", "sd3_m", "sd3_xl"]

# Judge-labeled data
JUDGE_BASE_DIR = Path("/scratch/khayes/finegrain_dev")
JUDGE_METADATA_DIR = JUDGE_BASE_DIR / "data/results/vlm_evaluation"
JUDGE_MODELS = [
    "flux", "flux2_dev", "flux2_pro", "gemini_image", "gemini_image_native",
    "gpt_image1", "gpt_image1_resized", "gpt_image15", "hidream",
    "nano_banana2", "qwen", "sd1", "sd2", "seedream", "wan22",
]

# Minimum sample threshold to include a judge model
MIN_JUDGE_SAMPLES = 10

# Default output
DEFAULT_OUTPUT_DIR = "data/finegrain_uq/finetune"


# ============================================================
# FAILURE MODE SLUG CONVERSION
# ============================================================

def failure_mode_to_slug(failure_mode: str) -> str:
    """Convert failure mode name to filesystem-safe slug.

    This matches the convention used in the finegrain_dev image directories:
    - Lowercase
    - Spaces replaced with underscores
    - Special characters like (), +, - are preserved
    """
    return failure_mode.lower().replace(" ", "_")


# ============================================================
# UQ SAMPLE FORMAT
# ============================================================

def make_uq_sample(
    sample_id: str,
    source_model: str,
    prompt_text: str,
    failure_mode: str,
    label: int,
    image_path: str,
) -> dict:
    """Create a UQ training sample dict.

    Args:
        sample_id: Unique identifier for this sample.
        source_model: Name of the T2I model that generated the image.
        prompt_text: The text prompt given to the T2I model.
        failure_mode: The specific failure category being tested.
        label: 0 = compliant (image matches prompt), 1 = failure (image has defect).
        image_path: Absolute path to the generated image.

    Returns:
        Dict matching the UQ training JSONL format.
    """
    question = (
        f"Prompt: {prompt_text}\n"
        f"Failure mode being tested: {failure_mode}\n"
        f"Does the generated image correctly depict the prompt without the specified failure?"
    )
    response = (
        "Yes, the image correctly depicts the prompt "
        "without exhibiting the specified failure mode."
    )
    # is_correct = True when image is compliant (label 0), False when failure (label 1)
    is_correct = (label == 0)

    return {
        "id": sample_id,
        "benchmark": "finegrain",
        "source_model": source_model,
        "question": question,
        "response": response,
        "is_correct": is_correct,
        "has_image": True,
        "image_path": image_path,
    }


# ============================================================
# HUMAN DATA LOADING
# ============================================================

def load_human_data(metadata_path: Path, image_dir: Path) -> list[dict]:
    """Load human-labeled FineGRAIN samples.

    Reads metadata.csv, filters to rows with valid human_labels,
    verifies image paths exist, and converts to UQ format.

    Returns:
        List of UQ sample dicts.
    """
    samples = []
    skipped_no_label = 0
    skipped_no_image = 0

    with open(metadata_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows without human labels
            label_str = row.get("human_labels", "").strip()
            if not label_str:
                skipped_no_label += 1
                continue

            try:
                label_float = float(label_str)
            except ValueError:
                skipped_no_label += 1
                continue

            if label_float not in (0.0, 1.0):
                skipped_no_label += 1
                continue

            label = int(label_float)  # 0 or 1
            model = row["model"]
            prompt_id = int(row["prompt_id"])
            prompt_text = row["prompt_text"]
            failure_mode = row["failure_mode"]

            # Build image path
            image_path = image_dir / model / f"{prompt_id:05d}.png"
            if not image_path.exists():
                skipped_no_image += 1
                continue

            sample_id = f"finegrain_human_{model}_{prompt_id:05d}"

            samples.append(make_uq_sample(
                sample_id=sample_id,
                source_model=model,
                prompt_text=prompt_text,
                failure_mode=failure_mode,
                label=label,
                image_path=str(image_path),
            ))

    print(f"[Human] Loaded {len(samples)} samples "
          f"(skipped: {skipped_no_label} no label, {skipped_no_image} no image)")
    return samples


# ============================================================
# JUDGE DATA LOADING
# ============================================================

def load_judge_data_for_model(model: str) -> list[dict]:
    """Load judge-labeled FineGRAIN samples for a single model.

    Reads the judged JSON file, filters to entries with valid boolean labels,
    verifies image paths exist, and converts to UQ format.

    Returns:
        List of UQ sample dicts.
    """
    json_path = JUDGE_METADATA_DIR / f"metadata_{model}_tailored_vllm_judged.json"
    if not json_path.exists():
        print(f"  [Judge] WARNING: {json_path} not found, skipping {model}")
        return []

    with open(json_path) as f:
        entries = json.load(f)

    samples = []
    skipped_no_eval = 0
    skipped_no_image = 0

    for entry in entries:
        # Extract judge evaluation
        eval_data = entry.get("llm_evaluation_tailored")
        if not isinstance(eval_data, dict):
            skipped_no_eval += 1
            continue

        boolean_val = eval_data.get("boolean")
        if boolean_val is None:
            skipped_no_eval += 1
            continue

        label = int(boolean_val)  # 1 = failure, 0 = compliant
        if label not in (0, 1):
            skipped_no_eval += 1
            continue

        # Build absolute image path
        # image_path in JSON is relative to JUDGE_BASE_DIR
        rel_image_path = entry.get("image_path", "")
        abs_image_path = JUDGE_BASE_DIR / rel_image_path
        if not abs_image_path.exists():
            skipped_no_image += 1
            continue

        prompt_text = entry.get("prompt", "")
        failure_mode = entry.get("failure_mode", "")
        index = entry.get("index", 0)

        # Build a unique ID using model, failure_mode slug, and index
        fm_slug = failure_mode_to_slug(failure_mode)
        sample_id = f"finegrain_judge_{model}_{fm_slug}_{index}"

        samples.append(make_uq_sample(
            sample_id=sample_id,
            source_model=model,
            prompt_text=prompt_text,
            failure_mode=failure_mode,
            label=label,
            image_path=str(abs_image_path),
        ))

    if skipped_no_eval > 0 or skipped_no_image > 0:
        print(f"  [Judge] {model}: {len(samples)} valid "
              f"(skipped: {skipped_no_eval} no eval, {skipped_no_image} no image)")
    return samples


def load_all_judge_data() -> list[dict]:
    """Load judge-labeled data across all models.

    Skips models with fewer than MIN_JUDGE_SAMPLES valid samples.

    Returns:
        List of UQ sample dicts.
    """
    all_samples = []
    skipped_models = []

    for model in JUDGE_MODELS:
        model_samples = load_judge_data_for_model(model)
        if len(model_samples) < MIN_JUDGE_SAMPLES:
            skipped_models.append((model, len(model_samples)))
            continue
        all_samples.extend(model_samples)

    if skipped_models:
        print(f"[Judge] Skipped {len(skipped_models)} models with < {MIN_JUDGE_SAMPLES} samples: "
              f"{skipped_models}")
    print(f"[Judge] Loaded {len(all_samples)} total samples")
    return all_samples


# ============================================================
# VALIDATION
# ============================================================

def validate_samples(samples: list[dict], label: str) -> bool:
    """Validate a list of UQ samples.

    Checks:
      - All image paths exist
      - No duplicate IDs
      - Label distribution is reasonable (neither class < 5%)

    Returns:
        True if validation passes.
    """
    issues = []

    # Check for missing images
    missing_images = []
    for s in samples:
        if not os.path.exists(s["image_path"]):
            missing_images.append(s["image_path"])
    if missing_images:
        issues.append(f"{len(missing_images)} missing image(s)")
        for p in missing_images[:5]:
            issues.append(f"  Missing: {p}")

    # Check for duplicate IDs
    ids = [s["id"] for s in samples]
    id_counts = Counter(ids)
    duplicates = {k: v for k, v in id_counts.items() if v > 1}
    if duplicates:
        issues.append(f"{len(duplicates)} duplicate ID(s)")
        for did, cnt in list(duplicates.items())[:5]:
            issues.append(f"  Duplicate: {did} (x{cnt})")

    # Check label distribution
    n_correct = sum(1 for s in samples if s["is_correct"])
    n_incorrect = len(samples) - n_correct
    total = len(samples)
    if total > 0:
        correct_rate = n_correct / total
        if correct_rate < 0.05 or correct_rate > 0.95:
            issues.append(
                f"Extreme label imbalance: {n_correct}/{total} correct "
                f"({correct_rate:.1%})"
            )

    if issues:
        print(f"\n[VALIDATION ISSUES - {label}]")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print(f"[VALIDATION OK - {label}] {total} samples, "
              f"{n_correct} correct ({n_correct/max(total,1):.1%}), "
              f"{n_incorrect} incorrect ({n_incorrect/max(total,1):.1%})")
        return True


# ============================================================
# OUTPUT
# ============================================================

def write_jsonl(samples: list[dict], output_path: Path):
    """Write samples to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"  Wrote {len(samples)} samples to {output_path}")


def compute_stats(human_samples: list[dict], judge_samples: list[dict]) -> dict:
    """Compute summary statistics across all data."""

    def _subset_stats(samples: list[dict], label: str) -> dict:
        n = len(samples)
        if n == 0:
            return {"n": 0}

        n_correct = sum(1 for s in samples if s["is_correct"])
        n_incorrect = n - n_correct
        models = Counter(s["source_model"] for s in samples)
        failure_modes = Counter()
        for s in samples:
            # Extract failure mode from the question text
            lines = s["question"].split("\n")
            for line in lines:
                if line.startswith("Failure mode being tested: "):
                    fm = line.replace("Failure mode being tested: ", "")
                    failure_modes[fm] += 1
                    break

        return {
            "n": n,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "correct_rate": round(n_correct / n, 4),
            "per_model": dict(sorted(models.items())),
            "n_failure_modes": len(failure_modes),
            "failure_mode_counts": dict(sorted(failure_modes.items())),
        }

    stats = {
        "human": _subset_stats(human_samples, "human"),
        "judge": _subset_stats(judge_samples, "judge"),
        "combined": _subset_stats(human_samples + judge_samples, "combined"),
    }

    # Per-model stats for human data
    human_per_model = defaultdict(list)
    for s in human_samples:
        human_per_model[s["source_model"]].append(s)
    stats["human_per_model"] = {}
    for model in sorted(human_per_model):
        ms = human_per_model[model]
        n_correct = sum(1 for s in ms if s["is_correct"])
        stats["human_per_model"][model] = {
            "n": len(ms),
            "n_correct": n_correct,
            "n_incorrect": len(ms) - n_correct,
            "correct_rate": round(n_correct / len(ms), 4),
        }

    return stats


def print_summary(human_samples: list[dict], judge_samples: list[dict]):
    """Print a formatted summary of the data."""
    print("\n" + "=" * 70)
    print("DATA CONVERSION SUMMARY")
    print("=" * 70)

    # Human data
    print(f"\n--- Human-Labeled Data ---")
    print(f"Total samples:  {len(human_samples)}")
    if human_samples:
        n_correct = sum(1 for s in human_samples if s["is_correct"])
        n_incorrect = len(human_samples) - n_correct
        print(f"Correct (compliant): {n_correct} ({n_correct/len(human_samples):.1%})")
        print(f"Incorrect (failure): {n_incorrect} ({n_incorrect/len(human_samples):.1%})")
        models = Counter(s["source_model"] for s in human_samples)
        print(f"Models ({len(models)}):")
        for m, c in sorted(models.items()):
            ms = [s for s in human_samples if s["source_model"] == m]
            mc = sum(1 for s in ms if s["is_correct"])
            print(f"  {m:20s}: {c:5d} samples, {mc:4d} correct ({mc/c:.1%})")

    # Judge data
    print(f"\n--- Judge-Labeled Data ---")
    print(f"Total samples:  {len(judge_samples)}")
    if judge_samples:
        n_correct = sum(1 for s in judge_samples if s["is_correct"])
        n_incorrect = len(judge_samples) - n_correct
        print(f"Correct (compliant): {n_correct} ({n_correct/len(judge_samples):.1%})")
        print(f"Incorrect (failure): {n_incorrect} ({n_incorrect/len(judge_samples):.1%})")
        models = Counter(s["source_model"] for s in judge_samples)
        print(f"Models ({len(models)}):")
        for m, c in sorted(models.items()):
            ms = [s for s in judge_samples if s["source_model"] == m]
            mc = sum(1 for s in ms if s["is_correct"])
            print(f"  {m:25s}: {c:5d} samples, {mc:4d} correct ({mc/c:.1%})")

    # Combined
    all_samples = human_samples + judge_samples
    print(f"\n--- Combined ---")
    print(f"Total samples:      {len(all_samples)}")
    print(f"Unique models:      {len(set(s['source_model'] for s in all_samples))}")

    # Failure mode distribution
    fms = Counter()
    for s in all_samples:
        lines = s["question"].split("\n")
        for line in lines:
            if line.startswith("Failure mode being tested: "):
                fm = line.replace("Failure mode being tested: ", "")
                fms[fm] += 1
                break
    print(f"Failure modes:      {len(fms)}")
    print()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert FineGRAIN T2I data to UQ training format"
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for JSONL files (default: %(default)s)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print stats without writing any files",
    )
    parser.add_argument(
        "--human_metadata",
        default=str(HUMAN_METADATA_CSV),
        help="Path to human-labeled metadata CSV",
    )
    parser.add_argument(
        "--human_image_dir",
        default=str(HUMAN_IMAGE_DIR),
        help="Path to human-labeled image directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    human_metadata = Path(args.human_metadata)
    human_image_dir = Path(args.human_image_dir)

    print("=" * 70)
    print("FineGRAIN -> UQ Training Data Conversion")
    print("=" * 70)

    # ----------------------------------------------------------
    # 1. Load human-labeled data
    # ----------------------------------------------------------
    print("\n[1/4] Loading human-labeled data...")
    if not human_metadata.exists():
        print(f"ERROR: Human metadata not found: {human_metadata}")
        sys.exit(1)
    if not human_image_dir.exists():
        print(f"ERROR: Human image directory not found: {human_image_dir}")
        sys.exit(1)

    human_samples = load_human_data(human_metadata, human_image_dir)

    # ----------------------------------------------------------
    # 2. Load judge-labeled data
    # ----------------------------------------------------------
    print("\n[2/4] Loading judge-labeled data...")
    if not JUDGE_METADATA_DIR.exists():
        print(f"ERROR: Judge metadata directory not found: {JUDGE_METADATA_DIR}")
        sys.exit(1)

    judge_samples = load_all_judge_data()

    # ----------------------------------------------------------
    # 3. Validate
    # ----------------------------------------------------------
    print("\n[3/4] Validating...")
    human_ok = validate_samples(human_samples, "human")
    judge_ok = validate_samples(judge_samples, "judge")

    # Print summary
    print_summary(human_samples, judge_samples)

    if not human_ok or not judge_ok:
        print("WARNING: Validation issues detected. Review above.")

    # ----------------------------------------------------------
    # 4. Write output files
    # ----------------------------------------------------------
    if args.dry_run:
        print("[DRY RUN] Skipping file writes.")
        print(f"Would write to: {output_dir}/")
        print(f"  human_all.jsonl       ({len(human_samples)} samples)")
        for model in HUMAN_MODELS:
            ms = [s for s in human_samples if s["source_model"] == model]
            print(f"  human_{model}.jsonl  ({len(ms)} samples)")
        print(f"  judge_all.jsonl       ({len(judge_samples)} samples)")
        print(f"  stats.json")
        return

    print(f"\n[4/4] Writing output files to {output_dir}/")
    output_dir.mkdir(parents=True, exist_ok=True)

    # human_all.jsonl
    write_jsonl(human_samples, output_dir / "human_all.jsonl")

    # Per-model human splits (for leave-one-model-out CV)
    for model in HUMAN_MODELS:
        model_samples = [s for s in human_samples if s["source_model"] == model]
        if model_samples:
            write_jsonl(model_samples, output_dir / f"human_{model}.jsonl")

    # judge_all.jsonl
    write_jsonl(judge_samples, output_dir / "judge_all.jsonl")

    # stats.json
    stats = compute_stats(human_samples, judge_samples)
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Wrote stats to {stats_path}")

    print(f"\nDone. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
