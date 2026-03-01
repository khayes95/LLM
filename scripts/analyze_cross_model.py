#!/usr/bin/env python3
"""Comprehensive cross-model analysis: per-benchmark AUROC, calibration, selective prediction.

Loads all cross-model JSON results from data/cross_model/ and produces:
  1. Cross-model transfer matrix (text calibrator + VLM judge)
  2. Per-benchmark AUROC heatmap data
  3. Calibration curves (reliability diagrams)
  4. Selective prediction curves (coverage vs accuracy)
  5. Verbalized confidence baseline comparison

No GPU needed — pure analysis on saved results.

Usage:
    python scripts/analyze_cross_model.py
    python scripts/analyze_cross_model.py --output_dir figures/analysis
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# ============================================================
# CONFIG
# ============================================================

CROSS_MODEL_DIR = Path("data/cross_model")
RUNS_DIR = Path("runs")

# Map of result files to display names
# Old calibrator (trained on GPT-5-mini only)
TEXT_V3_RESULTS = {
    "text_v3_on_gpt52.json": "GPT-5.2",
    "text_v3_on_qwen35.json": "Qwen3.5-397B",
    "text_v3_on_qwen3vl_updated.json": "Qwen3-VL-30B",
}

# New combined calibrator (trained on GPT-5-mini + GPT-5.2)
TEXT_COMBINED_RESULTS = {
    "text_combined_on_gpt52.json": "GPT-5.2",
    "text_combined_on_qwen35.json": "Qwen3.5-397B",
    "text_combined_on_qwen3vl.json": "Qwen3-VL-30B",
}

# GPT-5.2-only calibrator
TEXT_GPT52_RESULTS = {
    "text_gpt52cal_on_gpt52.json": "GPT-5.2",
    "text_gpt52cal_on_qwen35.json": "Qwen3.5-397B",
    "text_gpt52cal_on_qwen3vl.json": "Qwen3-VL-30B",
}

# Combined for backward compat
TEXT_RESULTS = TEXT_V3_RESULTS

VLM_RESULTS = {
    "vlm_judge_vsr_fixed_on_gpt52.json": "GPT-5.2",
    "vlm_judge_vsr_fixed_on_gpt5mini.json": "GPT-5-mini",
    "vlm_judge_vsr_fixed_on_qwen3vl_updated.json": "Qwen3-VL-30B",
    "vlm_judge_vsr_fixed_on_qwen35.json": "Qwen3.5-397B",
}

ALL_TEXT_CALIBRATORS = {
    "v3 (GPT-5-mini)": TEXT_V3_RESULTS,
    "GPT-5.2 only": TEXT_GPT52_RESULTS,
    "Combined (mini+5.2)": TEXT_COMBINED_RESULTS,
}

# Benchmark ordering for tables
BENCH_ORDER = [
    "simpleqa", "gpqa", "chembench", "bbeh", "hle", "omnimath",
    "livebench", "healthbench", "prbench", "arc_agi",
    "mmmu", "charxiv", "mathvista", "mathverse", "mathvision",
    "mmstar", "realworldqa", "hallusionbench", "vizwiz",
]


def load_results(filename):
    """Load a cross-model result JSON file."""
    path = CROSS_MODEL_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def print_header(title):
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def analyze_transfer_matrix():
    """Print the cross-model transfer matrix."""
    print_header("CROSS-MODEL TRANSFER MATRIX")

    for cal_name, cal_results in ALL_TEXT_CALIBRATORS.items():
        print(f"\n  TEXT CALIBRATOR: {cal_name}")
        print(f"  {'Target Model':<25} {'AUROC':>8} {'AUPRC':>8} {'ECE':>8} {'Brier':>8} {'N':>8}")
        print("  " + "-" * 70)

        for filename, model_name in cal_results.items():
            r = load_results(filename)
            if r is None:
                print(f"  {model_name:<25} {'—':>8} (file not found)")
                continue
            print(f"  {model_name:<25} {r['auroc']:>8.4f} {r.get('auprc', 0):>8.4f} "
                  f"{r.get('ece', 0):>8.4f} {r.get('brier', 0):>8.4f} {r['n_samples']:>8}")

    # VLM judge
    print("\n  VLM JUDGE (Qwen3-VL-8B, trained on InternVL3-78B)")
    print(f"  {'Target Model':<25} {'AUROC':>8} {'AUPRC':>8} {'ECE':>8} {'Brier':>8} {'N':>8}")
    print("  " + "-" * 70)

    for filename, model_name in VLM_RESULTS.items():
        r = load_results(filename)
        if r is None:
            print(f"  {model_name:<25} {'—':>8} (file not found)")
            continue
        print(f"  {model_name:<25} {r['auroc']:>8.4f} {r.get('auprc', 0):>8.4f} "
              f"{r.get('ece', 0):>8.4f} {r.get('brier', 0):>8.4f} {r['n_samples']:>8}")


def analyze_per_benchmark():
    """Print per-benchmark AUROC across all models and judges."""
    print_header("PER-BENCHMARK AUROC BREAKDOWN")

    # Collect all per-benchmark data
    all_results = {}
    for cal_name, cal_results in ALL_TEXT_CALIBRATORS.items():
        for filename, model_name in cal_results.items():
            r = load_results(filename)
            if r and "per_benchmark" in r:
                key = f"{cal_name} → {model_name}"
                all_results[key] = r["per_benchmark"]
    for filename, model_name in VLM_RESULTS.items():
        r = load_results(filename)
        if r and "per_benchmark" in r:
            key = f"VLM → {model_name}"
            all_results[key] = r["per_benchmark"]

    if not all_results:
        print("  No per-benchmark data found.")
        return

    # Print header
    keys = list(all_results.keys())
    header = f"  {'Benchmark':<20}"
    for k in keys:
        short = k[:15]
        header += f" {short:>15}"
    print(header)
    print("  " + "-" * (20 + 16 * len(keys)))

    # Print rows
    all_benchmarks = set()
    for pb in all_results.values():
        all_benchmarks.update(pb.keys())

    ordered = [b for b in BENCH_ORDER if b in all_benchmarks]
    ordered += sorted(all_benchmarks - set(BENCH_ORDER))

    for bench in ordered:
        row = f"  {bench:<20}"
        for k in keys:
            data = all_results[k].get(bench, {})
            auroc = data.get("auroc")
            n = data.get("n_samples", 0)
            if auroc is not None and n >= 10:
                row += f" {auroc:>15.3f}"
            elif n > 0:
                row += f" {'<10':>15}"
            else:
                row += f" {'—':>15}"
        print(row)


def analyze_calibration():
    """Analyze calibration (reliability) across models."""
    print_header("CALIBRATION ANALYSIS")

    print(f"\n  {'Model':<40} {'ECE':>8} {'Brier':>8} {'Overconf?':>12}")
    print("  " + "-" * 68)

    all_files = []
    for cal_name, cal_results in ALL_TEXT_CALIBRATORS.items():
        for filename, model_name in cal_results.items():
            all_files.append((filename, f"{cal_name} → {model_name}"))
    for filename, model_name in VLM_RESULTS.items():
        all_files.append((filename, f"VLM → {model_name}"))

    for filename, label in all_files:
        r = load_results(filename)
        if r is None:
            continue

        ece = r.get("ece", 0)
        brier = r.get("brier", 0)
        base_rate = r.get("base_rate", 0.5)

        # Check overconfidence: mean prediction vs base rate
        overconf = "unknown"
        if "per_benchmark" in r:
            mean_preds = []
            for pb in r["per_benchmark"].values():
                mp = pb.get("mean_p_correct")
                if mp is not None:
                    mean_preds.append(mp)
            if mean_preds:
                avg_pred = np.mean(mean_preds)
                if avg_pred > base_rate + 0.05:
                    overconf = f"yes ({avg_pred:.2f}>{base_rate:.2f})"
                elif avg_pred < base_rate - 0.05:
                    overconf = f"under ({avg_pred:.2f}<{base_rate:.2f})"
                else:
                    overconf = "well-cal."

        print(f"  {label:<40} {ece:>8.4f} {brier:>8.4f} {overconf:>12}")


def analyze_selective_prediction():
    """Compute selective prediction metrics from existing predictions."""
    print_header("SELECTIVE PREDICTION ANALYSIS")
    print("  (Coverage @ 90% accuracy, using P(correct) as selection criterion)")
    print()

    for filename in list(TEXT_RESULTS.keys()) + list(VLM_RESULTS.keys()):
        r = load_results(filename)
        if r is None or "per_benchmark" not in r:
            continue
        display = TEXT_RESULTS.get(filename) or VLM_RESULTS.get(filename)
        judge = "Text" if filename in TEXT_RESULTS else "VLM"
        label = f"{judge} cal. on {display}"

        n_total = r.get("n_samples", 0)
        n_correct = r.get("n_correct", 0)
        base_acc = n_correct / n_total if n_total > 0 else 0

        print(f"  {label}:")
        print(f"    Base accuracy: {base_acc:.1%} ({n_correct}/{n_total})")
        if base_acc >= 0.9:
            print(f"    Coverage@90%: 100% (base accuracy already >= 90%)")
        else:
            print(f"    Coverage@90%: requires raw predictions (not in summary JSON)")
        print()


def analyze_verbalized_confidence():
    """Analyze verbalized confidence from prediction files (self-reported confidence)."""
    print_header("VERBALIZED CONFIDENCE BASELINE")
    print("  Extracting self-reported confidence from model response JSON")
    print()

    prefixes = {
        "GPT-5.2": "gpt52_high_",
        "GPT-5-mini": "gpt5_mini_combined/",
        "Qwen3.5-397B": "qwen35_397b_",
    }

    for model_name, prefix in prefixes.items():
        all_confs = []
        all_labels = []
        per_bench = defaultdict(lambda: {"confs": [], "labels": []})

        if "/" in prefix:
            # GPT-5-mini uses combined dir structure
            base = RUNS_DIR / prefix.rstrip("/")
            if not base.exists():
                continue
            bench_dirs = sorted(base.iterdir())
        else:
            bench_dirs = sorted(RUNS_DIR.iterdir())
            bench_dirs = [d for d in bench_dirs if d.name.startswith(prefix)]

        for bench_dir in bench_dirs:
            if "/" in prefix:
                benchmark = bench_dir.name
            else:
                benchmark = bench_dir.name[len(prefix):]

            pred_file = bench_dir / "predictions.jsonl"
            if not pred_file.exists() or pred_file.stat().st_size == 0:
                continue

            with open(pred_file) as f:
                for line in f:
                    try:
                        pred = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Extract score
                    score = pred.get("score", {})
                    if isinstance(score, dict):
                        correct = score.get("correct", -1)
                    else:
                        correct = score
                    if correct not in (0, 1):
                        continue

                    # Extract verbalized confidence from response_text
                    conf = _extract_verbalized_confidence(pred)
                    if conf is None:
                        continue

                    all_confs.append(conf)
                    all_labels.append(float(correct == 1))
                    per_bench[benchmark]["confs"].append(conf)
                    per_bench[benchmark]["labels"].append(float(correct == 1))

        if len(all_confs) < 10:
            print(f"  {model_name}: too few samples with verbalized confidence ({len(all_confs)})")
            continue

        confs = np.array(all_confs)
        labels = np.array(all_labels)

        # Compute AUROC using verbalized confidence
        from sklearn.metrics import roc_auc_score, brier_score_loss
        auroc = roc_auc_score(labels, confs) if len(set(labels)) > 1 else 0.5
        brier = brier_score_loss(labels, confs)

        # ECE
        n_bins = 10
        ece = 0.0
        for j in range(n_bins):
            lo, hi = j / n_bins, (j + 1) / n_bins
            mask = (confs >= lo) & (confs < hi)
            if mask.sum() == 0:
                continue
            ece += (mask.sum() / len(confs)) * abs(labels[mask].mean() - confs[mask].mean())

        print(f"  {model_name}: AUROC={auroc:.4f}, Brier={brier:.4f}, ECE={ece:.4f} "
              f"(n={len(confs)}, mean_conf={confs.mean():.3f}, base_rate={labels.mean():.3f})")

        # Per-benchmark
        for bench in sorted(per_bench.keys()):
            bd = per_bench[bench]
            if len(bd["confs"]) < 5:
                continue
            bc = np.array(bd["confs"])
            bl = np.array(bd["labels"])
            try:
                ba = roc_auc_score(bl, bc) if len(set(bl)) > 1 else None
            except ValueError:
                ba = None
            auroc_str = f"{ba:.3f}" if ba is not None else "N/A"
            print(f"    {bench:<20} AUROC={auroc_str:>7} "
                  f"(n={len(bc)}, mean_conf={bc.mean():.2f}, acc={bl.mean():.2f})")
        print()


def _extract_verbalized_confidence(pred):
    """Extract verbalized confidence from a prediction record."""
    # Method 1: from score.brier field (computed from confidence)
    score = pred.get("score", {})
    if isinstance(score, dict) and "brier" in score:
        # Brier was computed from the confidence, so we can use it
        pass

    # Method 2: from response_text JSON
    response_text = pred.get("response_text", "")
    if isinstance(response_text, str):
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, dict) and "confidence" in parsed:
                conf = parsed["confidence"]
                if isinstance(conf, (int, float)) and 0 <= conf <= 1:
                    return float(conf)
        except (json.JSONDecodeError, ValueError):
            pass

    # Method 3: from prediction dict
    prediction = pred.get("prediction", {})
    if isinstance(prediction, dict) and "confidence" in prediction:
        conf = prediction["confidence"]
        if isinstance(conf, (int, float)) and 0 <= conf <= 1:
            return float(conf)

    # Method 4: top-level confidence field
    conf = pred.get("confidence")
    if conf is not None and isinstance(conf, (int, float)) and 0 <= conf <= 1:
        return float(conf)

    return None


def analyze_sample_counts():
    """Show current sample counts for overnight job monitoring."""
    print_header("CURRENT QWEN3.5-397B PREDICTION COUNTS")
    print("  (Resume targets for overnight job)")
    print()

    total = 0
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.name.startswith("qwen35_397b_"):
            continue
        if "backup" in d.name:
            continue
        bench = d.name[len("qwen35_397b_"):]
        pred_file = d / "predictions.jsonl"
        if pred_file.exists():
            n = sum(1 for _ in open(pred_file))
        else:
            n = 0
        # Determine target from GPT-5.2 sampled IDs
        target_file = d / "sampled_ids.json"
        target_n = "?"
        if target_file.exists():
            with open(target_file) as f:
                target_n = len(json.load(f))
        done_str = "DONE" if isinstance(target_n, int) and n >= target_n else ""
        print(f"  {bench:<20} {n:>5}/{str(target_n):>5}  {done_str}")
        total += n

    print(f"\n  Total: {total} predictions")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive cross-model analysis")
    parser.add_argument("--output_dir", default="figures/analysis",
                        help="Directory for output figures/data")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    analyze_transfer_matrix()
    analyze_per_benchmark()
    analyze_calibration()
    analyze_selective_prediction()
    analyze_verbalized_confidence()
    analyze_sample_counts()

    print_header("DONE")
    print(f"  Analysis complete. Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
