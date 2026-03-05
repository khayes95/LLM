#!/usr/bin/env python3
"""
Compute precision/recall/F1 across confidence thresholds for all scoring methods.

For each method (calibrator, verbalized, platt, combined, zeroshot):
  - Sweep 101 thresholds from 0.0 to 1.0
  - At each threshold t, predict "correct" if p >= t
  - Compute: precision, recall, F1, accuracy, TPR, FPR, coverage metrics
  - Find optimal threshold (max F1), thresholds for 90%/95% precision and recall
  - Compute PR-AUC

Also computes risk-coverage curves: if we only answer when p >= t,
what fraction of questions do we answer (coverage) and what accuracy do we get?

Input:  data/use_cases/scored_test_only/{gpt5mini,gpt52,qwen35}_scored.jsonl
Output: data/use_cases/results_test_only/decision_thresholds.json

Usage:
    python scripts/cpu_decision_thresholds.py
    python scripts/cpu_decision_thresholds.py --smoke_test
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

SCORED_DIR = "data/use_cases/scored_test_only"
SCORED_FILES = [
    "gpt5mini_scored.jsonl",
    "gpt52_scored.jsonl",
    "qwen35_scored.jsonl",
]

METHOD_KEYS = {
    "calibrator": "p_correct",
    "verbalized": "verbalized_confidence",
    "platt": "p_platt_verbalized",
    "combined": "p_combined_baseline",
    "zeroshot": "p_zeroshot",
}


def load_all_scored(scored_dir: str, max_examples: int | None = None) -> list[dict]:
    """Load all scored JSONL files and return list of dicts."""
    samples = []
    for fname in SCORED_FILES:
        path = os.path.join(scored_dir, fname)
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                samples.append(rec)
                if max_examples is not None and len(samples) >= max_examples:
                    return samples
    return samples


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

THRESHOLDS = np.linspace(0.0, 1.0, 101)  # 0.00, 0.01, ..., 1.00


def compute_threshold_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray = THRESHOLDS,
) -> dict:
    """
    Compute precision/recall/F1/accuracy/TPR/FPR at each threshold, plus
    optimal thresholds and PR-AUC.

    Parameters
    ----------
    labels : array of 0/1 ground truth (1 = correct)
    scores : array of predicted probabilities of correctness
    thresholds : array of thresholds to sweep

    Returns
    -------
    dict with all computed metrics
    """
    n = len(labels)
    if n == 0:
        return {"n": 0, "error": "no samples"}

    n_pos = int(labels.sum())
    n_neg = n - n_pos

    # Pre-sort for efficiency is not needed; numpy broadcasting is fine
    # for 101 thresholds.

    curve = []
    best_f1 = -1.0
    best_f1_threshold = 0.5
    best_f1_metrics = {}

    for t in thresholds:
        t_val = float(t)
        pred_pos = scores >= t_val
        pred_neg = ~pred_pos

        tp = int((pred_pos & (labels == 1)).sum())
        fp = int((pred_pos & (labels == 0)).sum())
        tn = int((pred_neg & (labels == 0)).sum())
        fn = int((pred_neg & (labels == 1)).sum())

        n_pred_pos = int(pred_pos.sum())
        n_pred_neg = int(pred_neg.sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / n if n > 0 else 0.0
        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0

        point = {
            "threshold": round(t_val, 2),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "accuracy": round(accuracy, 6),
            "tpr": round(tpr, 6),
            "fpr": round(fpr, 6),
            "n_predicted_positive": n_pred_pos,
            "n_predicted_negative": n_pred_neg,
        }
        curve.append(point)

        if f1 > best_f1:
            best_f1 = f1
            best_f1_threshold = t_val
            best_f1_metrics = point.copy()

    # Find thresholds for target precision levels
    def find_threshold_for_target(metric_name, target, direction=">="):
        """Find the lowest threshold where metric >= target (for precision)
        or the highest threshold where metric >= target (for recall)."""
        if metric_name == "precision":
            # For precision: sweep from low to high threshold.
            # Higher threshold -> higher precision (generally).
            # We want the LOWEST threshold that achieves target precision
            # (to maximize coverage).
            for pt in curve:
                # Skip points where nothing is predicted positive
                if pt["n_predicted_positive"] == 0:
                    continue
                if pt[metric_name] >= target:
                    return pt["threshold"]
            return None
        elif metric_name == "recall":
            # For recall: sweep from high to low threshold.
            # Lower threshold -> higher recall.
            # We want the HIGHEST threshold that still achieves target recall.
            for pt in reversed(curve):
                if pt[metric_name] >= target:
                    return pt["threshold"]
            return None
        return None

    threshold_90_precision = find_threshold_for_target("precision", 0.90)
    threshold_95_precision = find_threshold_for_target("precision", 0.95)
    threshold_90_recall = find_threshold_for_target("recall", 0.90)
    threshold_95_recall = find_threshold_for_target("recall", 0.95)

    # PR-AUC via sklearn
    try:
        pr_auc = float(average_precision_score(labels, scores))
    except Exception:
        pr_auc = None

    return {
        "n": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "prevalence": round(n_pos / n, 4) if n > 0 else 0.0,
        "pr_auc": round(pr_auc, 6) if pr_auc is not None else None,
        "optimal_f1": {
            "threshold": round(best_f1_threshold, 2),
            "f1": round(best_f1, 6),
            "precision": best_f1_metrics.get("precision", 0.0),
            "recall": best_f1_metrics.get("recall", 0.0),
            "accuracy": best_f1_metrics.get("accuracy", 0.0),
        },
        "threshold_90_precision": threshold_90_precision,
        "threshold_95_precision": threshold_95_precision,
        "threshold_90_recall": threshold_90_recall,
        "threshold_95_recall": threshold_95_recall,
        "threshold_curve": curve,
    }


def compute_risk_coverage(
    labels: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray = THRESHOLDS,
) -> list[dict]:
    """
    Compute risk-coverage curve: at each threshold t, if we only answer
    questions where score >= t, what fraction do we answer (coverage)
    and what is our accuracy on those?

    Returns list of dicts with threshold, coverage, accuracy, n_answered.
    """
    n = len(labels)
    if n == 0:
        return []

    points = []
    for t in thresholds:
        t_val = float(t)
        mask = scores >= t_val
        n_answered = int(mask.sum())
        coverage = n_answered / n if n > 0 else 0.0

        if n_answered > 0:
            accuracy = float(labels[mask].sum()) / n_answered
        else:
            accuracy = 0.0

        # Risk = 1 - accuracy (error rate on answered questions)
        risk = 1.0 - accuracy

        points.append({
            "threshold": round(t_val, 2),
            "coverage": round(coverage, 6),
            "accuracy": round(accuracy, 6),
            "risk": round(risk, 6),
            "n_answered": n_answered,
        })

    return points


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute decision thresholds for UQ methods"
    )
    parser.add_argument(
        "--scored_dir",
        default=SCORED_DIR,
        help="Directory containing scored JSONL files",
    )
    parser.add_argument(
        "--output",
        default="data/use_cases/results_test_only/decision_thresholds.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run on first 50 samples only (quick validation)",
    )
    args = parser.parse_args()

    max_examples = 50 if args.smoke_test else None
    samples = load_all_scored(args.scored_dir, max_examples=max_examples)
    print(f"Loaded {len(samples)} samples from {args.scored_dir}")

    if len(samples) == 0:
        print("ERROR: No samples loaded. Check scored_dir.", file=sys.stderr)
        sys.exit(1)

    # Extract labels and per-method scores
    labels_all = np.array([s["is_correct"] for s in samples], dtype=np.int32)

    # Group by target model
    model_groups = defaultdict(list)
    for i, s in enumerate(samples):
        model_groups[s["target_model"]].append(i)

    results = {}

    # -----------------------------------------------------------------------
    # For each method, compute thresholds on combined + per-model data
    # -----------------------------------------------------------------------
    for method_name, score_key in METHOD_KEYS.items():
        print(f"\n{'='*60}")
        print(f"Method: {method_name} (key: {score_key})")
        print(f"{'='*60}")

        # Extract scores, handling missing values
        scores_all = []
        valid_mask = []
        for s in samples:
            val = s.get(score_key)
            if val is not None:
                scores_all.append(float(val))
                valid_mask.append(True)
            else:
                scores_all.append(0.0)
                valid_mask.append(False)
        scores_all = np.array(scores_all, dtype=np.float64)
        valid_mask = np.array(valid_mask, dtype=bool)

        n_valid = int(valid_mask.sum())
        n_missing = len(valid_mask) - n_valid
        if n_missing > 0:
            print(f"  WARNING: {n_missing} samples missing {score_key}")

        # Filter to valid samples for this method
        labels_valid = labels_all[valid_mask]
        scores_valid = scores_all[valid_mask]

        # Combined (all data)
        print(f"\n  --- Combined (n={len(labels_valid)}) ---")
        combined_metrics = compute_threshold_metrics(labels_valid, scores_valid)
        combined_coverage = compute_risk_coverage(labels_valid, scores_valid)

        opt = combined_metrics["optimal_f1"]
        print(f"  Optimal F1:        {opt['f1']:.4f} @ threshold={opt['threshold']:.2f}"
              f"  (P={opt['precision']:.4f}, R={opt['recall']:.4f})")
        if combined_metrics["threshold_90_precision"] is not None:
            print(f"  90% Precision @:   threshold={combined_metrics['threshold_90_precision']:.2f}")
        else:
            print(f"  90% Precision @:   NOT ACHIEVABLE")
        if combined_metrics["threshold_95_precision"] is not None:
            print(f"  95% Precision @:   threshold={combined_metrics['threshold_95_precision']:.2f}")
        else:
            print(f"  95% Precision @:   NOT ACHIEVABLE")
        if combined_metrics["threshold_90_recall"] is not None:
            print(f"  90% Recall @:      threshold={combined_metrics['threshold_90_recall']:.2f}")
        else:
            print(f"  90% Recall @:      NOT ACHIEVABLE")
        if combined_metrics["threshold_95_recall"] is not None:
            print(f"  95% Recall @:      threshold={combined_metrics['threshold_95_recall']:.2f}")
        else:
            print(f"  95% Recall @:      NOT ACHIEVABLE")
        print(f"  PR-AUC:            {combined_metrics['pr_auc']}")

        # Print a few risk-coverage highlights
        for pt in combined_coverage:
            if pt["threshold"] in (0.5, 0.7, 0.9):
                print(f"  Coverage @ t={pt['threshold']:.1f}:  "
                      f"{pt['coverage']*100:.1f}% answered, "
                      f"{pt['accuracy']*100:.1f}% accuracy")

        method_result = {
            "combined": {
                "threshold_metrics": combined_metrics,
                "risk_coverage": combined_coverage,
            },
            "per_model": {},
        }

        # Per target model
        for model_name, indices in sorted(model_groups.items()):
            idx = np.array(indices)
            model_valid = valid_mask[idx]
            model_labels = labels_all[idx][model_valid]
            model_scores = scores_all[idx][model_valid]

            if len(model_labels) == 0:
                print(f"\n  --- {model_name}: no valid samples, skipping ---")
                continue

            print(f"\n  --- {model_name} (n={len(model_labels)}) ---")
            model_metrics = compute_threshold_metrics(model_labels, model_scores)
            model_coverage = compute_risk_coverage(model_labels, model_scores)

            mopt = model_metrics["optimal_f1"]
            print(f"  Optimal F1:        {mopt['f1']:.4f} @ threshold={mopt['threshold']:.2f}"
                  f"  (P={mopt['precision']:.4f}, R={mopt['recall']:.4f})")
            if model_metrics["threshold_90_precision"] is not None:
                print(f"  90% Precision @:   threshold={model_metrics['threshold_90_precision']:.2f}")
            if model_metrics["pr_auc"] is not None:
                print(f"  PR-AUC:            {model_metrics['pr_auc']:.4f}")

            method_result["per_model"][model_name] = {
                "threshold_metrics": model_metrics,
                "risk_coverage": model_coverage,
            }

        results[method_name] = method_result

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*80}")
    print("SUMMARY: Optimal F1 Thresholds (Combined)")
    print(f"{'='*80}")
    header = (f"{'Method':<15} {'PR-AUC':>8} {'Best F1':>8} {'Thresh':>8} "
              f"{'Prec':>8} {'Recall':>8} {'90%P @':>8} {'95%P @':>8}")
    print(header)
    print("-" * 80)

    for method_name in METHOD_KEYS:
        r = results[method_name]["combined"]["threshold_metrics"]
        opt = r["optimal_f1"]
        t90p = r["threshold_90_precision"]
        t95p = r["threshold_95_precision"]
        pr_auc_val = r["pr_auc"]

        pr_auc_str = f"{pr_auc_val:.4f}" if pr_auc_val is not None else "N/A"
        t90p_str = f"{t90p:.2f}" if t90p is not None else "N/A"
        t95p_str = f"{t95p:.2f}" if t95p is not None else "N/A"

        print(f"{method_name:<15} {pr_auc_str:>8} {opt['f1']:>8.4f} "
              f"{opt['threshold']:>8.2f} {opt['precision']:>8.4f} "
              f"{opt['recall']:>8.4f} {t90p_str:>8} {t95p_str:>8}")

    # -----------------------------------------------------------------------
    # Save output
    # -----------------------------------------------------------------------
    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = {
        "description": "Decision threshold analysis for UQ scoring methods",
        "scored_dir": args.scored_dir,
        "n_total_samples": len(samples),
        "smoke_test": args.smoke_test,
        "methods": list(METHOD_KEYS.keys()),
        "method_keys": METHOD_KEYS,
        "thresholds_swept": [round(t, 2) for t in THRESHOLDS],
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
