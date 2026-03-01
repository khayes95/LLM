#!/usr/bin/env python3
"""
Compute stronger UQ baselines and add them to scored data files.

Baselines added:
1. Platt-scaled verbalized confidence (logistic regression fit on train split)
2. Response length baseline (normalized token count → P(correct))
3. Combined (verbalized + length) ensemble

These are CPU-only and run on existing scored data.
The zero-shot base model baseline requires a separate GPU script.

Usage:
    python scripts/compute_baselines.py --scored_dir data/use_cases/scored_unified
    python scripts/compute_baselines.py --scored_dir data/use_cases/scored_unified --smoke_test
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr


def load_scored(path):
    """Load scored JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_scored(data, path):
    """Save scored JSONL file."""
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def compute_platt_scaled_verbalized(data, train_frac=0.5, seed=42):
    """
    Fit Platt scaling (logistic regression) on verbalized confidence.
    This turns the raw verbalized confidence into calibrated probabilities.

    Uses a train/test split to avoid overfitting.
    """
    rng = np.random.RandomState(seed)

    # Filter to samples with valid verbalized confidence
    valid = [(i, d) for i, d in enumerate(data)
             if d.get("verbalized_confidence") is not None
             and d.get("is_correct") is not None]

    if len(valid) < 20:
        print(f"    Too few samples with verbalized confidence ({len(valid)}), skipping")
        return data

    indices = [i for i, _ in valid]
    X = np.array([d["verbalized_confidence"] for _, d in valid]).reshape(-1, 1)
    y = np.array([d["is_correct"] for _, d in valid])

    # Stratified-ish split
    perm = rng.permutation(len(valid))
    n_train = int(len(valid) * train_frac)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    # Fit logistic regression
    lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    lr.fit(X[train_idx], y[train_idx])

    # Predict calibrated probabilities for ALL samples
    p_platt = lr.predict_proba(X)[:, 1]

    # Also fit isotonic regression (non-parametric)
    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
    iso.fit(X[train_idx].ravel(), y[train_idx])
    p_iso = iso.predict(X.ravel())

    # Add to data
    for j, (orig_idx, _) in enumerate(valid):
        data[orig_idx]["p_platt_verbalized"] = float(p_platt[j])
        data[orig_idx]["p_isotonic_verbalized"] = float(p_iso[j])

    # For samples without verbalized confidence, set to NaN
    valid_set = set(indices)
    for i, d in enumerate(data):
        if i not in valid_set:
            d["p_platt_verbalized"] = None
            d["p_isotonic_verbalized"] = None

    # Report
    test_auroc_raw = roc_auc_score(y[test_idx], X[test_idx].ravel())
    test_auroc_platt = roc_auc_score(y[test_idx], p_platt[test_idx])
    test_auroc_iso = roc_auc_score(y[test_idx], p_iso[test_idx])
    print(f"    Verbalized AUROC (raw): {test_auroc_raw:.4f}")
    print(f"    Verbalized AUROC (Platt): {test_auroc_platt:.4f}")
    print(f"    Verbalized AUROC (Isotonic): {test_auroc_iso:.4f}")

    return data


def compute_length_baseline(data):
    """
    Use normalized response length as a UQ signal.

    Hypothesis: Very short or very long responses may be less reliable.
    We fit a logistic regression: log(output_tokens) → P(correct).
    """
    valid = [(i, d) for i, d in enumerate(data)
             if d.get("total_tokens") is not None
             and d.get("is_correct") is not None
             and (d.get("output_tokens") or d.get("total_tokens", 0)) > 0]

    if len(valid) < 20:
        print(f"    Too few samples with token counts ({len(valid)}), skipping")
        return data

    # Use output_tokens if available, else total_tokens
    tokens = []
    for _, d in valid:
        t = d.get("output_tokens") or d.get("total_tokens", 1)
        tokens.append(max(t, 1))

    X = np.log1p(np.array(tokens)).reshape(-1, 1)
    y = np.array([d["is_correct"] for _, d in valid])

    # Fit on all data (length baseline is simple, not prone to overfitting)
    lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    lr.fit(X, y)
    p_length = lr.predict_proba(X)[:, 1]

    for j, (orig_idx, _) in enumerate(valid):
        data[orig_idx]["p_length_baseline"] = float(p_length[j])

    # AUROC
    auroc = roc_auc_score(y, p_length)
    print(f"    Length baseline AUROC: {auroc:.4f}")

    # For samples without token counts
    valid_set = set(i for i, _ in valid)
    for i, d in enumerate(data):
        if i not in valid_set:
            d["p_length_baseline"] = None

    return data


def compute_combined_baseline(data):
    """
    Combine verbalized confidence + response length into a single baseline.
    This is a stronger baseline than either alone.
    """
    valid = [(i, d) for i, d in enumerate(data)
             if d.get("verbalized_confidence") is not None
             and d.get("is_correct") is not None
             and (d.get("output_tokens") or d.get("total_tokens", 0)) > 0]

    if len(valid) < 20:
        print(f"    Too few samples for combined baseline ({len(valid)}), skipping")
        return data

    features = []
    for _, d in valid:
        verb = d["verbalized_confidence"]
        tokens = d.get("output_tokens") or d.get("total_tokens", 1)
        features.append([verb, np.log1p(max(tokens, 1))])

    X = np.array(features)
    y = np.array([d["is_correct"] for _, d in valid])

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(valid))
    n_train = int(len(valid) * 0.5)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    lr.fit(X[train_idx], y[train_idx])
    p_combined = lr.predict_proba(X)[:, 1]

    for j, (orig_idx, _) in enumerate(valid):
        data[orig_idx]["p_combined_baseline"] = float(p_combined[j])

    test_auroc = roc_auc_score(y[test_idx], p_combined[test_idx])
    print(f"    Combined baseline AUROC: {test_auroc:.4f}")

    valid_set = set(i for i, _ in valid)
    for i, d in enumerate(data):
        if i not in valid_set:
            d["p_combined_baseline"] = None

    return data


def main():
    parser = argparse.ArgumentParser(description="Compute UQ baselines on scored data")
    parser.add_argument("--scored_dir", default="data/use_cases/scored_unified",
                        help="Directory with scored JSONL files")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: same as scored_dir)")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run on first 100 samples per file (writes to separate dir)")
    args = parser.parse_args()

    # Smoke test writes to a separate directory to prevent data loss
    if args.smoke_test and args.output_dir is None:
        output_dir = args.scored_dir + "_baseline_smoke"
    else:
        output_dir = args.output_dir or args.scored_dir
    os.makedirs(output_dir, exist_ok=True)

    targets = ["gpt5mini", "gpt52", "qwen35"]

    for target in targets:
        path = Path(args.scored_dir) / f"{target}_scored.jsonl"
        if not path.exists():
            print(f"Skipping {target}: {path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {target}")
        print(f"{'='*60}")

        data = load_scored(path)
        if args.smoke_test:
            data = data[:100]
        print(f"  Loaded {len(data)} samples")

        # Compute baselines
        print("\n  Platt-scaled verbalized confidence:")
        data = compute_platt_scaled_verbalized(data)

        print("\n  Response length baseline:")
        data = compute_length_baseline(data)

        print("\n  Combined (verbalized + length) baseline:")
        data = compute_combined_baseline(data)

        # Report comparison with calibrator
        valid = [d for d in data if d.get("p_correct") is not None and d.get("is_correct") is not None]
        if valid:
            y = np.array([d["is_correct"] for d in valid])
            p_cal = np.array([d["p_correct"] for d in valid])
            cal_auroc = roc_auc_score(y, p_cal)
            print(f"\n  Summary for {target}:")
            print(f"    Calibrator AUROC:       {cal_auroc:.4f}")
            for key, label in [
                ("p_platt_verbalized", "Platt verbalized"),
                ("p_isotonic_verbalized", "Isotonic verbalized"),
                ("p_length_baseline", "Length"),
                ("p_combined_baseline", "Combined"),
            ]:
                valid_k = [d for d in valid if d.get(key) is not None]
                if valid_k:
                    yk = np.array([d["is_correct"] for d in valid_k])
                    pk = np.array([d[key] for d in valid_k])
                    auroc_k = roc_auc_score(yk, pk)
                    print(f"    {label:24s} {auroc_k:.4f}  (delta: {cal_auroc - auroc_k:+.4f})")

        # Save
        out_path = Path(output_dir) / f"{target}_scored.jsonl"
        save_scored(data, out_path)
        print(f"\n  Saved to {out_path}")

    print("\nDone! Baseline fields added: p_platt_verbalized, p_isotonic_verbalized, p_length_baseline, p_combined_baseline")


if __name__ == "__main__":
    main()
