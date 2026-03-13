#!/usr/bin/env python3
"""Compute AUPRC for v3 scored test-only data."""

import json
import os
import numpy as np
from sklearn.metrics import average_precision_score

SCORED_DIR = "/scratch/khayes/LLM/data/use_cases/scored_test_only_v3"
OUTPUT_PATH = "/scratch/khayes/LLM/data/use_cases/results_test_only_v3/auprc_v3.json"

FILES = {
    "gpt5mini": "gpt5mini_scored.jsonl",
    "gpt52": "gpt52_scored.jsonl",
    "qwen35": "qwen35_scored.jsonl",
}

SCORE_KEYS = [
    ("calibrator", "p_correct"),
    ("verbalized_raw", "verbalized_confidence"),
    ("isotonic_verbalized", "p_isotonic_verbalized"),
    ("length_baseline", "p_length_baseline"),
    ("combined_baseline", "p_combined_baseline"),
]


def load_scored(path):
    records = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("is_correct") is None or d.get("p_correct") is None:
                continue
            records.append(d)
    return records


def safe_auprc(y_true, y_score):
    """Compute AUPRC, filtering NaN scores. Returns dict or None."""
    mask = ~np.isnan(y_score)
    y_t = y_true[mask]
    y_s = y_score[mask]
    n_valid = int(mask.sum())
    n_nan = int((~mask).sum())
    if n_valid < 2 or len(np.unique(y_t)) < 2:
        return None
    ap = average_precision_score(y_t, y_s)
    return {
        "auprc": round(float(ap), 4),
        "n_samples": n_valid,
        "n_positive": int(np.sum(y_t)),
        "prevalence": round(float(np.mean(y_t)), 4),
        "n_nan_excluded": n_nan,
    }


results = {}
# For combined
all_y = []
all_scores = {k: [] for k, _ in SCORE_KEYS}

for model_key, fname in FILES.items():
    path = os.path.join(SCORED_DIR, fname)
    records = load_scored(path)

    y_true = np.array([r["is_correct"] for r in records], dtype=int)
    all_y.append(y_true)

    model_results = {}
    for score_name, field in SCORE_KEYS:
        scores = np.array([r.get(field, np.nan) for r in records], dtype=float)
        # Replace None with NaN
        res = safe_auprc(y_true, scores)
        if res is not None:
            model_results[score_name] = res
            all_scores[score_name].append(scores)
        else:
            # Still collect for combined if there are valid values
            all_scores[score_name].append(scores)

    results[model_key] = model_results
    cal = model_results.get("calibrator", {})
    verb = model_results.get("verbalized_raw", {})
    print(f"{model_key}: n={len(y_true)}, cal AUPRC={cal.get('auprc')}, "
          f"verb AUPRC={verb.get('auprc')}, prevalence={cal.get('prevalence')}")

# Combined
y_all = np.concatenate(all_y)
combined = {}
for score_name, _ in SCORE_KEYS:
    if all_scores[score_name]:
        s_all = np.concatenate(all_scores[score_name])
        res = safe_auprc(y_all, s_all)
        if res is not None:
            combined[score_name] = res

results["combined"] = combined

cal_c = combined.get("calibrator", {})
verb_c = combined.get("verbalized_raw", {})
print(f"\nCombined: n={len(y_all)}, cal AUPRC={cal_c.get('auprc')}, "
      f"verb AUPRC={verb_c.get('auprc')}, prevalence={cal_c.get('prevalence')}")

# Bootstrap CI for combined calibrator AUPRC
p_cal_all = np.concatenate(all_scores["calibrator"])
rng = np.random.RandomState(42)
n_boot = 10000
boot_auprc = []
for _ in range(n_boot):
    idx = rng.choice(len(y_all), size=len(y_all), replace=True)
    if len(np.unique(y_all[idx])) < 2:
        continue
    boot_auprc.append(average_precision_score(y_all[idx], p_cal_all[idx]))

boot_auprc = np.array(boot_auprc)
ci_lower = float(np.percentile(boot_auprc, 2.5))
ci_upper = float(np.percentile(boot_auprc, 97.5))
combined["calibrator"]["bootstrap_ci_95"] = [round(ci_lower, 4), round(ci_upper, 4)]
combined["calibrator"]["bootstrap_mean"] = round(float(np.mean(boot_auprc)), 4)

print(f"Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {OUTPUT_PATH}")
