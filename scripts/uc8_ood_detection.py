#!/usr/bin/env python3
"""UC8: OOD / Distribution Shift Detection — batch-level monitoring.

Monitors batch-level calibrator scores to detect when a model is failing
on a domain or experiencing distribution shift. Analyses:

1. Benchmark-level scatter: mean P(correct) vs actual accuracy, Spearman r
2. Bootstrap batch simulation: correlation at batch sizes {25, 50, 100, 200}
3. Cross-model shift detection: flag benchmarks with abnormally low z_batch
4. Alert system: threshold on batch mean P(correct) for accuracy < 50%

Baselines: verbalized confidence batch mean, random, hardcoded difficulty.

Usage:
    python scripts/uc8_ood_detection.py
    python scripts/uc8_ood_detection.py --scored_dir data/use_cases/scored_test_only_v2
    python scripts/uc8_ood_detection.py --smoke_test
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, ks_2samp


def load_scored(path):
    """Load scored JSONL."""
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def group_by_benchmark(samples):
    """Group samples by benchmark name."""
    groups = defaultdict(list)
    for s in samples:
        groups[s["benchmark"]].append(s)
    return dict(groups)


# ---------------------------------------------------------------------------
# Analysis 1: Benchmark-level scatter (mean P(correct) vs actual accuracy)
# ---------------------------------------------------------------------------

def benchmark_level_correlation(samples, score_key="p_correct", min_n=5):
    """Compute per-benchmark mean score vs actual accuracy.

    Returns dict with per-benchmark stats, Spearman r, and p-value.
    """
    groups = group_by_benchmark(samples)
    bench_stats = {}
    for bench, items in groups.items():
        if len(items) < min_n:
            continue
        scores = np.array([s[score_key] for s in items if s.get(score_key) is not None])
        labels = np.array([s["is_correct"] for s in items if s.get(score_key) is not None])
        if len(scores) < min_n:
            continue
        bench_stats[bench] = {
            "n": len(scores),
            "mean_score": float(scores.mean()),
            "actual_accuracy": float(labels.mean()),
        }

    if len(bench_stats) < 3:
        return {"benchmarks": bench_stats, "spearman_r": None, "p_value": None}

    benches = sorted(bench_stats.keys())
    mean_scores = [bench_stats[b]["mean_score"] for b in benches]
    actual_accs = [bench_stats[b]["actual_accuracy"] for b in benches]
    corr, pval = spearmanr(mean_scores, actual_accs)

    return {
        "benchmarks": bench_stats,
        "spearman_r": float(corr),
        "p_value": float(pval),
        "n_benchmarks": len(bench_stats),
    }


# ---------------------------------------------------------------------------
# Analysis 2: Bootstrap batch simulation
# ---------------------------------------------------------------------------

def bootstrap_batch_correlation(samples, batch_sizes=(25, 50, 100, 200),
                                n_simulations=1000, min_bench_samples=150,
                                rng_seed=42):
    """Draw random batches per benchmark, measure correlation at each batch size.

    For each batch size, draw n_simulations random batches from each benchmark,
    compute batch-level mean P(correct) and batch-level accuracy, then measure
    Spearman correlation across benchmarks per simulation. Report fraction of
    simulations where correlation is significant (p < 0.05).
    """
    rng = np.random.RandomState(rng_seed)
    groups = group_by_benchmark(samples)

    # Filter to benchmarks with enough samples
    eligible = {b: items for b, items in groups.items() if len(items) >= min_bench_samples}
    if len(eligible) < 3:
        return {"error": f"Only {len(eligible)} benchmarks have >= {min_bench_samples} samples",
                "n_eligible_benchmarks": len(eligible)}

    results = {}
    for bs in batch_sizes:
        sig_count = 0
        correlations = []

        for _ in range(n_simulations):
            bench_mean_p = []
            bench_acc = []
            for bench, items in eligible.items():
                # Draw a random batch of size bs (with replacement if needed)
                indices = rng.choice(len(items), size=min(bs, len(items)), replace=False)
                batch = [items[i] for i in indices]
                batch_p = np.mean([s["p_correct"] for s in batch])
                batch_a = np.mean([s["is_correct"] for s in batch])
                bench_mean_p.append(batch_p)
                bench_acc.append(batch_a)

            if len(bench_mean_p) >= 3:
                corr, pval = spearmanr(bench_mean_p, bench_acc)
                correlations.append(corr)
                if pval < 0.05:
                    sig_count += 1

        frac_sig = sig_count / n_simulations
        results[str(bs)] = {
            "batch_size": bs,
            "n_simulations": n_simulations,
            "n_benchmarks": len(eligible),
            "fraction_significant": float(frac_sig),
            "mean_correlation": float(np.mean(correlations)),
            "median_correlation": float(np.median(correlations)),
            "std_correlation": float(np.std(correlations)),
        }

    # Minimum reliable batch size: smallest where fraction_significant >= 0.95
    min_reliable = None
    for bs in sorted(batch_sizes):
        if results[str(bs)]["fraction_significant"] >= 0.95:
            min_reliable = bs
            break

    return {
        "batch_results": results,
        "eligible_benchmarks": list(eligible.keys()),
        "min_reliable_batch_size": min_reliable,
    }


# ---------------------------------------------------------------------------
# Analysis 3: Cross-model shift detection
# ---------------------------------------------------------------------------

def cross_model_shift_detection(all_scored, reference_target="gpt5mini",
                                z_threshold=-1.5, min_bench_n=10):
    """Compare batch distributions of other models to the reference.

    For each benchmark: compute mean P(correct) for reference model, then
    z_batch = (other_mean - ref_mean) / ref_std for each other model.
    Flag benchmarks where z_batch < z_threshold.
    Measure whether flagged benchmarks truly have lower accuracy.
    """
    if reference_target not in all_scored:
        return {"error": f"Reference target {reference_target} not found"}

    ref_groups = group_by_benchmark(all_scored[reference_target])

    # Compute reference per-benchmark stats
    ref_stats = {}
    for bench, items in ref_groups.items():
        if len(items) < min_bench_n:
            continue
        scores = np.array([s["p_correct"] for s in items])
        ref_stats[bench] = {
            "mean": float(scores.mean()),
            "std": float(scores.std() + 1e-8),
            "n": len(items),
            "accuracy": float(np.mean([s["is_correct"] for s in items])),
        }

    results = {}
    for target, samples in all_scored.items():
        if target == reference_target:
            continue

        target_groups = group_by_benchmark(samples)
        bench_shifts = {}
        flagged = []

        for bench, items in target_groups.items():
            if bench not in ref_stats or len(items) < min_bench_n:
                continue

            target_scores = np.array([s["p_correct"] for s in items])
            target_acc = float(np.mean([s["is_correct"] for s in items]))
            target_mean = float(target_scores.mean())

            ref_mean = ref_stats[bench]["mean"]
            ref_std = ref_stats[bench]["std"]
            z_batch = (target_mean - ref_mean) / ref_std

            is_flagged = z_batch < z_threshold
            bench_shifts[bench] = {
                "target_mean_p": target_mean,
                "ref_mean_p": ref_mean,
                "z_batch": float(z_batch),
                "target_accuracy": target_acc,
                "ref_accuracy": ref_stats[bench]["accuracy"],
                "n_target": len(items),
                "flagged": is_flagged,
            }
            if is_flagged:
                flagged.append(bench)

        # Evaluate: do flagged benchmarks truly have lower accuracy?
        flagged_accs = [bench_shifts[b]["target_accuracy"] for b in flagged]
        unflagged = [b for b in bench_shifts if not bench_shifts[b]["flagged"]]
        unflagged_accs = [bench_shifts[b]["target_accuracy"] for b in unflagged]

        flag_summary = {
            "n_benchmarks": len(bench_shifts),
            "n_flagged": len(flagged),
            "flagged_benchmarks": flagged,
            "flagged_mean_accuracy": float(np.mean(flagged_accs)) if flagged_accs else None,
            "unflagged_mean_accuracy": float(np.mean(unflagged_accs)) if unflagged_accs else None,
        }

        results[target] = {
            "benchmark_shifts": bench_shifts,
            "summary": flag_summary,
        }

    return {"reference": reference_target, "z_threshold": z_threshold, "targets": results}


# ---------------------------------------------------------------------------
# Analysis 4: Alert system (threshold on batch mean P(correct))
# ---------------------------------------------------------------------------

def alert_system(samples, accuracy_threshold=0.50, n_thresholds=200):
    """Threshold on per-benchmark mean P(correct) to detect low-accuracy domains.

    For each threshold t, benchmarks with mean P(correct) < t are "alerted".
    True positive = alerted benchmark truly has accuracy < accuracy_threshold.
    """
    groups = group_by_benchmark(samples)
    bench_stats = {}
    for bench, items in groups.items():
        if len(items) < 5:
            continue
        scores = np.array([s["p_correct"] for s in items])
        labels = np.array([s["is_correct"] for s in items])
        bench_stats[bench] = {
            "mean_p": float(scores.mean()),
            "accuracy": float(labels.mean()),
            "is_low_acc": float(labels.mean()) < accuracy_threshold,
        }

    if not bench_stats:
        return {"error": "No benchmarks with enough samples"}

    n_low = sum(1 for b in bench_stats.values() if b["is_low_acc"])
    n_high = len(bench_stats) - n_low

    if n_low == 0 or n_high == 0:
        return {
            "error": f"All benchmarks on same side of threshold ({n_low} low, {n_high} high)",
            "benchmark_stats": bench_stats,
            "accuracy_threshold": accuracy_threshold,
        }

    thresholds = np.linspace(0, 1, n_thresholds + 1)
    curve = []
    best_f1 = 0
    best_threshold = 0.5
    best_point = None

    for t in thresholds:
        tp = sum(1 for b in bench_stats.values() if b["mean_p"] < t and b["is_low_acc"])
        fp = sum(1 for b in bench_stats.values() if b["mean_p"] < t and not b["is_low_acc"])
        fn = sum(1 for b in bench_stats.values() if b["mean_p"] >= t and b["is_low_acc"])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        point = {
            "threshold": float(t),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": tp, "fp": fp, "fn": fn,
        }
        curve.append(point)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)
            best_point = point

    return {
        "accuracy_threshold": accuracy_threshold,
        "n_benchmarks": len(bench_stats),
        "n_low_accuracy": n_low,
        "n_high_accuracy": n_high,
        "best_f1": float(best_f1),
        "best_threshold": best_threshold,
        "best_point": best_point,
        "benchmark_stats": bench_stats,
        "pr_curve": curve,
    }


def alert_system_verbalized(samples, accuracy_threshold=0.50, n_thresholds=200):
    """Same as alert_system but using verbalized confidence as the score."""
    groups = group_by_benchmark(samples)
    bench_stats = {}
    for bench, items in groups.items():
        verb_items = [s for s in items if s.get("verbalized_confidence") is not None]
        if len(verb_items) < 5:
            continue
        scores = np.array([s["verbalized_confidence"] for s in verb_items])
        labels = np.array([s["is_correct"] for s in verb_items])
        bench_stats[bench] = {
            "mean_verb": float(scores.mean()),
            "accuracy": float(labels.mean()),
            "is_low_acc": float(labels.mean()) < accuracy_threshold,
        }

    if not bench_stats:
        return {"error": "No benchmarks with verbalized confidence"}

    n_low = sum(1 for b in bench_stats.values() if b["is_low_acc"])
    n_high = len(bench_stats) - n_low

    if n_low == 0 or n_high == 0:
        return {"error": f"All benchmarks on same side ({n_low} low, {n_high} high)"}

    thresholds = np.linspace(0, 1, n_thresholds + 1)
    best_f1 = 0
    best_threshold = 0.5

    for t in thresholds:
        tp = sum(1 for b in bench_stats.values() if b["mean_verb"] < t and b["is_low_acc"])
        fp = sum(1 for b in bench_stats.values() if b["mean_verb"] < t and not b["is_low_acc"])
        fn = sum(1 for b in bench_stats.values() if b["mean_verb"] >= t and b["is_low_acc"])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    return {
        "best_f1": float(best_f1),
        "best_threshold": best_threshold,
        "n_benchmarks": len(bench_stats),
    }


def alert_system_random(samples, accuracy_threshold=0.50, n_repeats=100,
                        rng_seed=42):
    """Random baseline: assign random scores to benchmarks, compute best F1."""
    rng = np.random.RandomState(rng_seed)
    groups = group_by_benchmark(samples)
    bench_accs = {}
    for bench, items in groups.items():
        if len(items) < 5:
            continue
        bench_accs[bench] = float(np.mean([s["is_correct"] for s in items]))

    if not bench_accs:
        return {"best_f1": 0.0}

    n_low = sum(1 for a in bench_accs.values() if a < accuracy_threshold)
    if n_low == 0 or n_low == len(bench_accs):
        return {"best_f1": 0.0}

    f1s = []
    for _ in range(n_repeats):
        random_scores = {b: rng.random() for b in bench_accs}
        best_f1 = 0
        for t in np.linspace(0, 1, 50):
            tp = sum(1 for b in bench_accs if random_scores[b] < t and bench_accs[b] < accuracy_threshold)
            fp = sum(1 for b in bench_accs if random_scores[b] < t and bench_accs[b] >= accuracy_threshold)
            fn = sum(1 for b in bench_accs if random_scores[b] >= t and bench_accs[b] < accuracy_threshold)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            best_f1 = max(best_f1, f1)
        f1s.append(best_f1)

    return {"best_f1": float(np.mean(f1s)), "std_f1": float(np.std(f1s))}


def alert_system_hardcoded(samples, accuracy_threshold=0.50):
    """Hardcoded difficulty baseline: flag known-hard benchmarks."""
    known_hard = {"hle", "omnimath", "arc_agi", "prbench", "simpleqa", "bbeh"}
    groups = group_by_benchmark(samples)
    bench_accs = {}
    for bench, items in groups.items():
        if len(items) < 5:
            continue
        bench_accs[bench] = float(np.mean([s["is_correct"] for s in items]))

    if not bench_accs:
        return {"best_f1": 0.0}

    tp = sum(1 for b in bench_accs if b in known_hard and bench_accs[b] < accuracy_threshold)
    fp = sum(1 for b in bench_accs if b in known_hard and bench_accs[b] >= accuracy_threshold)
    fn = sum(1 for b in bench_accs if b not in known_hard and bench_accs[b] < accuracy_threshold)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return {"best_f1": float(f1), "precision": float(prec), "recall": float(rec),
            "known_hard_list": sorted(known_hard)}


# ---------------------------------------------------------------------------
# Bootstrap CIs on alert F1
# ---------------------------------------------------------------------------

def bootstrap_alert_f1(samples, accuracy_threshold=0.50, n_bootstrap=1000,
                       rng_seed=42):
    """Bootstrap CI for the alert system's best F1 score.

    Resamples benchmarks (not individual samples) since the alert system
    operates at benchmark granularity.
    """
    rng = np.random.RandomState(rng_seed)
    groups = group_by_benchmark(samples)
    bench_list = []
    for bench, items in groups.items():
        if len(items) < 5:
            continue
        acc = float(np.mean([s["is_correct"] for s in items]))
        mean_p = float(np.mean([s["p_correct"] for s in items]))
        bench_list.append({
            "bench": bench,
            "n": len(items),
            "accuracy": acc,
            "mean_p": mean_p,
            "is_low_acc": acc < accuracy_threshold,
        })

    if len(bench_list) < 5:
        return {"error": "Too few benchmarks for bootstrap"}

    f1s = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(bench_list), size=len(bench_list), replace=True)
        boot_benches = [bench_list[i] for i in idx]

        n_low = sum(1 for b in boot_benches if b["is_low_acc"])
        n_high = len(boot_benches) - n_low
        if n_low == 0 or n_high == 0:
            f1s.append(0.0)
            continue

        best_f1 = 0
        for t in np.linspace(0, 1, 50):
            tp = sum(1 for b in boot_benches if b["mean_p"] < t and b["is_low_acc"])
            fp = sum(1 for b in boot_benches if b["mean_p"] < t and not b["is_low_acc"])
            fn = sum(1 for b in boot_benches if b["mean_p"] >= t and b["is_low_acc"])
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            best_f1 = max(best_f1, f1)
        f1s.append(best_f1)

    f1s = np.array(f1s)
    return {
        "mean_f1": float(f1s.mean()),
        "ci_lo": float(np.percentile(f1s, 2.5)),
        "ci_hi": float(np.percentile(f1s, 97.5)),
        "std_f1": float(f1s.std()),
        "n_benchmarks": len(bench_list),
    }


# ---------------------------------------------------------------------------
# KS test for distribution shift
# ---------------------------------------------------------------------------

def ks_test_shift_detection(all_scored, reference_target="gpt5mini",
                            min_bench_n=10):
    """Two-sample KS test on P(correct) distributions per benchmark.

    For each benchmark present in both reference and target, test whether
    the distributions of P(correct) differ significantly.
    """
    if reference_target not in all_scored:
        return {"error": f"Reference target {reference_target} not found"}

    ref_groups = group_by_benchmark(all_scored[reference_target])

    results = {}
    for target, samples in all_scored.items():
        if target == reference_target:
            continue
        target_groups = group_by_benchmark(samples)
        bench_results = {}
        n_sig = 0
        n_tested = 0

        for bench in sorted(set(ref_groups) & set(target_groups)):
            ref_items = ref_groups[bench]
            tgt_items = target_groups[bench]
            if len(ref_items) < min_bench_n or len(tgt_items) < min_bench_n:
                continue

            ref_scores = [s["p_correct"] for s in ref_items]
            tgt_scores = [s["p_correct"] for s in tgt_items]
            stat, pval = ks_2samp(ref_scores, tgt_scores)

            n_tested += 1
            is_sig = pval < 0.05
            if is_sig:
                n_sig += 1

            bench_results[bench] = {
                "ks_statistic": float(stat),
                "p_value": float(pval),
                "significant": is_sig,
                "n_ref": len(ref_items),
                "n_target": len(tgt_items),
            }

        results[target] = {
            "benchmarks": bench_results,
            "n_tested": n_tested,
            "n_significant": n_sig,
            "frac_significant": n_sig / n_tested if n_tested > 0 else 0,
        }

    return {"reference": reference_target, "targets": results}


# ---------------------------------------------------------------------------
# Calibration drift metric
# ---------------------------------------------------------------------------

def calibration_drift(all_scored, reference_target="gpt5mini", min_bench_n=10):
    """Measure whether mean P(correct) shifts proportionally with accuracy.

    For each benchmark, compare (accuracy_ref - accuracy_tgt) with
    (mean_p_ref - mean_p_tgt). If the calibrator tracks accuracy drops,
    these should correlate.
    """
    if reference_target not in all_scored:
        return {}

    ref_groups = group_by_benchmark(all_scored[reference_target])

    results = {}
    for target, samples in all_scored.items():
        if target == reference_target:
            continue
        target_groups = group_by_benchmark(samples)
        acc_deltas = []
        p_deltas = []
        bench_details = {}

        for bench in sorted(set(ref_groups) & set(target_groups)):
            ref_items = ref_groups[bench]
            tgt_items = target_groups[bench]
            if len(ref_items) < min_bench_n or len(tgt_items) < min_bench_n:
                continue

            ref_acc = np.mean([s["is_correct"] for s in ref_items])
            tgt_acc = np.mean([s["is_correct"] for s in tgt_items])
            ref_p = np.mean([s["p_correct"] for s in ref_items])
            tgt_p = np.mean([s["p_correct"] for s in tgt_items])

            acc_deltas.append(ref_acc - tgt_acc)
            p_deltas.append(ref_p - tgt_p)
            bench_details[bench] = {
                "acc_delta": float(ref_acc - tgt_acc),
                "p_delta": float(ref_p - tgt_p),
            }

        if len(acc_deltas) >= 5:
            corr, pval = spearmanr(acc_deltas, p_deltas)
            results[target] = {
                "spearman_r": float(corr),
                "p_value": float(pval),
                "n_benchmarks": len(acc_deltas),
                "benchmarks": bench_details,
            }
        else:
            results[target] = {
                "error": f"Only {len(acc_deltas)} overlapping benchmarks",
                "benchmarks": bench_details,
            }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ood_detection(all_results, output_path):
    """Four-panel figure for OOD detection use case."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    colors = {"gpt5mini": "C0", "gpt52": "C1", "qwen35": "C2"}
    names = {"gpt5mini": "GPT-5-mini", "gpt52": "GPT-5.2", "qwen35": "Qwen3.5"}

    # -----------------------------------------------------------------------
    # Panel A (top-left): Benchmark-level scatter — mean P(correct) vs accuracy
    # -----------------------------------------------------------------------
    ax = axes[0, 0]
    for target in ["gpt5mini", "gpt52", "qwen35"]:
        bl = all_results.get(target, {}).get("benchmark_level", {})
        bench_data = bl.get("benchmarks", {})
        if not bench_data:
            continue
        xs = [bench_data[b]["actual_accuracy"] for b in bench_data]
        ys = [bench_data[b]["mean_score"] for b in bench_data]
        ns = [bench_data[b]["n"] for b in bench_data]
        r = bl.get("spearman_r")
        label = f"{names.get(target, target)}"
        if r is not None:
            label += f" (r={r:.3f})"
        ax.scatter(xs, ys, s=[max(15, min(150, n * 0.5)) for n in ns],
                   alpha=0.6, color=colors.get(target, "gray"), label=label)
        # Annotate benchmark names for first target only
        if target == "gpt5mini":
            for b in bench_data:
                ax.annotate(b, (bench_data[b]["actual_accuracy"],
                                bench_data[b]["mean_score"]),
                            fontsize=6, alpha=0.7)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    ax.set_xlabel("Actual Accuracy", fontsize=11)
    ax.set_ylabel("Mean P(correct)", fontsize=11)
    ax.set_title("A. Benchmark-Level Calibration", fontsize=13)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # -----------------------------------------------------------------------
    # Panel B (top-right): Bootstrap batch simulation
    # -----------------------------------------------------------------------
    ax = axes[0, 1]
    for target in ["gpt5mini", "gpt52", "qwen35"]:
        bootstrap = all_results.get(target, {}).get("bootstrap", {})
        batch_results = bootstrap.get("batch_results", {})
        if not batch_results:
            continue
        batch_sizes = sorted(int(k) for k in batch_results.keys())
        mean_corrs = [batch_results[str(bs)]["mean_correlation"] for bs in batch_sizes]
        frac_sig = [batch_results[str(bs)]["fraction_significant"] for bs in batch_sizes]

        ax.plot(batch_sizes, frac_sig, marker="o", color=colors.get(target, "gray"),
                label=f"{names.get(target, target)}", linewidth=2)

    ax.axhline(y=0.95, color="gray", linestyle=":", alpha=0.5, label="95% target")
    ax.set_xlabel("Batch Size", fontsize=11)
    ax.set_ylabel("Fraction of Simulations with p < 0.05", fontsize=11)
    ax.set_title("B. Bootstrap: Reliability vs Batch Size", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.05)

    # -----------------------------------------------------------------------
    # Panel C (bottom-left): Cross-model shift detection
    # -----------------------------------------------------------------------
    ax = axes[1, 0]
    shift_data = all_results.get("cross_model_shift", {})
    targets_shift = shift_data.get("targets", {})
    bar_idx = 0
    bar_labels = []
    bar_vals = []
    bar_colors_list = []
    bar_hatches = []

    for target, tdata in targets_shift.items():
        bench_shifts = tdata.get("benchmark_shifts", {})
        for bench in sorted(bench_shifts.keys()):
            bs = bench_shifts[bench]
            bar_labels.append(f"{bench}\n({names.get(target, target)})")
            bar_vals.append(bs["z_batch"])
            bar_colors_list.append("C3" if bs["flagged"] else colors.get(target, "gray"))
            bar_hatches.append("//" if bs["flagged"] else "")

    if bar_vals:
        # Sort by z_batch for readability
        order = np.argsort(bar_vals)
        bar_labels = [bar_labels[i] for i in order]
        bar_vals = [bar_vals[i] for i in order]
        bar_colors_list = [bar_colors_list[i] for i in order]
        bar_hatches = [bar_hatches[i] for i in order]

        # Show at most 25 bars
        if len(bar_vals) > 25:
            bar_labels = bar_labels[:25]
            bar_vals = bar_vals[:25]
            bar_colors_list = bar_colors_list[:25]
            bar_hatches = bar_hatches[:25]

        y_pos = range(len(bar_vals))
        bars = ax.barh(y_pos, bar_vals, color=bar_colors_list)
        for bar, hatch in zip(bars, bar_hatches):
            bar.set_hatch(hatch)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(bar_labels, fontsize=6)
        z_thresh = shift_data.get("z_threshold", -1.5)
        ax.axvline(x=z_thresh, color="red", linestyle="--", alpha=0.7,
                   label=f"Flag threshold (z={z_thresh})")
        ax.axvline(x=0, color="k", linewidth=0.5)
        ax.set_xlabel("z_batch (vs GPT-5-mini reference)", fontsize=11)
        ax.set_title("C. Cross-Model Distribution Shift", fontsize=13)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="x")
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No cross-model data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12)
        ax.set_title("C. Cross-Model Distribution Shift", fontsize=13)

    # -----------------------------------------------------------------------
    # Panel D (bottom-right): Alert system precision-recall
    # -----------------------------------------------------------------------
    ax = axes[1, 1]
    for target in ["gpt5mini", "gpt52", "qwen35"]:
        alert = all_results.get(target, {}).get("alert_system", {})
        pr_curve = alert.get("pr_curve", [])
        if not pr_curve:
            continue

        recalls = [p["recall"] for p in pr_curve]
        precisions = [p["precision"] for p in pr_curve]
        best_f1 = alert.get("best_f1", 0)
        ax.plot(recalls, precisions, color=colors.get(target, "gray"),
                label=f"{names.get(target, target)} (F1={best_f1:.3f})", linewidth=2)

    # Add baseline markers
    for target in ["gpt5mini"]:
        baselines = all_results.get(target, {}).get("baselines", {})
        verb_f1 = baselines.get("verbalized", {}).get("best_f1", 0)
        rand_f1 = baselines.get("random", {}).get("best_f1", 0)
        hard_f1 = baselines.get("hardcoded", {}).get("best_f1", 0)
        if verb_f1:
            ax.axhline(y=verb_f1, color="C3", linestyle="--", alpha=0.5,
                       label=f"Verbalized (F1={verb_f1:.3f})")
        if rand_f1:
            ax.axhline(y=rand_f1, color="gray", linestyle=":", alpha=0.5,
                       label=f"Random (F1={rand_f1:.3f})")
        if hard_f1:
            ax.axhline(y=hard_f1, color="C4", linestyle="-.", alpha=0.5,
                       label=f"Hardcoded (F1={hard_f1:.3f})")

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("D. Alert System: Detecting Accuracy < 50%", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UC8: OOD/Distribution Shift Detection")
    parser.add_argument("--scored_dir", default="data/use_cases/scored_test_only_v2")
    parser.add_argument("--output_dir", default="data/use_cases/results_test_only_v2")
    parser.add_argument("--fig_dir", default="figures/use_cases_v2")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run on small subset for quick validation")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    targets = ["gpt5mini", "gpt52", "qwen35"]
    target_names = {"gpt5mini": "GPT-5-mini (in-dist)",
                    "gpt52": "GPT-5.2 (cross-model)",
                    "qwen35": "Qwen3.5 (cross-model)"}

    # Smoke test parameters
    bootstrap_sims = 50 if args.smoke_test else 1000
    min_bench_for_bootstrap = 10 if args.smoke_test else 150

    all_scored = {}
    for target in targets:
        scored_path = Path(args.scored_dir) / f"{target}_scored.jsonl"
        if not scored_path.exists():
            print(f"Skipping {target}: {scored_path} not found")
            continue
        samples = load_scored(scored_path)
        if args.smoke_test:
            samples = samples[:200]
        all_scored[target] = samples
        print(f"  Loaded {target}: {len(samples)} samples")

    if not all_scored:
        print("ERROR: No scored data found. Check --scored_dir.")
        return

    all_results = {}

    for target in targets:
        if target not in all_scored:
            continue
        samples = all_scored[target]

        print(f"\n{'='*70}")
        print(f"UC8: OOD/Distribution Shift Detection -- {target_names[target]}")
        print(f"{'='*70}")
        print(f"Samples: {len(samples)}, "
              f"Accuracy: {np.mean([s['is_correct'] for s in samples]):.3f}")

        target_results = {
            "n_samples": len(samples),
            "base_accuracy": float(np.mean([s["is_correct"] for s in samples])),
        }

        # -------------------------------------------------------------------
        # Analysis 1: Benchmark-level correlation
        # -------------------------------------------------------------------
        print("\n  --- Benchmark-Level Correlation ---")
        bl = benchmark_level_correlation(samples, "p_correct")
        target_results["benchmark_level"] = bl
        print(f"  N benchmarks: {bl.get('n_benchmarks', 0)}")
        if bl.get("spearman_r") is not None:
            print(f"  Spearman r (mean P vs accuracy): {bl['spearman_r']:.3f} "
                  f"(p={bl['p_value']:.4f})")

        # Verbalized baseline
        bl_verb = benchmark_level_correlation(samples, "verbalized_confidence")
        if bl_verb.get("spearman_r") is not None:
            print(f"  Verbalized Spearman r:            {bl_verb['spearman_r']:.3f} "
                  f"(p={bl_verb['p_value']:.4f})")
            target_results["benchmark_level_verbalized"] = bl_verb

        # Per-benchmark table
        bench_data = bl.get("benchmarks", {})
        if bench_data:
            print(f"\n  {'Benchmark':<20} {'N':>5} {'Accuracy':>9} {'MeanP':>7} {'Gap':>7}")
            print(f"  {'-'*52}")
            for b in sorted(bench_data, key=lambda x: bench_data[x]["actual_accuracy"]):
                bd = bench_data[b]
                gap = bd["mean_score"] - bd["actual_accuracy"]
                print(f"  {b:<20} {bd['n']:>5} {bd['actual_accuracy']:>9.3f} "
                      f"{bd['mean_score']:>7.3f} {gap:>+7.3f}")

        # -------------------------------------------------------------------
        # Analysis 2: Bootstrap batch simulation
        # -------------------------------------------------------------------
        print("\n  --- Bootstrap Batch Simulation ---")
        bootstrap = bootstrap_batch_correlation(
            samples, n_simulations=bootstrap_sims,
            min_bench_samples=min_bench_for_bootstrap)
        target_results["bootstrap"] = bootstrap

        if "error" in bootstrap:
            print(f"  {bootstrap['error']}")
        else:
            print(f"  Eligible benchmarks: {len(bootstrap['eligible_benchmarks'])}")
            batch_results = bootstrap["batch_results"]
            print(f"\n  {'Batch':>7} {'MeanCorr':>9} {'FracSig':>8} {'StdCorr':>8}")
            print(f"  {'-'*36}")
            for bs_key in sorted(batch_results.keys(), key=int):
                br = batch_results[bs_key]
                print(f"  {br['batch_size']:>7} {br['mean_correlation']:>9.3f} "
                      f"{br['fraction_significant']:>8.1%} {br['std_correlation']:>8.3f}")

            if bootstrap["min_reliable_batch_size"] is not None:
                print(f"\n  Min reliable batch size (95% sig): "
                      f"{bootstrap['min_reliable_batch_size']}")
            else:
                print(f"\n  Min reliable batch size: > {max(int(k) for k in batch_results)}"
                      f" (95% not reached)")

        # -------------------------------------------------------------------
        # Analysis 4: Alert system
        # -------------------------------------------------------------------
        print("\n  --- Alert System (detect accuracy < 50%) ---")
        alert = alert_system(samples)
        # Strip pr_curve from stored results to keep JSON small; keep for plotting
        target_results["alert_system"] = alert

        if "error" in alert:
            print(f"  {alert['error']}")
        else:
            print(f"  N benchmarks: {alert['n_benchmarks']} "
                  f"({alert['n_low_accuracy']} low, {alert['n_high_accuracy']} high)")
            print(f"  Best F1: {alert['best_f1']:.3f} (threshold={alert['best_threshold']:.3f})")
            if alert.get("best_point"):
                bp = alert["best_point"]
                print(f"  At best threshold: precision={bp['precision']:.3f}, "
                      f"recall={bp['recall']:.3f}")

        # Baselines
        baselines = {}

        verb_alert = alert_system_verbalized(samples)
        baselines["verbalized"] = verb_alert
        if verb_alert.get("best_f1") is not None:
            print(f"  Verbalized baseline F1: {verb_alert['best_f1']:.3f}")

        rand_alert = alert_system_random(samples)
        baselines["random"] = rand_alert
        print(f"  Random baseline F1: {rand_alert['best_f1']:.3f}")

        hard_alert = alert_system_hardcoded(samples)
        baselines["hardcoded"] = hard_alert
        print(f"  Hardcoded difficulty F1: {hard_alert['best_f1']:.3f}")

        target_results["baselines"] = baselines

        # -------------------------------------------------------------------
        # Bootstrap CIs on alert F1
        # -------------------------------------------------------------------
        n_boot = 50 if args.smoke_test else 1000
        boot_ci = bootstrap_alert_f1(samples, n_bootstrap=n_boot)
        target_results["alert_bootstrap_ci"] = boot_ci
        if "error" not in boot_ci:
            print(f"  Alert F1 bootstrap 95% CI: [{boot_ci['ci_lo']:.3f}, {boot_ci['ci_hi']:.3f}]")
            print(f"    (based on resampling {boot_ci['n_benchmarks']} benchmarks)")

        all_results[target] = target_results

    # -----------------------------------------------------------------------
    # Analysis 3: Cross-model shift detection (needs reference + others)
    # -----------------------------------------------------------------------
    if "gpt5mini" in all_scored and len(all_scored) >= 2:
        print(f"\n{'='*70}")
        print("Cross-Model Distribution Shift Detection")
        print(f"{'='*70}")

        shift = cross_model_shift_detection(all_scored, reference_target="gpt5mini")
        all_results["cross_model_shift"] = shift

        for target, tdata in shift.get("targets", {}).items():
            summary = tdata["summary"]
            print(f"\n  {target_names.get(target, target)}:")
            print(f"    Benchmarks analyzed: {summary['n_benchmarks']}")
            print(f"    Flagged (z < {shift['z_threshold']}): {summary['n_flagged']}")
            if summary["flagged_benchmarks"]:
                print(f"    Flagged: {', '.join(summary['flagged_benchmarks'])}")
                print(f"    Flagged mean accuracy: {summary['flagged_mean_accuracy']:.3f}")
                print(f"    Unflagged mean accuracy: {summary['unflagged_mean_accuracy']:.3f}")

            # Per-benchmark z_batch table
            bench_shifts = tdata["benchmark_shifts"]
            print(f"\n    {'Benchmark':<20} {'z_batch':>8} {'TargAcc':>8} {'RefAcc':>8} {'Flag':>5}")
            print(f"    {'-'*54}")
            for b in sorted(bench_shifts, key=lambda x: bench_shifts[x]["z_batch"]):
                bs = bench_shifts[b]
                flag_str = "***" if bs["flagged"] else ""
                print(f"    {b:<20} {bs['z_batch']:>8.2f} {bs['target_accuracy']:>8.3f} "
                      f"{bs['ref_accuracy']:>8.3f} {flag_str:>5}")

    # -----------------------------------------------------------------------
    # KS test for distribution shift
    # -----------------------------------------------------------------------
    if "gpt5mini" in all_scored and len(all_scored) >= 2:
        print(f"\n{'='*70}")
        print("KS Test: Distribution Shift Detection")
        print(f"{'='*70}")

        ks_results = ks_test_shift_detection(all_scored, reference_target="gpt5mini")
        all_results["ks_test"] = ks_results

        for target, tdata in ks_results.get("targets", {}).items():
            print(f"\n  {target_names.get(target, target)}:")
            print(f"    Benchmarks tested: {tdata['n_tested']}")
            print(f"    Significant shifts (p<0.05): {tdata['n_significant']} "
                  f"({tdata['frac_significant']:.0%})")
            for bench, bd in sorted(tdata["benchmarks"].items(),
                                     key=lambda x: x[1]["p_value"]):
                sig = " ***" if bd["significant"] else ""
                print(f"      {bench:<20} KS={bd['ks_statistic']:.3f} "
                      f"p={bd['p_value']:.4f} "
                      f"(n={bd['n_ref']}/{bd['n_target']}){sig}")

    # -----------------------------------------------------------------------
    # Calibration drift: does mean P(correct) track accuracy changes?
    # -----------------------------------------------------------------------
    if "gpt5mini" in all_scored and len(all_scored) >= 2:
        print(f"\n{'='*70}")
        print("Calibration Drift: P(correct) Tracking Accuracy Changes")
        print(f"{'='*70}")

        drift = calibration_drift(all_scored, reference_target="gpt5mini")
        all_results["calibration_drift"] = drift

        for target, tdata in drift.items():
            print(f"\n  {target_names.get(target, target)}:")
            if "error" in tdata:
                print(f"    {tdata['error']}")
            else:
                print(f"    Spearman r (acc_delta vs p_delta): {tdata['spearman_r']:.3f} "
                      f"(p={tdata['p_value']:.4f})")
                print(f"    Benchmarks: {tdata['n_benchmarks']}")
                if tdata["spearman_r"] > 0.5 and tdata["p_value"] < 0.05:
                    print(f"    --> Calibrator tracks accuracy changes well")
                else:
                    print(f"    --> Weak tracking of accuracy changes")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    if all_results:
        fig_path = f"{args.fig_dir}/uc8_ood_detection.pdf"
        plot_ood_detection(all_results, fig_path)

    # -----------------------------------------------------------------------
    # Save results (strip large arrays for JSON)
    # -----------------------------------------------------------------------
    save_results = {}
    for key, val in all_results.items():
        if isinstance(val, dict):
            # Deep copy to avoid mutating originals
            save_results[key] = json.loads(json.dumps(val, default=float))
        else:
            save_results[key] = val

    # Remove pr_curve arrays to keep JSON manageable
    for target in targets:
        if target in save_results:
            alert = save_results[target].get("alert_system", {})
            alert.pop("pr_curve", None)

    out_path = f"{args.output_dir}/uc8_results.json"
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")

    # Summary
    print(f"\n{'='*70}")
    print("UC8 Summary")
    print(f"{'='*70}")
    for target in targets:
        if target not in all_results:
            continue
        tr = all_results[target]
        bl_r = tr.get("benchmark_level", {}).get("spearman_r")
        n_bench = tr.get("benchmark_level", {}).get("n_benchmarks", 0)
        bs_min = tr.get("bootstrap", {}).get("min_reliable_batch_size")
        alert_f1 = tr.get("alert_system", {}).get("best_f1")
        alert_n = tr.get("alert_system", {}).get("n_benchmarks", 0)
        alert_n_low = tr.get("alert_system", {}).get("n_low_accuracy", 0)
        verb_f1 = tr.get("baselines", {}).get("verbalized", {}).get("best_f1")
        boot_ci = tr.get("alert_bootstrap_ci", {})
        print(f"  {target_names.get(target, target)} ({tr.get('n_samples', '?')} samples):")
        if bl_r is not None:
            print(f"    Benchmark-level Spearman r: {bl_r:.3f} (N={n_bench} benchmarks)")
        if bs_min is not None:
            print(f"    Min reliable batch size: {bs_min}")
        elif "bootstrap" in tr and "error" not in tr["bootstrap"]:
            print(f"    Min reliable batch size: not reached")
        if alert_f1 is not None:
            verb_str = f"{verb_f1:.3f}" if verb_f1 else "N/A"
            ci_str = ""
            if "ci_lo" in boot_ci:
                ci_str = f" [{boot_ci['ci_lo']:.3f}, {boot_ci['ci_hi']:.3f}]"
            print(f"    Alert F1: {alert_f1:.3f}{ci_str} (verbalized: {verb_str})")
            print(f"      (N={alert_n} benchmarks, {alert_n_low} low-accuracy)")
            print(f"      CAVEAT: F1 computed over {alert_n} benchmark-level units, not samples")


if __name__ == "__main__":
    main()
