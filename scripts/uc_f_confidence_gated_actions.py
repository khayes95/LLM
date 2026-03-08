#!/usr/bin/env python3
"""UC-F: Confidence-Gated Actions — "Think Before Acting".

In agentic settings, irreversible actions (code execution, API calls, sending
emails) should only proceed when the model is confident. When uncertain, the
system should verify first (web search, documentation lookup, human check).

This script simulates a gated action system and quantifies the safety benefit:

Analyses:
1. Action safety — what % of auto-executed actions would have been wrong?
2. Risk-adjusted cost — expected cost under various wrong_action/verification ratios.
3. Coverage-safety Pareto frontier — auto-execute rate vs error rate on executed.
4. Optimal threshold by risk profile — conservative, moderate, aggressive.
5. Method comparison — calibrator vs baselines as gating signal.
6. Per-benchmark safety — which domains are safest to auto-execute?

Outputs:
    {output_dir}/uc_f_results.json — all metrics
    {fig_dir}/uc_f_pareto.pdf — coverage vs safety Pareto frontier
    {fig_dir}/uc_f_risk_analysis.pdf — expected cost under risk profiles

Usage:
    python scripts/uc_f_confidence_gated_actions.py
    python scripts/uc_f_confidence_gated_actions.py --scored_dir data/use_cases/scored_test_only_v2
    python scripts/uc_f_confidence_gated_actions.py --smoke_test
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scored(path):
    """Load scored JSONL, converting fields to native types."""
    samples = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            row["is_correct"] = int(row["is_correct"])
            row["p_correct"] = float(row["p_correct"])
            row["output_tokens"] = int(row.get("output_tokens") or 0)
            vc = row.get("verbalized_confidence")
            if vc is None or str(vc).strip().lower() == "none":
                row["verbalized_confidence"] = None
            else:
                row["verbalized_confidence"] = float(vc)
            samples.append(row)
    return samples


# ---------------------------------------------------------------------------
# Analysis 1: Gated action simulation
# ---------------------------------------------------------------------------

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def gated_action_analysis(samples, score_key="p_correct", thresholds=THRESHOLDS):
    """Simulate: execute action if score >= threshold, verify otherwise.

    'Execute' = proceed without verification (fast, but risky if wrong).
    'Verify' = pause for verification (slow/costly, but prevents errors).
    """
    valid = [s for s in samples if s.get(score_key) is not None]
    if not valid:
        return []

    n_total = len(valid)
    n_wrong_total = sum(1 for s in valid if not s["is_correct"])
    base_error_rate = n_wrong_total / n_total if n_total > 0 else 0.0

    results = []
    for t in thresholds:
        executed = [s for s in valid if s[score_key] >= t]
        verified = [s for s in valid if s[score_key] < t]

        n_exec = len(executed)
        n_verify = len(verified)

        # Errors in auto-executed set (these are the dangerous ones)
        exec_errors = sum(1 for s in executed if not s["is_correct"])
        exec_error_rate = exec_errors / n_exec if n_exec > 0 else 0.0

        # Errors caught by verification (prevented)
        verify_errors = sum(1 for s in verified if not s["is_correct"])

        # Coverage = fraction auto-executed
        coverage = n_exec / n_total if n_total > 0 else 0.0

        # Safety improvement: how much safer is the executed set vs no gating?
        safety_improvement = base_error_rate - exec_error_rate

        results.append({
            "threshold": t,
            "n_executed": n_exec,
            "n_verified": n_verify,
            "coverage": float(coverage),
            "exec_errors": exec_errors,
            "exec_error_rate": float(exec_error_rate),
            "errors_prevented": verify_errors,
            "errors_prevented_pct": verify_errors / n_wrong_total if n_wrong_total > 0 else 0.0,
            "safety_improvement": float(safety_improvement),
            "exec_accuracy": float(1 - exec_error_rate),
        })
    return results


# ---------------------------------------------------------------------------
# Analysis 2: Risk-adjusted expected cost
# ---------------------------------------------------------------------------

def risk_adjusted_cost(gated_results, n_total, cost_ratios=None):
    """Compute expected cost per query under different cost models.

    Cost model:
    - Correct auto-execute: cost = 0 (ideal)
    - Wrong auto-execute: cost = wrong_cost
    - Verification (right or wrong): cost = verify_cost = 1 (normalized)

    Expected cost = (exec_errors * wrong_cost + n_verified * verify_cost) / n_total
    """
    if cost_ratios is None:
        cost_ratios = [5, 10, 20, 50, 100]  # wrong_cost / verify_cost

    results = {}
    for ratio in cost_ratios:
        threshold_costs = []
        for gr in gated_results:
            total_cost = gr["exec_errors"] * ratio + gr["n_verified"] * 1
            cost_per_query = total_cost / n_total if n_total > 0 else 0.0

            # Baseline: no gating (all executed)
            base_errors = gr["exec_errors"] + gr["errors_prevented"]
            base_cost = base_errors * ratio / n_total if n_total > 0 else 0.0

            # Savings vs no gating
            savings = base_cost - cost_per_query

            threshold_costs.append({
                "threshold": gr["threshold"],
                "cost_per_query": float(cost_per_query),
                "baseline_cost": float(base_cost),
                "savings": float(savings),
                "savings_pct": float(savings / base_cost) if base_cost > 0 else 0.0,
                "coverage": gr["coverage"],
            })

        # Find optimal threshold (minimize cost)
        best = min(threshold_costs, key=lambda x: x["cost_per_query"])

        results[f"ratio_{ratio}"] = {
            "wrong_cost_ratio": ratio,
            "optimal_threshold": best["threshold"],
            "optimal_cost": best["cost_per_query"],
            "baseline_cost": best["baseline_cost"],
            "optimal_savings_pct": best["savings_pct"],
            "optimal_coverage": best["coverage"],
            "all_thresholds": threshold_costs,
        }
    return results


# ---------------------------------------------------------------------------
# Analysis 3: Risk profiles
# ---------------------------------------------------------------------------

RISK_PROFILES = {
    "aggressive": {
        "description": "Maximize throughput, accept some risk",
        "max_error_rate": 0.10,
    },
    "moderate": {
        "description": "Balance speed and safety",
        "max_error_rate": 0.05,
    },
    "conservative": {
        "description": "Safety first, minimize errors",
        "max_error_rate": 0.02,
    },
    "paranoid": {
        "description": "Near-zero tolerance for errors",
        "max_error_rate": 0.01,
    },
}


def risk_profile_analysis(gated_results):
    """For each risk profile, find the highest-coverage threshold that meets
    the maximum error rate constraint."""
    results = {}
    for profile_name, profile in RISK_PROFILES.items():
        max_err = profile["max_error_rate"]

        # Find all thresholds that meet the error rate constraint
        candidates = [
            gr for gr in gated_results
            if gr["exec_error_rate"] <= max_err and gr["n_executed"] > 0
        ]

        if candidates:
            # Pick the one with highest coverage (lowest threshold)
            best = max(candidates, key=lambda x: x["coverage"])
            results[profile_name] = {
                "description": profile["description"],
                "max_error_rate": max_err,
                "achievable": True,
                "best_threshold": best["threshold"],
                "coverage": best["coverage"],
                "actual_error_rate": best["exec_error_rate"],
                "n_auto_executed": best["n_executed"],
                "errors_in_executed": best["exec_errors"],
            }
        else:
            results[profile_name] = {
                "description": profile["description"],
                "max_error_rate": max_err,
                "achievable": False,
                "best_threshold": None,
                "coverage": 0.0,
            }
    return results


# ---------------------------------------------------------------------------
# Analysis 4: Per-benchmark safety
# ---------------------------------------------------------------------------

def per_benchmark_safety(samples, threshold=0.7):
    """Which benchmarks are safest to auto-execute at a given threshold?"""
    by_bench = defaultdict(list)
    for s in samples:
        by_bench[s["benchmark"]].append(s)

    results = {}
    for bench, bs in sorted(by_bench.items()):
        n_total = len(bs)
        base_acc = np.mean([s["is_correct"] for s in bs])

        executed = [s for s in bs if s["p_correct"] >= threshold]
        n_exec = len(executed)
        exec_acc = np.mean([s["is_correct"] for s in executed]) if executed else None
        coverage = n_exec / n_total if n_total > 0 else 0.0

        results[bench] = {
            "n_total": n_total,
            "base_accuracy": float(base_acc),
            "n_auto_executed": n_exec,
            "coverage": float(coverage),
            "exec_accuracy": float(exec_acc) if exec_acc is not None else None,
            "safety_gain": float(exec_acc - base_acc) if exec_acc is not None else None,
        }
    return results


# ---------------------------------------------------------------------------
# Analysis 5: Method comparison
# ---------------------------------------------------------------------------

def method_pareto(samples, score_key="p_correct", thresholds=THRESHOLDS):
    """Compute coverage-safety pairs for a given score method."""
    return gated_action_analysis(samples, score_key=score_key, thresholds=thresholds)


def compute_area_under_safety_curve(gated_results):
    """Area under the (coverage, exec_accuracy) curve. Higher = better."""
    points = [(r["coverage"], r["exec_accuracy"]) for r in gated_results
              if r["n_executed"] > 0]
    if len(points) < 2:
        return 0.0
    points.sort(key=lambda p: p[0])
    coverages = np.array([p[0] for p in points])
    accuracies = np.array([p[1] for p in points])
    _trapz = getattr(np, "trapezoid", np.trapz)
    return float(_trapz(accuracies, coverages))


def compare_methods(samples):
    """Compare all methods on the coverage-safety tradeoff."""
    methods = {
        "calibrator": "p_correct",
        "verbalized": "verbalized_confidence",
        "platt_verbalized": "p_platt_verbalized",
        "isotonic_verbalized": "p_isotonic_verbalized",
        "length_baseline": "p_length_baseline",
        "combined_baseline": "p_combined_baseline",
    }
    results = {}
    for name, key in methods.items():
        ga = gated_action_analysis(samples, score_key=key)
        if not ga:
            continue
        ausc = compute_area_under_safety_curve(ga)
        # Find coverage at 95% exec accuracy
        cov_95 = 0.0
        for r in ga:
            if r["exec_accuracy"] >= 0.95 and r["coverage"] > cov_95:
                cov_95 = r["coverage"]
        results[name] = {
            "area_under_safety_curve": ausc,
            "coverage_at_95pct_accuracy": cov_95,
            "pareto_points": [
                {"coverage": r["coverage"], "exec_accuracy": r["exec_accuracy"],
                 "threshold": r["threshold"]}
                for r in ga if r["n_executed"] > 0
            ],
        }
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pareto(all_model_results, fig_path):
    """Coverage vs safety Pareto frontier for each model, all methods."""
    n_models = len(all_model_results)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5), squeeze=False)

    model_names = {
        "gpt5mini": "GPT-5-mini",
        "gpt52": "GPT-5.2",
        "qwen35": "Qwen3.5",
    }

    for col, (target, data) in enumerate(all_model_results.items()):
        ax = axes[0, col]
        mc = data["method_comparison"]

        styles = {
            "calibrator": ("o-", "C0", 2.5),
            "verbalized": ("s--", "C1", 2.0),
            "platt_verbalized": ("^:", "C2", 1.5),
            "combined_baseline": ("D-.", "C4", 1.5),
        }

        for method, md in mc.items():
            style = styles.get(method, ("+-", "gray", 1.0))
            points = md["pareto_points"]
            if not points:
                continue
            covs = [p["coverage"] for p in points]
            accs = [p["exec_accuracy"] for p in points]
            ax.plot(covs, accs, style[0], color=style[1], linewidth=style[2],
                    markersize=6,
                    label=f"{method} (AUSC={md['area_under_safety_curve']:.3f})")

        # Reference lines
        base_acc = data["base_accuracy"]
        ax.axhline(y=base_acc, color="black", linestyle=":", alpha=0.3,
                    label=f"No gating ({base_acc:.3f})")
        ax.axhline(y=0.95, color="red", linestyle=":", alpha=0.3, label="95% target")

        ax.set_xlabel("Coverage (% auto-executed)", fontsize=12)
        ax.set_ylabel("Accuracy of auto-executed", fontsize=12)
        ax.set_title(f"{model_names.get(target, target)} (N={data['n_samples']})",
                     fontsize=13)
        ax.legend(fontsize=8, loc="lower left")
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(max(0, base_acc - 0.1), 1.02)
        ax.grid(True, alpha=0.3)

    fig.suptitle("UC-F: Confidence-Gated Actions — Coverage vs Safety",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {fig_path}")


def plot_risk_analysis(all_model_results, fig_path):
    """Risk-adjusted cost vs threshold for different cost ratios."""
    n_models = len(all_model_results)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5), squeeze=False)

    model_names = {
        "gpt5mini": "GPT-5-mini",
        "gpt52": "GPT-5.2",
        "qwen35": "Qwen3.5",
    }
    colors = ["C0", "C1", "C2", "C3", "C4"]

    for col, (target, data) in enumerate(all_model_results.items()):
        ax = axes[0, col]
        rac = data["risk_adjusted_cost"]

        for i, (ratio_key, rd) in enumerate(rac.items()):
            ratio = rd["wrong_cost_ratio"]
            thresholds = [t["threshold"] for t in rd["all_thresholds"]]
            savings = [t["savings_pct"] * 100 for t in rd["all_thresholds"]]
            color = colors[i % len(colors)]
            ax.plot(thresholds, savings, "o-", color=color,
                    label=f"Wrong cost = {ratio}x (opt t={rd['optimal_threshold']:.2f})")
            ax.plot(rd["optimal_threshold"],
                    rd["optimal_savings_pct"] * 100,
                    "*", color=color, markersize=12)

        ax.axhline(y=0, color="black", linestyle=":", alpha=0.3)
        ax.set_xlabel("Gating threshold", fontsize=12)
        ax.set_ylabel("Cost savings vs no gating (%)", fontsize=12)
        ax.set_title(f"{model_names.get(target, target)}", fontsize=13)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("UC-F: Cost Savings from Confidence Gating", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {fig_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UC-F: Confidence-Gated Actions — Think Before Acting")
    parser.add_argument("--scored_dir", default="data/use_cases/scored_test_only_v2",
                        help="Directory with scored JSONL files")
    parser.add_argument("--output_dir", default="data/use_cases/results_test_only_v2",
                        help="Directory for results JSON")
    parser.add_argument("--fig_dir", default="figures/use_cases_v2",
                        help="Directory for output figures")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Only use first 50 samples per model")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    targets = ["gpt5mini", "gpt52", "qwen35"]
    target_names = {
        "gpt5mini": "GPT-5-mini (in-dist)",
        "gpt52": "GPT-5.2 (cross-model)",
        "qwen35": "Qwen3.5 (cross-model)",
    }

    all_results = {}

    for target in targets:
        scored_path = Path(args.scored_dir) / f"{target}_scored.jsonl"
        if not scored_path.exists():
            print(f"Skipping {target}: {scored_path} not found")
            continue

        samples = load_scored(scored_path)
        if args.smoke_test:
            samples = samples[:50]

        n_total = len(samples)
        base_acc = float(np.mean([s["is_correct"] for s in samples]))
        n_wrong = sum(1 for s in samples if not s["is_correct"])

        print(f"\n{'='*70}")
        print(f"UC-F: Confidence-Gated Actions — {target_names.get(target, target)}")
        print(f"{'='*70}")
        print(f"Samples: {n_total}, Base accuracy: {base_acc:.3f}, "
              f"Wrong: {n_wrong} ({n_wrong/n_total:.1%})")

        # --- 1. Gated action simulation ---
        print(f"\n--- Gated Action Simulation (calibrator) ---")
        ga = gated_action_analysis(samples)

        print(f"  {'Thresh':>7} {'AutoExec':>9} {'Cover%':>7} {'ExecErr':>8} "
              f"{'ErrRate':>8} {'Prevented':>9} {'SafetyGain':>11}")
        print(f"  {'-'*67}")
        for r in ga:
            print(f"  {r['threshold']:>7.2f} {r['n_executed']:>9} {r['coverage']:>6.1%} "
                  f"{r['exec_errors']:>8} {r['exec_error_rate']:>7.3f} "
                  f"{r['errors_prevented']:>9} {r['safety_improvement']:>+10.3f}")

        # --- 2. Risk-adjusted cost ---
        print(f"\n--- Risk-Adjusted Cost Analysis ---")
        rac = risk_adjusted_cost(ga, n_total)

        print(f"  {'WrongCost':>10} {'OptThresh':>10} {'CostSaving':>11} {'Coverage':>9}")
        print(f"  {'-'*44}")
        for ratio_key, rd in rac.items():
            print(f"  {rd['wrong_cost_ratio']:>10}x {rd['optimal_threshold']:>10.2f} "
                  f"{rd['optimal_savings_pct']:>10.1%} {rd['optimal_coverage']:>8.1%}")

        # --- 3. Risk profiles ---
        print(f"\n--- Risk Profile Analysis ---")
        rp = risk_profile_analysis(ga)

        print(f"  {'Profile':<14} {'MaxErr':>7} {'Achievable':>11} {'Thresh':>7} "
              f"{'Coverage':>9} {'ActualErr':>10}")
        print(f"  {'-'*62}")
        for name, rd in rp.items():
            if rd["achievable"]:
                print(f"  {name:<14} {rd['max_error_rate']:>6.1%} {'Yes':>11} "
                      f"{rd['best_threshold']:>7.2f} {rd['coverage']:>8.1%} "
                      f"{rd['actual_error_rate']:>9.3f}")
            else:
                print(f"  {name:<14} {rd['max_error_rate']:>6.1%} {'No':>11} "
                      f"{'N/A':>7} {'0.0%':>9} {'N/A':>10}")

        # --- 4. Per-benchmark ---
        print(f"\n--- Per-Benchmark Safety (t=0.7) ---")
        pbs = per_benchmark_safety(samples, threshold=0.7)

        print(f"  {'Benchmark':<18} {'N':>5} {'BaseAcc':>8} {'AutoExec':>9} "
              f"{'Cover%':>7} {'ExecAcc':>8} {'SafetyGain':>11}")
        print(f"  {'-'*70}")
        for bench, bd in sorted(pbs.items(),
                                key=lambda x: x[1].get("safety_gain") or 0,
                                reverse=True):
            exec_s = f"{bd['exec_accuracy']:.3f}" if bd['exec_accuracy'] is not None else "N/A"
            gain_s = f"{bd['safety_gain']:+.3f}" if bd['safety_gain'] is not None else "N/A"
            print(f"  {bench:<18} {bd['n_total']:>5} {bd['base_accuracy']:>8.3f} "
                  f"{bd['n_auto_executed']:>9} {bd['coverage']:>6.1%} "
                  f"{exec_s:>8} {gain_s:>11}")

        # --- 5. Method comparison ---
        print(f"\n--- Method Comparison (Area Under Safety Curve) ---")
        mc = compare_methods(samples)

        print(f"  {'Method':<25} {'AUSC':>7} {'Cov@95%acc':>11}")
        print(f"  {'-'*45}")
        for method, md in sorted(mc.items(),
                                 key=lambda x: x[1]["area_under_safety_curve"],
                                 reverse=True):
            print(f"  {method:<25} {md['area_under_safety_curve']:>7.3f} "
                  f"{md['coverage_at_95pct_accuracy']:>10.1%}")

        # Store
        all_results[target] = {
            "n_samples": n_total,
            "base_accuracy": base_acc,
            "n_wrong": n_wrong,
            "gated_analysis": ga,
            "risk_adjusted_cost": rac,
            "risk_profiles": rp,
            "per_benchmark_safety": pbs,
            "method_comparison": mc,
        }

    if not all_results:
        print("\nERROR: No scored data found.")
        return

    # --- Figures ---
    plot_pareto(all_results, f"{args.fig_dir}/uc_f_pareto.pdf")
    plot_risk_analysis(all_results, f"{args.fig_dir}/uc_f_risk_analysis.pdf")

    # --- Save JSON ---
    out_path = f"{args.output_dir}/uc_f_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # --- Final Summary ---
    print(f"\n{'='*70}")
    print("UC-F Summary: Confidence Gating Value")
    print(f"{'='*70}")
    for target in all_results:
        d = all_results[target]
        mc = d["method_comparison"]
        cal_ausc = mc.get("calibrator", {}).get("area_under_safety_curve", 0)
        cal_cov95 = mc.get("calibrator", {}).get("coverage_at_95pct_accuracy", 0)

        rp = d["risk_profiles"]
        mod_cov = rp.get("moderate", {}).get("coverage", 0)
        mod_thresh = rp.get("moderate", {}).get("best_threshold", "N/A")

        print(f"  {target_names.get(target, target)}:")
        print(f"    AUSC: {cal_ausc:.3f}  Coverage@95%acc: {cal_cov95:.1%}")
        print(f"    Moderate risk: t={mod_thresh}, coverage={mod_cov:.1%}")

        # Cost savings at 10x
        rac10 = d["risk_adjusted_cost"].get("ratio_10", {})
        if rac10:
            print(f"    Cost savings (10x): {rac10['optimal_savings_pct']:.1%} "
                  f"at t={rac10['optimal_threshold']:.2f}")


if __name__ == "__main__":
    main()
