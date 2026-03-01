#!/usr/bin/env python3
"""UC9: Annotation Prioritization / Labeling Efficiency.

Demonstrates that sorting by calibrator uncertainty finds errors faster than
random sampling. Computes error discovery rate curves and per-domain analysis
for calibrator vs baselines (random, verbalized, response length, oracle).

Usage:
    python scripts/uc9_annotation_prioritization.py
    python scripts/uc9_annotation_prioritization.py --scored_dir data/use_cases/scored_unified
    python scripts/uc9_annotation_prioritization.py --smoke_test
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DOMAIN_MAP = {
    "math": ["omnimath", "mathvision", "mathvista", "mathverse"],
    "coding": ["prbench"],
    "factual": ["simpleqa", "realworldqa", "vizwiz", "charxiv"],
    "knowledge": ["gpqa", "chembench", "mmmu"],
    "reasoning": ["bbeh", "arc_agi", "hallusionbench"],
    "multimodal": ["mmstar", "mmvet", "hle_multimodal"],
    "long_context": ["livebench", "hle"],
}

BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]


def load_scored(path):
    """Load scored JSONL."""
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def get_domain(benchmark):
    """Map a benchmark name to its domain."""
    for domain, benches in DOMAIN_MAP.items():
        if benchmark in benches:
            return domain
    return "other"


def compute_edr_curve(labels, sort_order, budgets=None):
    """Compute error discovery rate and labeling efficiency at various budgets.

    Args:
        labels: array of 0/1 (1 = correct).
        sort_order: indices that define the inspection order (first inspected first).
        budgets: list of budget fractions.

    Returns:
        dict with budgets, edr (error discovery rate / recall), efficiency (precision),
        and auedr (area under EDR curve).
    """
    if budgets is None:
        budgets = BUDGETS

    labels = np.array(labels)
    n = len(labels)
    total_errors = int((1 - labels).sum())

    if total_errors == 0:
        return {"budgets": budgets, "edr": [0.0] * len(budgets),
                "efficiency": [0.0] * len(budgets), "auedr": 0.0,
                "total_errors": 0, "n_samples": n}

    sorted_errors = (1 - labels[sort_order])  # 1 = error

    edr_curve = []
    eff_curve = []

    for budget in budgets:
        k = max(1, int(budget * n))
        errors_found = int(sorted_errors[:k].sum())
        edr = errors_found / total_errors
        eff = errors_found / k
        edr_curve.append(edr)
        eff_curve.append(eff)

    # AUEDR via trapezoidal rule
    auedr = float(np.trapezoid(edr_curve, budgets))

    return {
        "budgets": budgets,
        "edr": [float(x) for x in edr_curve],
        "efficiency": [float(x) for x in eff_curve],
        "auedr": auedr,
        "total_errors": total_errors,
        "n_samples": n,
    }


def compute_all_methods(samples, budgets=None):
    """Compute EDR curves for all methods on a set of samples.

    Returns dict mapping method name -> EDR result dict.
    """
    if budgets is None:
        budgets = BUDGETS

    labels = np.array([s["is_correct"] for s in samples])
    p_correct = np.array([s["p_correct"] for s in samples])
    n = len(samples)

    results = {}

    # 1. Calibrator: sort by ascending P(correct) (most uncertain first)
    cal_order = np.argsort(p_correct)
    results["Calibrator"] = compute_edr_curve(labels, cal_order, budgets)

    # 2. Random baseline (use fixed seed for reproducibility)
    rng = np.random.RandomState(42)
    rand_order = rng.permutation(n)
    results["Random"] = compute_edr_curve(labels, rand_order, budgets)

    # 3. Verbalized confidence: sort by ascending verbalized confidence
    verb_confs = np.array([
        s.get("verbalized_confidence") if s.get("verbalized_confidence") is not None else np.nan
        for s in samples
    ])
    has_verb = ~np.isnan(verb_confs)
    if has_verb.sum() >= 10:
        # Only score samples that have verbalized confidence; put the rest at the end
        verb_scores = np.where(has_verb, verb_confs, 2.0)  # 2.0 > max confidence, so pushed to end
        verb_order = np.argsort(verb_scores)
        results["Verbalized"] = compute_edr_curve(labels, verb_order, budgets)

    # 4. Response length: sort by ascending output_tokens (shorter = more uncertain)
    output_tokens = np.array([s.get("output_tokens") or 0 for s in samples])
    if output_tokens.sum() > 0:
        length_order = np.argsort(output_tokens)
        results["Response Length"] = compute_edr_curve(labels, length_order, budgets)

    # 5. Oracle: sort by actual correctness (errors first)
    oracle_order = np.argsort(labels)  # 0s (errors) come first
    results["Oracle"] = compute_edr_curve(labels, oracle_order, budgets)

    return results


def compute_domain_analysis(samples, budgets=None):
    """Per-domain EDR analysis for calibrator vs baselines.

    Returns dict mapping domain -> {method -> {auedr, ...}}.
    """
    if budgets is None:
        budgets = BUDGETS

    # Assign domains
    domain_samples = defaultdict(list)
    for s in samples:
        domain = get_domain(s["benchmark"])
        domain_samples[domain].append(s)

    domain_results = {}
    for domain in sorted(domain_samples.keys()):
        dom_s = domain_samples[domain]
        if len(dom_s) < 20:
            continue
        labels = np.array([s["is_correct"] for s in dom_s])
        total_errors = int((1 - labels).sum())
        if total_errors == 0:
            continue

        methods = compute_all_methods(dom_s, budgets)
        domain_results[domain] = {
            "n_samples": len(dom_s),
            "n_errors": total_errors,
            "error_rate": float(1 - labels.mean()),
            "methods": {
                method: {"auedr": data["auedr"]}
                for method, data in methods.items()
            },
        }

    return domain_results


def plot_annotation_prioritization(per_target_results, output_path):
    """Plot error discovery rate curves for calibrator vs baselines."""
    targets = list(per_target_results.keys())
    n_targets = len(targets)
    if n_targets == 0:
        return

    fig, axes = plt.subplots(1, n_targets, figsize=(7 * n_targets, 6), squeeze=False)

    method_styles = {
        "Calibrator": {"color": "C0", "linewidth": 2.5, "marker": "o", "markersize": 4},
        "Verbalized": {"color": "C1", "linewidth": 2, "linestyle": "--", "marker": "s", "markersize": 3},
        "Response Length": {"color": "C4", "linewidth": 1.5, "linestyle": "-.", "marker": "^", "markersize": 3},
        "Random": {"color": "gray", "linewidth": 2, "linestyle": ":"},
        "Oracle": {"color": "C2", "linewidth": 1.5, "linestyle": "-.", "marker": "d", "markersize": 3},
    }

    target_names = {
        "gpt5mini": "GPT-5-mini (in-dist)",
        "gpt52": "GPT-5.2 (cross-model)",
        "qwen35": "Qwen3.5 (cross-model)",
    }

    for col, target in enumerate(targets):
        ax = axes[0, col]
        data = per_target_results[target]
        methods = data.get("methods", {})

        for method_name in ["Oracle", "Calibrator", "Verbalized", "Response Length", "Random"]:
            if method_name not in methods:
                continue
            m = methods[method_name]
            budgets_pct = [b * 100 for b in m["budgets"]]
            style = method_styles.get(method_name, {})
            ax.plot(budgets_pct, m["edr"],
                    label=f"{method_name} (AUEDR={m['auedr']:.3f})",
                    **style)

        # Diagonal reference (random expectation)
        ax.plot([0, 100], [0, 1], color="gray", linewidth=0.5, alpha=0.3)

        ax.set_xlabel("Annotation Budget (% of data labeled)", fontsize=12)
        ax.set_ylabel("Error Discovery Rate (recall)", fontsize=12)
        ax.set_title(f"Annotation Prioritization\n{target_names.get(target, target)}", fontsize=13)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="UC9: Annotation Prioritization / Labeling Efficiency"
    )
    parser.add_argument("--scored_dir", default="data/use_cases/scored_unified")
    parser.add_argument("--output_dir", default="data/use_cases/results_unified")
    parser.add_argument("--fig_dir", default="figures/use_cases_unified")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run on a tiny subset (first 50 samples per target) for quick validation")
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

        n_samples = len(samples)
        labels = np.array([s["is_correct"] for s in samples])
        n_errors = int((1 - labels).sum())
        base_acc = float(labels.mean())

        print(f"\n{'='*70}")
        print(f"UC9: Annotation Prioritization — {target_names[target]}")
        print(f"{'='*70}")
        print(f"Samples: {n_samples}, Errors: {n_errors} ({n_errors/max(n_samples,1):.1%}), "
              f"Accuracy: {base_acc:.3f}")

        if n_errors == 0:
            print("  No errors found — skipping (nothing to discover).")
            continue

        # ---- Compute EDR curves for all methods ----
        methods = compute_all_methods(samples)

        # Print summary table
        print(f"\n  {'Method':<20} {'AUEDR':>8}  ", end="")
        budgets_to_show = [0.10, 0.20, 0.30, 0.50]
        for b in budgets_to_show:
            print(f"{'EDR@'+str(int(b*100))+'%':>9}", end="")
        print()
        print(f"  {'-'*72}")

        for method_name in ["Calibrator", "Verbalized", "Response Length", "Random", "Oracle"]:
            if method_name not in methods:
                continue
            m = methods[method_name]
            line = f"  {method_name:<20} {m['auedr']:>8.3f}  "
            for b in budgets_to_show:
                idx = m["budgets"].index(b) if b in m["budgets"] else None
                if idx is not None:
                    line += f"{m['edr'][idx]:>9.3f}"
                else:
                    line += f"{'N/A':>9}"
            print(line)

        # Labeling efficiency (precision) at key budgets
        print(f"\n  Labeling Efficiency (errors found / labels used):")
        print(f"  {'Method':<20} ", end="")
        for b in budgets_to_show:
            print(f"{'Eff@'+str(int(b*100))+'%':>9}", end="")
        print()
        print(f"  {'-'*58}")

        for method_name in ["Calibrator", "Verbalized", "Response Length", "Random"]:
            if method_name not in methods:
                continue
            m = methods[method_name]
            line = f"  {method_name:<20} "
            for b in budgets_to_show:
                idx = m["budgets"].index(b) if b in m["budgets"] else None
                if idx is not None:
                    line += f"{m['efficiency'][idx]:>9.3f}"
                else:
                    line += f"{'N/A':>9}"
            print(line)

        # Calibrator advantage over baselines
        cal_auedr = methods["Calibrator"]["auedr"]
        rand_auedr = methods["Random"]["auedr"]
        print(f"\n  Calibrator advantage over Random: "
              f"+{cal_auedr - rand_auedr:.3f} AUEDR ({(cal_auedr - rand_auedr) / max(rand_auedr, 1e-6) * 100:.1f}% relative)")
        if "Verbalized" in methods:
            verb_auedr = methods["Verbalized"]["auedr"]
            print(f"  Calibrator advantage over Verbalized: "
                  f"+{cal_auedr - verb_auedr:.3f} AUEDR ({(cal_auedr - verb_auedr) / max(verb_auedr, 1e-6) * 100:.1f}% relative)")

        # ---- Per-domain analysis ----
        print(f"\n  Per-Domain Analysis:")
        domain_results = compute_domain_analysis(samples)

        print(f"  {'Domain':<15} {'N':>5} {'Errors':>7} {'Cal AUEDR':>10} {'Verb AUEDR':>11} "
              f"{'Rand AUEDR':>11} {'Cal Advantage':>14}")
        print(f"  {'-'*78}")

        for domain in sorted(domain_results.keys()):
            dr = domain_results[domain]
            dm = dr["methods"]
            cal = dm.get("Calibrator", {}).get("auedr", 0)
            verb = dm.get("Verbalized", {}).get("auedr", 0)
            rand = dm.get("Random", {}).get("auedr", 0)
            advantage = cal - rand
            verb_str = f"{verb:.3f}" if "Verbalized" in dm else "N/A"
            print(f"  {domain:<15} {dr['n_samples']:>5} {dr['n_errors']:>7} "
                  f"{cal:>10.3f} {verb_str:>11} {rand:>11.3f} {advantage:>+14.3f}")

        # ---- Store results ----
        # Strip full curve arrays from domain results for compact JSON
        domain_summary = {}
        for domain, dr in domain_results.items():
            domain_summary[domain] = {
                "n_samples": dr["n_samples"],
                "n_errors": dr["n_errors"],
                "error_rate": dr["error_rate"],
                "calibrator_auedr": dr["methods"].get("Calibrator", {}).get("auedr"),
                "verbalized_auedr": dr["methods"].get("Verbalized", {}).get("auedr"),
                "random_auedr": dr["methods"].get("Random", {}).get("auedr"),
                "oracle_auedr": dr["methods"].get("Oracle", {}).get("auedr"),
            }

        all_results[target] = {
            "n_samples": n_samples,
            "n_errors": n_errors,
            "base_accuracy": base_acc,
            "methods": methods,
            "per_domain": domain_summary,
        }

    # ---- Plot ----
    if all_results:
        plot_annotation_prioritization(all_results, f"{args.fig_dir}/uc9_annotation_prioritization.pdf")

    # ---- Save results ----
    # Make JSON-serializable: strip large arrays from per-method curves in top-level
    # but keep AUEDR and key budget EDR values
    json_results = {}
    for target, data in all_results.items():
        target_json = {
            "n_samples": data["n_samples"],
            "n_errors": data["n_errors"],
            "base_accuracy": data["base_accuracy"],
            "per_domain": data["per_domain"],
            "methods": {},
        }
        for method_name, m in data["methods"].items():
            target_json["methods"][method_name] = {
                "auedr": m["auedr"],
                "total_errors": m["total_errors"],
                "n_samples": m["n_samples"],
                "edr_at_budgets": {
                    f"{int(b*100)}%": m["edr"][i]
                    for i, b in enumerate(m["budgets"])
                },
                "efficiency_at_budgets": {
                    f"{int(b*100)}%": m["efficiency"][i]
                    for i, b in enumerate(m["budgets"])
                },
            }
        json_results[target] = target_json

    out_path = f"{args.output_dir}/uc9_results.json"
    with open(out_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
