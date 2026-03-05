#!/usr/bin/env python3
"""UC3: Error Discovery — flag errors and prioritize annotation.

Merged from UC3 (error flagging) and UC9 (annotation prioritization). Provides
two evaluation protocols:

Protocol A: Binary error detection — threshold-based flagging with precision,
    recall, F1, AUROC, and AUPRC.
Protocol B: Ranked annotation efficiency — sort by score, measure error
    discovery rate (EDR) at various annotation budgets, and AUEDR.

Baselines: calibrator, verbalized confidence, response length, random, oracle.
All metrics include bootstrap 95% CIs.

Usage:
    python scripts/uc3_error_discovery.py
    python scripts/uc3_error_discovery.py --scored_dir data/use_cases/scored_test_only_v2
    python scripts/uc3_error_discovery.py --smoke_test
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score
)


DOMAIN_MAP = {
    "factual": ["simpleqa", "gpqa", "chembench", "hallusionbench"],
    "reasoning": ["bbeh", "arc_agi", "multichallenge"],
    "math": ["omnimath", "mathvista", "mathvision", "mathverse"],
    "knowledge": ["mmlu_pro", "mmmu", "hle"],
    "multimodal": ["charxiv", "mmstar", "mmvet", "realworldqa", "vizwiz"],
    "coding": ["prbench"],
    "long_context": ["livebench", "oolong", "longbench"],
}

BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]


def load_scored(path):
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def get_domain(benchmark):
    for domain, benches in DOMAIN_MAP.items():
        if benchmark in benches:
            return domain
    return "other"


# ---------------------------------------------------------------------------
# Protocol A: Binary Error Detection
# ---------------------------------------------------------------------------

def compute_detection_metrics(labels, scores):
    """Compute binary error detection metrics.
    labels: 1=correct, scores: P(correct). We detect errors (label=0).
    """
    error_labels = 1 - np.array(labels)
    error_scores = 1 - np.array(scores)

    metrics = {}
    if len(set(error_labels)) > 1:
        metrics["auroc"] = float(roc_auc_score(error_labels, error_scores))
        metrics["auprc"] = float(average_precision_score(error_labels, error_scores))

        # Best F1 across thresholds
        best_f1 = 0
        best_threshold = 0.5
        for t in np.linspace(0.1, 0.9, 81):
            preds = (error_scores >= t).astype(int)
            if preds.sum() == 0 or preds.sum() == len(preds):
                continue
            f1 = f1_score(error_labels, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        metrics["best_f1"] = float(best_f1)
        metrics["best_threshold"] = float(best_threshold)

        # Precision at target recall levels
        precisions, recalls, _ = precision_recall_curve(error_labels, error_scores)
        for target_recall in [0.50, 0.80, 0.90]:
            idx = np.argmin(np.abs(recalls - target_recall))
            metrics[f"precision_at_{int(target_recall*100)}recall"] = float(precisions[idx])

    metrics["n_samples"] = len(labels)
    metrics["n_errors"] = int((1 - np.array(labels)).sum())
    metrics["error_rate"] = float(1 - np.mean(labels))

    return metrics


def bootstrap_detection_metrics(labels, scores, n_bootstrap=1000, seed=42):
    """Bootstrap 95% CI for AUROC and best F1."""
    rng = np.random.RandomState(seed)
    labels = np.array(labels)
    scores = np.array(scores)
    n = len(labels)

    aurocs = []
    f1s = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        bl = labels[idx]
        bs = scores[idx]
        if len(set(bl)) < 2:
            continue
        el = 1 - bl
        es = 1 - bs
        aurocs.append(roc_auc_score(el, es))
        # Best F1
        best_f1 = 0
        for t in np.linspace(0.2, 0.8, 31):
            preds = (es >= t).astype(int)
            if preds.sum() == 0 or preds.sum() == len(preds):
                continue
            f1 = f1_score(el, preds)
            best_f1 = max(best_f1, f1)
        f1s.append(best_f1)

    ci = {}
    if aurocs:
        ci["auroc_ci"] = [float(np.percentile(aurocs, 2.5)),
                          float(np.percentile(aurocs, 97.5))]
    if f1s:
        ci["f1_ci"] = [float(np.percentile(f1s, 2.5)),
                       float(np.percentile(f1s, 97.5))]
    return ci


# ---------------------------------------------------------------------------
# Protocol B: Ranked Annotation Efficiency (EDR)
# ---------------------------------------------------------------------------

def compute_edr_curve(labels, sort_order, budgets=None):
    """Compute error discovery rate at various annotation budgets.
    labels: 1=correct. sort_order: indices defining inspection order.
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

    sorted_errors = (1 - labels[sort_order])

    edr_curve = []
    eff_curve = []
    for budget in budgets:
        k = max(1, int(budget * n))
        errors_found = int(sorted_errors[:k].sum())
        edr_curve.append(errors_found / total_errors)
        eff_curve.append(errors_found / k)

    _trapz = getattr(np, "trapezoid", np.trapz)
    auedr = float(_trapz(edr_curve, budgets))

    return {
        "budgets": budgets,
        "edr": [float(x) for x in edr_curve],
        "efficiency": [float(x) for x in eff_curve],
        "auedr": auedr,
        "total_errors": total_errors,
        "n_samples": n,
    }


def compute_all_edr_methods(samples, budgets=None):
    """Compute EDR curves for calibrator, verbalized, length, random, oracle."""
    if budgets is None:
        budgets = BUDGETS

    labels = np.array([s["is_correct"] for s in samples])
    p_correct = np.array([s["p_correct"] for s in samples])
    n = len(samples)
    results = {}

    # Calibrator: ascending P(correct)
    results["Calibrator"] = compute_edr_curve(labels, np.argsort(p_correct), budgets)

    # Random
    rng = np.random.RandomState(42)
    results["Random"] = compute_edr_curve(labels, rng.permutation(n), budgets)

    # Verbalized confidence
    verb_confs = np.array([
        s.get("verbalized_confidence") if s.get("verbalized_confidence") is not None else np.nan
        for s in samples
    ])
    has_verb = ~np.isnan(verb_confs)
    if has_verb.sum() >= 10:
        verb_scores = np.where(has_verb, verb_confs, 2.0)
        results["Verbalized"] = compute_edr_curve(labels, np.argsort(verb_scores), budgets)

    # Response length: ascending tokens
    output_tokens = np.array([s.get("output_tokens") or 0 for s in samples])
    if output_tokens.sum() > 0:
        results["Response Length"] = compute_edr_curve(labels, np.argsort(output_tokens), budgets)

    # Oracle: errors first
    results["Oracle"] = compute_edr_curve(labels, np.argsort(labels), budgets)

    return results


def bootstrap_auedr(labels, p_correct, n_bootstrap=1000, seed=42):
    """Bootstrap 95% CI for calibrator AUEDR."""
    rng = np.random.RandomState(seed)
    labels = np.array(labels)
    p_correct = np.array(p_correct)
    n = len(labels)
    auedrs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        bl = labels[idx]
        bp = p_correct[idx]
        if (1 - bl).sum() == 0:
            continue
        order = np.argsort(bp)
        res = compute_edr_curve(bl, order)
        auedrs.append(res["auedr"])
    if auedrs:
        return [float(np.percentile(auedrs, 2.5)), float(np.percentile(auedrs, 97.5))]
    return None


# ---------------------------------------------------------------------------
# Per-domain analysis
# ---------------------------------------------------------------------------

def per_domain_analysis(samples, budgets=None):
    """Per-domain error detection AUROC + AUEDR."""
    domain_samples = defaultdict(list)
    for s in samples:
        domain_samples[get_domain(s["benchmark"])].append(s)

    results = {}
    for domain in sorted(domain_samples.keys()):
        ds = domain_samples[domain]
        if len(ds) < 20:
            continue
        labels = np.array([s["is_correct"] for s in ds])
        p_correct = np.array([s["p_correct"] for s in ds])
        total_errors = int((1 - labels).sum())
        if total_errors == 0:
            continue

        d = {"n": len(ds), "n_errors": total_errors, "error_rate": float(1 - labels.mean())}

        if len(set(labels)) > 1:
            d["calibrator_auroc"] = float(roc_auc_score(1 - labels, 1 - p_correct))
            # Verbalized AUROC
            verb = [s for s in ds if s.get("verbalized_confidence") is not None]
            if verb and len(set([s["is_correct"] for s in verb])) > 1:
                d["verbalized_auroc"] = float(roc_auc_score(
                    1 - np.array([s["is_correct"] for s in verb]),
                    1 - np.array([s["verbalized_confidence"] for s in verb])
                ))

        # AUEDR
        edr = compute_edr_curve(labels, np.argsort(p_correct), budgets)
        d["calibrator_auedr"] = edr["auedr"]

        results[domain] = d
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_error_discovery(per_target_results, output_path):
    """Combined 2x2 figure: PR curves, per-domain, EDR curves, EDR comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    colors = {"gpt5mini": "C0", "gpt52": "C1", "qwen35": "C2"}
    names = {"gpt5mini": "GPT-5-mini", "gpt52": "GPT-5.2", "qwen35": "Qwen3.5"}
    method_styles = {
        "Calibrator": {"color": "C0", "linewidth": 2.5, "marker": "o", "markersize": 4},
        "Verbalized": {"color": "C1", "linewidth": 2, "linestyle": "--", "marker": "s", "markersize": 3},
        "Response Length": {"color": "C4", "linewidth": 1.5, "linestyle": "-.", "marker": "^", "markersize": 3},
        "Random": {"color": "gray", "linewidth": 2, "linestyle": ":"},
        "Oracle": {"color": "C2", "linewidth": 1.5, "linestyle": "-.", "marker": "d", "markersize": 3},
    }

    # Top-left: Error detection F1/AUROC comparison across targets
    ax = axes[0, 0]
    targets = list(per_target_results.keys())
    x = np.arange(len(targets))
    w = 0.3
    for i, (method_key, label, color) in enumerate([
        ("calibrator", "Calibrator", "C0"),
        ("verbalized", "Verbalized", "C1"),
    ]):
        aurocs = []
        for t in targets:
            m = per_target_results[t].get(f"{method_key}_detection", {})
            aurocs.append(m.get("auroc", 0.5))
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, aurocs, w, label=label, color=color)
        for j, v in enumerate(aurocs):
            ax.text(x[j] + offset, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([names.get(t, t) for t in targets])
    ax.set_ylabel("AUROC (error detection)")
    ax.set_title("A. Error Detection AUROC by Method", fontsize=13)
    ax.legend()
    ax.set_ylim(0.3, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    # Top-right: Per-domain AUROC for first target
    ax = axes[0, 1]
    first_target = targets[0] if targets else None
    if first_target:
        domain_data = per_target_results[first_target].get("per_domain", {})
        if domain_data:
            domains = sorted(domain_data.keys())
            cal_aurocs = [domain_data[d].get("calibrator_auroc", 0.5) for d in domains]
            verb_aurocs = [domain_data[d].get("verbalized_auroc", 0.5) for d in domains]
            x_d = np.arange(len(domains))
            w = 0.35
            ax.bar(x_d - w/2, cal_aurocs, w, label="Calibrator", color="C0")
            ax.bar(x_d + w/2, verb_aurocs, w, label="Verbalized", color="C1")
            ax.set_xticks(x_d)
            ax.set_xticklabels(domains, rotation=30, ha="right")
            ax.set_ylabel("AUROC")
            ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
            ax.legend()
    ax.set_title(f"B. Per-Domain Error Detection ({names.get(first_target, '')})", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom-left: EDR curves for all targets (calibrator only)
    ax = axes[1, 0]
    for target in targets:
        edr_methods = per_target_results[target].get("edr_methods", {})
        cal_edr = edr_methods.get("Calibrator", {})
        if cal_edr:
            budgets_pct = [b * 100 for b in cal_edr["budgets"]]
            auedr = cal_edr["auedr"]
            ci = per_target_results[target].get("auedr_ci")
            ci_str = f" [{ci[0]:.3f},{ci[1]:.3f}]" if ci else ""
            ax.plot(budgets_pct, cal_edr["edr"], color=colors.get(target, "gray"),
                    marker="o", markersize=4, linewidth=2,
                    label=f"{names.get(target, target)} (AUEDR={auedr:.3f}{ci_str})")

    # Random diagonal
    ax.plot([0, 100], [0, 1], color="gray", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Annotation Budget (% of data)")
    ax.set_ylabel("Error Discovery Rate (recall)")
    ax.set_title("C. Annotation Efficiency: Calibrator", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 1.05)

    # Bottom-right: EDR method comparison for first target
    ax = axes[1, 1]
    if first_target:
        edr_methods = per_target_results[first_target].get("edr_methods", {})
        for method_name in ["Oracle", "Calibrator", "Verbalized", "Response Length", "Random"]:
            if method_name not in edr_methods:
                continue
            m = edr_methods[method_name]
            budgets_pct = [b * 100 for b in m["budgets"]]
            style = method_styles.get(method_name, {})
            ax.plot(budgets_pct, m["edr"],
                    label=f"{method_name} (AUEDR={m['auedr']:.3f})", **style)

    ax.plot([0, 100], [0, 1], color="gray", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Annotation Budget (% of data)")
    ax.set_ylabel("Error Discovery Rate")
    ax.set_title(f"D. Method Comparison ({names.get(first_target, '')})", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UC3: Error Discovery (merged UC3+UC9)")
    parser.add_argument("--scored_dir", default="data/use_cases/scored_test_only_v2")
    parser.add_argument("--output_dir", default="data/use_cases/results_test_only_v2")
    parser.add_argument("--fig_dir", default="figures/use_cases_v2")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    targets = ["gpt5mini", "gpt52", "qwen35"]
    names = {"gpt5mini": "GPT-5-mini (in-dist)", "gpt52": "GPT-5.2 (cross-model)",
             "qwen35": "Qwen3.5 (cross-model)"}
    n_bootstrap = 50 if args.smoke_test else args.n_bootstrap

    all_results = {}

    for target in targets:
        scored_path = Path(args.scored_dir) / f"{target}_scored.jsonl"
        if not scored_path.exists():
            print(f"Skipping {target}: not found")
            continue

        samples = load_scored(scored_path)
        if args.smoke_test:
            samples = samples[:100]

        labels = np.array([s["is_correct"] for s in samples])
        p_correct = np.array([s["p_correct"] for s in samples])
        n_errors = int((1 - labels).sum())

        print(f"\n{'='*70}")
        print(f"UC3: Error Discovery — {names[target]}")
        print(f"{'='*70}")
        print(f"Samples: {len(samples)}, Errors: {n_errors} ({n_errors/len(samples):.1%})")

        target_results = {}

        # === Protocol A: Binary Error Detection ===
        print(f"\n  --- Protocol A: Binary Error Detection ---")

        # Calibrator
        cal_det = compute_detection_metrics(labels, p_correct)
        cal_ci = bootstrap_detection_metrics(labels, p_correct, n_bootstrap)
        cal_det.update(cal_ci)
        target_results["calibrator_detection"] = cal_det

        auroc_ci_str = f" [{cal_ci['auroc_ci'][0]:.3f},{cal_ci['auroc_ci'][1]:.3f}]" if "auroc_ci" in cal_ci else ""
        f1_ci_str = f" [{cal_ci['f1_ci'][0]:.3f},{cal_ci['f1_ci'][1]:.3f}]" if "f1_ci" in cal_ci else ""
        print(f"  Calibrator AUROC: {cal_det.get('auroc', 'N/A')}{auroc_ci_str}")
        print(f"  Calibrator F1:    {cal_det.get('best_f1', 'N/A')}{f1_ci_str}")
        print(f"  Precision@80%recall: {cal_det.get('precision_at_80recall', 'N/A')}")

        # Verbalized
        verb_samples = [s for s in samples if s.get("verbalized_confidence") is not None]
        if verb_samples:
            verb_labels = np.array([s["is_correct"] for s in verb_samples])
            verb_scores = np.array([s["verbalized_confidence"] for s in verb_samples])
            verb_det = compute_detection_metrics(verb_labels, verb_scores)
            verb_ci = bootstrap_detection_metrics(verb_labels, verb_scores, n_bootstrap)
            verb_det.update(verb_ci)
            target_results["verbalized_detection"] = verb_det
            print(f"  Verbalized AUROC: {verb_det.get('auroc', 'N/A')}")
            print(f"  Verbalized F1:    {verb_det.get('best_f1', 'N/A')}")

        # === Protocol B: Ranked Annotation Efficiency ===
        print(f"\n  --- Protocol B: Annotation Efficiency ---")

        edr_methods = compute_all_edr_methods(samples)
        target_results["edr_methods"] = {
            method: {k: v for k, v in data.items() if k not in ("edr", "efficiency")}
            for method, data in edr_methods.items()
        }
        # Keep full curves for plotting
        target_results["_edr_full"] = edr_methods

        # Summary table
        print(f"\n  {'Method':<20} {'AUEDR':>8}  ", end="")
        budgets_to_show = [0.10, 0.20, 0.30, 0.50]
        for b in budgets_to_show:
            print(f"{'EDR@'+str(int(b*100))+'%':>9}", end="")
        print()
        print(f"  {'-'*72}")

        for method_name in ["Calibrator", "Verbalized", "Response Length", "Random", "Oracle"]:
            if method_name not in edr_methods:
                continue
            m = edr_methods[method_name]
            line = f"  {method_name:<20} {m['auedr']:>8.3f}  "
            for b in budgets_to_show:
                idx = m["budgets"].index(b) if b in m["budgets"] else None
                if idx is not None:
                    line += f"{m['edr'][idx]:>9.3f}"
                else:
                    line += f"{'N/A':>9}"
            print(line)

        # Bootstrap CI for calibrator AUEDR
        auedr_ci = bootstrap_auedr(labels, p_correct, n_bootstrap)
        target_results["auedr_ci"] = auedr_ci
        if auedr_ci:
            print(f"\n  Calibrator AUEDR 95% CI: [{auedr_ci[0]:.3f}, {auedr_ci[1]:.3f}]")

        # Advantage metrics
        cal_auedr = edr_methods["Calibrator"]["auedr"]
        rand_auedr = edr_methods["Random"]["auedr"]
        print(f"  Calibrator advantage over Random: +{cal_auedr - rand_auedr:.3f} AUEDR")
        if "Verbalized" in edr_methods:
            verb_auedr = edr_methods["Verbalized"]["auedr"]
            print(f"  Calibrator advantage over Verbalized: +{cal_auedr - verb_auedr:.3f} AUEDR")

        # === Per-domain analysis ===
        print(f"\n  --- Per-Domain Analysis ---")
        domain_results = per_domain_analysis(samples)
        target_results["per_domain"] = domain_results

        print(f"  {'Domain':<15} {'N':>5} {'Errors':>7} {'Cal AUROC':>10} {'Verb AUROC':>10} {'Cal AUEDR':>10}")
        print(f"  {'-'*62}")
        for domain in sorted(domain_results.keys()):
            d = domain_results[domain]
            cal_a = f"{d.get('calibrator_auroc', 0):.3f}" if "calibrator_auroc" in d else "N/A"
            verb_a = f"{d.get('verbalized_auroc', 0):.3f}" if "verbalized_auroc" in d else "N/A"
            print(f"  {domain:<15} {d['n']:>5} {d['n_errors']:>7} {cal_a:>10} {verb_a:>10} {d['calibrator_auedr']:>10.3f}")

        all_results[target] = target_results

    # Plot
    if all_results:
        # Rebuild plot data with full curves
        plot_data = {}
        for target, tr in all_results.items():
            plot_data[target] = {
                "calibrator_detection": tr.get("calibrator_detection", {}),
                "verbalized_detection": tr.get("verbalized_detection", {}),
                "per_domain": tr.get("per_domain", {}),
                "edr_methods": tr.get("_edr_full", {}),
                "auedr_ci": tr.get("auedr_ci"),
            }
        plot_error_discovery(plot_data, f"{args.fig_dir}/uc3_error_discovery.pdf")

    # Save (strip internal fields)
    for target in all_results:
        all_results[target].pop("_edr_full", None)

    out_path = f"{args.output_dir}/uc3_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
