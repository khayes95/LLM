#!/usr/bin/env python3
"""UC3: Error Flagging — flag responses likely to be wrong for human review.

Frames the calibrator as a binary error detector: P(correct) < threshold = "flagged".
Computes precision/recall curves, per-domain breakdown, and labeling efficiency
(how many errors are found per label at various annotation budgets).

Usage:
    python scripts/uc3_hallucination_detection.py
    python scripts/uc3_hallucination_detection.py --scored_dir data/use_cases/scored_test_only_v2
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


def compute_detection_metrics(labels, scores):
    """Compute error detection metrics.

    Here: label=1 means correct, score = P(correct).
    For error detection: we want to detect label=0 (errors).
    So we flip: error_score = 1 - P(correct), error_label = 1 - is_correct.
    """
    error_labels = 1 - np.array(labels)  # 1 = error
    error_scores = 1 - np.array(scores)  # higher = more likely error

    metrics = {}
    if len(set(error_labels)) > 1:
        metrics["auroc"] = float(roc_auc_score(error_labels, error_scores))
        metrics["auprc"] = float(average_precision_score(error_labels, error_scores))

        # Precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(error_labels, error_scores)
        metrics["pr_curve"] = {
            "precisions": precisions.tolist(),
            "recalls": recalls.tolist(),
        }

        # F1 at various thresholds
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
        for target_recall in [0.50, 0.80, 0.90]:
            idx = np.argmin(np.abs(recalls - target_recall))
            metrics[f"precision_at_{int(target_recall*100)}recall"] = float(precisions[idx])
            metrics[f"actual_recall_at_{int(target_recall*100)}target"] = float(recalls[idx])

        # Recall at target precision levels
        for target_prec in [0.50, 0.80]:
            valid = precisions >= target_prec
            if valid.any():
                best_recall = recalls[valid].max()
                metrics[f"recall_at_{int(target_prec*100)}precision"] = float(best_recall)

    metrics["n_samples"] = len(labels)
    metrics["n_errors"] = int(error_labels.sum())
    metrics["error_rate"] = float(error_labels.mean())

    return metrics


def compute_labeling_efficiency(labels, scores, budgets=None):
    """Compute error discovery rate and labeling efficiency at various budgets.

    Sort by ascending score (lowest P(correct) first = most uncertain).
    At each budget k%, look at the top k% of samples and count errors found.

    Returns dict with curves and summary metrics.
    """
    if budgets is None:
        budgets = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]

    labels = np.array(labels)
    scores = np.array(scores)
    n = len(labels)
    total_errors = int((1 - labels).sum())

    if total_errors == 0:
        return {"budgets": budgets, "edr": [0.0] * len(budgets),
                "efficiency": [0.0] * len(budgets), "auedr": 0.0,
                "total_errors": 0, "n_samples": n}

    # Sort by ascending score (most uncertain first)
    order = np.argsort(scores)
    sorted_errors = (1 - labels[order])  # 1 = error

    edr_curve = []  # error discovery rate (recall)
    eff_curve = []  # labeling efficiency (precision)

    for budget in budgets:
        k = max(1, int(budget * n))
        errors_found = int(sorted_errors[:k].sum())
        edr = errors_found / total_errors
        eff = errors_found / k
        edr_curve.append(edr)
        eff_curve.append(eff)

    # AUEDR via trapezoidal rule (area under error discovery rate curve)
    auedr = float(np.trapz(edr_curve, budgets))

    return {
        "budgets": budgets,
        "edr": [float(x) for x in edr_curve],
        "efficiency": [float(x) for x in eff_curve],
        "auedr": auedr,
        "total_errors": total_errors,
        "n_samples": n,
    }


def plot_detection(per_target_results, output_path):
    """Plot error detection results."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Left: PR curves by method for each target
    ax = axes[0]
    colors = {"gpt5mini": "C0", "gpt52": "C1", "qwen35": "C2"}
    names = {"gpt5mini": "GPT-5-mini", "gpt52": "GPT-5.2",
             "qwen35": "Qwen3.5"}

    for target, data in per_target_results.items():
        cal_data = data.get("calibrator", {})
        pr = cal_data.get("pr_curve", {})
        if pr:
            precs = pr["precisions"]
            recs = pr["recalls"]
            auroc = cal_data.get("auroc", 0)
            ax.plot(recs, precs, color=colors.get(target, "gray"),
                    label=f"{names.get(target, target)} (AUROC={auroc:.3f})")

    ax.set_xlabel("Recall (fraction of errors caught)", fontsize=12)
    ax.set_ylabel("Precision (fraction of flags that are real errors)", fontsize=12)
    ax.set_title("Error Detection: Precision-Recall", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Middle: Per-domain AUROC comparison
    ax = axes[1]
    for target, data in per_target_results.items():
        domain_data = data.get("per_domain", {})
        if domain_data:
            domains = sorted(domain_data.keys())
            aurocs_cal = [domain_data[d].get("calibrator_auroc", 0.5) for d in domains]
            aurocs_verb = [domain_data[d].get("verbalized_auroc", 0.5) for d in domains]
            x = np.arange(len(domains))
            w = 0.35
            ax.bar(x - w/2, aurocs_cal, w, label="Calibrator", color="C0")
            ax.bar(x + w/2, aurocs_verb, w, label="Verbalized", color="C1")
            ax.set_xticks(x)
            ax.set_xticklabels(domains, rotation=30, ha="right")
            ax.set_ylabel("AUROC (error detection)", fontsize=12)
            ax.set_title(f"Per-Domain Error Detection — {names.get(target, target)}", fontsize=13)
            ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
            ax.legend()
            break

    # Right: Labeling efficiency (error discovery rate curves)
    ax = axes[2]
    for target, data in per_target_results.items():
        le = data.get("labeling_efficiency", {})
        if le.get("calibrator"):
            cal_le = le["calibrator"]
            budgets = cal_le["budgets"]
            ax.plot([b * 100 for b in budgets], cal_le["edr"],
                    color=colors.get(target, "gray"), marker="o", markersize=4,
                    label=f"{names.get(target, target)} (AUEDR={cal_le['auedr']:.3f})")

        # Show verbalized and random baselines for first target only
        if target == list(per_target_results.keys())[0]:
            verb_le = le.get("verbalized")
            rand_le = le.get("random")
            if verb_le:
                ax.plot([b * 100 for b in verb_le["budgets"]], verb_le["edr"],
                        color="C3", linestyle="--", marker="s", markersize=3,
                        label=f"Verbalized (AUEDR={verb_le['auedr']:.3f})")
            if rand_le:
                ax.plot([b * 100 for b in rand_le["budgets"]], rand_le["edr"],
                        color="gray", linestyle=":", linewidth=2,
                        label=f"Random (AUEDR={rand_le['auedr']:.3f})")

    ax.set_xlabel("Annotation Budget (% of data)", fontsize=12)
    ax.set_ylabel("Error Discovery Rate (recall)", fontsize=12)
    ax.set_title("Labeling Efficiency: Errors Found per Budget", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored_dir", default="data/use_cases/scored_test_only_v2")
    parser.add_argument("--output_dir", default="data/use_cases/results_test_only_v2")
    parser.add_argument("--fig_dir", default="figures/use_cases")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    targets = ["gpt5mini", "gpt52", "qwen35"]
    names = {"gpt5mini": "GPT-5-mini", "gpt52": "GPT-5.2",
             "qwen35": "Qwen3.5"}

    all_results = {}

    for target in targets:
        scored_path = Path(args.scored_dir) / f"{target}_scored.jsonl"
        if not scored_path.exists():
            continue

        samples = load_scored(scored_path)
        print(f"\n{'='*70}")
        print(f"UC3: Error Flagging — {names[target]}")
        print(f"{'='*70}")

        labels = np.array([s["is_correct"] for s in samples])
        p_correct = np.array([s["p_correct"] for s in samples])
        n_errors = (1 - labels).sum()
        print(f"Samples: {len(samples)}, Errors: {n_errors} ({n_errors/len(samples):.1%})")

        target_results = {}

        # Calibrator metrics
        cal_metrics = compute_detection_metrics(labels, p_correct)
        target_results["calibrator"] = cal_metrics
        print(f"\n  Calibrator:")
        print(f"    AUROC: {cal_metrics.get('auroc', 'N/A')}")
        print(f"    AUPRC: {cal_metrics.get('auprc', 'N/A')}")
        print(f"    Best F1: {cal_metrics.get('best_f1', 'N/A')} (threshold={cal_metrics.get('best_threshold', 'N/A')})")
        print(f"    Precision@80%recall: {cal_metrics.get('precision_at_80recall', 'N/A')}")

        # Verbalized confidence
        verb_samples = [s for s in samples if s.get("verbalized_confidence") is not None]
        if verb_samples:
            verb_labels = np.array([s["is_correct"] for s in verb_samples])
            verb_scores = np.array([s["verbalized_confidence"] for s in verb_samples])
            verb_metrics = compute_detection_metrics(verb_labels, verb_scores)
            target_results["verbalized"] = verb_metrics
            print(f"  Verbalized:")
            print(f"    AUROC: {verb_metrics.get('auroc', 'N/A')}")
            print(f"    Best F1: {verb_metrics.get('best_f1', 'N/A')}")

        # Per-domain breakdown
        domain_results = {}
        for s in samples:
            s["_domain"] = get_domain(s["benchmark"])

        domains = set(s["_domain"] for s in samples)
        print(f"\n  Per-domain error detection:")
        print(f"  {'Domain':<15} {'N':>5} {'Errors':>7} {'Cal AUROC':>10} {'Verb AUROC':>10}")
        print(f"  {'-'*52}")

        for domain in sorted(domains):
            dom_samples = [s for s in samples if s["_domain"] == domain]
            dom_labels = np.array([s["is_correct"] for s in dom_samples])
            dom_scores = np.array([s["p_correct"] for s in dom_samples])

            dom_result = {"n": len(dom_samples), "n_errors": int((1-dom_labels).sum())}

            cal_auroc_str = "N/A"
            verb_auroc_str = "N/A"

            if len(set(dom_labels)) > 1 and len(dom_labels) >= 10:
                dom_auroc = roc_auc_score(1-dom_labels, 1-dom_scores)
                dom_result["calibrator_auroc"] = float(dom_auroc)
                cal_auroc_str = f"{dom_auroc:.3f}"

                # Verbalized for this domain
                dom_verb = [s for s in dom_samples if s.get("verbalized_confidence") is not None]
                if dom_verb and len(set([s["is_correct"] for s in dom_verb])) > 1:
                    v_labels = np.array([s["is_correct"] for s in dom_verb])
                    v_scores = np.array([s["verbalized_confidence"] for s in dom_verb])
                    v_auroc = roc_auc_score(1-v_labels, 1-v_scores)
                    dom_result["verbalized_auroc"] = float(v_auroc)
                    verb_auroc_str = f"{v_auroc:.3f}"

            print(f"  {domain:<15} {len(dom_samples):>5} {dom_result['n_errors']:>7} "
                  f"{cal_auroc_str:>10} {verb_auroc_str:>10}")
            domain_results[domain] = dom_result

        target_results["per_domain"] = domain_results

        # Labeling efficiency analysis
        print(f"\n  Labeling Efficiency (annotation prioritization):")
        le_results = {}

        # Calibrator: sort by ascending P(correct)
        cal_le = compute_labeling_efficiency(labels, p_correct)
        le_results["calibrator"] = cal_le
        print(f"    Calibrator AUEDR: {cal_le['auedr']:.3f}")

        # Verbalized baseline
        if verb_samples:
            # Need aligned arrays — only samples with verbalized confidence
            verb_labels_le = np.array([s["is_correct"] for s in verb_samples])
            verb_scores_le = np.array([s["verbalized_confidence"] for s in verb_samples])
            verb_le = compute_labeling_efficiency(verb_labels_le, verb_scores_le)
            le_results["verbalized"] = verb_le
            print(f"    Verbalized AUEDR: {verb_le['auedr']:.3f}")

        # Random baseline (diagonal)
        rand_le = compute_labeling_efficiency(labels, np.random.RandomState(42).rand(len(labels)))
        le_results["random"] = rand_le
        print(f"    Random AUEDR:     {rand_le['auedr']:.3f}")

        # Response length baseline (sort by ascending length → shorter = more uncertain)
        resp_lengths = np.array([s.get("output_tokens") or 0 for s in samples])
        if resp_lengths.sum() > 0:
            length_le = compute_labeling_efficiency(labels, resp_lengths)
            le_results["response_length"] = length_le
            print(f"    Resp Length AUEDR: {length_le['auedr']:.3f}")

        # Print error discovery at key budgets
        budgets_to_show = [0.10, 0.20, 0.30, 0.50]
        print(f"\n    {'Budget':<8} {'Calibrator':>10} {'Verbalized':>10} {'Random':>10}")
        print(f"    {'-'*42}")
        for b in budgets_to_show:
            idx = cal_le["budgets"].index(b) if b in cal_le["budgets"] else None
            if idx is not None:
                cal_edr = cal_le["edr"][idx]
                verb_edr = le_results.get("verbalized", {}).get("edr", [0]*len(cal_le["budgets"]))[idx]
                rand_edr = rand_le["edr"][idx]
                print(f"    {b:>5.0%}   {cal_edr:>10.3f} {verb_edr:>10.3f} {rand_edr:>10.3f}")

        # Per-domain labeling efficiency
        print(f"\n    Per-domain AUEDR (calibrator):")
        domain_le = {}
        for domain in sorted(domains):
            dom_s = [s for s in samples if s["_domain"] == domain]
            if len(dom_s) < 20:
                continue
            dom_l = np.array([s["is_correct"] for s in dom_s])
            dom_p = np.array([s["p_correct"] for s in dom_s])
            if (1 - dom_l).sum() == 0:
                continue
            dle = compute_labeling_efficiency(dom_l, dom_p)
            domain_le[domain] = {"auedr": dle["auedr"], "n": len(dom_s),
                                 "n_errors": dle["total_errors"]}
            print(f"      {domain:<15}: AUEDR={dle['auedr']:.3f} (N={len(dom_s)}, errors={dle['total_errors']})")

        le_results["per_domain"] = domain_le
        target_results["labeling_efficiency"] = le_results

        all_results[target] = target_results

    # Plot
    if all_results:
        plot_detection(all_results, f"{args.fig_dir}/uc3_error_detection.pdf")

    # Save
    # Remove large array fields for JSON
    for target in all_results:
        for method in ["calibrator", "verbalized"]:
            if method in all_results[target]:
                all_results[target][method].pop("pr_curve", None)

    out_path = f"{args.output_dir}/uc3_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
