#!/usr/bin/env python3
"""UC5: Calibrator-Guided Response Selection (merged UC5 + UC-B).

Given multiple models' responses to the same question, select the best response
using the calibrator P(correct) score. Evaluates:

1. Pairwise ranking accuracy (N=2): does higher P(correct) pick the correct one?
2. Best-of-N selection (N=2, 3, 5): accuracy of selecting response with highest score
3. Disagreement analysis: accuracy when models disagree
4. Bootstrap CIs on all key metrics
5. Per-benchmark breakdown

Baselines: calibrator (raw, z-score, percentile), verbalized confidence,
response length, random, majority vote, oracle.

Usage:
    python scripts/uc5_response_selection.py
    python scripts/uc5_response_selection.py --scored_dir data/use_cases/scored_test_only_v2
    python scripts/uc5_response_selection.py --smoke_test
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest


def load_scored(path):
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


def build_question_index(all_scored):
    question_data = defaultdict(dict)
    for target, samples in all_scored.items():
        for s in samples:
            key = (s["benchmark"], s["id"])
            question_data[key][target] = s
    return question_data


def apply_normalizations(all_scored):
    """Add z-score and percentile fields to each sample."""
    for target, samples in all_scored.items():
        scores = [s["p_correct"] for s in samples]
        mu = np.mean(scores)
        sigma = np.std(scores) + 1e-8
        sorted_scores = np.sort(scores)
        n = len(scores)
        for s in samples:
            s["z_score_global"] = (s["p_correct"] - mu) / sigma
            s["percentile"] = np.searchsorted(sorted_scores, s["p_correct"]) / n
            s["response_length"] = s.get("output_tokens") or 0


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def pairwise_ranking(question_data, model_a, model_b, score_key):
    """For discriminative pairs, does higher score pick the correct response?"""
    n_disc = 0
    n_correct = 0
    for key, models in question_data.items():
        if model_a not in models or model_b not in models:
            continue
        a, b = models[model_a], models[model_b]
        if a["is_correct"] == b["is_correct"]:
            continue
        n_disc += 1
        sa = a.get(score_key)
        sb = b.get(score_key)
        if sa is None or sb is None:
            continue
        if (a["is_correct"] > b["is_correct"] and sa > sb) or \
           (b["is_correct"] > a["is_correct"] and sb > sa):
            n_correct += 1

    acc = n_correct / n_disc if n_disc > 0 else 0.5
    p_value = None
    if n_disc >= 10:
        p_value = float(binomtest(n_correct, n_disc, 0.5, alternative="greater").pvalue)

    return {
        "pairwise_accuracy": float(acc),
        "n_discriminative": n_disc,
        "n_correct": n_correct,
        "p_value": p_value,
        "significant": p_value is not None and p_value < 0.05,
    }


def best_of_n_accuracy(question_data, models, score_key, n_models,
                        rng, n_random_trials=1000):
    """Best-of-N selection accuracy for a given score key.
    For N>len(available_models), we use subsets of available models.
    """
    # Filter to questions with >= n_models
    questions = {k: v for k, v in question_data.items() if len(v) >= n_models}
    if not questions:
        return None

    n_q = len(questions)
    cal_correct = 0
    rand_correct_sum = 0
    oracle_correct = 0

    for (bench, qid), mdict in questions.items():
        responses = list(mdict.values())
        if len(responses) > n_models:
            # Take first n_models alphabetically for consistency
            models_here = sorted(mdict.keys())[:n_models]
            responses = [mdict[m] for m in models_here]

        # Score-based selection
        valid = [r for r in responses if r.get(score_key) is not None]
        if valid:
            best = max(valid, key=lambda r: r.get(score_key, 0))
            cal_correct += best["is_correct"]
        else:
            cal_correct += responses[0]["is_correct"]

        # Oracle
        oracle_correct += max(r["is_correct"] for r in responses)

        # Random (expected)
        n_cor = sum(r["is_correct"] for r in responses)
        rand_correct_sum += n_cor / len(responses)

    # Individual model accuracies
    all_model_names = set()
    for mdict in questions.values():
        all_model_names.update(mdict.keys())
    model_accs = {}
    for mn in sorted(all_model_names):
        mc = sum(1 for mdict in questions.values() if mn in mdict and mdict[mn]["is_correct"])
        mt = sum(1 for mdict in questions.values() if mn in mdict)
        if mt > 0:
            model_accs[mn] = mc / mt
    best_individual = max(model_accs.values()) if model_accs else 0

    # Majority vote
    maj_correct = 0
    for (bench, qid), mdict in questions.items():
        responses = list(mdict.values())
        n_cor = sum(r["is_correct"] for r in responses)
        maj_correct += 1 if n_cor > len(responses) / 2 else 0

    return {
        "n_questions": n_q,
        "n_models": n_models,
        "calibrator": float(cal_correct / n_q),
        "random": float(rand_correct_sum / n_q),
        "oracle": float(oracle_correct / n_q),
        "majority_vote": float(maj_correct / n_q),
        "best_individual": float(best_individual),
        "individual_models": model_accs,
    }


def bootstrap_selection_ci(question_data, models, score_key, n_models,
                           n_bootstrap=1000, seed=42):
    """Bootstrap 95% CI for calibrator selection accuracy."""
    questions = [(k, v) for k, v in question_data.items() if len(v) >= n_models]
    if not questions:
        return None
    rng = np.random.RandomState(seed)
    n = len(questions)
    accs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        correct = 0
        for i in idx:
            (bench, qid), mdict = questions[i]
            responses = list(mdict.values())
            if len(responses) > n_models:
                responses = responses[:n_models]
            valid = [r for r in responses if r.get(score_key) is not None]
            if valid:
                best = max(valid, key=lambda r: r.get(score_key, 0))
                correct += best["is_correct"]
        accs.append(correct / n)
    return [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]


def disagreement_analysis(question_data, n_models, rng):
    """Selection accuracy on questions where models disagree."""
    questions = {}
    for k, v in question_data.items():
        if len(v) != n_models:
            continue
        correctness = [r["is_correct"] for r in v.values()]
        if not (all(c == 1 for c in correctness) or all(c == 0 for c in correctness)):
            questions[k] = v

    if not questions:
        return None

    n_q = len(questions)
    results = {"n_questions": n_q, "n_models": n_models}

    for method, score_key in [("calibrator", "p_correct"),
                              ("verbalized", "verbalized_confidence"),
                              ("length", "response_length")]:
        correct = 0
        valid = 0
        for (b, qid), mdict in questions.items():
            responses = list(mdict.values())
            scored = [r for r in responses if r.get(score_key) is not None]
            if not scored:
                continue
            valid += 1
            best = max(scored, key=lambda r: r.get(score_key, 0))
            correct += best["is_correct"]
        results[method] = float(correct / valid) if valid > 0 else None

    # Random
    rand_trials = []
    for _ in range(1000):
        trial = 0
        for (b, qid), mdict in questions.items():
            responses = list(mdict.values())
            trial += responses[rng.randint(0, len(responses))]["is_correct"]
        rand_trials.append(trial / n_q)
    results["random"] = float(np.mean(rand_trials))

    # Oracle
    oracle = sum(max(r["is_correct"] for r in mdict.values())
                 for mdict in questions.values())
    results["oracle"] = float(oracle / n_q)

    return results


def per_benchmark_breakdown(question_data, rng):
    """Per-benchmark calibrator selection accuracy for N=3."""
    q3 = {k: v for k, v in question_data.items() if len(v) == 3}
    if not q3:
        return {}

    by_bench = defaultdict(dict)
    for (bench, qid), mdict in q3.items():
        by_bench[bench][(bench, qid)] = mdict

    results = {}
    for bench in sorted(by_bench.keys()):
        bq = by_bench[bench]
        nq = len(bq)
        if nq == 0:
            continue
        cal_correct = sum(
            max(list(mdict.values()), key=lambda r: r["p_correct"])["is_correct"]
            for mdict in bq.values()
        )
        # Random
        rand_accs = []
        for _ in range(500):
            tc = sum(list(mdict.values())[rng.randint(0, 3)]["is_correct"]
                     for mdict in bq.values())
            rand_accs.append(tc / nq)

        oracle_correct = sum(max(r["is_correct"] for r in mdict.values())
                             for mdict in bq.values())
        results[bench] = {
            "n_questions": nq,
            "calibrator": float(cal_correct / nq),
            "random": float(np.mean(rand_accs)),
            "oracle": float(oracle_correct / nq),
            "lift_over_random": float(cal_correct / nq - np.mean(rand_accs)),
        }
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_response_selection(results_by_n, disagree_by_n, output_path):
    """2-panel figure: selection accuracy by N and disagreement analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    target_names = {"gpt5mini": "GPT-5-mini", "gpt52": "GPT-5.2", "qwen35": "Qwen3.5"}

    # Left: Best-of-N accuracy across methods
    ax = axes[0]
    methods = ["calibrator", "verbalized", "length", "random", "majority_vote", "oracle"]
    method_labels = {
        "calibrator": "Calibrator",
        "verbalized": "Verbalized",
        "length": "Longest",
        "random": "Random",
        "majority_vote": "Majority",
        "oracle": "Oracle",
    }

    n_values = sorted(results_by_n.keys())
    n_groups = len(n_values)
    present = [m for m in methods
               if any(results_by_n.get(n, {}).get(m) is not None for n in n_values)]
    n_methods = len(present)

    bar_width = 0.8 / max(n_groups, 1)
    x = np.arange(n_methods)

    for i, n in enumerate(n_values):
        vals = []
        for m in present:
            v = results_by_n.get(n, {}).get(m)
            vals.append(v if v is not None else 0)
        offset = (i - (n_groups - 1) / 2) * bar_width
        nq = results_by_n.get(n, {}).get("n_questions", "?")
        ax.bar(x + offset, vals, bar_width,
               label=f"N={n} ({nq} Qs)", alpha=0.85, edgecolor="black", linewidth=0.5)
        for j, v in enumerate(vals):
            if v > 0:
                ax.text(x[j] + offset, v + 0.008, f"{v:.3f}",
                        ha="center", fontsize=7)

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([method_labels.get(m, m) for m in present], fontsize=9)
    ax.set_ylabel("Selection Accuracy")
    ax.set_title("UC5: Best-of-N Response Selection", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.grid(True, alpha=0.3, axis="y")

    # Right: Disagreement analysis
    ax = axes[1]
    if disagree_by_n:
        dm = ["calibrator", "verbalized", "length", "random", "oracle"]
        dm_labels = ["Calibrator", "Verbalized", "Longest", "Random", "Oracle"]
        present_d = [m for m in dm
                     if any(disagree_by_n.get(n, {}).get(m) is not None
                            for n in disagree_by_n)]
        x_d = np.arange(len(present_d))
        n_d_values = sorted(disagree_by_n.keys())
        bw = 0.8 / max(len(n_d_values), 1)

        for i, n in enumerate(n_d_values):
            vals = [disagree_by_n[n].get(m, 0) or 0 for m in present_d]
            offset = (i - (len(n_d_values) - 1) / 2) * bw
            nq = disagree_by_n[n].get("n_questions", "?")
            ax.bar(x_d + offset, vals, bw,
                   label=f"N={n} ({nq} disagree)", alpha=0.85,
                   edgecolor="black", linewidth=0.5)

        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
        ax.set_xticks(x_d)
        ax.set_xticklabels([dict(zip(dm, dm_labels)).get(m, m) for m in present_d],
                           fontsize=9)
        ax.set_ylabel("Selection Accuracy")
        ax.set_title("Disagreement Questions Only", fontsize=13)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UC5: Response Selection (merged UC5+UC-B)")
    parser.add_argument("--scored_dir", default="data/use_cases/scored_test_only_v2")
    parser.add_argument("--output_dir", default="data/use_cases/results_test_only_v2")
    parser.add_argument("--fig_dir", default="figures/use_cases_v2")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    all_scored = {}
    for target in ["gpt5mini", "gpt52", "qwen35"]:
        path = Path(args.scored_dir) / f"{target}_scored.jsonl"
        if path.exists():
            samples = load_scored(path)
            if args.smoke_test:
                samples = samples[:50]
            all_scored[target] = samples
            print(f"  {target}: {len(samples)} samples")

    if len(all_scored) < 2:
        print("ERROR: Need at least 2 scored targets")
        return

    print("=" * 70)
    print("UC5: Calibrator-Guided Response Selection")
    print("=" * 70)

    apply_normalizations(all_scored)
    question_data = build_question_index(all_scored)
    models = list(all_scored.keys())

    n_total = len(question_data)
    n_2plus = sum(1 for v in question_data.values() if len(v) >= 2)
    n_3 = sum(1 for v in question_data.values() if len(v) == 3)
    print(f"Total questions: {n_total}, 2+ models: {n_2plus}, 3 models: {n_3}")

    rng = np.random.RandomState(42)
    n_bootstrap = 50 if args.smoke_test else args.n_bootstrap

    # === Pairwise ranking ===
    print(f"\n--- Pairwise Ranking Accuracy ---")
    score_methods = {
        "p_correct": "Calibrator (raw P)",
        "z_score_global": "Z-score (global)",
        "percentile": "Percentile",
        "verbalized_confidence": "Verbalized Conf.",
        "response_length": "Response Length",
    }

    all_pairwise = {}
    model_pairs = []
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            model_pairs.append((m1, m2, f"{m1}_vs_{m2}"))

    print(f"  {'Method':<22}", end="")
    for _, _, pair in model_pairs:
        print(f"  {pair:>20}", end="")
    print()
    print(f"  {'-'*(22 + 22*len(model_pairs))}")

    for method_key, method_label in score_methods.items():
        all_pairwise[method_key] = {}
        print(f"  {method_label:<22}", end="")
        for m1, m2, pair in model_pairs:
            res = pairwise_ranking(question_data, m1, m2, method_key)
            all_pairwise[method_key][pair] = res
            sig = "*" if res.get("significant") else " "
            print(f"  {res['pairwise_accuracy']:>6.3f}{sig} (n={res['n_discriminative']:>4})", end="")
        print()

    # === Best-of-N ===
    print(f"\n--- Best-of-N Response Selection ---")
    results_by_n = {}
    disagree_by_n = {}

    for n_models in [2, 3]:
        # Verbalized and length selection
        for score_key, method_name in [
            ("p_correct", "calibrator"),
            ("verbalized_confidence", "verbalized"),
            ("response_length", "length"),
        ]:
            bon = best_of_n_accuracy(question_data, models, score_key, n_models, rng)
            if bon is None:
                continue
            if n_models not in results_by_n:
                results_by_n[n_models] = {"n_questions": bon["n_questions"],
                                          "n_models": n_models}
            if method_name == "calibrator":
                results_by_n[n_models]["calibrator"] = bon["calibrator"]
                results_by_n[n_models]["random"] = bon["random"]
                results_by_n[n_models]["oracle"] = bon["oracle"]
                results_by_n[n_models]["majority_vote"] = bon["majority_vote"]
                results_by_n[n_models]["best_individual"] = bon["best_individual"]
                results_by_n[n_models]["individual_models"] = bon["individual_models"]
            else:
                results_by_n[n_models][method_name] = bon["calibrator"]

        # Bootstrap CI
        ci = bootstrap_selection_ci(question_data, models, "p_correct",
                                    n_models, n_bootstrap)
        if ci and n_models in results_by_n:
            results_by_n[n_models]["calibrator_ci"] = ci

        # Disagreement
        disagree = disagreement_analysis(question_data, n_models, rng)
        if disagree:
            disagree_by_n[n_models] = disagree

    # Print summary
    print(f"\n  {'Method':<25}", end="")
    for n in sorted(results_by_n.keys()):
        print(f" {'N='+str(n)+' Acc':>10}", end="")
    print(f" {'Lift/Rand':>10} {'Lift/Best':>10}")
    print(f"  {'-'*75}")

    for method, label in [("calibrator", "Calibrator (p_correct)"),
                          ("verbalized", "Verbalized Conf."),
                          ("length", "Longest Response"),
                          ("random", "Random"),
                          ("majority_vote", "Majority Vote"),
                          ("oracle", "Oracle"),
                          ("best_individual", "Best Single Model")]:
        row = f"  {label:<25}"
        for n in sorted(results_by_n.keys()):
            v = results_by_n[n].get(method)
            ci = results_by_n[n].get(f"{method}_ci")
            if v is not None:
                ci_str = ""
                if ci:
                    ci_str = f" [{ci[0]:.3f},{ci[1]:.3f}]"
                row += f" {v:>10.4f}"
            else:
                row += f" {'N/A':>10}"
        # Lift columns for largest N
        largest_n = max(results_by_n.keys())
        cal = results_by_n[largest_n].get("calibrator", 0)
        rand = results_by_n[largest_n].get("random", 0)
        best_ind = results_by_n[largest_n].get("best_individual", 0)
        if method == "calibrator":
            row += f" {cal-rand:>+10.4f} {cal-best_ind:>+10.4f}"
        print(row)

    # Print CIs
    for n in sorted(results_by_n.keys()):
        ci = results_by_n[n].get("calibrator_ci")
        if ci:
            print(f"\n  N={n} Calibrator 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    # Disagreement
    for n, disagree in disagree_by_n.items():
        print(f"\n  Disagreement N={n}: {disagree['n_questions']} questions")
        for m in ["calibrator", "verbalized", "length", "random", "oracle"]:
            v = disagree.get(m)
            if v is not None:
                print(f"    {m:<20}: {v:.4f}")

    # Per-benchmark
    bench_results = per_benchmark_breakdown(question_data, rng)
    if bench_results:
        print(f"\n--- Per-Benchmark (N=3) ---")
        print(f"  {'Benchmark':<20} {'N':>5} {'Cal':>8} {'Rand':>8} {'Oracle':>8} {'Lift':>8}")
        print(f"  {'-'*55}")
        for b in sorted(bench_results, key=lambda b: -bench_results[b]["n_questions"]):
            br = bench_results[b]
            print(f"  {b:<20} {br['n_questions']:>5} {br['calibrator']:>8.3f} "
                  f"{br['random']:>8.3f} {br['oracle']:>8.3f} {br['lift_over_random']:>+8.3f}")

    # Plot
    plot_response_selection(results_by_n, disagree_by_n,
                            f"{args.fig_dir}/uc5_response_selection.pdf")

    # Save
    output = {
        "pairwise_ranking": all_pairwise,
        "selection_by_n": {str(n): v for n, v in results_by_n.items()},
        "disagreement_by_n": {str(n): v for n, v in disagree_by_n.items()},
        "per_benchmark_n3": bench_results,
        "question_counts": {"total": n_total, "n_2plus": n_2plus, "n_3": n_3},
    }
    out_path = f"{args.output_dir}/uc5_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
