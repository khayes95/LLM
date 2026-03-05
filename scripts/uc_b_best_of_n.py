#!/usr/bin/env python3
"""UC-B: Best-of-N Verification via Calibrator — Stage 1 (Cross-Model Selection).

For questions answered by multiple models, simulates "Best-of-N" selection:
pick the response with the highest calibrator p_correct.  Tests whether the
calibrator can act as an inference-time verifier by selecting the best response
among N candidates from different models.

Analyses:
1. Cross-Model Selection (N=2 and N=3)
   - Calibrator selection (highest p_correct)
   - Verbalized selection (highest verbalized_confidence)
   - Length selection (longest response, max output_tokens)
   - Random selection (uniform random, averaged over 1000 trials)
   - Oracle selection (pick a correct response if any exist)
   - Individual model accuracy
   - Majority vote (majority of N correct => "correct")

2. Conditional Analysis
   - When models disagree: accuracy of each method on disagreement questions
   - Confidence threshold: coverage vs accuracy tradeoff for selected response

3. Per-Benchmark Breakdown for N=3 questions

4. Accuracy Gain Analysis
   - Lift over best single model
   - Lift over random selection
   - Oracle gap (room for improvement)

5. Summary Table: Method | N=2 accuracy | N=3 accuracy | Lift over random | Lift over best model

Outputs:
    {output_dir}/uc_b_results.json — all metrics and tables
    {fig_dir}/uc_b_selection_accuracy.pdf — bar chart: methods x N
    {fig_dir}/uc_b_disagreement.pdf — accuracy on disagreement questions only

Usage:
    python scripts/uc_b_best_of_n.py
    python scripts/uc_b_best_of_n.py --scored_dir data/use_cases/scored_test_only_v2
    python scripts/uc_b_best_of_n.py --smoke_test
"""
import argparse
import json
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
# Question grouping
# ---------------------------------------------------------------------------

def build_question_index(all_scored):
    """Index samples by (benchmark, id) -> {model: sample}.

    Returns a dict where keys are (benchmark, question_id) tuples and values
    are dicts mapping model name to the scored sample dict.
    """
    question_data = defaultdict(dict)
    for target, samples in all_scored.items():
        for s in samples:
            key = (s["benchmark"], s["id"])
            question_data[key][target] = s
    return question_data


# ---------------------------------------------------------------------------
# Selection methods
# ---------------------------------------------------------------------------

def select_by_calibrator(responses):
    """Pick the response with the highest p_correct."""
    best = max(responses, key=lambda r: r["p_correct"])
    return best


def select_by_verbalized(responses):
    """Pick the response with the highest verbalized_confidence.
    Returns None if no response has a valid verbalized_confidence.
    """
    valid = [r for r in responses if r["verbalized_confidence"] is not None]
    if not valid:
        return None
    return max(valid, key=lambda r: r["verbalized_confidence"])


def select_by_length(responses):
    """Pick the response with the most output_tokens."""
    return max(responses, key=lambda r: r["output_tokens"])


def oracle_select(responses):
    """Pick a correct response if any exist, otherwise pick any."""
    correct = [r for r in responses if r["is_correct"] == 1]
    if correct:
        return correct[0]
    return responses[0]


def majority_vote(responses):
    """Return 1 if the majority of responses are correct, else 0."""
    n_correct = sum(r["is_correct"] for r in responses)
    return 1 if n_correct > len(responses) / 2 else 0


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def compute_selection_accuracy(question_data, n_models, rng, n_random_trials=1000):
    """Compute selection accuracy for each method on questions with exactly
    n_models model responses.

    Returns dict of {method_name: accuracy} and supporting metadata.
    """
    # Filter to questions with exactly n_models responses
    questions = {k: v for k, v in question_data.items() if len(v) == n_models}
    n_questions = len(questions)

    if n_questions == 0:
        return None

    # Collect all model names that appear
    all_models = set()
    for models_dict in questions.values():
        all_models.update(models_dict.keys())

    # --- Calibrator selection ---
    cal_correct = 0
    for (bench, qid), models_dict in questions.items():
        responses = list(models_dict.values())
        selected = select_by_calibrator(responses)
        cal_correct += selected["is_correct"]
    cal_acc = cal_correct / n_questions

    # --- Verbalized selection ---
    verb_total = 0
    verb_correct = 0
    for (bench, qid), models_dict in questions.items():
        responses = list(models_dict.values())
        selected = select_by_verbalized(responses)
        if selected is not None:
            verb_total += 1
            verb_correct += selected["is_correct"]
    verb_acc = verb_correct / verb_total if verb_total > 0 else None

    # --- Length selection ---
    len_correct = 0
    for (bench, qid), models_dict in questions.items():
        responses = list(models_dict.values())
        selected = select_by_length(responses)
        len_correct += selected["is_correct"]
    len_acc = len_correct / n_questions

    # --- Random selection (average over trials) ---
    random_corrects = []
    for _ in range(n_random_trials):
        trial_correct = 0
        for (bench, qid), models_dict in questions.items():
            responses = list(models_dict.values())
            idx = rng.randint(0, len(responses))
            trial_correct += responses[idx]["is_correct"]
        random_corrects.append(trial_correct / n_questions)
    rand_acc = float(np.mean(random_corrects))

    # --- Oracle selection ---
    oracle_correct = 0
    for (bench, qid), models_dict in questions.items():
        responses = list(models_dict.values())
        selected = oracle_select(responses)
        oracle_correct += selected["is_correct"]
    oracle_acc = oracle_correct / n_questions

    # --- Individual model accuracy ---
    model_accuracies = {}
    for model_name in sorted(all_models):
        model_correct = 0
        model_total = 0
        for (bench, qid), models_dict in questions.items():
            if model_name in models_dict:
                model_total += 1
                model_correct += models_dict[model_name]["is_correct"]
        if model_total > 0:
            model_accuracies[model_name] = model_correct / model_total

    best_individual_acc = max(model_accuracies.values()) if model_accuracies else 0.0

    # --- Majority vote ---
    maj_correct = 0
    for (bench, qid), models_dict in questions.items():
        responses = list(models_dict.values())
        maj_correct += majority_vote(responses)
    maj_acc = maj_correct / n_questions

    results = {
        "n_questions": n_questions,
        "n_models": n_models,
        "calibrator": cal_acc,
        "verbalized": verb_acc,
        "verbalized_n": verb_total,
        "length": len_acc,
        "random": rand_acc,
        "oracle": oracle_acc,
        "majority_vote": maj_acc,
        "individual_models": model_accuracies,
        "best_individual": best_individual_acc,
        "best_individual_model": max(model_accuracies, key=model_accuracies.get) if model_accuracies else None,
    }
    return results


def compute_disagreement_accuracy(question_data, n_models, rng, n_random_trials=1000):
    """Same as compute_selection_accuracy but only on questions where models
    disagree (not all correct and not all incorrect).
    """
    # Filter to questions with exactly n_models and disagreement
    questions = {}
    for k, v in question_data.items():
        if len(v) != n_models:
            continue
        responses = list(v.values())
        correctness = [r["is_correct"] for r in responses]
        if not (all(c == 1 for c in correctness) or all(c == 0 for c in correctness)):
            questions[k] = v

    n_questions = len(questions)
    if n_questions == 0:
        return None

    all_models = set()
    for models_dict in questions.values():
        all_models.update(models_dict.keys())

    # Calibrator
    cal_correct = 0
    for (bench, qid), models_dict in questions.items():
        responses = list(models_dict.values())
        selected = select_by_calibrator(responses)
        cal_correct += selected["is_correct"]
    cal_acc = cal_correct / n_questions

    # Verbalized
    verb_total = 0
    verb_correct = 0
    for (bench, qid), models_dict in questions.items():
        responses = list(models_dict.values())
        selected = select_by_verbalized(responses)
        if selected is not None:
            verb_total += 1
            verb_correct += selected["is_correct"]
    verb_acc = verb_correct / verb_total if verb_total > 0 else None

    # Length
    len_correct = 0
    for (bench, qid), models_dict in questions.items():
        responses = list(models_dict.values())
        selected = select_by_length(responses)
        len_correct += selected["is_correct"]
    len_acc = len_correct / n_questions

    # Random
    random_corrects = []
    for _ in range(n_random_trials):
        trial_correct = 0
        for (bench, qid), models_dict in questions.items():
            responses = list(models_dict.values())
            idx = rng.randint(0, len(responses))
            trial_correct += responses[idx]["is_correct"]
        random_corrects.append(trial_correct / n_questions)
    rand_acc = float(np.mean(random_corrects))

    # Oracle
    oracle_correct = 0
    for (bench, qid), models_dict in questions.items():
        responses = list(models_dict.values())
        selected = oracle_select(responses)
        oracle_correct += selected["is_correct"]
    oracle_acc = oracle_correct / n_questions

    # Majority vote
    maj_correct = 0
    for (bench, qid), models_dict in questions.items():
        responses = list(models_dict.values())
        maj_correct += majority_vote(responses)
    maj_acc = maj_correct / n_questions

    # Individual models
    model_accuracies = {}
    for model_name in sorted(all_models):
        model_correct = 0
        model_total = 0
        for (bench, qid), models_dict in questions.items():
            if model_name in models_dict:
                model_total += 1
                model_correct += models_dict[model_name]["is_correct"]
        if model_total > 0:
            model_accuracies[model_name] = model_correct / model_total

    best_individual_acc = max(model_accuracies.values()) if model_accuracies else 0.0

    return {
        "n_questions": n_questions,
        "n_models": n_models,
        "calibrator": cal_acc,
        "verbalized": verb_acc,
        "verbalized_n": verb_total,
        "length": len_acc,
        "random": rand_acc,
        "oracle": oracle_acc,
        "majority_vote": maj_acc,
        "individual_models": model_accuracies,
        "best_individual": best_individual_acc,
    }


def compute_confidence_threshold_curve(question_data, n_models, n_points=50):
    """Sweep a threshold on max p_correct for the selected response.

    Only "answer" if the calibrator's best p_correct exceeds the threshold;
    otherwise abstain.  Measures coverage (fraction answered) vs accuracy
    of the selected response.
    """
    questions = {k: v for k, v in question_data.items() if len(v) == n_models}
    if not questions:
        return []

    thresholds = np.linspace(0.0, 1.0, n_points + 1)
    curve = []
    for t in thresholds:
        n_answered = 0
        n_correct = 0
        for (bench, qid), models_dict in questions.items():
            responses = list(models_dict.values())
            best = max(responses, key=lambda r: r["p_correct"])
            if best["p_correct"] >= t:
                n_answered += 1
                n_correct += best["is_correct"]
        coverage = n_answered / len(questions) if questions else 0.0
        accuracy = n_correct / n_answered if n_answered > 0 else None
        curve.append({
            "threshold": float(t),
            "coverage": float(coverage),
            "accuracy": float(accuracy) if accuracy is not None else None,
            "n_answered": n_answered,
        })
    return curve


def per_benchmark_breakdown(question_data, rng):
    """For N=3 questions, compute selection accuracy per benchmark."""
    # Filter to questions with 3 models
    questions_n3 = {k: v for k, v in question_data.items() if len(v) == 3}
    if not questions_n3:
        return {}

    # Group by benchmark
    by_bench = defaultdict(dict)
    for (bench, qid), models_dict in questions_n3.items():
        by_bench[bench][(bench, qid)] = models_dict

    results = {}
    for bench in sorted(by_bench.keys()):
        bench_questions = by_bench[bench]
        n_q = len(bench_questions)
        if n_q == 0:
            continue

        # Calibrator
        cal_correct = 0
        for (b, qid), models_dict in bench_questions.items():
            responses = list(models_dict.values())
            selected = select_by_calibrator(responses)
            cal_correct += selected["is_correct"]
        cal_acc = cal_correct / n_q

        # Random (1000 trials)
        random_corrects = []
        for _ in range(1000):
            trial_correct = 0
            for (b, qid), models_dict in bench_questions.items():
                responses = list(models_dict.values())
                idx = rng.randint(0, len(responses))
                trial_correct += responses[idx]["is_correct"]
            random_corrects.append(trial_correct / n_q)
        rand_acc = float(np.mean(random_corrects))

        # Oracle
        oracle_correct = 0
        for (b, qid), models_dict in bench_questions.items():
            responses = list(models_dict.values())
            oracle_correct += oracle_select(responses)["is_correct"]
        oracle_acc = oracle_correct / n_q

        # Individual model accuracies
        all_models = set()
        for models_dict in bench_questions.values():
            all_models.update(models_dict.keys())
        model_accs = {}
        for model_name in sorted(all_models):
            mc = 0
            mt = 0
            for (b, qid), models_dict in bench_questions.items():
                if model_name in models_dict:
                    mt += 1
                    mc += models_dict[model_name]["is_correct"]
            if mt > 0:
                model_accs[model_name] = mc / mt
        best_ind = max(model_accs.values()) if model_accs else 0.0

        results[bench] = {
            "n_questions": n_q,
            "calibrator": cal_acc,
            "random": rand_acc,
            "oracle": oracle_acc,
            "best_individual": best_ind,
            "individual_models": model_accs,
            "lift_over_random": cal_acc - rand_acc,
            "lift_over_best_model": cal_acc - best_ind,
            "oracle_gap": oracle_acc - cal_acc,
        }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_selection_accuracy(results_by_n, output_path):
    """Bar chart comparing selection methods for N=2 and N=3."""
    methods_order = ["calibrator", "verbalized", "length", "random",
                     "majority_vote", "oracle"]
    method_labels = {
        "calibrator": "Calibrator\n(max p_correct)",
        "verbalized": "Verbalized\nConfidence",
        "length": "Longest\nResponse",
        "random": "Random",
        "majority_vote": "Majority\nVote",
        "oracle": "Oracle",
    }
    colors = {
        "calibrator": "C0",
        "verbalized": "C1",
        "length": "C2",
        "random": "gray",
        "majority_vote": "C3",
        "oracle": "C4",
    }

    n_values = sorted(results_by_n.keys())
    n_groups = len(n_values)
    present_methods = [m for m in methods_order
                       if any(results_by_n[n].get(m) is not None for n in n_values)]
    n_methods = len(present_methods)

    fig, ax = plt.subplots(figsize=(max(10, n_methods * 1.5), 6))
    bar_width = 0.8 / n_groups
    x = np.arange(n_methods)

    for i, n in enumerate(n_values):
        vals = []
        for m in present_methods:
            v = results_by_n[n].get(m)
            vals.append(v if v is not None else 0.0)
        offset = (i - (n_groups - 1) / 2) * bar_width
        bars = ax.bar(x + offset, vals, bar_width,
                      label=f"N={n} ({results_by_n[n].get('n_questions', '?')} Qs)",
                      alpha=0.85, edgecolor="black", linewidth=0.5)
        # Value labels
        for xi, v in zip(x, vals):
            if v > 0:
                ax.text(xi + offset, v + 0.008, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8, rotation=0)

    # Also plot best individual model accuracy as horizontal lines
    for i, n in enumerate(n_values):
        best_ind = results_by_n[n].get("best_individual", 0)
        if best_ind > 0:
            ax.axhline(y=best_ind, color=f"C{5 + i}", linestyle=":",
                       alpha=0.6, linewidth=1.2,
                       label=f"Best single model (N={n}): {best_ind:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels([method_labels.get(m, m) for m in present_methods], fontsize=9)
    ax.set_ylabel("Selection Accuracy", fontsize=12)
    ax.set_title("UC-B: Best-of-N Selection Accuracy by Method", fontsize=14)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, 1.08)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {output_path}")


def plot_disagreement(disagree_by_n, output_path):
    """Bar chart showing accuracy on disagreement questions only."""
    methods_order = ["calibrator", "verbalized", "length", "random",
                     "majority_vote", "oracle"]
    method_labels = {
        "calibrator": "Calibrator\n(max p_correct)",
        "verbalized": "Verbalized\nConfidence",
        "length": "Longest\nResponse",
        "random": "Random",
        "majority_vote": "Majority\nVote",
        "oracle": "Oracle",
    }

    n_values = sorted(disagree_by_n.keys())
    n_groups = len(n_values)
    present_methods = [m for m in methods_order
                       if any(disagree_by_n[n].get(m) is not None for n in n_values)]
    n_methods = len(present_methods)

    if n_methods == 0:
        print("  No disagreement data to plot")
        return

    fig, ax = plt.subplots(figsize=(max(10, n_methods * 1.5), 6))
    bar_width = 0.8 / n_groups
    x = np.arange(n_methods)

    for i, n in enumerate(n_values):
        vals = []
        for m in present_methods:
            v = disagree_by_n[n].get(m)
            vals.append(v if v is not None else 0.0)
        offset = (i - (n_groups - 1) / 2) * bar_width
        bars = ax.bar(x + offset, vals, bar_width,
                      label=f"N={n} ({disagree_by_n[n].get('n_questions', '?')} disagree Qs)",
                      alpha=0.85, edgecolor="black", linewidth=0.5)
        for xi, v in zip(x, vals):
            if v > 0:
                ax.text(xi + offset, v + 0.008, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8)

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([method_labels.get(m, m) for m in present_methods], fontsize=9)
    ax.set_ylabel("Selection Accuracy", fontsize=12)
    ax.set_title("UC-B: Selection Accuracy on Disagreement Questions Only", fontsize=14)
    ax.legend(fontsize=9, loc="lower right")
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
        description="UC-B: Best-of-N Verification via Calibrator "
                    "(Stage 1 — Cross-Model Selection, CPU-only)")
    parser.add_argument("--scored_dir", default="data/use_cases/scored_test_only_v2",
                        help="Directory with scored JSONL files")
    parser.add_argument("--output_dir", default="data/use_cases/results_test_only_v2",
                        help="Directory for results JSON")
    parser.add_argument("--fig_dir", default="figures/use_cases_unified",
                        help="Directory for output figures")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Only use first 50 samples per model")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    targets = ["gpt5mini", "gpt52", "qwen35"]
    target_names = {
        "gpt5mini": "GPT-5-mini",
        "gpt52": "GPT-5.2",
        "qwen35": "Qwen3.5",
    }
    all_scored = {}
    for target in targets:
        path = Path(args.scored_dir) / f"{target}_scored.jsonl"
        if not path.exists():
            print(f"  Skipping {target}: {path} not found")
            continue
        samples = load_scored(path)
        if args.smoke_test:
            samples = samples[:50]
        all_scored[target] = samples
        print(f"  {target}: {len(samples)} samples loaded")

    if len(all_scored) < 2:
        print("ERROR: Need at least 2 scored targets for Best-of-N selection.")
        return

    print()
    print("=" * 70)
    print("UC-B: Best-of-N Verification via Calibrator")
    print("  Stage 1 — Cross-Model Selection (CPU-only)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Build question index
    # ------------------------------------------------------------------
    question_data = build_question_index(all_scored)
    n_total = len(question_data)
    n_2plus = sum(1 for v in question_data.values() if len(v) >= 2)
    n_exactly2 = sum(1 for v in question_data.values() if len(v) == 2)
    n_exactly3 = sum(1 for v in question_data.values() if len(v) == 3)

    print(f"\nTotal unique questions: {n_total}")
    print(f"  Questions with 1 model:  {n_total - n_2plus}")
    print(f"  Questions with 2 models: {n_exactly2}")
    print(f"  Questions with 3 models: {n_exactly3}")

    rng = np.random.RandomState(42)

    # ------------------------------------------------------------------
    # 1. Cross-Model Selection (N=2 and N=3)
    # ------------------------------------------------------------------
    results_by_n = {}
    disagree_by_n = {}

    for n_models in [2, 3]:
        # Use questions with >= n_models; for "exactly N" we use ==
        # For N=2, include questions with 2+ models (use all pairs);
        # actually the spec says "N models", so use exactly N.
        # But to maximize data, for N=2 we also include the 3-model
        # questions (pick best of any 2 out of 3). For simplicity,
        # we use questions with exactly n_models responses.

        # However, for N=2 it makes more sense to use questions with
        # exactly 2 models AND questions with 3 models (treating them
        # as having 3 models is N=3). Let's use >= n_models for each N.
        # Actually, the cleanest interpretation: N=2 means we pick from
        # all questions with >=2 models (using all responses), and N=3
        # from questions with ==3 models. But then N=2 accuracy would
        # be a mix. Let's use "exactly N" for clarity.

        sel_results = compute_selection_accuracy(question_data, n_models, rng)
        if sel_results is None:
            print(f"\n  No questions with exactly {n_models} models. Skipping N={n_models}.")
            continue

        results_by_n[n_models] = sel_results
        print(f"\n--- N={n_models}: {sel_results['n_questions']} questions ---")
        print(f"  {'Method':<25} {'Accuracy':>10}")
        print(f"  {'-'*38}")

        method_display = [
            ("calibrator", "Calibrator (max p_correct)"),
            ("verbalized", "Verbalized Confidence"),
            ("length", "Longest Response"),
            ("random", "Random"),
            ("majority_vote", "Majority Vote"),
            ("oracle", "Oracle"),
        ]
        for key, label in method_display:
            val = sel_results.get(key)
            if val is not None:
                n_note = ""
                if key == "verbalized" and sel_results.get("verbalized_n"):
                    n_note = f" (n={sel_results['verbalized_n']})"
                print(f"  {label:<25} {val:>10.4f}{n_note}")
            else:
                print(f"  {label:<25} {'N/A':>10}")

        print(f"\n  Individual model accuracies:")
        for model_name, acc in sorted(sel_results["individual_models"].items()):
            print(f"    {target_names.get(model_name, model_name):<20} {acc:.4f}")
        print(f"  Best individual: {sel_results['best_individual']:.4f} "
              f"({target_names.get(sel_results['best_individual_model'], sel_results['best_individual_model'])})")

        # Lifts
        cal_acc = sel_results["calibrator"]
        rand_acc = sel_results["random"]
        best_ind = sel_results["best_individual"]
        oracle_acc = sel_results["oracle"]

        print(f"\n  Accuracy gain analysis:")
        print(f"    Lift over random:       {cal_acc - rand_acc:+.4f}")
        print(f"    Lift over best model:   {cal_acc - best_ind:+.4f}")
        print(f"    Oracle gap:             {oracle_acc - cal_acc:+.4f}")

        # Disagreement analysis
        disagree = compute_disagreement_accuracy(question_data, n_models, rng)
        if disagree is not None:
            disagree_by_n[n_models] = disagree
            print(f"\n  Disagreement questions (N={n_models}): {disagree['n_questions']}")
            print(f"  {'Method':<25} {'Accuracy':>10}")
            print(f"  {'-'*38}")
            for key, label in method_display:
                val = disagree.get(key)
                if val is not None:
                    print(f"  {label:<25} {val:>10.4f}")
                else:
                    print(f"  {label:<25} {'N/A':>10}")
        else:
            print(f"\n  No disagreement questions for N={n_models}")

    if not results_by_n:
        print("\nERROR: No valid N=2 or N=3 groups found. Check that questions overlap across models.")
        return

    # ------------------------------------------------------------------
    # 2. Confidence threshold curve (N=3 if available, else N=2)
    # ------------------------------------------------------------------
    best_n = max(results_by_n.keys())
    conf_curve = compute_confidence_threshold_curve(question_data, best_n)

    if conf_curve:
        print(f"\n--- Confidence Threshold Curve (N={best_n}) ---")
        print(f"  {'Threshold':>10} {'Coverage':>10} {'Accuracy':>10} {'N answered':>12}")
        print(f"  {'-'*45}")
        # Show a subset of thresholds
        show_thresholds = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for point in conf_curve:
            if any(abs(point["threshold"] - t) < 0.015 for t in show_thresholds):
                acc_str = f"{point['accuracy']:.4f}" if point["accuracy"] is not None else "N/A"
                print(f"  {point['threshold']:>10.2f} {point['coverage']:>10.4f} "
                      f"{acc_str:>10} {point['n_answered']:>12}")

    # ------------------------------------------------------------------
    # 3. Per-Benchmark Breakdown (N=3)
    # ------------------------------------------------------------------
    bench_results = per_benchmark_breakdown(question_data, rng)

    if bench_results:
        print(f"\n--- Per-Benchmark Breakdown (N=3) ---")
        print(f"  {'Benchmark':<22} {'N Qs':>6} {'Calibrator':>10} {'Random':>8} "
              f"{'Best Model':>10} {'Lift/Rand':>10} {'Lift/Best':>10} {'Oracle Gap':>10}")
        print(f"  {'-'*90}")
        for bench in sorted(bench_results.keys(), key=lambda b: -bench_results[b]["n_questions"]):
            br = bench_results[bench]
            print(f"  {bench:<22} {br['n_questions']:>6} {br['calibrator']:>10.4f} "
                  f"{br['random']:>8.4f} {br['best_individual']:>10.4f} "
                  f"{br['lift_over_random']:>+10.4f} {br['lift_over_best_model']:>+10.4f} "
                  f"{br['oracle_gap']:>+10.4f}")

    # ------------------------------------------------------------------
    # 4. Summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("UC-B Summary: Best-of-N Verification via Calibrator")
    print(f"{'='*70}")

    header = f"  {'Method':<25}"
    for n in sorted(results_by_n.keys()):
        header += f" {'N=' + str(n) + ' Acc':>10}"
    # Add lift columns for the largest N
    header += f" {'Lift/Rand':>10} {'Lift/Best':>10}"
    print(header)
    print(f"  {'-'*len(header)}")

    method_display = [
        ("calibrator", "Calibrator (p_correct)"),
        ("verbalized", "Verbalized Confidence"),
        ("length", "Longest Response"),
        ("random", "Random"),
        ("majority_vote", "Majority Vote"),
        ("oracle", "Oracle"),
        ("best_individual", "Best Single Model"),
    ]

    for key, label in method_display:
        row = f"  {label:<25}"
        for n in sorted(results_by_n.keys()):
            val = results_by_n[n].get(key)
            if val is not None:
                row += f" {val:>10.4f}"
            else:
                row += f" {'N/A':>10}"
        # Lift columns (use largest N)
        largest_n = max(results_by_n.keys())
        cal_acc = results_by_n[largest_n].get("calibrator", 0)
        rand_acc = results_by_n[largest_n].get("random", 0)
        best_ind = results_by_n[largest_n].get("best_individual", 0)
        if key == "calibrator":
            row += f" {cal_acc - rand_acc:>+10.4f} {cal_acc - best_ind:>+10.4f}"
        else:
            row += f" {'':>10} {'':>10}"
        print(row)

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    output = {
        "question_counts": {
            "total": n_total,
            "n_2plus": n_2plus,
            "n_exactly2": n_exactly2,
            "n_exactly3": n_exactly3,
        },
        "selection_results": {},
        "disagreement_results": {},
        "confidence_threshold_curve": conf_curve,
        "per_benchmark_n3": bench_results,
    }

    for n, res in results_by_n.items():
        output["selection_results"][f"N={n}"] = res
    for n, res in disagree_by_n.items():
        output["disagreement_results"][f"N={n}"] = res

    # Add summary
    summary = {}
    for n in sorted(results_by_n.keys()):
        r = results_by_n[n]
        summary[f"N={n}"] = {
            "calibrator_accuracy": r["calibrator"],
            "random_accuracy": r["random"],
            "best_individual_accuracy": r["best_individual"],
            "oracle_accuracy": r["oracle"],
            "lift_over_random": r["calibrator"] - r["random"],
            "lift_over_best_model": r["calibrator"] - r["best_individual"],
            "oracle_gap": r["oracle"] - r["calibrator"],
            "n_questions": r["n_questions"],
        }
    output["summary"] = summary

    results_path = Path(args.output_dir) / "uc_b_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    plot_selection_accuracy(results_by_n,
                            f"{args.fig_dir}/uc_b_selection_accuracy.pdf")

    if disagree_by_n:
        plot_disagreement(disagree_by_n,
                          f"{args.fig_dir}/uc_b_disagreement.pdf")

    print("\nDone.")


if __name__ == "__main__":
    main()
