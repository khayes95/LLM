#!/usr/bin/env python3
"""Best-of-N model selection and DPO pair generation for FineGRAIN.

Uses existing UQ scores to:
1. Best-of-N MODEL selection: For each prompt, pick the model whose image
   the UQ model rates most compliant. Compare accuracy vs random model selection.
2. DPO pairs: For each prompt, create (winner, loser) pairs based on UQ scores
   for potential diffusion model finetuning.
3. Failure-mode-specific routing: Route each prompt to the best model per
   failure mode category.

Uses pre-computed scores from data/finegrain_uq/scored_samples.jsonl.

Usage:
    python scripts/finegrain_bestofn_dpo.py
    python scripts/finegrain_bestofn_dpo.py --output_dir data/finegrain_uq/bestofn
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# ============================================================
# PATHS
# ============================================================
FINEGRAIN_METADATA = Path("/scratch/khayes/diff/t2i-finegrain/metadata.csv")
UQ_SCORED = Path("data/finegrain_uq/scored_samples.jsonl")
OUTPUT_DIR = Path("data/finegrain_uq/bestofn")

LABELED_MODELS = ["flux", "sd3.5_large", "sd3.5_medium", "sd3_m", "sd3_xl"]


def load_data():
    """Load metadata and UQ scores, aligned by (model, prompt_id)."""
    # Load metadata
    meta = {}
    with open(FINEGRAIN_METADATA) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["model"] not in LABELED_MODELS:
                continue
            if row.get("human_labels") in (None, "", "nan"):
                continue
            key = (row["model"], int(row["prompt_id"]))
            meta[key] = {
                "prompt_text": row["prompt_text"],
                "failure_mode": row["failure_mode"],
                "human_label": int(float(row["human_labels"])),
            }

    # Load UQ scores
    uq = {}
    with open(UQ_SCORED) as f:
        for line in f:
            s = json.loads(line)
            key = (s["model"], s["prompt_id"])
            uq[key] = s.get("p_compliant", s.get("p_correct", 0.5))

    # Group by prompt_id
    prompts = defaultdict(dict)
    for (model, pid), m in meta.items():
        if (model, pid) in uq:
            prompts[pid][model] = {
                "human_label": m["human_label"],
                "uq_score": uq[(model, pid)],
                "prompt_text": m["prompt_text"],
                "failure_mode": m["failure_mode"],
            }

    print(f"Loaded {len(prompts)} prompts with UQ scores across models")
    return prompts


def best_of_n_model_selection(prompts):
    """For each prompt, select the model with highest UQ score."""
    print("\n" + "=" * 70)
    print("EXPERIMENT: Best-of-N Model Selection")
    print("=" * 70)

    results = {
        "random_model": {"correct": 0, "total": 0},
        "best_uq": {"correct": 0, "total": 0},
        "worst_uq": {"correct": 0, "total": 0},
        "oracle": {"correct": 0, "total": 0},
    }

    per_n_results = {}  # best-of-N for N=1,2,3,4,5
    rng = np.random.RandomState(42)

    for pid, models_data in prompts.items():
        if len(models_data) < 2:
            continue

        model_list = list(models_data.keys())
        scores = {m: models_data[m]["uq_score"] for m in model_list}
        labels = {m: models_data[m]["human_label"] for m in model_list}

        # Random model selection
        random_model = rng.choice(model_list)
        results["random_model"]["correct"] += (1 - labels[random_model])
        results["random_model"]["total"] += 1

        # Best UQ score (highest P(compliant))
        best_model = max(scores, key=scores.get)
        results["best_uq"]["correct"] += (1 - labels[best_model])
        results["best_uq"]["total"] += 1

        # Worst UQ score
        worst_model = min(scores, key=scores.get)
        results["worst_uq"]["correct"] += (1 - labels[worst_model])
        results["worst_uq"]["total"] += 1

        # Oracle (best actually correct model)
        oracle_correct = any(labels[m] == 0 for m in model_list)
        results["oracle"]["correct"] += int(oracle_correct)
        results["oracle"]["total"] += 1

        # Best-of-N for various N
        for n in [1, 2, 3, 4, 5]:
            if n not in per_n_results:
                per_n_results[n] = {"correct": 0, "total": 0}
            if n >= len(model_list):
                selected = model_list
            else:
                selected = list(rng.choice(model_list, n, replace=False))
            best_in_n = max(selected, key=lambda m: scores[m])
            per_n_results[n]["correct"] += (1 - labels[best_in_n])
            per_n_results[n]["total"] += 1

    print(f"\nModel Selection Results (N={results['random_model']['total']} prompts):")
    print(f"  Random model:     {results['random_model']['correct']}/{results['random_model']['total']} "
          f"= {results['random_model']['correct']/results['random_model']['total']:.3f}")
    print(f"  Best by UQ:       {results['best_uq']['correct']}/{results['best_uq']['total']} "
          f"= {results['best_uq']['correct']/results['best_uq']['total']:.3f}")
    print(f"  Worst by UQ:      {results['worst_uq']['correct']}/{results['worst_uq']['total']} "
          f"= {results['worst_uq']['correct']/results['worst_uq']['total']:.3f}")
    print(f"  Oracle:           {results['oracle']['correct']}/{results['oracle']['total']} "
          f"= {results['oracle']['correct']/results['oracle']['total']:.3f}")

    print(f"\n  Best-of-N accuracy (selecting highest UQ from N random models):")
    for n in sorted(per_n_results):
        r = per_n_results[n]
        print(f"    N={n}: {r['correct']}/{r['total']} = {r['correct']/r['total']:.3f}")

    # UQ score gap between best and worst
    uq_gaps = []
    label_gaps = []
    for pid, models_data in prompts.items():
        if len(models_data) < 2:
            continue
        scores = [models_data[m]["uq_score"] for m in models_data]
        labels_list = [models_data[m]["human_label"] for m in models_data]
        uq_gaps.append(max(scores) - min(scores))
        label_gaps.append(sum(l == 0 for l in labels_list) / len(labels_list))

    print(f"\n  UQ score gap (best - worst model): mean={np.mean(uq_gaps):.3f}, "
          f"std={np.std(uq_gaps):.3f}")
    print(f"  Fraction of models compliant per prompt: mean={np.mean(label_gaps):.3f}")

    results["per_n"] = {str(n): {
        "accuracy": r["correct"] / r["total"],
        "correct": r["correct"],
        "total": r["total"],
    } for n, r in per_n_results.items()}

    return results


def failure_mode_routing(prompts):
    """Route each prompt to the best model per failure mode."""
    print("\n" + "=" * 70)
    print("EXPERIMENT: Failure-Mode-Specific Routing")
    print("=" * 70)

    # Group by failure mode
    fm_data = defaultdict(lambda: defaultdict(list))

    for pid, models_data in prompts.items():
        for model, data in models_data.items():
            fm = data["failure_mode"]
            fm_data[fm][model].append({
                "pid": pid,
                "label": data["human_label"],
                "score": data["uq_score"],
            })

    results = {}
    for fm in sorted(fm_data):
        model_scores = {}
        for model in LABELED_MODELS:
            if model in fm_data[fm]:
                items = fm_data[fm][model]
                # Mean UQ compliance score and actual compliance rate
                mean_score = np.mean([i["score"] for i in items])
                actual_rate = np.mean([1 - i["label"] for i in items])
                model_scores[model] = {"mean_uq": mean_score, "actual_compliance": actual_rate, "n": len(items)}

        if len(model_scores) < 2:
            continue

        # Which model does UQ recommend?
        uq_best = max(model_scores, key=lambda m: model_scores[m]["mean_uq"])
        actual_best = max(model_scores, key=lambda m: model_scores[m]["actual_compliance"])

        results[fm] = {
            "uq_recommended": uq_best,
            "actually_best": actual_best,
            "correct_recommendation": uq_best == actual_best,
            "per_model": model_scores,
        }

        match = "✓" if uq_best == actual_best else "✗"
        print(f"  {match} {fm}:")
        print(f"      UQ picks: {uq_best} ({model_scores[uq_best]['mean_uq']:.3f})")
        print(f"      Actual best: {actual_best} ({model_scores[actual_best]['actual_compliance']:.3f})")

    correct = sum(1 for r in results.values() if r["correct_recommendation"])
    total = len(results)
    print(f"\n  Routing accuracy: {correct}/{total} = {correct/total:.3f}")

    # Spearman correlation across all (model, fm) pairs
    uq_scores = []
    actual_scores = []
    for fm, r in results.items():
        for model, ms in r["per_model"].items():
            uq_scores.append(ms["mean_uq"])
            actual_scores.append(ms["actual_compliance"])
    rho, p = spearmanr(uq_scores, actual_scores)
    print(f"  Spearman rho (UQ vs actual compliance): {rho:.3f} (p={p:.2e})")

    results["_summary"] = {
        "routing_accuracy": correct / total,
        "n_failure_modes": total,
        "spearman_rho": float(rho),
        "spearman_p": float(p),
    }

    return results


def generate_dpo_pairs(prompts):
    """Generate DPO preference pairs from UQ scores."""
    print("\n" + "=" * 70)
    print("EXPERIMENT: DPO Pair Generation")
    print("=" * 70)

    pairs = []
    for pid, models_data in prompts.items():
        if len(models_data) < 2:
            continue

        model_list = list(models_data.keys())
        # All pairwise comparisons
        for i in range(len(model_list)):
            for j in range(i + 1, len(model_list)):
                m_i, m_j = model_list[i], model_list[j]
                s_i = models_data[m_i]["uq_score"]
                s_j = models_data[m_j]["uq_score"]
                l_i = models_data[m_i]["human_label"]
                l_j = models_data[m_j]["human_label"]

                # Skip if UQ scores are too close (uninformative)
                if abs(s_i - s_j) < 0.05:
                    continue

                # Winner = higher UQ score (more likely compliant)
                if s_i > s_j:
                    winner, loser = m_i, m_j
                    w_label, l_label = l_i, l_j
                    w_score, l_score = s_i, s_j
                else:
                    winner, loser = m_j, m_i
                    w_label, l_label = l_j, l_i
                    w_score, l_score = s_j, s_i

                pairs.append({
                    "prompt_id": pid,
                    "prompt_text": models_data[m_i]["prompt_text"],
                    "failure_mode": models_data[m_i]["failure_mode"],
                    "winner_model": winner,
                    "loser_model": loser,
                    "winner_uq_score": float(w_score),
                    "loser_uq_score": float(l_score),
                    "winner_actually_correct": int(1 - w_label),
                    "loser_actually_correct": int(1 - l_label),
                    "uq_gap": float(w_score - l_score),
                })

    # Analyze pair quality
    n_correct_winner = sum(1 for p in pairs if p["winner_actually_correct"] == 1)
    n_correct_loser = sum(1 for p in pairs if p["loser_actually_correct"] == 1)
    n_informative = sum(1 for p in pairs
                        if p["winner_actually_correct"] != p["loser_actually_correct"])
    n_winner_truly_better = sum(1 for p in pairs
                                if p["winner_actually_correct"] == 1 and p["loser_actually_correct"] == 0)

    print(f"  Total pairs generated: {len(pairs)}")
    print(f"  Winner actually correct: {n_correct_winner}/{len(pairs)} = {n_correct_winner/len(pairs):.3f}")
    print(f"  Loser actually correct: {n_correct_loser}/{len(pairs)} = {n_correct_loser/len(pairs):.3f}")
    print(f"  Informative pairs (different labels): {n_informative}/{len(pairs)} = {n_informative/len(pairs):.3f}")
    print(f"  Winner truly better: {n_winner_truly_better}/{n_informative} "
          f"= {n_winner_truly_better/n_informative:.3f}" if n_informative > 0 else "  No informative pairs")
    print(f"  Mean UQ gap: {np.mean([p['uq_gap'] for p in pairs]):.3f}")

    results = {
        "n_pairs": len(pairs),
        "winner_correct_rate": n_correct_winner / len(pairs),
        "loser_correct_rate": n_correct_loser / len(pairs),
        "informative_pair_rate": n_informative / len(pairs),
        "precision_on_informative": n_winner_truly_better / n_informative if n_informative > 0 else 0,
        "mean_uq_gap": float(np.mean([p["uq_gap"] for p in pairs])),
    }

    return results, pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_data()

    # Best-of-N
    bon_results = best_of_n_model_selection(prompts)

    # Failure-mode routing
    routing_results = failure_mode_routing(prompts)

    # DPO pairs
    dpo_results, dpo_pairs = generate_dpo_pairs(prompts)

    # Save
    all_results = {
        "best_of_n": bon_results,
        "routing": routing_results,
        "dpo": dpo_results,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(output_dir / "dpo_pairs.jsonl", "w") as f:
        for p in dpo_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"\n{'=' * 70}")
    print(f"Results saved to {output_dir / 'results.json'}")
    print(f"DPO pairs saved to {output_dir / 'dpo_pairs.jsonl'} ({len(dpo_pairs)} pairs)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
