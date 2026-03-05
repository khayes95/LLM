#!/usr/bin/env python3
"""Proxy Semantic Entropy + Cross-Model Self-Consistency baselines.

Two baselines for comparison against our trained UQ calibrator:

1. Cross-model self-consistency (CPU only):
   For questions answered by multiple target models, use inter-model
   agreement as a confidence proxy.

2. Proxy semantic entropy (1 GPU):
   Generate N responses per question from Qwen3-VL-8B-Instruct (a small
   proxy model), cluster by answer similarity, compute entropy.
   Low entropy = high confidence.

Usage:
    # CPU only: cross-model self-consistency
    python scripts/semantic_entropy_baseline.py --stage self_consistency

    # GPU: proxy semantic entropy (text-only questions)
    python scripts/semantic_entropy_baseline.py --stage proxy_se --gpu 0

    # Both
    python scripts/semantic_entropy_baseline.py --stage all --gpu 0

    # Smoke test
    python scripts/semantic_entropy_baseline.py --stage all --gpu 0 --smoke_test
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss


# ── helpers ──────────────────────────────────────────────────────────────────

def extract_question_text(input_data) -> str:
    """Extract question text from prediction input field."""
    if isinstance(input_data, str):
        return input_data
    if isinstance(input_data, list):
        # messages format — concatenate user messages
        parts = []
        for msg in input_data:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append(item.get("text", ""))
        return "\n".join(parts) if parts else str(input_data)[:2000]
    if isinstance(input_data, dict):
        for key in ["question", "query", "query_cot", "prompt", "text"]:
            if key in input_data and input_data[key]:
                val = input_data[key]
                if isinstance(val, str):
                    return val
        if "messages" in input_data:
            return extract_question_text(input_data["messages"])
        clean = {k: v for k, v in input_data.items() if k != "images"}
        return json.dumps(clean)[:2000]
    return str(input_data)[:2000]


def normalize_answer(text: str) -> str:
    """Normalize an answer for comparison."""
    text = text.strip().lower()
    # Remove common prefixes
    for prefix in ["the answer is", "answer:", "final answer:", "therefore,"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    # Remove punctuation at end
    text = text.rstrip(".,;:!?")
    return text.strip()


def extract_short_answer(response: str) -> str:
    """Extract a short answer from a model response for clustering."""
    text = response.strip()

    # Check for boxed math answer
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return normalize_answer(boxed[-1])

    # Check for "The answer is X" pattern
    match = re.search(r"(?:the answer is|answer:)\s*(.+?)(?:\.|$)", text, re.I)
    if match:
        return normalize_answer(match.group(1))

    # Check for MCQ letter pattern (A), (B), etc.
    match = re.search(r"\b([A-H])\)?(?:\s|$|\.|,)", text[-200:])
    if match:
        return match.group(1).upper()

    # Check for standalone number at end
    match = re.search(r"(\d+(?:\.\d+)?)\s*$", text[-100:])
    if match:
        return match.group(1)

    # Fall back to last line
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        return normalize_answer(lines[-1][:200])
    return normalize_answer(text[:200])


def compute_semantic_entropy(responses: list[str]) -> float:
    """Compute entropy over clustered responses."""
    answers = [extract_short_answer(r) for r in responses]
    counts = Counter(answers)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    entropy = -sum(p * np.log(p + 1e-12) for p in probs)
    return entropy


def compute_self_consistency_confidence(responses: list[str]) -> float:
    """Fraction of responses that agree with the majority answer."""
    answers = [extract_short_answer(r) for r in responses]
    counts = Counter(answers)
    majority_count = counts.most_common(1)[0][1]
    return majority_count / len(answers)


def ece_score(y_true, y_pred, n_bins=10):
    """Expected Calibration Error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = y_pred[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += mask.sum() / len(y_true) * abs(avg_conf - avg_acc)
    return ece


def compute_metrics(y_true, scores, label=""):
    """Compute AUROC, Brier, ECE for a set of scores."""
    y_true = np.array(y_true)

    # Replace None with NaN
    scores = np.array([s if s is not None else np.nan for s in scores], dtype=float)
    valid = ~(np.isnan(scores) | np.isinf(scores))
    if valid.sum() < 10:
        return {"auroc": None, "n": int(valid.sum()), "label": label}

    y_true = y_true[valid]
    scores = scores[valid]

    if len(np.unique(y_true)) < 2:
        return {"auroc": None, "n": len(y_true), "label": label}

    auroc = roc_auc_score(y_true, scores)

    result = {
        "auroc": round(auroc, 4),
        "n": len(y_true),
        "label": label,
        "pos_rate": round(y_true.mean(), 4),
    }

    if scores.min() >= 0 and scores.max() <= 1:
        result["brier"] = round(brier_score_loss(y_true, scores), 4)
        result["ece"] = round(ece_score(y_true, scores), 4)

    return result


# ── Data loading ─────────────────────────────────────────────────────────────

def load_full_questions(runs_dir: str = "runs") -> dict:
    """Load full question text from predictions.jsonl files.

    Returns: {id: {"question": str, "response": str, "correct": int, "benchmark": str}}
    """
    runs_path = Path(runs_dir)
    question_map = {}

    pred_files = list(runs_path.rglob("predictions.jsonl"))
    print(f"Found {len(pred_files)} prediction files")

    for pf in pred_files:
        parent = pf.parent.name
        # Determine target model from directory name
        target_model = None
        for m in ["gpt-5-mini", "gpt5mini", "gpt52", "gpt-5.2", "qwen35", "qwen3.5"]:
            if m in parent.lower():
                target_model = m
                break
        if not target_model:
            continue

        # Normalize model name
        model_map = {
            "gpt-5-mini": "gpt5mini", "gpt5mini": "gpt5mini",
            "gpt52": "gpt52", "gpt-5.2": "gpt52",
            "qwen35": "qwen35", "qwen3.5": "qwen35",
        }
        target_model = model_map.get(target_model, target_model)

        try:
            with open(pf) as f:
                for line in f:
                    try:
                        pred = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    pid = pred.get("id", "")
                    if not pid:
                        continue

                    question = extract_question_text(pred.get("input", ""))
                    response = pred.get("response_text", "") or pred.get("response", "")

                    # Get correctness
                    score = pred.get("score", {})
                    if isinstance(score, dict):
                        correct = score.get("correct")
                    else:
                        correct = pred.get("correct")

                    # Determine benchmark
                    benchmark = pred.get("benchmark", "")
                    if not benchmark:
                        # Try to infer from ID
                        for bname in ["bbeh", "simpleqa", "gpqa", "hle", "healthbench",
                                      "math", "gsm8k", "mmlu", "arc", "drop", "triviaqa",
                                      "omnimath", "livebench", "longbench", "multichallenge",
                                      "oolong", "prbench", "tutorbench", "chembench",
                                      "babilong", "mmmu", "charxiv", "hallusionbench",
                                      "mathvista", "mathverse", "mathvision", "mmstar",
                                      "mmvet", "realworldqa", "vizwiz", "vsr", "aokvqa",
                                      "erqa"]:
                            if pid.startswith(bname):
                                benchmark = bname
                                break

                    # Check for image in input (VLM indicator)
                    has_image = False
                    inp = pred.get("input", {})
                    if isinstance(inp, dict):
                        has_image = "images" in inp or "image" in inp
                    elif isinstance(inp, list):
                        for msg in inp:
                            if isinstance(msg, dict):
                                content = msg.get("content", "")
                                if isinstance(content, list):
                                    for item in content:
                                        if isinstance(item, dict) and item.get("type") == "image":
                                            has_image = True

                    key = f"{target_model}_{pid}"
                    if key not in question_map or len(question) > len(question_map[key].get("question", "")):
                        question_map[key] = {
                            "question": question,
                            "response": response,
                            "correct": correct,
                            "benchmark": benchmark,
                            "target_model": target_model,
                            "has_image": has_image,
                            "id": pid,
                        }
        except Exception as e:
            print(f"  Warning: error reading {pf}: {e}")

    print(f"Loaded {len(question_map)} question entries from runs/")
    return question_map


def load_scored_test_only(scored_dir: str = "data/use_cases/scored_test_only") -> list:
    """Load scored test-only data."""
    samples = []
    scored_path = Path(scored_dir)
    for f in sorted(scored_path.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                    samples.append(d)
                except json.JSONDecodeError:
                    continue
    print(f"Loaded {len(samples)} scored test-only samples")
    return samples


# ── Stage 1: Cross-model self-consistency ────────────────────────────────────

def run_self_consistency(scored_samples: list, output_dir: str):
    """Cross-model self-consistency baseline.

    For questions answered by multiple models, use agreement as confidence.
    """
    print("\n" + "=" * 60)
    print("STAGE 1: Cross-Model Self-Consistency")
    print("=" * 60)

    # Group by question ID (strip model prefix)
    by_question = defaultdict(list)
    for s in scored_samples:
        qid = s["id"]
        by_question[qid].append(s)

    # Find questions with 2+ model responses
    multi_model = {qid: responses for qid, responses in by_question.items()
                   if len(responses) >= 2}
    three_model = {qid: responses for qid, responses in by_question.items()
                   if len(responses) >= 3}

    print(f"Questions with 2+ models: {len(multi_model)}")
    print(f"Questions with 3 models: {len(three_model)}")

    if not multi_model:
        print("No multi-model questions found. Skipping.")
        return

    # For each response, compute "agreement" with other models' responses
    # We use the correctness agreement: if 2/3 models got it right, probably easier
    y_true_all = []
    agreement_scores = []
    calibrator_scores = []
    verbalized_scores = []
    sample_ids = []

    for qid, responses in multi_model.items():
        n_models = len(responses)
        correctness = [r["is_correct"] for r in responses]
        n_correct = sum(correctness)

        for r in responses:
            # Agreement: fraction of OTHER models that also got it correct/incorrect
            others_correct = (n_correct - r["is_correct"]) / (n_models - 1)
            # If this response is correct, agreement = others_correct
            # If this response is incorrect, agreement = 1 - others_correct
            # But we don't know correctness at inference time.
            # So agreement = fraction of models agreeing (all correct or all incorrect)
            # Actually: the proxy signal is "most models agree on correctness direction"
            # Since we can't see correctness, we use p_correct agreement instead

            # Use calibrator p_correct as proxy for agreement
            p_values = [x["p_correct"] for x in responses]
            # Agreement: average of other models' p_correct
            other_ps = [p for i, p in enumerate(p_values) if i != responses.index(r)]
            mean_other_p = np.mean(other_ps)

            # The "agreement signal" is: are calibrator scores consistent?
            # If all models have high p_correct → likely easy question → high confidence
            # If scores disagree → uncertain

            # Method 1: Mean p_correct across models (model-level consensus)
            mean_all_p = np.mean(p_values)

            # Method 2: Std of p_correct (low std = high consensus)
            std_p = np.std(p_values)

            # Method 3: Min p_correct (conservative)
            min_p = np.min(p_values)

            y_true_all.append(r["is_correct"])
            agreement_scores.append({
                "mean_p": mean_all_p,
                "std_p": std_p,
                "min_p": min_p,
                "this_p": r["p_correct"],
                "mean_other_p": mean_other_p,
            })
            calibrator_scores.append(r["p_correct"])
            verbalized_scores.append(r.get("verbalized_confidence", 0.5))
            sample_ids.append(f"{r['target_model']}_{qid}")

    y_true = np.array(y_true_all)
    print(f"\nTotal samples in multi-model questions: {len(y_true)}")
    print(f"Positive rate: {y_true.mean():.3f}")

    # Compute metrics for each approach
    results = {}

    # Calibrator baseline (our model, per-response)
    results["calibrator"] = compute_metrics(y_true, calibrator_scores, "Calibrator (per-response)")
    print(f"Calibrator AUROC: {results['calibrator']['auroc']}")

    # Verbalized confidence
    results["verbalized"] = compute_metrics(y_true, verbalized_scores, "Verbalized confidence")
    print(f"Verbalized AUROC: {results['verbalized']['auroc']}")

    # Agreement baselines
    mean_ps = [s["mean_p"] for s in agreement_scores]
    results["mean_p_consensus"] = compute_metrics(y_true, mean_ps, "Mean P(correct) consensus")
    print(f"Mean P consensus AUROC: {results['mean_p_consensus']['auroc']}")

    neg_stds = [-s["std_p"] for s in agreement_scores]
    results["low_std_consensus"] = compute_metrics(y_true, neg_stds, "Low-std consensus (neg std)")
    print(f"Low-std consensus AUROC: {results['low_std_consensus']['auroc']}")

    min_ps = [s["min_p"] for s in agreement_scores]
    results["min_p_consensus"] = compute_metrics(y_true, min_ps, "Min P(correct) consensus")
    print(f"Min P consensus AUROC: {results['min_p_consensus']['auroc']}")

    # Combined: calibrator + consensus
    combined = [0.7 * s["this_p"] + 0.3 * s["mean_p"] for s in agreement_scores]
    results["combined_cal_consensus"] = compute_metrics(y_true, combined, "0.7*cal + 0.3*consensus")
    print(f"Combined AUROC: {results['combined_cal_consensus']['auroc']}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "self_consistency_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    return results


# ── Stage 2: Proxy Semantic Entropy ──────────────────────────────────────────

def run_proxy_se(scored_samples: list, question_map: dict, output_dir: str,
                 gpu: int = 0, n_samples: int = 5, max_tokens: int = 512,
                 smoke_test: bool = False):
    """Proxy semantic entropy using Qwen3-VL-8B-Instruct."""
    print("\n" + "=" * 60)
    print("STAGE 2: Proxy Semantic Entropy (Qwen3-VL-8B)")
    print("=" * 60)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Match scored samples to full question text, text-only
    matched = []
    for s in scored_samples:
        if s.get("has_image"):
            continue  # Skip VLM for now

        # Try to find full question in question_map
        key = f"{s['target_model']}_{s['id']}"
        if key in question_map:
            q = question_map[key]["question"]
            if len(q) > 50:  # Ensure it's a real question, not truncated
                matched.append({
                    "id": s["id"],
                    "target_model": s["target_model"],
                    "benchmark": s["benchmark"],
                    "is_correct": s["is_correct"],
                    "p_correct": s["p_correct"],
                    "verbalized_confidence": s.get("verbalized_confidence", 0.5),
                    "question": q,
                })

    print(f"Text-only scored samples: {sum(1 for s in scored_samples if not s.get('has_image'))}")
    print(f"Matched with full question text: {len(matched)}")

    if smoke_test:
        matched = matched[:10]
        print(f"Smoke test: using {len(matched)} samples")

    min_samples = 5 if smoke_test else 20
    if len(matched) < min_samples:
        print(f"Too few matched samples ({len(matched)} < {min_samples}). Cannot run proxy SE.")
        return None

    # Load model via vLLM
    print(f"\nLoading Qwen3-VL-8B-Instruct via vLLM on GPU {gpu}...")
    t0 = time.time()

    from vllm import LLM, SamplingParams

    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    print(f"Model loaded in {time.time() - t0:.0f}s")

    # Build prompts — just the question, asking the model to answer
    # Truncate to ~2000 chars to stay under 8192 token limit with responses
    system_prompt = "Answer the following question concisely. Give your final answer clearly."
    prompts = []
    skipped = 0
    kept_matched = []
    for m in matched:
        q = m["question"][:2000]
        prompt = f"{system_prompt}\n\nQuestion: {q}\n\nAnswer:"
        # Rough token estimate: 1 token per 3.5 chars
        est_tokens = len(prompt) / 3.5
        if est_tokens > 7000:
            skipped += 1
            continue
        prompts.append(prompt)
        kept_matched.append(m)
    matched = kept_matched
    if skipped:
        print(f"Skipped {skipped} prompts that were too long")

    # Generate N responses per question
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=max_tokens,
        n=n_samples,
    )

    print(f"\nGenerating {n_samples} responses for {len(prompts)} questions...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"Generation done in {elapsed:.0f}s ({len(prompts) * n_samples / elapsed:.1f} responses/s)")

    # Compute semantic entropy for each question
    results_per_sample = []
    for i, (m, output) in enumerate(zip(matched, outputs)):
        responses = [o.text for o in output.outputs]
        se = compute_semantic_entropy(responses)
        sc = compute_self_consistency_confidence(responses)

        results_per_sample.append({
            "id": m["id"],
            "target_model": m["target_model"],
            "benchmark": m["benchmark"],
            "is_correct": m["is_correct"],
            "p_correct": m["p_correct"],
            "verbalized_confidence": m["verbalized_confidence"],
            "semantic_entropy": se,
            "self_consistency": sc,
            "n_unique_answers": len(set(extract_short_answer(r) for r in responses)),
            "answers": [extract_short_answer(r) for r in responses],
        })

    # Compute aggregate metrics
    y_true = [r["is_correct"] for r in results_per_sample]

    # Semantic entropy: LOW entropy → confident → more likely correct
    # So we use NEGATIVE entropy as the "confidence" score for AUROC
    neg_entropy = [-r["semantic_entropy"] for r in results_per_sample]
    se_metrics = compute_metrics(y_true, neg_entropy, "Proxy Semantic Entropy")
    print(f"\nProxy SE AUROC: {se_metrics['auroc']}")

    # Self-consistency: HIGH agreement → confident
    sc_scores = [r["self_consistency"] for r in results_per_sample]
    sc_metrics = compute_metrics(y_true, sc_scores, "Proxy Self-Consistency (N=5)")
    print(f"Proxy Self-Consistency AUROC: {sc_metrics['auroc']}")

    # Calibrator (our model) on the same subset
    cal_scores = [r["p_correct"] for r in results_per_sample]
    cal_metrics = compute_metrics(y_true, cal_scores, "Calibrator (same subset)")
    print(f"Calibrator AUROC (same subset): {cal_metrics['auroc']}")

    # Verbalized confidence
    verb_scores = [r["verbalized_confidence"] for r in results_per_sample]
    verb_metrics = compute_metrics(y_true, verb_scores, "Verbalized (same subset)")
    print(f"Verbalized AUROC (same subset): {verb_metrics['auroc']}")

    # Combined: calibrator + SE
    combined_1 = [r["p_correct"] + 0.2 * (-r["semantic_entropy"]) for r in results_per_sample]
    comb1_metrics = compute_metrics(y_true, combined_1, "Calibrator + 0.2*neg_SE")
    print(f"Combined (cal + SE) AUROC: {comb1_metrics['auroc']}")

    # Summary statistics
    entropies = [r["semantic_entropy"] for r in results_per_sample]
    correct_ent = [r["semantic_entropy"] for r in results_per_sample if r["is_correct"]]
    incorrect_ent = [r["semantic_entropy"] for r in results_per_sample if not r["is_correct"]]

    summary = {
        "n_samples": len(results_per_sample),
        "n_text_only": len(results_per_sample),
        "pos_rate": round(np.mean(y_true), 4),
        "proxy_model": "Qwen3-VL-8B-Instruct",
        "n_generations": n_samples,
        "temperature": 0.7,
        "generation_time_s": round(elapsed, 1),
        "entropy_stats": {
            "mean": round(np.mean(entropies), 4),
            "std": round(np.std(entropies), 4),
            "correct_mean": round(np.mean(correct_ent), 4) if correct_ent else None,
            "incorrect_mean": round(np.mean(incorrect_ent), 4) if incorrect_ent else None,
        },
        "metrics": {
            "proxy_se": se_metrics,
            "proxy_self_consistency": sc_metrics,
            "calibrator": cal_metrics,
            "verbalized": verb_metrics,
            "combined_cal_se": comb1_metrics,
        },
    }

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "proxy_se_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Save per-sample results
    detail_path = os.path.join(output_dir, "proxy_se_per_sample.jsonl")
    with open(detail_path, "w") as f:
        for r in results_per_sample:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved summary to {out_path}")
    print(f"Saved per-sample to {detail_path}")

    # Print nice comparison table
    print("\n" + "=" * 60)
    print("COMPARISON (text-only subset)")
    print("=" * 60)
    print(f"{'Method':<40} {'AUROC':>8}")
    print("-" * 50)
    for name, m in summary["metrics"].items():
        auroc = m["auroc"] if m["auroc"] is not None else "N/A"
        if isinstance(auroc, float):
            auroc = f"{auroc:.4f}"
        print(f"{m['label']:<40} {auroc:>8}")

    return summary


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Semantic Entropy + Self-Consistency baselines")
    parser.add_argument("--stage", choices=["self_consistency", "proxy_se", "all"],
                        default="all")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--scored_dir", default="data/use_cases/scored_test_only")
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--output_dir", default="data/ablations/semantic_entropy")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of generations per question for SE")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    print(f"Stage: {args.stage}")
    print(f"Scored dir: {args.scored_dir}")
    print(f"Output dir: {args.output_dir}")
    if args.smoke_test:
        print("SMOKE TEST MODE")

    # Load scored data
    scored = load_scored_test_only(args.scored_dir)

    if args.stage in ["self_consistency", "all"]:
        run_self_consistency(scored, args.output_dir)

    if args.stage in ["proxy_se", "all"]:
        # Load full questions from prediction files
        question_map = load_full_questions(args.runs_dir)
        run_proxy_se(
            scored, question_map, args.output_dir,
            gpu=args.gpu, n_samples=args.n_samples,
            max_tokens=args.max_tokens, smoke_test=args.smoke_test,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
