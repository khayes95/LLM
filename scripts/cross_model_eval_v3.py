#!/usr/bin/env python3
"""Cross-model evaluation of text_calibrator_v3.

Evaluates the trained UQ calibrator (trained on GPT-5-mini responses)
on target model responses to test cross-model transfer.

Supports multiple target models:
  1. Qwen3-VL-30B predictions (hardcoded paths)
  2. GPT-5.2 predictions (prefix: gpt52_high_*)
  3. Qwen3.5-397B predictions (prefix: qwen35_397b_*)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/cross_model_eval_v3.py --target qwen3vl
    CUDA_VISIBLE_DEVICES=0 python scripts/cross_model_eval_v3.py --target gpt52
    CUDA_VISIBLE_DEVICES=0 python scripts/cross_model_eval_v3.py --target qwen35
    CUDA_VISIBLE_DEVICES=0 python scripts/cross_model_eval_v3.py --target qwen35 --smoke_test
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# CONFIG
# ============================================================

CALIBRATOR_PATH = "uq_models/text_calibrator_v3"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
GPT5_COMBINED = "runs/gpt5_mini_combined"

# Best Qwen3-VL run directories (largest complete runs per benchmark)
QWEN3_RUNS = {
    "gpqa": "runs/20260103_170932_full_Qwen3_VL_30B_A3B_Thinking/gpqa",
    "hallusionbench": "runs/20260103_170932_full_Qwen3_VL_30B_A3B_Thinking/hallusionbench",
    "chembench": "runs/20260103_170932_full_Qwen3_VL_30B_A3B_Thinking/chembench",
    "mmmu": "runs/20260103_170933_full_Qwen3_VL_30B_A3B_Thinking/mmmu",
    "hle": "runs/20260103_170933_full_Qwen3_VL_30B_A3B_Thinking/hle",
    "mathvision": "runs/20260103_130359_full_Qwen3_VL_30B_A3B_Thinking/mathvision",
    "mathverse": "runs/20260103_130359_full_Qwen3_VL_30B_A3B_Thinking/mathverse",
    "mathvista": "runs/20260103_130359_full_Qwen3_VL_30B_A3B_Thinking/mathvista",
    "mmstar": "runs/20260103_130359_full_Qwen3_VL_30B_A3B_Thinking/mmstar",
    "realworldqa": "runs/20260103_130359_full_Qwen3_VL_30B_A3B_Thinking/realworldqa",
    "vizwiz": "runs/20260103_130359_full_Qwen3_VL_30B_A3B_Thinking/vizwiz",
    "hle_multimodal": "runs/20260103_130359_full_Qwen3_VL_30B_A3B_Thinking/hle_multimodal",
    "charxiv": "runs/20260103_214937_charxiv_Qwen3_VL_30B_A3B_Thinking/charxiv",
    "mmvet": "runs/20260103_214252_mmvet_Qwen3_VL_30B_A3B_Thinking/mmvet",
    # Overnight Feb 21 runs (missing benchmarks)
    "omnimath": "runs/qwen3vl_30b_overnight/omnimath",
    "bbeh": "runs/qwen3vl_30b_overnight/bbeh",
    "livebench": "runs/qwen3vl_30b_overnight/livebench",
    "simpleqa": "runs/qwen3vl_30b_overnight/simpleqa",
}

TARGET_NAMES = {
    "qwen3vl": "Qwen3-VL-30B-A3B-Thinking",
    "gpt52": "GPT-5.2 (high reasoning)",
    "qwen35": "Qwen3.5-397B-A17B-FP8",
}

# Benchmarks to exclude
EXCLUDED_BENCHMARKS = {
    "triviaqa", "babilong", "vsr", "aokvqa", "erqa",
    "tutorbench", "healthbench",
}

PROMPT_TEMPLATE = """Question: {question}

Answer: {response}

Is the answer correct? (i) No (ii) Yes"""


# ============================================================
# DATA LOADING
# ============================================================

def load_gpt5_questions(benchmark: str) -> dict:
    """Load question text from GPT-5-mini combined data, keyed by sample ID."""
    pred_file = Path(GPT5_COMBINED) / benchmark / "predictions.jsonl"
    if not pred_file.exists():
        return {}
    questions = {}
    with open(pred_file) as f:
        for line in f:
            try:
                pred = json.loads(line)
                sid = str(pred.get("id", ""))
                input_data = pred.get("input", {})
                question = extract_question_text(input_data)
                if question:
                    questions[sid] = question
            except json.JSONDecodeError:
                continue
    return questions


def extract_question_text(input_data) -> str:
    """Extract question text from input field."""
    if isinstance(input_data, str):
        return input_data
    if isinstance(input_data, dict):
        for key in ["question", "query", "query_cot", "prompt", "text"]:
            if key in input_data and input_data[key]:
                val = input_data[key]
                if isinstance(val, str):
                    return val
        if "messages" in input_data:
            for msg in input_data["messages"]:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        texts = [p.get("text", "") for p in content
                                 if isinstance(p, dict) and "text" in p]
                        return " ".join(texts)
        clean = {k: v for k, v in input_data.items() if k != "images"}
        return json.dumps(clean)[:1000]
    return str(input_data)[:1000]


def load_cross_model_samples(max_per_benchmark=None):
    """Load Qwen3-VL responses enriched with GPT-5-mini questions."""
    samples = []
    stats = {}

    for benchmark, run_path in sorted(QWEN3_RUNS.items()):
        if benchmark in EXCLUDED_BENCHMARKS:
            continue

        pred_file = Path(run_path) / "predictions.jsonl"
        if not pred_file.exists():
            print(f"  {benchmark}: predictions.jsonl not found at {run_path}")
            continue

        # Load GPT-5-mini questions for this benchmark
        gpt5_questions = load_gpt5_questions(benchmark)

        # Load Qwen3-VL predictions
        bench_samples = []
        n_total = 0
        n_matched = 0
        n_no_question = 0

        with open(pred_file) as f:
            for line in f:
                try:
                    pred = json.loads(line)
                except json.JSONDecodeError:
                    continue

                n_total += 1

                # Get score - handle both int and dict formats
                score = pred.get("score", -1)
                if isinstance(score, dict):
                    correct = score.get("correct", -1)
                else:
                    correct = score

                if correct not in (0, 1):
                    continue

                sid = str(pred.get("id", ""))

                # Get question from GPT-5-mini data
                question = gpt5_questions.get(sid, "")
                if not question:
                    n_no_question += 1
                    continue

                n_matched += 1
                response = pred.get("response_text", "")
                if not response:
                    response = str(pred.get("prediction", ""))

                bench_samples.append({
                    "id": sid,
                    "benchmark": benchmark,
                    "question": question[:2000],
                    "response": response[:1000],
                    "is_correct": bool(correct == 1),
                })

        if max_per_benchmark and len(bench_samples) > max_per_benchmark:
            np.random.seed(42)
            indices = np.random.choice(len(bench_samples), max_per_benchmark, replace=False)
            bench_samples = [bench_samples[i] for i in indices]

        n_correct = sum(1 for s in bench_samples if s["is_correct"])
        stats[benchmark] = {
            "total": n_total,
            "matched": n_matched,
            "no_question": n_no_question,
            "used": len(bench_samples),
            "correct": n_correct,
            "accuracy": n_correct / len(bench_samples) if bench_samples else 0,
        }
        print(f"  {benchmark}: {len(bench_samples)} samples "
              f"({n_correct} correct, {n_matched - n_correct} incorrect, "
              f"{n_no_question} unmatched)")
        samples.extend(bench_samples)

    return samples, stats


def load_prefixed_run_samples(prefix: str, runs_dir: str = "runs", max_per_benchmark=None):
    """Load predictions from runs matching a prefix (e.g. gpt52_high_, qwen35_397b_).

    These runs store input/response/score directly in predictions.jsonl.
    """
    samples = []
    stats = {}
    runs_path = Path(runs_dir)

    for run_dir in sorted(runs_path.iterdir()):
        if not run_dir.name.startswith(prefix):
            continue
        benchmark = run_dir.name[len(prefix):]
        if benchmark in EXCLUDED_BENCHMARKS:
            continue

        pred_file = run_dir / "predictions.jsonl"
        if not pred_file.exists() or pred_file.stat().st_size == 0:
            continue

        bench_samples = []
        with open(pred_file) as f:
            for line in f:
                try:
                    pred = json.loads(line)
                except json.JSONDecodeError:
                    continue

                score = pred.get("score", {})
                if isinstance(score, dict):
                    correct = score.get("correct", -1)
                else:
                    correct = score
                if correct not in (0, 1):
                    continue

                question = extract_question_text(pred.get("input", {}))
                response = pred.get("response_text", "")
                if not response:
                    response = str(pred.get("prediction", {}).get("answer", ""))
                if not question or not response:
                    continue

                bench_samples.append({
                    "id": str(pred.get("id", "")),
                    "benchmark": benchmark,
                    "question": question[:2000],
                    "response": response[:1000],
                    "is_correct": bool(correct == 1),
                })

        if max_per_benchmark and len(bench_samples) > max_per_benchmark:
            np.random.seed(42)
            indices = np.random.choice(len(bench_samples), max_per_benchmark, replace=False)
            bench_samples = [bench_samples[i] for i in indices]

        n_correct = sum(1 for s in bench_samples if s["is_correct"])
        stats[benchmark] = {
            "used": len(bench_samples),
            "correct": n_correct,
            "accuracy": n_correct / len(bench_samples) if bench_samples else 0,
        }
        print(f"  {benchmark}: {len(bench_samples)} samples ({n_correct} correct)")
        samples.extend(bench_samples)

    return samples, stats


# ============================================================
# EVALUATION
# ============================================================

def get_p_correct(model, tokenizer, question: str, response: str, device) -> float:
    """Extract P(correct) from model logits."""
    prompt = PROMPT_TEMPLATE.format(question=question[:800], response=response[:400])
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]
    token_i = tokenizer.encode("i", add_special_tokens=False)[-1]
    token_ii = tokenizer.encode("ii", add_special_tokens=False)[-1]
    probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
    return probs[1].item()


def evaluate(model, tokenizer, samples, device):
    """Run evaluation and compute metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    per_benchmark = defaultdict(lambda: {"preds": [], "labels": []})

    for i, sample in enumerate(samples):
        if i % 50 == 0:
            print(f"  Evaluating {i}/{len(samples)}...")

        try:
            p_correct = get_p_correct(
                model, tokenizer,
                sample["question"], sample["response"], device
            )
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            p_correct = 0.5

        all_preds.append(p_correct)
        all_labels.append(float(sample["is_correct"]))
        per_benchmark[sample["benchmark"]]["preds"].append(p_correct)
        per_benchmark[sample["benchmark"]]["labels"].append(float(sample["is_correct"]))

        # Save intermediate results every 200 samples
        if (i + 1) % 200 == 0:
            _save_intermediate(all_preds, all_labels, per_benchmark, i + 1)

    # Compute metrics
    results = compute_metrics(all_preds, all_labels, per_benchmark)
    return results, all_preds, all_labels


def _save_intermediate(preds, labels, per_benchmark, n_done):
    """Save intermediate results to disk."""
    try:
        interim = {
            "n_evaluated": n_done,
            "auroc": roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5,
            "n_correct": int(sum(labels)),
        }
        with open("data/cross_model/cross_model_interim.json", "w") as f:
            json.dump(interim, f, indent=2)
    except Exception:
        pass


def compute_metrics(all_preds, all_labels, per_benchmark):
    """Compute all evaluation metrics."""
    preds = np.array(all_preds)
    labels = np.array(all_labels)

    results = {
        "auroc": float(roc_auc_score(labels, preds)) if len(set(labels)) > 1 else 0.5,
        "auprc": float(average_precision_score(labels, preds)) if len(set(labels)) > 1 else 0.5,
        "n_samples": len(labels),
        "n_correct": int(sum(labels)),
        "base_rate": float(sum(labels) / len(labels)) if len(labels) > 0 else 0.5,
        "brier": float(np.mean((preds - labels) ** 2)),
    }

    # ECE
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for j in range(n_bins):
        in_bin = (preds >= bin_boundaries[j]) & (preds < bin_boundaries[j + 1])
        if in_bin.sum() == 0:
            continue
        bin_conf = preds[in_bin].mean()
        bin_acc = labels[in_bin].mean()
        ece += (in_bin.sum() / len(preds)) * abs(bin_acc - bin_conf)
    results["ece"] = float(ece)

    # Per-benchmark metrics
    results["per_benchmark"] = {}
    for bench, data in sorted(per_benchmark.items()):
        bench_preds = np.array(data["preds"])
        bench_labels = np.array(data["labels"])
        n = len(bench_labels)
        n_correct = int(sum(bench_labels))

        entry = {
            "n_samples": n,
            "n_correct": n_correct,
            "accuracy": float(n_correct / n) if n > 0 else 0,
            "mean_p_correct": float(bench_preds.mean()),
            "std_p_correct": float(bench_preds.std()),
        }

        if len(set(bench_labels)) > 1:
            entry["auroc"] = float(roc_auc_score(bench_labels, bench_preds))
        else:
            entry["auroc"] = None
            entry["note"] = "Only one class present"

        results["per_benchmark"][bench] = entry

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Cross-model evaluation of text_calibrator_v3")
    parser.add_argument("--target", choices=["qwen3vl", "gpt52", "qwen35"],
                        default="qwen3vl", help="Target model data to evaluate on")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run on 5 samples per benchmark")
    parser.add_argument("--max_per_benchmark", type=int, default=None,
                        help="Max samples per benchmark")
    parser.add_argument("--calibrator", type=str, default=CALIBRATOR_PATH,
                        help="Path to calibrator checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results (auto-generated if not set)")
    args = parser.parse_args()

    if args.smoke_test:
        args.max_per_benchmark = 5

    if args.output is None:
        args.output = f"data/cross_model/text_v3_on_{args.target}.json"

    target_name = TARGET_NAMES[args.target]

    print("=" * 70)
    print(f"CROSS-MODEL EVALUATION: text_calibrator_v3 on {target_name}")
    print("=" * 70)
    print(f"Calibrator: {args.calibrator}")
    print(f"Source model (training): GPT-5-mini")
    print(f"Target model (testing): {target_name}")
    print()

    # Load cross-model samples
    print(f"Loading {target_name} responses...")
    if args.target == "qwen3vl":
        samples, data_stats = load_cross_model_samples(
            max_per_benchmark=args.max_per_benchmark
        )
    elif args.target == "gpt52":
        samples, data_stats = load_prefixed_run_samples(
            "gpt52_high_", max_per_benchmark=args.max_per_benchmark)
    elif args.target == "qwen35":
        samples, data_stats = load_prefixed_run_samples(
            "qwen35_397b_", max_per_benchmark=args.max_per_benchmark)
    print(f"\nTotal samples: {len(samples)}")
    n_correct = sum(1 for s in samples if s["is_correct"])
    print(f"Correct: {n_correct} ({100*n_correct/len(samples):.1f}%)")
    print(f"Incorrect: {len(samples) - n_correct} ({100*(len(samples)-n_correct)/len(samples):.1f}%)")

    if len(samples) == 0:
        print("ERROR: No samples loaded!")
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load calibrator
    print(f"\nLoading calibrator from {args.calibrator}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.calibrator)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.calibrator)
    model.eval()
    print(f"Model loaded on {device}")

    # Run evaluation
    print(f"\nEvaluating on {len(samples)} Qwen3-VL responses...")
    results, all_preds, all_labels = evaluate(model, tokenizer, samples, device)

    # Add metadata
    results["source_model"] = "GPT-5-mini"
    results["target_model"] = target_name
    results["calibrator"] = args.calibrator
    results["data_stats"] = data_stats

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Print results
    print()
    print("=" * 70)
    print("CROSS-MODEL EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nOverall AUROC: {results['auroc']:.4f}")
    print(f"Overall AUPRC: {results['auprc']:.4f}")
    print(f"ECE: {results['ece']:.4f}")
    print(f"Brier: {results['brier']:.4f}")
    print(f"Base rate: {results['base_rate']:.2%}")
    print(f"N samples: {results['n_samples']}")
    print()
    print("Per-benchmark AUROC:")
    for bench, data in sorted(results["per_benchmark"].items(),
                               key=lambda x: x[1].get("auroc") or 0,
                               reverse=True):
        auroc = data.get("auroc")
        if auroc is not None:
            print(f"  {bench}: {auroc:.4f} "
                  f"(n={data['n_samples']}, {data['n_correct']} correct, "
                  f"acc={data['accuracy']:.1%})")
        else:
            print(f"  {bench}: N/A ({data.get('note', 'unknown')})")

    # Compare with in-distribution results
    indist_path = Path(args.calibrator) / "results.json"
    if indist_path.exists():
        with open(indist_path) as f:
            indist = json.load(f)
        print(f"\n{'='*70}")
        print("COMPARISON: In-Distribution vs Cross-Model")
        print(f"{'='*70}")
        print(f"  In-dist AUROC:    {indist['auroc']:.4f}")
        print(f"  Cross-model AUROC: {results['auroc']:.4f}")
        print(f"  Delta:             {results['auroc'] - indist['auroc']:+.4f}")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
