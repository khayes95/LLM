#!/usr/bin/env python3
"""Cross-model evaluation of text_calibrator_v3 on GPT-5.2 or Qwen3.5 responses.

Evaluates the trained UQ calibrator (trained on GPT-5-mini responses) on
closed-source (GPT-5.2) or open-source (Qwen3.5-397B) responses.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/cross_model_eval_gpt52.py --target gpt52
    CUDA_VISIBLE_DEVICES=0 python scripts/cross_model_eval_gpt52.py --target qwen35
    CUDA_VISIBLE_DEVICES=0 python scripts/cross_model_eval_gpt52.py --target gpt52 --smoke_test
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

CALIBRATOR_PATH = "uq_models/text_calibrator_v3"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Run directories per target
TARGET_CONFIGS = {
    "gpt52": {
        "prefix": "gpt52_high_",
        "runs_dir": "runs",
        "label": "GPT-5.2 (high reasoning)",
    },
    "qwen35": {
        "prefix": "qwen35_397b_",
        "runs_dir": "runs",
        "label": "Qwen3.5-397B-A17B-FP8",
    },
}

# Skip benchmarks where grading is unreliable or not applicable
EXCLUDED_BENCHMARKS = {
    "triviaqa", "babilong", "vsr", "aokvqa", "erqa",
}

PROMPT_TEMPLATE = """Question: {question}

Answer: {response}

Is the answer correct? (i) No (ii) Yes"""


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


def load_target_samples(target: str, max_per_benchmark=None):
    """Load predictions from target model runs."""
    config = TARGET_CONFIGS[target]
    prefix = config["prefix"]
    runs_dir = Path(config["runs_dir"])

    samples = []
    stats = {}

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.name.startswith(prefix):
            continue
        benchmark = run_dir.name[len(prefix):]
        if benchmark in EXCLUDED_BENCHMARKS:
            continue

        pred_file = run_dir / "predictions.jsonl"
        if not pred_file.exists():
            continue

        bench_samples = []
        n_total = 0
        n_scored = 0

        with open(pred_file) as f:
            for line in f:
                try:
                    pred = json.loads(line)
                except json.JSONDecodeError:
                    continue

                n_total += 1

                score = pred.get("score", {})
                if isinstance(score, dict):
                    correct = score.get("correct", -1)
                else:
                    correct = score

                if correct not in (0, 1):
                    continue

                n_scored += 1
                question = extract_question_text(pred.get("input", {}))
                response = pred.get("response_text", "")
                if not response:
                    response = str(pred.get("prediction", {}).get("answer", ""))

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
            "total": n_total,
            "scored": n_scored,
            "used": len(bench_samples),
            "correct": n_correct,
            "accuracy": n_correct / len(bench_samples) if bench_samples else 0,
        }
        print(f"  {benchmark}: {len(bench_samples)} samples "
              f"({n_correct} correct, {len(bench_samples) - n_correct} incorrect)")
        samples.extend(bench_samples)

    return samples, stats


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

    results = compute_metrics(all_preds, all_labels, per_benchmark)
    return results, all_preds, all_labels


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

    # Per-benchmark
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
        }

        if len(set(bench_labels)) > 1:
            entry["auroc"] = float(roc_auc_score(bench_labels, bench_preds))
        else:
            entry["auroc"] = None
            entry["note"] = "Only one class present"

        results["per_benchmark"][bench] = entry

    return results


def main():
    parser = argparse.ArgumentParser(description="Cross-model eval on GPT-5.2/Qwen3.5")
    parser.add_argument("--target", required=True, choices=["gpt52", "qwen35"],
                        help="Target model to evaluate on")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run on 5 samples per benchmark")
    parser.add_argument("--max_per_benchmark", type=int, default=None)
    parser.add_argument("--calibrator", default=CALIBRATOR_PATH)
    parser.add_argument("--output", default=None,
                        help="Output path (default: data/cross_model/text_v3_on_<target>.json)")
    args = parser.parse_args()

    if args.smoke_test:
        args.max_per_benchmark = 5

    if args.output is None:
        args.output = f"data/cross_model/text_v3_on_{args.target}.json"

    config = TARGET_CONFIGS[args.target]

    print("=" * 70)
    print(f"CROSS-MODEL EVALUATION: text_calibrator_v3 on {config['label']}")
    print("=" * 70)
    print(f"Calibrator: {args.calibrator}")
    print(f"Target model: {config['label']}")
    print()

    # Load samples
    print(f"Loading {args.target} responses...")
    samples, data_stats = load_target_samples(
        args.target, max_per_benchmark=args.max_per_benchmark
    )
    print(f"\nTotal samples: {len(samples)}")

    if len(samples) == 0:
        print("ERROR: No samples loaded!")
        sys.exit(1)

    n_correct = sum(1 for s in samples if s["is_correct"])
    print(f"Correct: {n_correct} ({100 * n_correct / len(samples):.1f}%)")

    # Load calibrator
    os.makedirs(Path(args.output).parent, exist_ok=True)
    print(f"\nLoading calibrator from {args.calibrator}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.calibrator)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.calibrator)
    model.eval()
    print(f"Model loaded on {device}")

    # Evaluate
    print(f"\nEvaluating on {len(samples)} {config['label']} responses...")
    results, all_preds, all_labels = evaluate(model, tokenizer, samples, device)

    results["source_model"] = "GPT-5-mini"
    results["target_model"] = config["label"]
    results["calibrator"] = args.calibrator
    results["data_stats"] = data_stats

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Overall AUROC: {results['auroc']:.4f}")
    print(f"Overall AUPRC: {results['auprc']:.4f}")
    print(f"ECE: {results['ece']:.4f}")
    print(f"Brier: {results['brier']:.4f}")
    print(f"Base rate: {results['base_rate']:.2%}")
    print()
    print("Per-benchmark AUROC:")
    for bench, data in sorted(results["per_benchmark"].items(),
                               key=lambda x: x[1].get("auroc") or 0, reverse=True):
        auroc = data.get("auroc")
        if auroc is not None:
            print(f"  {bench}: {auroc:.4f} (n={data['n_samples']}, acc={data['accuracy']:.1%})")
        else:
            print(f"  {bench}: N/A ({data.get('note', '')})")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
