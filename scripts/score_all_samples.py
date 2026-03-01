#!/usr/bin/env python3
"""Score all predictions with the UQ calibrator and save per-sample P(correct).

This is the prerequisite for all 7 use case experiments. It runs the calibrator
on every prediction and saves per-sample scores to JSONL.

Usage:
    # Smoke test (5 samples/benchmark, ~2 min)
    CUDA_VISIBLE_DEVICES=0 python scripts/score_all_samples.py --target gpt5mini --smoke_test

    # Full run (~30-45 min per target on 1 GPU)
    CUDA_VISIBLE_DEVICES=0 python scripts/score_all_samples.py --target gpt5mini
    CUDA_VISIBLE_DEVICES=1 python scripts/score_all_samples.py --target gpt52
    CUDA_VISIBLE_DEVICES=2 python scripts/score_all_samples.py --target qwen35
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_CALIBRATOR_PATH = "uq_models/text_calibrator_v3"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Model-specific calibrators (trained on each model's own data)
MODEL_SPECIFIC_CALIBRATORS = {
    "gpt5mini": "uq_models/text_calibrator_v3",          # trained on GPT-5-mini
    "gpt52": "uq_models/text_calibrator_gpt52",          # trained on GPT-5.2
    "qwen35": "uq_models/text_calibrator_qwen35",        # trained on Qwen3.5
}

EXCLUDED_BENCHMARKS = {
    "triviaqa", "babilong", "vsr", "aokvqa", "erqa",
    "tutorbench", "healthbench",
}

PROMPT_TEMPLATE = """Question: {question}

Answer: {response}

Is the answer correct? (i) No (ii) Yes"""

TARGET_CONFIGS = {
    "gpt5mini": {
        "name": "GPT-5-mini",
        "data_dir": "runs/gpt5_mini_combined",
        "mode": "combined",  # subdirectories per benchmark
    },
    "gpt52": {
        "name": "GPT-5.2 (high reasoning)",
        "prefix": "gpt52_high_",
        "mode": "prefixed",
    },
    "qwen35": {
        "name": "Qwen3.5-397B-A17B-FP8",
        "prefix": "qwen35_397b_",
        "mode": "prefixed",
    },
}


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


def load_training_ids(train_path="data/finetune/train_v2.jsonl") -> set:
    """Load IDs from training data to exclude from in-distribution scoring."""
    ids = set()
    if not os.path.exists(train_path):
        return ids
    with open(train_path) as f:
        for line in f:
            try:
                s = json.loads(line)
                ids.add(str(s.get("id", "")))
            except json.JSONDecodeError:
                continue
    return ids


def load_samples_combined(data_dir: str, max_per_benchmark=None, exclude_ids=None):
    """Load predictions from combined directory (subdirs per benchmark)."""
    samples = []
    stats = {}
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"WARNING: {data_dir} does not exist")
        return samples, stats

    for bench_dir in sorted(data_path.iterdir()):
        if not bench_dir.is_dir():
            continue
        benchmark = bench_dir.name
        if benchmark in EXCLUDED_BENCHMARKS:
            continue

        pred_file = bench_dir / "predictions.jsonl"
        if not pred_file.exists() or pred_file.stat().st_size == 0:
            continue

        bench_samples = []
        n_excluded = 0
        with open(pred_file) as f:
            for line in f:
                try:
                    pred = json.loads(line)
                except json.JSONDecodeError:
                    continue

                sid = str(pred.get("id", ""))

                # Skip training samples to avoid leakage
                if exclude_ids and sid in exclude_ids:
                    n_excluded += 1
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

                # Extract verbalized confidence and token usage
                prediction = pred.get("prediction") or {}
                usage = pred.get("usage") or {}

                bench_samples.append({
                    "id": sid,
                    "benchmark": benchmark,
                    "question": question[:2000],
                    "response": response[:1000],
                    "is_correct": int(correct == 1),
                    "verbalized_confidence": prediction.get("confidence"),
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                })

        if max_per_benchmark and len(bench_samples) > max_per_benchmark:
            np.random.seed(42)
            indices = np.random.choice(len(bench_samples), max_per_benchmark, replace=False)
            bench_samples = [bench_samples[i] for i in sorted(indices)]

        n_correct = sum(s["is_correct"] for s in bench_samples)
        stats[benchmark] = {
            "used": len(bench_samples), "correct": n_correct,
            "excluded_training": n_excluded,
        }
        print(f"  {benchmark}: {len(bench_samples)} samples "
              f"({n_correct} correct, {n_excluded} excluded as training)")
        samples.extend(bench_samples)

    return samples, stats


def load_samples_prefixed(prefix: str, runs_dir="runs", max_per_benchmark=None):
    """Load predictions from runs matching a prefix."""
    samples = []
    stats = {}
    runs_path = Path(runs_dir)

    for run_dir in sorted(runs_path.iterdir()):
        if not run_dir.name.startswith(prefix):
            continue
        benchmark = run_dir.name[len(prefix):]
        if benchmark in EXCLUDED_BENCHMARKS:
            continue
        if "backup" in run_dir.name:
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

                prediction = pred.get("prediction") or {}
                usage = pred.get("usage") or {}

                bench_samples.append({
                    "id": str(pred.get("id", "")),
                    "benchmark": benchmark,
                    "question": question[:2000],
                    "response": response[:1000],
                    "is_correct": int(correct == 1),
                    "verbalized_confidence": prediction.get("confidence") if isinstance(prediction, dict) else None,
                    "input_tokens": usage.get("input_tokens") if isinstance(usage, dict) else None,
                    "output_tokens": usage.get("output_tokens") if isinstance(usage, dict) else None,
                    "total_tokens": usage.get("total_tokens") if isinstance(usage, dict) else None,
                })

        if max_per_benchmark and len(bench_samples) > max_per_benchmark:
            np.random.seed(42)
            indices = np.random.choice(len(bench_samples), max_per_benchmark, replace=False)
            bench_samples = [bench_samples[i] for i in sorted(indices)]

        n_correct = sum(s["is_correct"] for s in bench_samples)
        stats[benchmark] = {"used": len(bench_samples), "correct": n_correct}
        print(f"  {benchmark}: {len(bench_samples)} samples ({n_correct} correct)")
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


def main():
    parser = argparse.ArgumentParser(description="Score all samples with UQ calibrator")
    parser.add_argument("--target", choices=list(TARGET_CONFIGS.keys()),
                        required=True, help="Which model's predictions to score")
    parser.add_argument("--calibrator", default=None,
                        help="Path to calibrator checkpoint (default: model-specific)")
    parser.add_argument("--model_specific", action="store_true", default=True,
                        help="Use model-specific calibrator for each target (default: True)")
    parser.add_argument("--output_dir", default="data/use_cases/scored",
                        help="Output directory for scored JSONL")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Score 5 samples per benchmark only")
    parser.add_argument("--max_per_benchmark", type=int, default=None)
    args = parser.parse_args()

    if args.smoke_test:
        args.max_per_benchmark = 5

    # Resolve calibrator path
    if args.calibrator is None:
        if args.model_specific and args.target in MODEL_SPECIFIC_CALIBRATORS:
            args.calibrator = MODEL_SPECIFIC_CALIBRATORS[args.target]
        else:
            args.calibrator = DEFAULT_CALIBRATOR_PATH

    config = TARGET_CONFIGS[args.target]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.target}_scored.jsonl"

    print("=" * 70)
    print(f"SCORING: {config['name']} predictions with UQ calibrator")
    print("=" * 70)
    print(f"Calibrator: {args.calibrator}")
    print(f"Output: {output_path}")
    if args.smoke_test:
        print("MODE: Smoke test (5 samples/benchmark)")
    print()

    # Load samples
    print("Loading predictions...")
    if config["mode"] == "combined":
        exclude_ids = load_training_ids() if args.target == "gpt5mini" else None
        if exclude_ids:
            print(f"Excluding {len(exclude_ids)} training IDs to prevent leakage")
        samples, stats = load_samples_combined(
            config["data_dir"],
            max_per_benchmark=args.max_per_benchmark,
            exclude_ids=exclude_ids,
        )
    else:
        samples, stats = load_samples_prefixed(
            config["prefix"],
            max_per_benchmark=args.max_per_benchmark,
        )

    if not samples:
        print("ERROR: No samples loaded!")
        sys.exit(1)

    n_correct = sum(s["is_correct"] for s in samples)
    print(f"\nTotal: {len(samples)} samples ({n_correct} correct, "
          f"{len(samples) - n_correct} incorrect)")

    # Load calibrator
    print(f"\nLoading calibrator from {args.calibrator}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.calibrator)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.calibrator)
    model.eval()
    print(f"Model loaded on {device}")

    # Score all samples, saving incrementally
    print(f"\nScoring {len(samples)} samples...")
    start_time = time.time()
    all_preds = []
    all_labels = []

    with open(output_path, "w") as f_out:
        for i, sample in enumerate(samples):
            try:
                p_correct = get_p_correct(
                    model, tokenizer,
                    sample["question"], sample["response"], device
                )
            except Exception as e:
                print(f"  Error on sample {i} ({sample['id']}): {e}")
                p_correct = 0.5

            sample["p_correct"] = p_correct
            sample["target_model"] = args.target
            all_preds.append(p_correct)
            all_labels.append(sample["is_correct"])

            # Write immediately (incremental save)
            # Don't write full question/response to keep file manageable
            out_record = {
                "id": sample["id"],
                "benchmark": sample["benchmark"],
                "target_model": args.target,
                "is_correct": sample["is_correct"],
                "p_correct": round(p_correct, 6),
                "verbalized_confidence": sample.get("verbalized_confidence"),
                "input_tokens": sample.get("input_tokens"),
                "output_tokens": sample.get("output_tokens"),
                "total_tokens": sample.get("total_tokens"),
                "question_preview": sample["question"][:200],
                "response_preview": sample["response"][:200],
            }
            f_out.write(json.dumps(out_record) + "\n")

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(samples) - i - 1) / rate
                interim_auroc = "N/A"
                if len(set(all_labels)) > 1:
                    interim_auroc = f"{roc_auc_score(all_labels, all_preds):.4f}"
                print(f"  [{i+1}/{len(samples)}] "
                      f"{rate:.1f} samples/s, ETA {eta/60:.1f}m, "
                      f"interim AUROC={interim_auroc}")
                f_out.flush()

    elapsed = time.time() - start_time

    # Final metrics
    print(f"\n{'='*70}")
    print("SCORING COMPLETE")
    print(f"{'='*70}")
    print(f"Scored {len(samples)} samples in {elapsed:.0f}s ({len(samples)/elapsed:.1f}/s)")
    print(f"Output: {output_path}")

    if len(set(all_labels)) > 1:
        auroc = roc_auc_score(all_labels, all_preds)
        print(f"AUROC: {auroc:.4f}")
        print(f"Mean P(correct): {np.mean(all_preds):.4f}")
        print(f"Accuracy: {np.mean(all_labels):.4f}")

        # Per-benchmark AUROC
        bench_data = defaultdict(lambda: {"preds": [], "labels": []})
        for sample, p, l in zip(samples, all_preds, all_labels):
            bench_data[sample["benchmark"]]["preds"].append(p)
            bench_data[sample["benchmark"]]["labels"].append(l)

        print(f"\nPer-benchmark AUROC:")
        for bench in sorted(bench_data.keys()):
            bd = bench_data[bench]
            if len(set(bd["labels"])) > 1:
                ba = roc_auc_score(bd["labels"], bd["preds"])
                print(f"  {bench:<20} AUROC={ba:.3f} (n={len(bd['labels'])})")
            else:
                print(f"  {bench:<20} single class (n={len(bd['labels'])})")

    # Save summary
    summary = {
        "target_model": args.target,
        "target_name": config["name"],
        "calibrator": args.calibrator,
        "n_samples": len(samples),
        "n_correct": int(sum(all_labels)),
        "auroc": float(roc_auc_score(all_labels, all_preds)) if len(set(all_labels)) > 1 else None,
        "mean_p_correct": float(np.mean(all_preds)),
        "elapsed_seconds": elapsed,
        "data_stats": stats,
    }
    summary_path = output_dir / f"{args.target}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
