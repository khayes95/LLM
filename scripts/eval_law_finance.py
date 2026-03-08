#!/usr/bin/env python3
"""Evaluate UQ calibrator on law & finance benchmarks using Qwen3.5-0.8B as target model.

Step 1: Generate responses from Qwen3.5-0.8B on law/finance benchmarks
Step 2: Auto-grade responses (MCQ matching or exact match)
Step 3: Score with UQ calibrator (Qwen3-VL-8B + LoRA)
Step 4: Compute AUROC

Benchmarks:
  Legal:
    - MMLU professional_law (1534)
    - MMLU jurisprudence (108)
    - MMLU international_law (121)
    - LexGLUE SCOTUS (1400, 14-class classification)
  Finance:
    - MMLU professional_accounting (282)
    - MMLU econometrics (114)
    - ConvFinQA (1490, numerical QA)

Usage:
    # Smoke test (5 examples per benchmark)
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_law_finance.py --smoke_test

    # Full run
    CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_law_finance.py

    # Score only (skip generation, use existing predictions)
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_law_finance.py --score_only
"""

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
)

# ============================================================
# CONFIG
# ============================================================

TARGET_MODEL = "Qwen/Qwen3.5-0.8B"
CALIBRATOR_BASE = "Qwen/Qwen3-VL-8B-Instruct"
CALIBRATOR_CHECKPOINT = "uq_models/best_v2_r32_combined"

PROMPT_TEMPLATE = """Benchmark: {benchmark}
Source model: {source_model}

Question: {question}

Answer: {response}

Analyze whether the answer above is correct. Consider:
- Does the answer address the question?
- Are there factual errors or logical flaws?
- Is the answer complete?

Based on your analysis, is the answer correct? (i) No (ii) Yes"""

Q_LEN = 1500
R_LEN = 800

PRED_DIR = "data/law_finance/predictions"
RESULTS_DIR = "data/law_finance/results"

# LexGLUE SCOTUS labels
SCOTUS_LABELS = [
    "Criminal Procedure", "Civil Rights", "First Amendment",
    "Due Process", "Privacy", "Attorneys", "Unions",
    "Economic Activity", "Judicial Power", "Federalism",
    "Interstate Relations", "Federal Taxation", "Miscellaneous",
    "Private Action"
]

# ============================================================
# BENCHMARK DEFINITIONS
# ============================================================

BENCHMARKS = {
    # Legal
    "mmlu_professional_law": {
        "domain": "legal",
        "type": "mcq",
        "dataset": "cais/mmlu",
        "config": "professional_law",
        "split": "test",
    },
    "mmlu_jurisprudence": {
        "domain": "legal",
        "type": "mcq",
        "dataset": "cais/mmlu",
        "config": "jurisprudence",
        "split": "test",
    },
    "mmlu_international_law": {
        "domain": "legal",
        "type": "mcq",
        "dataset": "cais/mmlu",
        "config": "international_law",
        "split": "test",
    },
    "lexglue_scotus": {
        "domain": "legal",
        "type": "classification",
        "dataset": "coastalcph/lex_glue",
        "config": "scotus",
        "split": "test",
    },
    # Finance
    "mmlu_professional_accounting": {
        "domain": "finance",
        "type": "mcq",
        "dataset": "cais/mmlu",
        "config": "professional_accounting",
        "split": "test",
    },
    "mmlu_econometrics": {
        "domain": "finance",
        "type": "mcq",
        "dataset": "cais/mmlu",
        "config": "econometrics",
        "split": "test",
    },
    "convfinqa": {
        "domain": "finance",
        "type": "numeric_qa",
        "dataset": "AdaptLLM/finance-tasks",
        "config": "ConvFinQA",
        "split": "test",
    },
}


# ============================================================
# DATA LOADING
# ============================================================

def load_benchmark_data(bench_name: str, bench_cfg: dict, max_examples: int | None = None) -> list:
    """Load benchmark data and format as QA pairs."""
    ds = load_dataset(bench_cfg["dataset"], bench_cfg["config"], split=bench_cfg["split"])
    samples = []

    if bench_cfg["type"] == "mcq":
        # MMLU format
        choices_letters = ["A", "B", "C", "D"]
        for i, row in enumerate(ds):
            if max_examples and i >= max_examples:
                break
            q = row["question"]
            choices = row["choices"]
            answer_idx = row["answer"]

            # Format question with choices
            options_str = "\n".join(
                f"({choices_letters[j]}) {c}" for j, c in enumerate(choices)
            )
            formatted_q = f"{q}\n\n{options_str}"
            gold = f"({choices_letters[answer_idx]}) {choices[answer_idx]}"

            samples.append({
                "id": f"{bench_name}_{i}",
                "benchmark": bench_name,
                "question": formatted_q,
                "gold_answer": gold,
                "gold_index": answer_idx,
                "type": "mcq",
            })

    elif bench_cfg["type"] == "classification":
        # LexGLUE SCOTUS
        for i, row in enumerate(ds):
            if max_examples and i >= max_examples:
                break
            text = row["text"][:2000]  # Truncate long legal texts
            label = row["label"]
            label_str = SCOTUS_LABELS[label] if label < len(SCOTUS_LABELS) else str(label)

            options_str = "\n".join(
                f"({j+1}) {SCOTUS_LABELS[j]}" for j in range(len(SCOTUS_LABELS))
            )
            formatted_q = (
                f"Classify the following Supreme Court case into one of these issue areas:\n\n"
                f"{options_str}\n\n"
                f"Case excerpt:\n{text}\n\n"
                f"Which issue area does this case belong to?"
            )

            samples.append({
                "id": f"{bench_name}_{i}",
                "benchmark": bench_name,
                "question": formatted_q,
                "gold_answer": label_str,
                "gold_index": label,
                "type": "classification",
            })

    elif bench_cfg["type"] == "numeric_qa":
        # ConvFinQA
        for i, row in enumerate(ds):
            if max_examples and i >= max_examples:
                break
            input_text = row["input"]
            label = row["label"]

            samples.append({
                "id": f"{bench_name}_{i}",
                "benchmark": bench_name,
                "question": input_text[:3000],
                "gold_answer": str(label),
                "gold_index": -1,
                "type": "numeric_qa",
            })

    return samples


# ============================================================
# RESPONSE GENERATION
# ============================================================

def generate_responses(model, tokenizer, samples: list, batch_size: int = 1) -> list:
    """Generate responses from target model."""
    results = []
    t0 = time.time()

    for i, sample in enumerate(samples):
        q = sample["question"]

        if sample["type"] == "mcq":
            prompt = f"Answer the following multiple choice question. Give your answer as the letter choice (A, B, C, or D) followed by a brief explanation.\n\n{q}"
        elif sample["type"] == "classification":
            prompt = f"{q}\n\nAnswer with just the issue area name."
        elif sample["type"] == "numeric_qa":
            prompt = f"{q}\n\nProvide the numerical answer."
        else:
            prompt = q

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode only new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Grade response
        is_correct = grade_response(sample, response)

        result = {
            **sample,
            "response": response,
            "is_correct": int(is_correct),
        }
        results.append(result)

        if (i + 1) % 25 == 0 or (i + 1) == len(samples):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            acc = sum(r["is_correct"] for r in results) / len(results)
            print(f"    [{i+1}/{len(samples)}] {rate:.1f}/sec, acc={acc:.1%}")

    return results


def grade_response(sample: dict, response: str) -> bool:
    """Grade a response against the gold answer."""
    response_lower = response.lower().strip()

    if sample["type"] == "mcq":
        # Extract letter choice from response
        gold_idx = sample["gold_index"]
        gold_letter = ["a", "b", "c", "d"][gold_idx]

        # Look for patterns like "(A)", "A.", "A)", "Answer: A", etc.
        patterns = [
            rf'\({gold_letter}\)',
            rf'^{gold_letter}[\.\)\s:,]',
            rf'answer\s*(?:is|:)\s*\(?{gold_letter}\)?',
            rf'correct\s+answer\s*(?:is|:)\s*\(?{gold_letter}\)?',
        ]
        for pat in patterns:
            if re.search(pat, response_lower):
                return True

        # Check if response starts with the letter
        if response_lower.startswith(gold_letter) and (
            len(response_lower) == 1 or not response_lower[1].isalpha()
        ):
            return True

        return False

    elif sample["type"] == "classification":
        gold = sample["gold_answer"].lower()
        return gold in response_lower

    elif sample["type"] == "numeric_qa":
        gold = sample["gold_answer"]
        # Try to find the number in the response
        try:
            gold_num = float(gold)
        except ValueError:
            return gold.lower() in response_lower

        # Extract numbers from response
        numbers = re.findall(r'-?\d+\.?\d*', response)
        for num_str in numbers:
            try:
                num = float(num_str)
                # Allow 1% tolerance for floating point
                if abs(gold_num) < 1e-9:
                    if abs(num) < 1e-9:
                        return True
                elif abs(num - gold_num) / max(abs(gold_num), 1e-9) < 0.01:
                    return True
            except ValueError:
                continue
        return False

    return False


# ============================================================
# CALIBRATOR SCORING
# ============================================================

def score_with_calibrator(model, processor, samples: list, device,
                          source_model: str = "Qwen3.5-0.8B") -> list:
    """Score samples with the UQ calibrator."""
    t0 = time.time()

    for i, sample in enumerate(samples):
        prompt = PROMPT_TEMPLATE.format(
            question=sample["question"][:Q_LEN],
            response=sample["response"][:R_LEN],
            benchmark=sample["benchmark"],
            source_model=source_model,
        )

        # Gray placeholder image (text-only)
        image = Image.new("RGB", (224, 224), color="gray")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text], images=[image], return_tensors="pt", padding=True,
            min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[0, -1, :]
        token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
        token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]
        probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
        sample["p_correct"] = probs[1].item()

        if (i + 1) % 25 == 0 or (i + 1) == len(samples):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"    [{i+1}/{len(samples)}] {rate:.1f}/sec")

    return samples


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--score_only", action="store_true",
                        help="Skip generation, use existing predictions")
    parser.add_argument("--generate_only", action="store_true",
                        help="Only generate responses, skip calibrator scoring")
    parser.add_argument("--target_model", default=TARGET_MODEL)
    parser.add_argument("--calibrator_checkpoint", default=CALIBRATOR_CHECKPOINT)
    parser.add_argument("--benchmarks", type=str, default=None,
                        help="Comma-separated list of benchmarks to run (default: all)")
    parser.add_argument("--pred_dir", default=PRED_DIR)
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    args = parser.parse_args()

    if args.smoke_test:
        args.max_examples = 5

    os.makedirs(args.pred_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Select benchmarks
    if args.benchmarks:
        bench_names = [b.strip() for b in args.benchmarks.split(",")]
    else:
        bench_names = list(BENCHMARKS.keys())

    # --------------------------------------------------------
    # Step 1: Generate responses (or load existing)
    # --------------------------------------------------------
    all_predictions = {}

    if not args.score_only:
        print(f"\n{'='*60}")
        print(f"Step 1: Generating responses with {args.target_model}")
        print(f"{'='*60}")

        print(f"Loading target model: {args.target_model}")
        target_tokenizer = AutoTokenizer.from_pretrained(args.target_model)
        target_model = AutoModelForCausalLM.from_pretrained(
            args.target_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        target_model.eval()

        for bench_name in bench_names:
            if bench_name not in BENCHMARKS:
                print(f"  SKIP {bench_name}: not found")
                continue

            bench_cfg = BENCHMARKS[bench_name]
            pred_file = os.path.join(args.pred_dir, f"{bench_name}.jsonl")

            print(f"\n  Loading {bench_name} ({bench_cfg['domain']})...")
            try:
                samples = load_benchmark_data(bench_name, bench_cfg, args.max_examples)
            except Exception as e:
                print(f"  ERROR loading {bench_name}: {e}")
                continue
            print(f"  Loaded {len(samples)} examples")

            print(f"  Generating responses...")
            results = generate_responses(target_model, target_tokenizer, samples)

            # Save predictions
            with open(pred_file, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            print(f"  Saved to {pred_file}")

            acc = sum(r["is_correct"] for r in results) / len(results)
            print(f"  Accuracy: {acc:.1%} ({sum(r['is_correct'] for r in results)}/{len(results)})")

            all_predictions[bench_name] = results

        # Free target model memory
        del target_model
        del target_tokenizer
        torch.cuda.empty_cache()

    else:
        # Load existing predictions
        for bench_name in bench_names:
            pred_file = os.path.join(args.pred_dir, f"{bench_name}.jsonl")
            if not os.path.exists(pred_file):
                print(f"  SKIP {bench_name}: {pred_file} not found")
                continue
            with open(pred_file) as f:
                all_predictions[bench_name] = [json.loads(line) for line in f]
            print(f"  Loaded {len(all_predictions[bench_name])} predictions from {pred_file}")

    if args.generate_only:
        print("\n--generate_only: skipping calibrator scoring")
        _print_generation_summary(all_predictions)
        return

    # --------------------------------------------------------
    # Step 2: Score with calibrator
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Step 2: Scoring with UQ calibrator")
    print(f"{'='*60}")

    print(f"Loading calibrator base: {CALIBRATOR_BASE}")
    cal_model = Qwen3VLForConditionalGeneration.from_pretrained(
        CALIBRATOR_BASE, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(CALIBRATOR_BASE)
    print(f"Loading LoRA: {args.calibrator_checkpoint}")
    cal_model = PeftModel.from_pretrained(cal_model, args.calibrator_checkpoint)
    cal_model.eval()
    device = next(cal_model.parameters()).device

    all_results = {}

    for bench_name, predictions in all_predictions.items():
        print(f"\n  Scoring {bench_name} ({len(predictions)} samples)...")
        scored = score_with_calibrator(
            cal_model, processor, predictions, device,
            source_model="Qwen3.5-0.8B",
        )

        # Save scored results
        scored_file = os.path.join(args.results_dir, f"{bench_name}_scored.jsonl")
        with open(scored_file, "w") as f:
            for s in scored:
                out = {
                    "id": s["id"],
                    "benchmark": s["benchmark"],
                    "domain": BENCHMARKS.get(bench_name, {}).get("domain", "unknown"),
                    "is_correct": s["is_correct"],
                    "p_correct": s["p_correct"],
                    "response_preview": s["response"][:200],
                }
                f.write(json.dumps(out) + "\n")

        # Compute metrics
        labels = np.array([s["is_correct"] for s in scored])
        scores = np.array([s["p_correct"] for s in scored])

        if len(set(labels)) >= 2:
            auroc = roc_auc_score(labels, scores)
        else:
            auroc = float("nan")

        acc = labels.mean()
        all_results[bench_name] = {
            "benchmark": bench_name,
            "domain": BENCHMARKS.get(bench_name, {}).get("domain", "unknown"),
            "n_samples": len(scored),
            "accuracy": float(acc),
            "auroc": float(auroc) if not math.isnan(auroc) else None,
            "mean_p_correct": float(scores.mean()),
            "mean_p_when_correct": float(scores[labels == 1].mean()) if labels.sum() > 0 else None,
            "mean_p_when_wrong": float(scores[labels == 0].mean()) if (1 - labels).sum() > 0 else None,
        }

        auroc_str = f"{auroc:.3f}" if not math.isnan(auroc) else "N/A"
        print(f"  {bench_name}: acc={acc:.1%}, AUROC={auroc_str}")

    # --------------------------------------------------------
    # Step 3: Summary
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"LAW & FINANCE BENCHMARK RESULTS")
    print(f"Target model: Qwen3.5-0.8B | Calibrator: best_v2_r32_combined")
    print(f"{'='*60}")

    print(f"\n{'Benchmark':>30} {'Domain':>8} {'N':>6} {'Acc':>6} {'AUROC':>8}")
    print("-" * 65)

    domain_results = {"legal": [], "finance": []}
    for key, res in sorted(all_results.items()):
        auroc_str = f"{res['auroc']:.3f}" if res['auroc'] is not None else "N/A"
        print(f"{res['benchmark']:>30} {res['domain']:>8} {res['n_samples']:>6} "
              f"{res['accuracy']:>6.1%} {auroc_str:>8}")
        if res["auroc"] is not None:
            domain_results[res["domain"]].append(res)

    # Domain averages
    print("-" * 65)
    for domain in ["legal", "finance"]:
        if domain_results[domain]:
            avg_auroc = np.mean([r["auroc"] for r in domain_results[domain]])
            total_n = sum(r["n_samples"] for r in domain_results[domain])
            avg_acc = np.mean([r["accuracy"] for r in domain_results[domain]])
            print(f"{'[' + domain.upper() + ' AVG]':>30} {domain:>8} {total_n:>6} "
                  f"{avg_acc:>6.1%} {avg_auroc:>8.3f}")

    # Overall
    all_valid = [r for r in all_results.values() if r["auroc"] is not None]
    if all_valid:
        overall_auroc = np.mean([r["auroc"] for r in all_valid])
        overall_n = sum(r["n_samples"] for r in all_valid)
        overall_acc = np.mean([r["accuracy"] for r in all_valid])
        print(f"{'[OVERALL AVG]':>30} {'all':>8} {overall_n:>6} "
              f"{overall_acc:>6.1%} {overall_auroc:>8.3f}")

    # Save summary
    summary = {
        "description": "Law & finance benchmark evaluation (OOD)",
        "target_model": "Qwen3.5-0.8B",
        "calibrator": args.calibrator_checkpoint,
        "per_benchmark": all_results,
        "domain_averages": {
            domain: {
                "mean_auroc": float(np.mean([r["auroc"] for r in results])),
                "n_benchmarks": len(results),
                "total_samples": sum(r["n_samples"] for r in results),
            }
            for domain, results in domain_results.items()
            if results
        },
    }
    summary_path = os.path.join(args.results_dir, "law_finance_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


def _print_generation_summary(all_predictions):
    """Print summary of generated predictions."""
    print(f"\n{'='*60}")
    print("GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Benchmark':>30} {'N':>6} {'Acc':>6}")
    for name, preds in sorted(all_predictions.items()):
        acc = sum(p["is_correct"] for p in preds) / len(preds) if preds else 0
        print(f"{name:>30} {len(preds):>6} {acc:>6.1%}")


if __name__ == "__main__":
    main()
