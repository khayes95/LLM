#!/usr/bin/env python3
"""Score PRBench legal predictions with the v2 calibrator and compute AUROC.

Reads graded predictions from runs/prbench_legal_*/predictions.jsonl,
scores each (question, response) with the UQ calibrator, and outputs
AUROC for the legal domain.

Usage:
    # Smoke test
    CUDA_VISIBLE_DEVICES=0 python scripts/score_prbench_legal.py --smoke_test

    # Full
    CUDA_VISIBLE_DEVICES=0 python scripts/score_prbench_legal.py
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
CHECKPOINT = "uq_models/best_v2_r32_combined"
Q_LEN = 1500
R_LEN = 800

PROMPT_TEMPLATE = """Benchmark: prbench
Source model: {source_model}

Question: {question}

Answer: {response}

Analyze whether the answer above is correct. Consider:
- Does the answer address the question?
- Are there factual errors or logical flaws?
- Is the answer complete?

Based on your analysis, is the answer correct? (i) No (ii) Yes"""

OUTPUT_DIR = Path("data/prbench_legal_scored")


def load_predictions(run_dirs):
    """Load graded predictions from multiple run directories."""
    samples = []
    for run_dir in run_dirs:
        pred_path = Path(run_dir) / "predictions.jsonl"
        if not pred_path.exists():
            print(f"  SKIP: {pred_path} not found")
            continue

        seen_ids = set()
        with open(pred_path) as f:
            for line in f:
                d = json.loads(line)
                eid = d["id"]
                correct = d.get("score", {}).get("correct", -1)

                # Skip ungraded and duplicates
                if correct == -1:
                    continue
                if eid in seen_ids:
                    continue
                seen_ids.add(eid)

                # Extract question from request messages
                question = ""
                msgs = d.get("request", {}).get("messages", [])
                for msg in msgs:
                    if msg.get("role") == "user":
                        question = msg["content"]
                        break
                if not question:
                    question = str(d.get("input", ""))

                response = d.get("response_text", "") or d.get("prediction", {}).get("answer", "")
                domain = d.get("meta", {}).get("domain", "?")

                # Determine source model from directory name
                if "gpt5mini" in str(run_dir):
                    source_model = "gpt-5-mini"
                elif "gpt52" in str(run_dir):
                    source_model = "gpt-5.2"
                else:
                    source_model = "unknown"

                samples.append({
                    "id": eid,
                    "question": question[:Q_LEN],
                    "response": response[:R_LEN],
                    "is_correct": correct,
                    "source_model": source_model,
                    "domain": domain,
                })

        print(f"  Loaded {len(seen_ids)} graded from {run_dir}")

    return samples


def load_model(checkpoint):
    """Load calibrator model."""
    print(f"Loading {BASE_MODEL} + LoRA from {checkpoint}...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model = PeftModel.from_pretrained(model, checkpoint)
    model.eval()

    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    return model, processor


def score_batch(model, processor, samples, batch_size=1):
    """Score samples with calibrator, return p_correct for each."""
    device = next(model.parameters()).device
    gray = Image.new("RGB", (28, 28), (128, 128, 128))
    results = []

    for i, sample in enumerate(samples):
        prompt = PROMPT_TEMPLATE.format(
            source_model=sample["source_model"],
            question=sample["question"],
            response=sample["response"],
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": gray},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[gray],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]

        # Get token IDs for (i) No and (ii) Yes
        no_tokens = processor.tokenizer.encode("No", add_special_tokens=False)
        yes_tokens = processor.tokenizer.encode("Yes", add_special_tokens=False)

        no_logit = logits[no_tokens[0]].float()
        yes_logit = logits[yes_tokens[0]].float()

        probs = torch.softmax(torch.stack([no_logit, yes_logit]), dim=0)
        p_correct = probs[1].item()

        results.append({
            **sample,
            "p_correct": p_correct,
        })

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Scored {i+1}/{len(samples)}, last p_correct={p_correct:.3f}")

    return results


def compute_metrics(results):
    """Compute AUROC and other metrics."""
    labels = np.array([r["is_correct"] for r in results])
    scores = np.array([r["p_correct"] for r in results])

    metrics = {
        "n": len(results),
        "accuracy": float(labels.mean()),
        "n_correct": int(labels.sum()),
        "n_wrong": int((1 - labels).sum()),
    }

    if len(set(labels)) >= 2:
        metrics["auroc"] = float(roc_auc_score(labels, scores))

    # Per-domain breakdown
    domains = {}
    for r in results:
        dom = r["domain"]
        if dom not in domains:
            domains[dom] = {"labels": [], "scores": []}
        domains[dom]["labels"].append(r["is_correct"])
        domains[dom]["scores"].append(r["p_correct"])

    domain_metrics = {}
    for dom, data in domains.items():
        dl = np.array(data["labels"])
        ds = np.array(data["scores"])
        dm = {"n": len(dl), "accuracy": float(dl.mean())}
        if len(set(dl)) >= 2:
            dm["auroc"] = float(roc_auc_score(dl, ds))
        domain_metrics[dom] = dm

    metrics["per_domain"] = domain_metrics

    # Per source model
    models = {}
    for r in results:
        sm = r["source_model"]
        if sm not in models:
            models[sm] = {"labels": [], "scores": []}
        models[sm]["labels"].append(r["is_correct"])
        models[sm]["scores"].append(r["p_correct"])

    model_metrics = {}
    for sm, data in models.items():
        ml = np.array(data["labels"])
        ms = np.array(data["scores"])
        mm = {"n": len(ml), "accuracy": float(ml.mean())}
        if len(set(ml)) >= 2:
            mm["auroc"] = float(roc_auc_score(ml, ms))
        model_metrics[sm] = mm

    metrics["per_model"] = model_metrics

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    args = parser.parse_args()

    max_examples = 10 if args.smoke_test else args.max_examples

    # Find all PRBench legal runs
    run_dirs = [
        "runs/prbench_legal_gpt5mini",
        "runs/prbench_legal_gpt52",
        "runs/prbench_legal_hard_gpt5mini",
        "runs/prbench_legal_hard_gpt52",
    ]

    print("Loading predictions...")
    samples = load_predictions(run_dirs)
    print(f"Total graded samples: {len(samples)}")

    if max_examples:
        samples = samples[:max_examples]
        print(f"Limited to {max_examples} for smoke test")

    if not samples:
        print("No graded samples found!")
        return

    # Load model
    model, processor = load_model(args.checkpoint)

    # Score
    print(f"\nScoring {len(samples)} samples...")
    t0 = time.time()
    results = score_batch(model, processor, samples)
    elapsed = time.time() - t0
    print(f"Scoring took {elapsed:.0f}s ({elapsed/len(results):.1f}s/sample)")

    # Compute metrics
    metrics = compute_metrics(results)

    # Print results
    print(f"\n{'='*60}")
    print(f"PRBench Legal - Calibrator AUROC")
    print(f"{'='*60}")
    print(f"Total: {metrics['n']} samples, accuracy: {metrics['accuracy']:.1%}")
    if "auroc" in metrics:
        print(f"Overall AUROC: {metrics['auroc']:.3f}")

    print(f"\nPer domain:")
    for dom, dm in sorted(metrics.get("per_domain", {}).items()):
        auroc_str = f", AUROC={dm['auroc']:.3f}" if "auroc" in dm else ""
        print(f"  {dom}: n={dm['n']}, acc={dm['accuracy']:.1%}{auroc_str}")

    print(f"\nPer model:")
    for sm, mm in sorted(metrics.get("per_model", {}).items()):
        auroc_str = f", AUROC={mm['auroc']:.3f}" if "auroc" in mm else ""
        print(f"  {sm}: n={mm['n']}, acc={mm['accuracy']:.1%}{auroc_str}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "scored_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
