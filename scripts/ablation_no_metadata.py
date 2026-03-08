#!/usr/bin/env python3
"""Ablation: Score test-only samples with and without metadata in the prompt.

Tests whether the calibrator relies on benchmark/source_model metadata or
actually reads the question/response content.

Conditions:
  1. "combined" — full prompt with real benchmark + source_model (reproduces existing scores)
  2. "no_metadata" — combined template but benchmark="" and source_model=""
  3. "baseline_template" — baseline template (no metadata fields at all)

Usage:
    # Smoke test
    CUDA_VISIBLE_DEVICES=0 python scripts/ablation_no_metadata.py --smoke_test

    # Full run (~60-90 min on 1 GPU, 3 conditions x ~4400 samples)
    CUDA_VISIBLE_DEVICES=0 python scripts/ablation_no_metadata.py

    # Single condition
    CUDA_VISIBLE_DEVICES=0 python scripts/ablation_no_metadata.py --conditions no_metadata
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
from PIL import Image
from peft import PeftModel
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_CHECKPOINT = "uq_models/best_v2_r32_combined"
IMAGE_CACHE_DIR = Path("data/training_images")
OUTPUT_DIR = Path("data/ablations/no_metadata")

VLM_BENCHMARKS = {
    "charxiv", "mmmu", "mmstar", "hallusionbench", "mathverse",
    "mathvision", "mathvista", "realworldqa", "vizwiz", "hle_multimodal", "mmvet",
}

EXCLUDED_BENCHMARKS = {
    "triviaqa", "babilong", "vsr", "aokvqa", "erqa",
    "tutorbench", "healthbench", "arc", "oolong",
}

TARGET_CONFIGS = {
    "gpt5mini": {
        "name": "GPT-5-mini",
        "data_dir": "runs/gpt5_mini_combined",
        "mode": "combined",
    },
    "gpt52": {
        "name": "GPT-5.2",
        "prefix": "gpt52_high_",
        "mode": "prefixed",
    },
    "qwen35": {
        "name": "Qwen3.5",
        "prefix": "qwen35_397b_",
        "mode": "prefixed",
    },
}

PROMPT_COMBINED = """Benchmark: {benchmark}
Source model: {source_model}

Question: {question}

Answer: {response}

Analyze whether the answer above is correct. Consider:
- Does the answer address the question?
- Are there factual errors or logical flaws?
- Is the answer complete?

Based on your analysis, is the answer correct? (i) No (ii) Yes"""

PROMPT_BASELINE = """Question: {question}

Answer: {response}

Is the answer correct? (i) No (ii) Yes"""

CONDITIONS = {
    "combined": {
        "template": PROMPT_COMBINED,
        "q_len": 1500,
        "r_len": 800,
        "use_metadata": True,
    },
    "no_metadata": {
        "template": PROMPT_COMBINED,
        "q_len": 1500,
        "r_len": 800,
        "use_metadata": False,
    },
    "baseline_template": {
        "template": PROMPT_BASELINE,
        "q_len": 1500,
        "r_len": 800,
        "use_metadata": False,
    },
}


# ============================================================
# DATA LOADING (from score_all_unified.py)
# ============================================================

def extract_question_text(input_data) -> str:
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
        return json.dumps(clean)[:2000]
    return str(input_data)[:2000]


def load_predictions(pred_file: Path, benchmark: str) -> list:
    samples = []
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

            samples.append({
                "id": str(pred.get("id", "")),
                "benchmark": benchmark,
                "question": question[:2000],
                "response": response[:1000],
                "is_correct": int(correct == 1),
                "has_image": benchmark in VLM_BENCHMARKS,
            })
    return samples


def load_all_samples_for_target(target: str, config: dict):
    """Load all prediction samples for a target model."""
    samples = []
    if config["mode"] == "combined":
        data_path = Path(config["data_dir"])
        if not data_path.exists():
            return samples
        for bench_dir in sorted(data_path.iterdir()):
            if not bench_dir.is_dir():
                continue
            benchmark = bench_dir.name
            if benchmark in EXCLUDED_BENCHMARKS:
                continue
            pred_file = bench_dir / "predictions.jsonl"
            if not pred_file.exists():
                continue
            bench_samples = load_predictions(pred_file, benchmark)
            samples.extend(bench_samples)
    else:
        prefix = config["prefix"]
        runs_path = Path("runs")
        for run_dir in sorted(runs_path.iterdir()):
            if not run_dir.name.startswith(prefix):
                continue
            benchmark = run_dir.name[len(prefix):]
            if benchmark in EXCLUDED_BENCHMARKS:
                continue
            if "backup" in run_dir.name:
                continue
            pred_file = run_dir / "predictions.jsonl"
            if not pred_file.exists():
                continue
            bench_samples = load_predictions(pred_file, benchmark)
            samples.extend(bench_samples)
    return samples


def filter_to_test_ids(samples: list, test_ids: set) -> list:
    """Keep only samples whose ID is in the test set."""
    return [s for s in samples if s["id"] in test_ids]


# ============================================================
# INFERENCE
# ============================================================

def load_image_for_sample(sample: dict) -> Image.Image:
    fallback = Image.new('RGB', (224, 224), color='gray')
    if not sample["has_image"]:
        return fallback
    cache_path = IMAGE_CACHE_DIR / sample["benchmark"] / f"{sample['id']}.jpg"
    if cache_path.exists():
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception:
            return fallback
    return fallback


def get_p_correct(model, processor, sample, device, template, q_len, r_len,
                  use_metadata=True, target_model=""):
    """Score a single sample."""
    fmt_kwargs = {
        "question": sample["question"][:q_len],
        "response": sample["response"][:r_len],
    }
    if "{benchmark}" in template:
        fmt_kwargs["benchmark"] = sample["benchmark"] if use_metadata else ""
        fmt_kwargs["source_model"] = target_model if use_metadata else ""

    prompt = template.format(**fmt_kwargs)
    image = load_image_for_sample(sample)

    is_vlm = sample["has_image"]
    min_px = 256 * 28 * 28
    max_px = 512 * 28 * 28 if is_vlm else 256 * 28 * 28

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=[image], return_tensors="pt", padding=True,
        min_pixels=min_px, max_pixels=max_px,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]
    token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
    token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]
    probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
    return probs[1].item()


def compute_metrics(labels, scores):
    labels = np.array(labels)
    scores = np.array(scores)
    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "n": len(labels)}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "n": len(labels),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--conditions", nargs="+",
                        default=list(CONDITIONS.keys()),
                        choices=list(CONDITIONS.keys()))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load split info to filter to test-only
    split_info_path = Path(args.checkpoint) / "split_info.json"
    if not split_info_path.exists():
        print(f"ERROR: split_info.json not found at {split_info_path}")
        sys.exit(1)

    with open(split_info_path) as f:
        split_info = json.load(f)
    test_ids = set(split_info["test_ids"])
    print(f"Loaded {len(test_ids)} test IDs from split_info.json")

    # Load all test-only samples from original prediction files
    print("\nLoading test-only samples from prediction files...")
    all_samples = []
    for target, config in TARGET_CONFIGS.items():
        target_samples = load_all_samples_for_target(target, config)
        test_samples = filter_to_test_ids(target_samples, test_ids)
        for s in test_samples:
            s["target_model"] = target
        print(f"  {target}: {len(test_samples)} test samples (from {len(target_samples)} total)")
        all_samples.extend(test_samples)

    if args.smoke_test:
        # Take 5 samples per target for smoke test
        from collections import Counter
        target_counts = Counter()
        filtered = []
        for s in all_samples:
            if target_counts[s["target_model"]] < 5:
                filtered.append(s)
                target_counts[s["target_model"]] += 1
        all_samples = filtered

    print(f"\nTotal test samples: {len(all_samples)}")
    n_benchmarks = len(set(s["benchmark"] for s in all_samples))
    print(f"Unique benchmarks: {n_benchmarks}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()
    print("Model loaded.")

    # Run each condition
    all_results = {}
    for cond_name in args.conditions:
        cond = CONDITIONS[cond_name]
        print(f"\n{'='*70}")
        print(f"CONDITION: {cond_name}")
        print(f"  Template: {'combined' if '{benchmark}' in cond['template'] else 'baseline'}")
        print(f"  Metadata: {'yes' if cond['use_metadata'] else 'no'}")
        print(f"  Truncation: Q={cond['q_len']}, R={cond['r_len']}")
        print(f"{'='*70}")

        labels = []
        scores = []
        per_benchmark = defaultdict(lambda: {"labels": [], "scores": []})
        per_model = defaultdict(lambda: {"labels": [], "scores": []})
        t0 = time.time()

        # Save per-sample predictions
        preds_path = output_dir / f"{cond_name}_predictions.jsonl"
        with open(preds_path, "w") as f_out:
            for i, sample in enumerate(all_samples):
                try:
                    p = get_p_correct(
                        model, processor, sample, device,
                        template=cond["template"],
                        q_len=cond["q_len"],
                        r_len=cond["r_len"],
                        use_metadata=cond["use_metadata"],
                        target_model=sample["target_model"],
                    )
                except Exception as e:
                    if i < 5:
                        print(f"  Error on sample {i} ({sample['id']}): {e}")
                    p = 0.5

                label = sample["is_correct"]
                labels.append(label)
                scores.append(p)

                bench = sample["benchmark"]
                target = sample["target_model"]
                per_benchmark[bench]["labels"].append(label)
                per_benchmark[bench]["scores"].append(p)
                per_model[target]["labels"].append(label)
                per_model[target]["scores"].append(p)

                f_out.write(json.dumps({
                    "id": sample["id"],
                    "benchmark": bench,
                    "target_model": target,
                    "is_correct": label,
                    "p_correct": round(p, 6),
                }) + "\n")

                if (i + 1) % 100 == 0 or (i + 1) == len(all_samples):
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    interim = ""
                    if len(set(labels)) > 1:
                        interim = f", AUROC={roc_auc_score(labels, scores):.4f}"
                    print(f"  [{i+1}/{len(all_samples)}] {rate:.1f} samples/sec, "
                          f"elapsed {elapsed:.0f}s{interim}")
                    f_out.flush()

        # Compute metrics
        overall = compute_metrics(labels, scores)
        bench_metrics = {}
        for bench in sorted(per_benchmark):
            data = per_benchmark[bench]
            bench_metrics[bench] = compute_metrics(data["labels"], data["scores"])
        model_metrics = {}
        for target in sorted(per_model):
            data = per_model[target]
            model_metrics[target] = compute_metrics(data["labels"], data["scores"])

        result = {
            "condition": cond_name,
            "overall": overall,
            "per_benchmark": bench_metrics,
            "per_model": model_metrics,
            "config": {
                "template_type": "combined" if "{benchmark}" in cond["template"] else "baseline",
                "use_metadata": cond["use_metadata"],
                "q_len": cond["q_len"],
                "r_len": cond["r_len"],
            },
        }
        all_results[cond_name] = result

        cond_path = output_dir / f"{cond_name}_results.json"
        with open(cond_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n  AUROC: {overall['auroc']:.4f} | AUPRC: {overall['auprc']:.4f} | N: {overall['n']}")
        for target in sorted(model_metrics):
            m = model_metrics[target]
            print(f"    {target}: AUROC={m['auroc']:.4f}")
        print(f"  Saved: {cond_path}")

    # Summary comparison
    summary = {
        "description": "Metadata ablation: does the calibrator rely on benchmark/model metadata?",
        "note": "WARNING: These scores use the current (leaky) train/test split. "
                "Absolute numbers may be inflated, but relative comparisons between "
                "conditions are valid.",
        "checkpoint": args.checkpoint,
        "n_samples": len(all_samples),
        "conditions": {},
    }
    for cond_name, result in all_results.items():
        summary["conditions"][cond_name] = {
            "auroc": result["overall"]["auroc"],
            "auprc": result["overall"]["auprc"],
            "per_model": {k: v["auroc"] for k, v in result["per_model"].items()},
            "config": result["config"],
        }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")
    print(f"{'Condition':<20} {'AUROC':>8}  {'Per-model':}")
    print("-" * 70)
    for cond_name in args.conditions:
        r = all_results[cond_name]
        per_m = " | ".join(f"{k}={v['auroc']:.3f}" for k, v in sorted(r["per_model"].items()))
        print(f"  {cond_name:<18} {r['overall']['auroc']:>8.4f}  {per_m}")
    if "combined" in all_results and "no_metadata" in all_results:
        delta = all_results["no_metadata"]["overall"]["auroc"] - all_results["combined"]["overall"]["auroc"]
        print(f"\n  Delta (no_metadata - combined): {delta:+.4f}")
        if abs(delta) > 0.05:
            print(f"  SIGNIFICANT: Metadata contributes substantially to AUROC.")
        elif abs(delta) > 0.02:
            print(f"  MODERATE: Metadata has a meaningful but modest effect.")
        else:
            print(f"  MINIMAL: Metadata has negligible impact — model reads content.")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
