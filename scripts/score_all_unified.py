#!/usr/bin/env python3
"""Score all predictions with the unified UQ model (Qwen3-VL-8B + LoRA).

This is the prerequisite for all use case experiments. It runs the unified
calibrator on every prediction and saves per-sample P(correct) to JSONL.
Unlike score_all_samples.py, this uses the VLM model with real images for
VLM benchmarks.

Usage:
    # Smoke test (5 samples/benchmark, ~2 min)
    CUDA_VISIBLE_DEVICES=0 python scripts/score_all_unified.py --target gpt5mini --smoke_test

    # Full run (~30-45 min per target on 1 GPU)
    CUDA_VISIBLE_DEVICES=0 python scripts/score_all_unified.py --target gpt5mini
    CUDA_VISIBLE_DEVICES=1 python scripts/score_all_unified.py --target gpt52
    CUDA_VISIBLE_DEVICES=2 python scripts/score_all_unified.py --target qwen35

    # Score all 3 targets
    CUDA_VISIBLE_DEVICES=0 python scripts/score_all_unified.py --target all
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
from sklearn.metrics import roc_auc_score
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_CHECKPOINT = "uq_models/best_unified"
IMAGE_CACHE_DIR = Path("data/training_images")

VLM_BENCHMARKS = {
    "charxiv", "mmmu", "mmstar", "hallusionbench", "mathverse",
    "mathvision", "mathvista", "realworldqa", "vizwiz", "hle_multimodal", "mmvet",
}

EXCLUDED_BENCHMARKS = {
    "triviaqa", "babilong", "vsr", "aokvqa", "erqa",
    "tutorbench", "healthbench", "arc", "oolong",
}

PROMPT_TEMPLATES = {
    "baseline": """Question: {question}

Answer: {response}

Is the answer correct? (i) No (ii) Yes""",

    "combined": """Benchmark: {benchmark}
Source model: {source_model}

Question: {question}

Answer: {response}

Analyze whether the answer above is correct. Consider:
- Does the answer address the question?
- Are there factual errors or logical flaws?
- Is the answer complete?

Based on your analysis, is the answer correct? (i) No (ii) Yes""",
}

TRUNCATION_LENGTHS = {
    "baseline": (500, 300),
    "combined": (1500, 800),
}

# Default (overridden by --prompt_variant)
PROMPT_TEMPLATE = PROMPT_TEMPLATES["baseline"]

TARGET_CONFIGS = {
    "gpt5mini": {
        "name": "GPT-5-mini",
        "data_dir": "runs/gpt5_mini_combined",
        "mode": "combined",
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


# ============================================================
# DATA LOADING
# ============================================================

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
        return json.dumps(clean)[:2000]
    return str(input_data)[:2000]


def load_samples_combined(data_dir: str, max_per_benchmark=None):
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

        bench_samples = _load_predictions(pred_file, benchmark)

        if max_per_benchmark and len(bench_samples) > max_per_benchmark:
            np.random.seed(42)
            indices = np.random.choice(len(bench_samples), max_per_benchmark, replace=False)
            bench_samples = [bench_samples[i] for i in sorted(indices)]

        n_correct = sum(s["is_correct"] for s in bench_samples)
        stats[benchmark] = {"used": len(bench_samples), "correct": n_correct}
        print(f"  {benchmark}: {len(bench_samples)} samples ({n_correct} correct)")
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

        bench_samples = _load_predictions(pred_file, benchmark)

        if max_per_benchmark and len(bench_samples) > max_per_benchmark:
            np.random.seed(42)
            indices = np.random.choice(len(bench_samples), max_per_benchmark, replace=False)
            bench_samples = [bench_samples[i] for i in sorted(indices)]

        n_correct = sum(s["is_correct"] for s in bench_samples)
        stats[benchmark] = {"used": len(bench_samples), "correct": n_correct}
        print(f"  {benchmark}: {len(bench_samples)} samples ({n_correct} correct)")
        samples.extend(bench_samples)

    return samples, stats


def _load_predictions(pred_file: Path, benchmark: str) -> list:
    """Load and filter predictions from a single JSONL file."""
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

            prediction = pred.get("prediction") or {}
            usage = pred.get("usage") or {}

            samples.append({
                "id": str(pred.get("id", "")),
                "benchmark": benchmark,
                "question": question[:2000],
                "response": response[:1000],
                "is_correct": int(correct == 1),
                "has_image": benchmark in VLM_BENCHMARKS,
                "verbalized_confidence": prediction.get("confidence") if isinstance(prediction, dict) else None,
                "input_tokens": usage.get("input_tokens") if isinstance(usage, dict) else None,
                "output_tokens": usage.get("output_tokens") if isinstance(usage, dict) else None,
                "total_tokens": usage.get("total_tokens") if isinstance(usage, dict) else None,
            })

    return samples


# ============================================================
# INFERENCE
# ============================================================

def get_p_correct_vlm(model, processor, question: str, response: str,
                      image: Image.Image, device,
                      prompt_template=None, q_len=500, r_len=300,
                      benchmark="", source_model="") -> float:
    """Extract P(correct) from unified VLM model logits."""
    template = prompt_template or PROMPT_TEMPLATE
    prompt = template.format(
        question=question[:q_len],
        response=response[:r_len],
        benchmark=benchmark,
        source_model=source_model,
    )
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
    return probs[1].item()


def load_image_for_sample(sample: dict) -> Image.Image:
    """Load cached image for a sample, or gray fallback."""
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


# ============================================================
# MAIN
# ============================================================

def score_target(model, processor, device, target: str, config: dict,
                 output_dir: Path, max_per_benchmark=None,
                 prompt_variant="baseline"):
    """Score all samples for a single target model."""
    output_path = output_dir / f"{target}_scored.jsonl"

    print(f"\n{'='*70}")
    print(f"SCORING: {config['name']} predictions with unified UQ model")
    print(f"{'='*70}")
    print(f"Output: {output_path}")

    # Load samples
    print("Loading predictions...")
    if config["mode"] == "combined":
        samples, stats = load_samples_combined(
            config["data_dir"], max_per_benchmark=max_per_benchmark,
        )
    else:
        samples, stats = load_samples_prefixed(
            config["prefix"], max_per_benchmark=max_per_benchmark,
        )

    if not samples:
        print("ERROR: No samples loaded!")
        return None

    n_correct = sum(s["is_correct"] for s in samples)
    n_vlm = sum(1 for s in samples if s["has_image"])
    print(f"\nTotal: {len(samples)} samples ({n_correct} correct, {n_vlm} VLM)")

    # Score all samples
    print(f"\nScoring {len(samples)} samples...")
    start_time = time.time()
    all_preds, all_labels = [], []

    with open(output_path, "w") as f_out:
        for i, sample in enumerate(samples):
            try:
                image = load_image_for_sample(sample)
                q_len, r_len = TRUNCATION_LENGTHS.get(prompt_variant, (500, 300))
                p_correct = get_p_correct_vlm(
                    model, processor,
                    sample["question"], sample["response"],
                    image, device,
                    prompt_template=PROMPT_TEMPLATES.get(prompt_variant, PROMPT_TEMPLATE),
                    q_len=q_len, r_len=r_len,
                    benchmark=sample.get("benchmark", ""),
                    source_model=target,
                )
            except Exception as e:
                if i < 5:
                    print(f"  Error on sample {i} ({sample['id']}): {e}")
                p_correct = 0.5

            all_preds.append(p_correct)
            all_labels.append(sample["is_correct"])

            out_record = {
                "id": sample["id"],
                "benchmark": sample["benchmark"],
                "target_model": target,
                "is_correct": sample["is_correct"],
                "p_correct": round(p_correct, 6),
                "has_image": sample["has_image"],
                "verbalized_confidence": sample.get("verbalized_confidence"),
                "input_tokens": sample.get("input_tokens"),
                "output_tokens": sample.get("output_tokens"),
                "total_tokens": sample.get("total_tokens"),
                "question_preview": sample["question"][:200],
                "response_preview": sample["response"][:200],
            }
            f_out.write(json.dumps(out_record) + "\n")

            if (i + 1) % 50 == 0:
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
    print(f"SCORING COMPLETE: {config['name']}")
    print(f"{'='*70}")
    print(f"Scored {len(samples)} samples in {elapsed:.0f}s ({len(samples)/elapsed:.1f}/s)")
    print(f"Output: {output_path}")

    results = {
        "target_model": target,
        "target_name": config["name"],
        "calibrator": "unified (Qwen3-VL-8B + LoRA)",
        "n_samples": len(samples),
        "n_correct": int(sum(all_labels)),
        "n_vlm": n_vlm,
        "mean_p_correct": float(np.mean(all_preds)),
        "elapsed_seconds": elapsed,
        "data_stats": stats,
    }

    if len(set(all_labels)) > 1:
        auroc = roc_auc_score(all_labels, all_preds)
        results["auroc"] = float(auroc)
        print(f"Overall AUROC: {auroc:.4f}")

    # Per-benchmark AUROC
    bench_data = defaultdict(lambda: {"preds": [], "labels": [], "is_vlm": False})
    for sample, p, l in zip(samples, all_preds, all_labels):
        bench_data[sample["benchmark"]]["preds"].append(p)
        bench_data[sample["benchmark"]]["labels"].append(l)
        bench_data[sample["benchmark"]]["is_vlm"] = sample["has_image"]

    print(f"\nPer-benchmark AUROC:")
    results["per_benchmark"] = {}
    for bench in sorted(bench_data.keys()):
        bd = bench_data[bench]
        entry = {"n_samples": len(bd["labels"]), "is_vlm": bd["is_vlm"]}
        if len(set(bd["labels"])) > 1:
            ba = roc_auc_score(bd["labels"], bd["preds"])
            entry["auroc"] = float(ba)
            tag = "[VLM]" if bd["is_vlm"] else "[TXT]"
            print(f"  {bench:<20} AUROC={ba:.3f} (n={len(bd['labels'])}) {tag}")
        else:
            print(f"  {bench:<20} single class (n={len(bd['labels'])})")
        results["per_benchmark"][bench] = entry

    # VLM vs text aggregate
    vlm_p, vlm_l, txt_p, txt_l = [], [], [], []
    for bench, data in bench_data.items():
        if data["is_vlm"]:
            vlm_p.extend(data["preds"])
            vlm_l.extend(data["labels"])
        else:
            txt_p.extend(data["preds"])
            txt_l.extend(data["labels"])

    if vlm_l and len(set(vlm_l)) > 1:
        vlm_auroc = roc_auc_score(vlm_l, vlm_p)
        results["vlm_auroc"] = float(vlm_auroc)
        print(f"\nVLM AUROC: {vlm_auroc:.4f} ({len(vlm_l)} samples)")
    if txt_l and len(set(txt_l)) > 1:
        txt_auroc = roc_auc_score(txt_l, txt_p)
        results["text_auroc"] = float(txt_auroc)
        print(f"Text AUROC: {txt_auroc:.4f} ({len(txt_l)} samples)")

    # Save summary
    summary_path = output_dir / f"{target}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary: {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Score all samples with unified UQ model")
    parser.add_argument("--target", choices=list(TARGET_CONFIGS.keys()) + ["all"],
                        required=True, help="Which model's predictions to score (or 'all')")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Path to unified model checkpoint")
    parser.add_argument("--output_dir", default="data/use_cases/scored_unified",
                        help="Output directory for scored JSONL")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Score 5 samples per benchmark only")
    parser.add_argument("--max_per_benchmark", type=int, default=None)
    parser.add_argument("--prompt_variant", default="baseline",
                        choices=list(PROMPT_TEMPLATES.keys()),
                        help="Prompt template variant (must match training)")
    args = parser.parse_args()

    if args.smoke_test:
        args.max_per_benchmark = 5

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve checkpoint (find best checkpoint subdir if available)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint {args.checkpoint} does not exist!")
        sys.exit(1)

    # Check for checkpoint subdirectories (HuggingFace Trainer saves checkpoints as subdirs)
    checkpoint_subdirs = sorted(checkpoint_path.glob("checkpoint-*"))
    if checkpoint_subdirs:
        # Use the last checkpoint (highest step)
        best_ckpt = checkpoint_subdirs[-1]
        print(f"Using checkpoint: {best_ckpt}")
        lora_path = str(best_ckpt)
    else:
        # Assume the path itself contains the LoRA weights
        lora_path = str(checkpoint_path)
    print(f"LoRA adapter: {lora_path}")

    # Load model
    print(f"\nLoading base model: {BASE_MODEL}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Loading LoRA adapter from {lora_path}...")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    print(f"Model loaded on {device}")

    # Score targets
    targets = list(TARGET_CONFIGS.keys()) if args.target == "all" else [args.target]
    all_results = {}

    for target in targets:
        config = TARGET_CONFIGS[target]
        results = score_target(
            model, processor, device, target, config,
            output_dir, max_per_benchmark=args.max_per_benchmark,
            prompt_variant=args.prompt_variant,
        )
        if results:
            all_results[target] = results

    # Summary across all targets
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("OVERALL SUMMARY")
        print(f"{'='*70}")
        for target, r in all_results.items():
            auroc = r.get("auroc", "N/A")
            vlm = r.get("vlm_auroc", "N/A")
            txt = r.get("text_auroc", "N/A")
            print(f"  {target:<10} Overall={auroc:.4f}  VLM={vlm}  Text={txt}  (n={r['n_samples']})")

        combined_path = output_dir / "all_targets_summary.json"
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nCombined summary: {combined_path}")


if __name__ == "__main__":
    main()
