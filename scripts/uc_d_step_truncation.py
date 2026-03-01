#!/usr/bin/env python3
"""UC-D Stage 2: Step-Truncation Confidence Scoring.

Loads the unified calibrator (Qwen3-VL-8B + LoRA), then for each multi-step
response from GPT-5-mini and GPT-5.2:
  1. Loads the FULL response text from the raw predictions in runs/
  2. Splits the response into reasoning steps
  3. Creates truncated versions (after step 1, after step 2, ..., full)
  4. Scores each truncation through the calibrator -> P(correct) at each level
  5. Saves step-level confidence trajectories
  6. Analyzes: correct responses show stable/increasing confidence, incorrect
     show "confidence crash"

Usage:
    # Smoke test (10 samples, ~2 min)
    CUDA_VISIBLE_DEVICES=0 python scripts/uc_d_step_truncation.py --smoke_test

    # Full run (~1-3 hours on 1 GPU)
    CUDA_VISIBLE_DEVICES=0 python scripts/uc_d_step_truncation.py

    # Limit samples
    CUDA_VISIBLE_DEVICES=0 python scripts/uc_d_step_truncation.py --max_samples 200
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
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

PROMPT_TEMPLATE = """Question: {question}

Answer: {response}

Is the answer correct? (i) No (ii) Yes"""

# Run directories for each target model
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
}

# Minimum requirements for a response to be considered multi-step
MIN_STEPS = 3
MIN_RESPONSE_LEN = 200


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


def load_raw_predictions_combined(data_dir: str):
    """Load raw predictions from combined directory (subdirs per benchmark)."""
    samples = []
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"WARNING: {data_dir} does not exist")
        return samples

    for bench_dir in sorted(data_path.iterdir()):
        if not bench_dir.is_dir():
            continue
        benchmark = bench_dir.name
        if benchmark in EXCLUDED_BENCHMARKS:
            continue

        pred_file = bench_dir / "predictions.jsonl"
        if not pred_file.exists() or pred_file.stat().st_size == 0:
            continue

        bench_samples = _load_raw_predictions(pred_file, benchmark)
        samples.extend(bench_samples)

    return samples


def load_raw_predictions_prefixed(prefix: str, runs_dir="runs"):
    """Load raw predictions from runs matching a prefix."""
    samples = []
    runs_path = Path(runs_dir)

    for run_dir in sorted(runs_path.iterdir()):
        if not run_dir.name.startswith(prefix):
            continue
        benchmark = run_dir.name[len(prefix):]
        if benchmark in EXCLUDED_BENCHMARKS:
            continue
        if "backup" in run_dir.name or "combined" in run_dir.name:
            continue

        pred_file = run_dir / "predictions.jsonl"
        if not pred_file.exists() or pred_file.stat().st_size == 0:
            continue

        bench_samples = _load_raw_predictions(pred_file, benchmark)
        samples.extend(bench_samples)

    return samples


def _load_raw_predictions(pred_file: Path, benchmark: str) -> list:
    """Load raw predictions with FULL response_text from a single JSONL file."""
    samples = []
    with open(pred_file) as f:
        for line in f:
            try:
                pred = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Determine correctness
            score = pred.get("score", {})
            if isinstance(score, dict):
                correct = score.get("correct", -1)
            else:
                correct = score
            if correct not in (0, 1):
                continue

            question = extract_question_text(pred.get("input", {}))
            response_text = pred.get("response_text", "")
            if not question or not response_text:
                continue
            if len(response_text) < MIN_RESPONSE_LEN:
                continue

            usage = pred.get("usage") or {}
            output_tokens = usage.get("output_tokens")
            if output_tokens is not None:
                output_tokens = int(output_tokens)
            else:
                output_tokens = 0

            samples.append({
                "id": str(pred.get("id", "")),
                "benchmark": benchmark,
                "question": question[:2000],
                "response_text": response_text,
                "is_correct": int(correct == 1),
                "has_image": benchmark in VLM_BENCHMARKS,
                "output_tokens": output_tokens,
            })

    return samples


# ============================================================
# STEP SPLITTING
# ============================================================

# Regex for numbered step patterns: "1. ", "1) ", "Step 1:", etc.
_NUMBERED_STEP_RE = re.compile(r'(?:^|\n)\s*(?:\d+[\.\)]\s|Step\s+\d)', re.IGNORECASE)


def split_into_steps(response_text: str):
    """Split a response into reasoning steps.

    Returns (steps, separator) where:
      - steps is a list of non-empty step strings
      - separator is the string used to rejoin steps for truncation

    Heuristics (in order):
    1. JSON with "reasoning" key that is a list -> each list item is a step
    2. Numbered steps like "1. ...\n2. ..." -> split on numbered pattern
    3. Paragraph breaks ("\n\n") -> split on double newline
    """
    # 1. Try JSON with "reasoning" list
    try:
        parsed = json.loads(response_text)
        if isinstance(parsed, dict):
            reasoning = parsed.get("reasoning")
            if isinstance(reasoning, list) and len(reasoning) >= MIN_STEPS:
                steps = [str(s).strip() for s in reasoning if str(s).strip()]
                if len(steps) >= MIN_STEPS:
                    return steps, "\n"

            # If reasoning is a string, continue to heuristic splitting on it
            if isinstance(reasoning, str) and len(reasoning) > MIN_RESPONSE_LEN:
                response_text = reasoning
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # 2. Numbered steps
    matches = list(_NUMBERED_STEP_RE.finditer(response_text))
    if len(matches) >= MIN_STEPS:
        steps = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(response_text)
            step_text = response_text[start:end].strip()
            if step_text:
                steps.append(step_text)
        if len(steps) >= MIN_STEPS:
            return steps, "\n"

    # 3. Paragraph breaks
    paragraphs = response_text.split("\n\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if len(paragraphs) >= MIN_STEPS:
        return paragraphs, "\n\n"

    return [], ""


def create_truncated_responses(steps, separator):
    """Create K truncated versions from K steps.

    Returns list of (truncated_text, step_index) where step_index is 1-based.
    """
    truncations = []
    for k in range(1, len(steps) + 1):
        truncated = separator.join(steps[:k])
        truncations.append((truncated, k))
    return truncations


# ============================================================
# MODEL LOADING & INFERENCE
# ============================================================

def load_model(checkpoint_path: str):
    """Load the unified calibrator model (Qwen3-VL-8B + LoRA)."""
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        print(f"ERROR: Checkpoint {checkpoint_path} does not exist!")
        sys.exit(1)

    # Check for checkpoint subdirectories
    checkpoint_subdirs = sorted(checkpoint.glob("checkpoint-*"))
    if checkpoint_subdirs:
        lora_path = str(checkpoint_subdirs[-1])
        print(f"Using checkpoint: {lora_path}")
    else:
        lora_path = str(checkpoint)
    print(f"LoRA adapter: {lora_path}")

    print(f"Loading base model: {BASE_MODEL}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    print(f"Model loaded on {device}")

    return model, processor, device


def get_p_correct(model, processor, question: str, response: str,
                  image: Image.Image, device) -> float:
    """Extract P(correct) from unified VLM model logits."""
    prompt = PROMPT_TEMPLATE.format(
        question=question[:500],
        response=response[:300],
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
# ANALYSIS
# ============================================================

def compute_confidence_drop(step_scores):
    """Compute confidence drop metrics from a step-level trajectory.

    Returns dict with:
      - max_drop: largest single-step drop in P(correct)
      - total_drop: first step score minus last step score
      - max_drop_step: step index (1-based) where the largest drop occurs
      - trajectory_slope: linear regression slope across steps
      - final_minus_max: final score minus maximum score seen
    """
    if len(step_scores) < 2:
        return {
            "max_drop": 0.0,
            "total_drop": 0.0,
            "max_drop_step": 0,
            "trajectory_slope": 0.0,
            "final_minus_max": 0.0,
        }

    scores = np.array(step_scores)
    diffs = np.diff(scores)  # score[i+1] - score[i]

    # Largest single-step drop (negative diff = drop)
    max_drop_idx = np.argmin(diffs)
    max_drop = -float(diffs[max_drop_idx])  # positive = big drop

    # Total drop
    total_drop = float(scores[0] - scores[-1])  # positive = ended lower

    # Linear regression slope
    x = np.arange(len(scores))
    if len(scores) > 1:
        slope = float(np.polyfit(x, scores, 1)[0])
    else:
        slope = 0.0

    # Final minus max
    final_minus_max = float(scores[-1] - scores.max())

    return {
        "max_drop": max_drop,
        "total_drop": total_drop,
        "max_drop_step": int(max_drop_idx + 2),  # 1-based, +2 because diff is between i and i+1
        "trajectory_slope": slope,
        "final_minus_max": final_minus_max,
    }


def analyze_trajectories(trajectories):
    """Analyze step-level confidence trajectories.

    Returns analysis dict with aggregate metrics, per-benchmark breakdown,
    and anomaly detection results.
    """
    if not trajectories:
        return {"error": "No trajectories to analyze"}

    # Separate correct vs incorrect
    correct_trajs = [t for t in trajectories if t["is_correct"] == 1]
    incorrect_trajs = [t for t in trajectories if t["is_correct"] == 0]

    print(f"\n  Trajectories: {len(trajectories)} total "
          f"({len(correct_trajs)} correct, {len(incorrect_trajs)} incorrect)")

    results = {
        "n_total": len(trajectories),
        "n_correct": len(correct_trajs),
        "n_incorrect": len(incorrect_trajs),
    }

    # --- Mean trajectories (normalize to relative step position 0..1) ---
    n_interp = 20  # Interpolation points
    interp_x = np.linspace(0, 1, n_interp)

    def interpolate_trajectories(trajs):
        """Interpolate trajectories to common x-axis."""
        interp_scores = []
        for t in trajs:
            scores = t["step_scores"]
            if len(scores) < 2:
                continue
            x = np.linspace(0, 1, len(scores))
            interp_y = np.interp(interp_x, x, scores)
            interp_scores.append(interp_y)
        if not interp_scores:
            return None, None
        arr = np.array(interp_scores)
        return arr.mean(axis=0), arr.std(axis=0)

    correct_mean, correct_std = interpolate_trajectories(correct_trajs)
    incorrect_mean, incorrect_std = interpolate_trajectories(incorrect_trajs)

    if correct_mean is not None:
        results["correct_trajectory_mean"] = correct_mean.tolist()
        results["correct_trajectory_std"] = correct_std.tolist()
    if incorrect_mean is not None:
        results["incorrect_trajectory_mean"] = incorrect_mean.tolist()
        results["incorrect_trajectory_std"] = incorrect_std.tolist()
    results["interp_x"] = interp_x.tolist()

    # --- Divergence point ---
    if correct_mean is not None and incorrect_mean is not None:
        diff = correct_mean - incorrect_mean
        # Find first point where difference exceeds a threshold
        threshold = 0.05
        diverge_idx = None
        for i in range(len(diff)):
            if diff[i] > threshold:
                diverge_idx = i
                break
        if diverge_idx is not None:
            results["divergence_relative_step"] = float(interp_x[diverge_idx])
            results["divergence_gap"] = float(diff[diverge_idx])
        else:
            results["divergence_relative_step"] = None
            results["divergence_gap"] = None

        results["final_gap"] = float(diff[-1])
        results["max_gap"] = float(diff.max())
        results["mean_gap"] = float(diff.mean())

    # --- Confidence drop analysis ---
    all_drops = []
    for t in trajectories:
        drop_metrics = compute_confidence_drop(t["step_scores"])
        t["_drop_metrics"] = drop_metrics
        all_drops.append({
            "is_correct": t["is_correct"],
            "max_drop": drop_metrics["max_drop"],
            "total_drop": drop_metrics["total_drop"],
            "trajectory_slope": drop_metrics["trajectory_slope"],
            "final_minus_max": drop_metrics["final_minus_max"],
        })

    correct_drops = [d for d in all_drops if d["is_correct"] == 1]
    incorrect_drops = [d for d in all_drops if d["is_correct"] == 0]

    if correct_drops and incorrect_drops:
        results["drop_stats"] = {
            "correct": {
                "mean_max_drop": float(np.mean([d["max_drop"] for d in correct_drops])),
                "mean_total_drop": float(np.mean([d["total_drop"] for d in correct_drops])),
                "mean_slope": float(np.mean([d["trajectory_slope"] for d in correct_drops])),
                "mean_final_minus_max": float(np.mean([d["final_minus_max"] for d in correct_drops])),
            },
            "incorrect": {
                "mean_max_drop": float(np.mean([d["max_drop"] for d in incorrect_drops])),
                "mean_total_drop": float(np.mean([d["total_drop"] for d in incorrect_drops])),
                "mean_slope": float(np.mean([d["trajectory_slope"] for d in incorrect_drops])),
                "mean_final_minus_max": float(np.mean([d["final_minus_max"] for d in incorrect_drops])),
            },
        }

    # --- Anomaly detection: use max_drop and final_minus_max to detect incorrect ---
    # For each metric, sweep thresholds and compute precision/recall for
    # detecting incorrect responses (is_correct=0 is the positive class)
    anomaly_metrics = {}
    for metric_name in ["max_drop", "total_drop", "final_minus_max"]:
        values = np.array([d[metric_name] for d in all_drops])
        # For max_drop and total_drop: higher = more likely incorrect
        # For final_minus_max: more negative = more likely incorrect
        if metric_name == "final_minus_max":
            scores_for_detection = -values  # Negate so higher = more anomalous
        else:
            scores_for_detection = values

        labels = np.array([1 - d["is_correct"] for d in all_drops])  # 1 = incorrect

        if len(set(labels)) < 2:
            continue

        try:
            precision, recall, thresholds = precision_recall_curve(labels, scores_for_detection)
            pr_auc = auc(recall, precision)
            auroc = roc_auc_score(labels, scores_for_detection)

            # Find best F1
            with np.errstate(divide='ignore', invalid='ignore'):
                f1_scores = 2 * (precision * recall) / (precision + recall)
            f1_scores = np.nan_to_num(f1_scores)
            best_f1_idx = np.argmax(f1_scores)

            anomaly_metrics[metric_name] = {
                "pr_auc": float(pr_auc),
                "auroc": float(auroc),
                "best_f1": float(f1_scores[best_f1_idx]),
                "best_f1_precision": float(precision[best_f1_idx]),
                "best_f1_recall": float(recall[best_f1_idx]),
                "best_f1_threshold": float(thresholds[best_f1_idx]) if best_f1_idx < len(thresholds) else None,
                # Store full curves for plotting
                "_precision": precision.tolist(),
                "_recall": recall.tolist(),
            }
        except Exception as e:
            print(f"  WARNING: Could not compute anomaly metrics for {metric_name}: {e}")

    results["anomaly_detection"] = anomaly_metrics

    # --- Also try: full_p_correct as a baseline detector ---
    full_scores = np.array([t["full_p_correct"] for t in trajectories])
    labels = np.array([1 - t["is_correct"] for t in trajectories])  # 1 = incorrect
    if len(set(labels)) > 1:
        try:
            # Lower p_correct = more likely incorrect
            precision_base, recall_base, _ = precision_recall_curve(labels, -full_scores)
            pr_auc_base = auc(recall_base, precision_base)
            auroc_base = roc_auc_score(labels, -full_scores)
            anomaly_metrics["full_p_correct_baseline"] = {
                "pr_auc": float(pr_auc_base),
                "auroc": float(auroc_base),
                "_precision": precision_base.tolist(),
                "_recall": recall_base.tolist(),
            }
        except Exception:
            pass

    # --- Per-benchmark breakdown ---
    bench_data = defaultdict(lambda: {"correct": [], "incorrect": []})
    for t in trajectories:
        key = "correct" if t["is_correct"] == 1 else "incorrect"
        bench_data[t["benchmark"]][key].append(t)

    per_benchmark = {}
    for bench in sorted(bench_data.keys()):
        bd = bench_data[bench]
        n_c = len(bd["correct"])
        n_i = len(bd["incorrect"])
        entry = {
            "n_correct": n_c,
            "n_incorrect": n_i,
            "n_total": n_c + n_i,
        }

        if n_c > 0:
            entry["correct_mean_steps"] = float(np.mean([t["n_steps"] for t in bd["correct"]]))
            entry["correct_mean_full_p"] = float(np.mean([t["full_p_correct"] for t in bd["correct"]]))
        if n_i > 0:
            entry["incorrect_mean_steps"] = float(np.mean([t["n_steps"] for t in bd["incorrect"]]))
            entry["incorrect_mean_full_p"] = float(np.mean([t["full_p_correct"] for t in bd["incorrect"]]))

        # Per-benchmark anomaly detection using max_drop
        if n_c > 0 and n_i > 0:
            bench_trajs = bd["correct"] + bd["incorrect"]
            bench_labels = np.array([1 - t["is_correct"] for t in bench_trajs])
            bench_drops = np.array([
                compute_confidence_drop(t["step_scores"])["max_drop"] for t in bench_trajs
            ])
            if len(set(bench_labels)) > 1:
                try:
                    entry["max_drop_auroc"] = float(roc_auc_score(bench_labels, bench_drops))
                except Exception:
                    pass

        per_benchmark[bench] = entry

    results["per_benchmark"] = per_benchmark

    return results


# ============================================================
# PLOTTING
# ============================================================

def plot_trajectories(analysis, fig_path):
    """Plot mean confidence trajectory for correct vs incorrect responses."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: Mean trajectory with confidence bands ---
    ax = axes[0]
    interp_x = np.array(analysis.get("interp_x", []))

    has_correct = "correct_trajectory_mean" in analysis
    has_incorrect = "incorrect_trajectory_mean" in analysis

    if has_correct and len(interp_x) > 0:
        c_mean = np.array(analysis["correct_trajectory_mean"])
        c_std = np.array(analysis["correct_trajectory_std"])
        ax.plot(interp_x, c_mean, color="C0", linewidth=2.5, label="Correct", zorder=3)
        ax.fill_between(interp_x, c_mean - c_std, c_mean + c_std,
                        alpha=0.2, color="C0", zorder=2)

    if has_incorrect and len(interp_x) > 0:
        i_mean = np.array(analysis["incorrect_trajectory_mean"])
        i_std = np.array(analysis["incorrect_trajectory_std"])
        ax.plot(interp_x, i_mean, color="C3", linewidth=2.5, label="Incorrect", zorder=3)
        ax.fill_between(interp_x, i_mean - i_std, i_mean + i_std,
                        alpha=0.2, color="C3", zorder=2)

    # Annotate divergence point
    div_step = analysis.get("divergence_relative_step")
    if div_step is not None:
        ax.axvline(x=div_step, color="gray", linestyle="--", alpha=0.6, linewidth=1)
        ax.text(div_step + 0.02, 0.05,
                f"Diverge at {div_step:.0%}\nof reasoning",
                fontsize=9, color="gray", transform=ax.get_xaxis_transform())

    # Annotate final gap
    final_gap = analysis.get("final_gap")
    if final_gap is not None:
        ax.text(0.98, 0.02,
                f"Final gap: {final_gap:.3f}",
                fontsize=10, transform=ax.transAxes,
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel("Relative reasoning progress (0 = first step, 1 = full response)", fontsize=11)
    ax.set_ylabel("Calibrator P(correct)", fontsize=11)
    ax.set_title("Mean Confidence Trajectory: Correct vs Incorrect", fontsize=12)
    ax.legend(fontsize=11, loc="lower left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # --- Right: Per-benchmark heatmap of gaps ---
    ax = axes[1]
    per_bench = analysis.get("per_benchmark", {})

    # Filter to benchmarks with both correct and incorrect
    bench_names = []
    bench_aurocs = []
    bench_n = []
    for bench, bd in sorted(per_bench.items()):
        if bd.get("max_drop_auroc") is not None and bd["n_total"] >= 5:
            bench_names.append(bench)
            bench_aurocs.append(bd["max_drop_auroc"])
            bench_n.append(bd["n_total"])

    if bench_names:
        # Sort by AUROC descending
        sorted_idx = np.argsort(bench_aurocs)[::-1]
        bench_names = [bench_names[i] for i in sorted_idx]
        bench_aurocs = [bench_aurocs[i] for i in sorted_idx]
        bench_n = [bench_n[i] for i in sorted_idx]

        y = np.arange(len(bench_names))
        colors = ["C0" if a >= 0.5 else "C3" for a in bench_aurocs]
        bars = ax.barh(y, bench_aurocs, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)

        for yi, (auroc_val, n) in enumerate(zip(bench_aurocs, bench_n)):
            ax.text(auroc_val + 0.01, yi, f"{auroc_val:.3f} (n={n})",
                    va="center", fontsize=8)

        ax.set_yticks(y)
        ax.set_yticklabels(bench_names, fontsize=9)
        ax.set_xlabel("AUROC (max_drop for detecting incorrect)", fontsize=10)
        ax.set_title("Per-Benchmark Anomaly Detection\n(confidence drop AUROC)", fontsize=11)
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="x")
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "Insufficient per-benchmark data",
                transform=ax.transAxes, ha="center", va="center", fontsize=12)

    fig.suptitle("UC-D Stage 2: Step-Truncation Confidence Scoring", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {fig_path}")


def plot_anomaly_detection(analysis, fig_path):
    """Plot precision/recall curves for detecting incorrect responses
    via confidence drop vs baseline (full P(correct))."""
    anomaly = analysis.get("anomaly_detection", {})
    if not anomaly:
        print("  No anomaly detection data for plotting, skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: Precision-Recall curves ---
    ax = axes[0]
    metric_labels = {
        "max_drop": "Max single-step drop",
        "total_drop": "Total drop (first - last)",
        "final_minus_max": "Final minus max score",
        "full_p_correct_baseline": "Baseline: full P(correct)",
    }
    metric_colors = {
        "max_drop": "C0",
        "total_drop": "C1",
        "final_minus_max": "C2",
        "full_p_correct_baseline": "gray",
    }
    metric_styles = {
        "max_drop": "-",
        "total_drop": "--",
        "final_minus_max": "-.",
        "full_p_correct_baseline": ":",
    }

    for metric_name in ["max_drop", "total_drop", "final_minus_max", "full_p_correct_baseline"]:
        if metric_name not in anomaly:
            continue
        m = anomaly[metric_name]
        recall = m.get("_recall", [])
        precision = m.get("_precision", [])
        pr_auc_val = m.get("pr_auc", 0)
        label = f"{metric_labels.get(metric_name, metric_name)} (AUC={pr_auc_val:.3f})"
        ax.plot(recall, precision,
                color=metric_colors.get(metric_name, "C4"),
                linestyle=metric_styles.get(metric_name, "-"),
                linewidth=2, label=label)

    ax.set_xlabel("Recall (detecting incorrect responses)", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall: Confidence Drop Anomaly Detection", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)

    # --- Right: Summary table ---
    ax = axes[1]
    ax.axis("off")

    table_data = []
    col_headers = ["Metric", "AUROC", "PR-AUC", "Best F1", "Precision", "Recall"]

    for metric_name in ["max_drop", "total_drop", "final_minus_max", "full_p_correct_baseline"]:
        if metric_name not in anomaly:
            continue
        m = anomaly[metric_name]
        row = [
            metric_labels.get(metric_name, metric_name),
            f"{m.get('auroc', 0):.3f}" if 'auroc' in m else "-",
            f"{m.get('pr_auc', 0):.3f}",
            f"{m.get('best_f1', 0):.3f}" if 'best_f1' in m else "-",
            f"{m.get('best_f1_precision', 0):.3f}" if 'best_f1_precision' in m else "-",
            f"{m.get('best_f1_recall', 0):.3f}" if 'best_f1_recall' in m else "-",
        ]
        table_data.append(row)

    if table_data:
        table = ax.table(cellText=table_data, colLabels=col_headers,
                         loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.3, 1.8)

        # Style header
        for j in range(len(col_headers)):
            table[0, j].set_facecolor("#4472C4")
            table[0, j].set_text_props(color="white", fontweight="bold")
        # Alternate row colors
        for i in range(len(table_data)):
            color = "#D6E4F0" if i % 2 == 0 else "white"
            for j in range(len(col_headers)):
                table[i + 1, j].set_facecolor(color)

    ax.set_title("Anomaly Detection Summary\n(positive class = incorrect response)", fontsize=12)

    # Add sample count annotation
    n_total = analysis.get("n_total", 0)
    n_incorrect = analysis.get("n_incorrect", 0)
    ax.text(0.5, -0.05,
            f"N={n_total} responses ({n_incorrect} incorrect, {n_total - n_incorrect} correct)",
            transform=ax.transAxes, ha="center", fontsize=10, style="italic")

    fig.suptitle("UC-D Stage 2: Anomaly Detection via Confidence Drop", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {fig_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="UC-D Stage 2: Step-Truncation Confidence Scoring")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Path to unified model checkpoint")
    parser.add_argument("--output_dir", default="data/use_cases/results_unified",
                        help="Output directory for results")
    parser.add_argument("--fig_dir", default="figures/use_cases_unified",
                        help="Output directory for figures")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Maximum samples to process (0 = all)")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Only process 10 samples (quick test)")
    args = parser.parse_args()

    if args.smoke_test:
        args.max_samples = 10

    output_dir = Path(args.output_dir)
    fig_dir = Path(args.fig_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    trajectories_path = output_dir / "uc_d_step_trajectories.jsonl"
    results_path = output_dir / "uc_d_s2_results.json"

    # ---- Load raw predictions from all target models ----
    print("=" * 70)
    print("UC-D Stage 2: Step-Truncation Confidence Scoring")
    print("=" * 70)

    all_samples = []
    for target, config in TARGET_CONFIGS.items():
        print(f"\nLoading {config['name']} predictions...")
        if config["mode"] == "combined":
            samples = load_raw_predictions_combined(config["data_dir"])
        else:
            samples = load_raw_predictions_prefixed(config["prefix"])

        # Tag with model name
        for s in samples:
            s["model"] = target

        print(f"  Loaded {len(samples)} samples with response_text >= {MIN_RESPONSE_LEN} chars")
        all_samples.extend(samples)

    print(f"\nTotal raw samples: {len(all_samples)}")

    # ---- Split into steps and filter to multi-step responses ----
    print("\nSplitting responses into steps...")
    multi_step_samples = []
    step_count_dist = defaultdict(int)
    skipped_reasons = defaultdict(int)

    for sample in all_samples:
        steps, separator = split_into_steps(sample["response_text"])
        if len(steps) < MIN_STEPS:
            skipped_reasons["< 3 steps"] += 1
            continue

        sample["_steps"] = steps
        sample["_separator"] = separator
        sample["_n_steps"] = len(steps)
        multi_step_samples.append(sample)
        step_count_dist[len(steps)] += 1

    print(f"  Multi-step responses (>= {MIN_STEPS} steps): {len(multi_step_samples)}")
    print(f"  Skipped: {dict(skipped_reasons)}")

    # Print step count distribution
    print(f"  Step count distribution:")
    for k in sorted(step_count_dist.keys()):
        print(f"    {k} steps: {step_count_dist[k]}")

    # Per-model breakdown
    for target in TARGET_CONFIGS:
        n = sum(1 for s in multi_step_samples if s["model"] == target)
        print(f"  {TARGET_CONFIGS[target]['name']}: {n} multi-step samples")

    if not multi_step_samples:
        print("\nERROR: No multi-step samples found. Nothing to do.")
        sys.exit(1)

    # ---- Apply max_samples limit ----
    if args.max_samples > 0 and len(multi_step_samples) > args.max_samples:
        np.random.seed(42)
        indices = np.random.choice(len(multi_step_samples), args.max_samples, replace=False)
        multi_step_samples = [multi_step_samples[i] for i in sorted(indices)]
        print(f"\nSubsampled to {len(multi_step_samples)} samples (--max_samples={args.max_samples})")

    # ---- Compute total inference calls ----
    total_calls = sum(s["_n_steps"] for s in multi_step_samples)
    print(f"\nTotal inference calls needed: {total_calls}")
    print(f"  (mean {total_calls / len(multi_step_samples):.1f} per sample)")

    # ---- Load model ----
    print()
    model, processor, device = load_model(args.checkpoint)

    # ---- Score all truncations ----
    print(f"\nScoring {len(multi_step_samples)} samples with step-by-step truncation...")
    start_time = time.time()

    trajectories = []
    n_errors = 0
    calls_done = 0

    # Open output file for incremental saving
    f_out = open(trajectories_path, "w")

    try:
        for i, sample in enumerate(multi_step_samples):
            steps = sample["_steps"]
            separator = sample["_separator"]
            n_steps = sample["_n_steps"]
            question = sample["question"]

            # Load image
            image = load_image_for_sample(sample)

            # Score each truncation level
            step_scores = []
            step_lengths = []
            had_error = False

            for k in range(1, n_steps + 1):
                truncated = separator.join(steps[:k])
                step_lengths.append(len(truncated))

                try:
                    p = get_p_correct(model, processor, question, truncated, image, device)
                    step_scores.append(round(p, 6))
                except Exception as e:
                    if n_errors < 5:
                        print(f"  ERROR on sample {i} ({sample['id']}), step {k}: {e}")
                    step_scores.append(0.5)  # fallback
                    n_errors += 1
                    had_error = True

                calls_done += 1

            # Build trajectory record
            record = {
                "id": sample["id"],
                "benchmark": sample["benchmark"],
                "model": sample["model"],
                "is_correct": sample["is_correct"],
                "n_steps": n_steps,
                "full_p_correct": step_scores[-1] if step_scores else 0.5,
                "step_scores": step_scores,
                "step_lengths": step_lengths,
            }
            trajectories.append(record)

            # Write to file
            f_out.write(json.dumps(record) + "\n")

            # Flush every 100 samples
            if (i + 1) % 100 == 0:
                f_out.flush()

            # Progress reporting every 50 samples
            if (i + 1) % 50 == 0 or (i + 1) == len(multi_step_samples):
                elapsed = time.time() - start_time
                rate = calls_done / elapsed if elapsed > 0 else 0
                eta = (total_calls - calls_done) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(multi_step_samples)}] "
                      f"{calls_done}/{total_calls} calls, "
                      f"{rate:.1f} calls/s, "
                      f"ETA {eta/60:.1f}m, "
                      f"errors={n_errors}")

    finally:
        f_out.close()

    elapsed_total = time.time() - start_time
    print(f"\nScoring complete: {len(trajectories)} trajectories in {elapsed_total:.0f}s")
    print(f"  ({calls_done} total inference calls, {n_errors} errors)")
    print(f"  Trajectories saved to: {trajectories_path}")

    # ---- Analyze trajectories ----
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")

    analysis = analyze_trajectories(trajectories)

    # Print summary
    if "drop_stats" in analysis:
        ds = analysis["drop_stats"]
        print(f"\n  Confidence Drop Statistics:")
        print(f"  {'Metric':<25} {'Correct':>10} {'Incorrect':>10} {'Delta':>10}")
        print(f"  {'-'*57}")
        for key in ["mean_max_drop", "mean_total_drop", "mean_slope", "mean_final_minus_max"]:
            c_val = ds["correct"].get(key, 0)
            i_val = ds["incorrect"].get(key, 0)
            delta = i_val - c_val
            print(f"  {key:<25} {c_val:>10.4f} {i_val:>10.4f} {delta:>+10.4f}")

    if "anomaly_detection" in analysis:
        ad = analysis["anomaly_detection"]
        print(f"\n  Anomaly Detection (detecting incorrect via confidence drop):")
        print(f"  {'Metric':<30} {'AUROC':>8} {'PR-AUC':>8} {'Best F1':>8}")
        print(f"  {'-'*56}")
        for metric_name in ["max_drop", "total_drop", "final_minus_max", "full_p_correct_baseline"]:
            if metric_name in ad:
                m = ad[metric_name]
                auroc_str = f"{m['auroc']:.3f}" if 'auroc' in m else "-"
                pr_auc_str = f"{m['pr_auc']:.3f}"
                f1_str = f"{m.get('best_f1', 0):.3f}" if 'best_f1' in m else "-"
                label = {
                    "max_drop": "Max single-step drop",
                    "total_drop": "Total drop (first - last)",
                    "final_minus_max": "Final minus max score",
                    "full_p_correct_baseline": "Baseline: full P(correct)",
                }.get(metric_name, metric_name)
                print(f"  {label:<30} {auroc_str:>8} {pr_auc_str:>8} {f1_str:>8}")

    div = analysis.get("divergence_relative_step")
    if div is not None:
        print(f"\n  Divergence point: correct and incorrect trajectories diverge at "
              f"{div:.0%} of reasoning (gap={analysis.get('divergence_gap', 0):.3f})")
    print(f"  Final gap (correct - incorrect): {analysis.get('final_gap', 0):.3f}")

    # Per-benchmark summary
    if "per_benchmark" in analysis:
        print(f"\n  Per-Benchmark Summary:")
        print(f"  {'Benchmark':<22} {'N':>6} {'Correct':>8} {'Incorrect':>10} {'Drop AUROC':>12}")
        print(f"  {'-'*60}")
        for bench in sorted(analysis["per_benchmark"].keys()):
            bd = analysis["per_benchmark"][bench]
            auroc_str = f"{bd['max_drop_auroc']:.3f}" if 'max_drop_auroc' in bd else "-"
            print(f"  {bench:<22} {bd['n_total']:>6} {bd['n_correct']:>8} "
                  f"{bd['n_incorrect']:>10} {auroc_str:>12}")

    # ---- Save analysis results ----
    # Clean up internal fields before saving
    save_analysis = {}
    for k, v in analysis.items():
        if k == "anomaly_detection":
            # Remove internal plotting arrays from saved JSON
            cleaned_ad = {}
            for mk, mv in v.items():
                cleaned_mv = {kk: vv for kk, vv in mv.items() if not kk.startswith("_")}
                cleaned_ad[mk] = cleaned_mv
            save_analysis[k] = cleaned_ad
        else:
            save_analysis[k] = v

    save_analysis["config"] = {
        "checkpoint": args.checkpoint,
        "max_samples": args.max_samples,
        "smoke_test": args.smoke_test,
        "min_steps": MIN_STEPS,
        "min_response_len": MIN_RESPONSE_LEN,
        "targets": list(TARGET_CONFIGS.keys()),
        "elapsed_seconds": elapsed_total,
        "total_inference_calls": calls_done,
        "n_errors": n_errors,
    }

    with open(results_path, "w") as f:
        json.dump(save_analysis, f, indent=2, default=str)
    print(f"\nAnalysis saved to: {results_path}")

    # ---- Generate figures ----
    print(f"\nGenerating figures...")
    plot_trajectories(analysis, str(fig_dir / "uc_d_trajectories.pdf"))
    plot_anomaly_detection(analysis, str(fig_dir / "uc_d_anomaly_detection.pdf"))

    # ---- Final summary ----
    print(f"\n{'='*70}")
    print("UC-D Stage 2 Complete")
    print(f"{'='*70}")
    print(f"  Trajectories:  {trajectories_path}")
    print(f"  Results:       {results_path}")
    print(f"  Figures:       {fig_dir}/uc_d_trajectories.pdf")
    print(f"                 {fig_dir}/uc_d_anomaly_detection.pdf")
    print(f"  Samples:       {len(trajectories)}")
    print(f"  Inference:     {calls_done} calls in {elapsed_total:.0f}s")
    if n_errors > 0:
        print(f"  Errors:        {n_errors}")
    print()


if __name__ == "__main__":
    main()
