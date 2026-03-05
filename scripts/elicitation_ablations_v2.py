#!/usr/bin/env python3
"""Additional elicitation ablations for the UQ judge (v2 checkpoint).

Three strategies:
  1. temperature_scaling — Post-hoc calibration of existing logits. CPU-only.
     Standard baseline reviewers expect. Preserves AUROC, improves ECE/Brier.
  2. token_entropy — Full vocabulary entropy at classification position.
     Tests whether broader uncertainty signal beats binary (i)/(ii) logits.
  3. hidden_state_probing — Extract last hidden state, train MLP classifier.
     Tests whether hidden representations carry richer signal than logits.

Usage:
    # Temperature scaling (CPU only, uses existing scored data)
    python scripts/elicitation_ablations_v2.py --strategy temperature_scaling

    # Token entropy (needs GPU)
    CUDA_VISIBLE_DEVICES=2 python scripts/elicitation_ablations_v2.py --strategy token_entropy

    # Hidden state probing (needs GPU)
    CUDA_VISIBLE_DEVICES=2 python scripts/elicitation_ablations_v2.py --strategy hidden_state_probing

    # All GPU strategies together (single model load)
    CUDA_VISIBLE_DEVICES=2 python scripts/elicitation_ablations_v2.py --strategy gpu_all

    # Smoke test
    CUDA_VISIBLE_DEVICES=2 python scripts/elicitation_ablations_v2.py --strategy gpu_all --smoke_test
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss

# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_CHECKPOINT = "uq_models/best_v2_r32_combined"
IMAGE_CACHE_DIR = Path("data/training_images")
SCORED_DIR = "data/use_cases/scored_test_only_v2"
OUTPUT_DIR = "data/ablations/elicitation_v2"

VLM_BENCHMARKS = {
    "charxiv", "mmmu", "mmstar", "hallusionbench", "mathverse",
    "mathvision", "mathvista", "realworldqa", "vizwiz", "hle_multimodal", "mmvet",
}

PROMPT_TEMPLATE = """Benchmark: {benchmark}
Source model: {source_model}

Question: {question}

Answer: {response}

Analyze whether the answer above is correct. Consider:
- Does the answer address the question?
- Are there factual errors or logical flaws?
- Is the answer complete?

Based on your analysis, is the answer correct? (i) No (ii) Yes"""


# ============================================================
# HELPERS
# ============================================================

def load_scored_data(scored_dir, test_only_dir=None):
    """Load scored data with p_correct and labels.

    If test_only_dir is provided, loads full data from scored_dir but
    filters to only IDs present in test_only_dir (for full Q/A text).
    """
    # Build test-only ID set if needed
    test_ids = None
    if test_only_dir and Path(test_only_dir).exists():
        test_ids = set()
        for fname in Path(test_only_dir).glob("*_scored.jsonl"):
            model_name = fname.stem.replace("_scored", "")
            with open(fname) as f:
                for line in f:
                    try:
                        pred = json.loads(line)
                        test_ids.add((model_name, str(pred.get("id", ""))))
                    except json.JSONDecodeError:
                        continue
        print(f"Test-only filter: {len(test_ids)} IDs")

    samples = []
    for fname in sorted(Path(scored_dir).glob("*_scored.jsonl")):
        model_name = fname.stem.replace("_scored", "")
        with open(fname) as f:
            for line in f:
                try:
                    pred = json.loads(line)
                except json.JSONDecodeError:
                    continue

                sid = str(pred.get("id", ""))

                # Filter to test-only if requested
                if test_ids is not None and (model_name, sid) not in test_ids:
                    continue

                p = pred.get("p_correct")
                # Try is_correct first (test-only format), then score.correct
                correct = pred.get("is_correct", -1)
                if correct == -1:
                    score = pred.get("score", {})
                    if isinstance(score, dict):
                        correct = score.get("correct", -1)
                if p is None or correct == -1:
                    continue

                # Get full Q/A text (from full scored files)
                question = pred.get("question", pred.get("question_preview", ""))
                response = pred.get("response", pred.get("response_preview", ""))

                samples.append({
                    "id": sid,
                    "benchmark": pred.get("benchmark", ""),
                    "question": question[:1500],
                    "response": response[:800],
                    "p_correct": float(p),
                    "correct": int(correct),
                    "source_model": model_name,
                    "has_image": pred.get("benchmark", "") in VLM_BENCHMARKS,
                })
    return samples


def compute_metrics(labels, preds, per_benchmark=None):
    """Compute AUROC, Brier, ECE and VLM/text splits."""
    labels = np.array(labels)
    preds = np.array(preds)
    results = {}
    if len(set(labels)) > 1:
        results["auroc"] = float(roc_auc_score(labels, preds))
    else:
        results["auroc"] = 0.5
    results["n_samples"] = len(labels)

    # Brier and ECE only make sense for probability-valued predictions [0, 1]
    if preds.min() >= 0 and preds.max() <= 1:
        results["brier"] = float(brier_score_loss(labels, preds))
        bins = np.linspace(0, 1, 11)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (preds >= lo) & (preds < hi)
            if mask.sum() > 0:
                bin_acc = labels[mask].mean()
                bin_conf = preds[mask].mean()
                ece += mask.sum() / len(labels) * abs(bin_acc - bin_conf)
        results["ece"] = float(ece)

    # VLM/text split
    if per_benchmark:
        vlm_p, vlm_l, txt_p, txt_l = [], [], [], []
        for bench, data in per_benchmark.items():
            if bench in VLM_BENCHMARKS:
                vlm_p.extend(data["preds"])
                vlm_l.extend(data["labels"])
            else:
                txt_p.extend(data["preds"])
                txt_l.extend(data["labels"])
        if vlm_l and len(set(vlm_l)) > 1:
            results["vlm_auroc"] = float(roc_auc_score(vlm_l, vlm_p))
        if txt_l and len(set(txt_l)) > 1:
            results["text_auroc"] = float(roc_auc_score(txt_l, txt_p))

    return results


# ============================================================
# STRATEGY 1: TEMPERATURE SCALING (CPU only)
# ============================================================

def run_temperature_scaling(args):
    """Fit temperature T to minimize NLL on validation set, report calibration improvement."""
    from scipy.optimize import minimize_scalar

    print(f"\n{'='*60}")
    print("TEMPERATURE SCALING (post-hoc calibration)")
    print(f"Scored dir: {args.scored_dir}")
    print(f"{'='*60}")

    samples = load_scored_data(args.scored_dir, test_only_dir=args.test_only_dir)
    if args.smoke_test:
        samples = samples[:100]
    print(f"Loaded {len(samples)} scored samples")

    # Split into calibration and evaluation sets
    np.random.seed(42)
    indices = np.random.permutation(len(samples))
    cal_size = len(samples) // 3  # 33% for fitting T
    cal_idx = indices[:cal_size]
    eval_idx = indices[cal_size:]

    cal_p = np.array([samples[i]["p_correct"] for i in cal_idx])
    cal_y = np.array([samples[i]["correct"] for i in cal_idx])
    eval_p = np.array([samples[i]["p_correct"] for i in eval_idx])
    eval_y = np.array([samples[i]["correct"] for i in eval_idx])

    # Convert p_correct to logits (inverse sigmoid)
    eps = 1e-7
    cal_logits = np.log(np.clip(cal_p, eps, 1 - eps) / (1 - np.clip(cal_p, eps, 1 - eps)))
    eval_logits = np.log(np.clip(eval_p, eps, 1 - eps) / (1 - np.clip(eval_p, eps, 1 - eps)))

    # Fit T on calibration set (minimize NLL)
    def nll(T):
        scaled = 1 / (1 + np.exp(-cal_logits / T))
        scaled = np.clip(scaled, eps, 1 - eps)
        return -np.mean(cal_y * np.log(scaled) + (1 - cal_y) * np.log(1 - scaled))

    result = minimize_scalar(nll, bounds=(0.01, 10.0), method='bounded')
    T_opt = result.x
    print(f"Optimal temperature: T = {T_opt:.4f}")

    # Apply to eval set
    eval_scaled = 1 / (1 + np.exp(-eval_logits / T_opt))

    # Collect per-benchmark
    per_bench_before = defaultdict(lambda: {"preds": [], "labels": []})
    per_bench_after = defaultdict(lambda: {"preds": [], "labels": []})
    for idx_pos, idx in enumerate(eval_idx):
        s = samples[idx]
        per_bench_before[s["benchmark"]]["preds"].append(eval_p[idx_pos])
        per_bench_before[s["benchmark"]]["labels"].append(eval_y[idx_pos])
        per_bench_after[s["benchmark"]]["preds"].append(eval_scaled[idx_pos])
        per_bench_after[s["benchmark"]]["labels"].append(eval_y[idx_pos])

    before = compute_metrics(eval_y, eval_p, per_bench_before)
    after = compute_metrics(eval_y, eval_scaled, per_bench_after)

    print(f"\nBefore temperature scaling:")
    print(f"  AUROC: {before['auroc']:.4f}  Brier: {before['brier']:.4f}  ECE: {before['ece']:.4f}")
    print(f"After temperature scaling (T={T_opt:.4f}):")
    print(f"  AUROC: {after['auroc']:.4f}  Brier: {after['brier']:.4f}  ECE: {after['ece']:.4f}")

    results = {
        "strategy": "temperature_scaling",
        "temperature": float(T_opt),
        "cal_size": int(cal_size),
        "eval_size": int(len(eval_idx)),
        "before": before,
        "after": after,
        "auroc_delta": after["auroc"] - before["auroc"],
        "brier_delta": after["brier"] - before["brier"],
        "ece_delta": after["ece"] - before["ece"],
    }

    out_dir = Path(args.output_dir) / "temperature_scaling"
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_dir / 'results.json'}")

    return results


# ============================================================
# STRATEGY 2 & 3: TOKEN ENTROPY + HIDDEN STATE PROBING
# (Combined — single model load)
# ============================================================

def run_gpu_strategies(args):
    """Run token entropy and hidden state probing with a single model load."""
    import torch
    from PIL import Image
    from peft import PeftModel
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    print(f"\n{'='*60}")
    print("GPU STRATEGIES: Token Entropy + Hidden State Probing")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'='*60}")

    # Load scored data for labels and metadata
    samples = load_scored_data(args.scored_dir, test_only_dir=args.test_only_dir)
    if args.smoke_test:
        samples = samples[:50]
    print(f"Loaded {len(samples)} scored samples")

    # Split
    np.random.seed(42)
    indices = np.random.permutation(len(samples))
    train_size = int(len(samples) * 0.7)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

    # Load model
    print(f"\nLoading model...")
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.checkpoint)
    model.eval()

    device = next(model.parameters()).device
    fallback = Image.new('RGB', (224, 224), color='gray')

    token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
    token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]

    # Collect: logit p_correct, entropy, hidden states
    all_logit_p = []
    all_entropy = []
    all_hidden = []
    all_labels = []
    all_benchmarks = []

    t0 = time.time()
    for idx, sample in enumerate(samples):
        if idx % 50 == 0:
            elapsed = time.time() - t0
            rate = idx / max(elapsed, 1)
            eta = (len(samples) - idx) / max(rate, 0.01) / 60
            print(f"  [{idx}/{len(samples)}] {rate:.1f} samples/s, ETA {eta:.1f}m")

        # Load image
        if sample["has_image"]:
            cache_path = IMAGE_CACHE_DIR / sample["benchmark"] / f"{sample['id']}.jpg"
            if cache_path.exists():
                try:
                    image = Image.open(cache_path).convert("RGB")
                except Exception:
                    image = fallback
            else:
                image = fallback
        else:
            image = fallback

        prompt = PROMPT_TEMPLATE.format(
            question=sample["question"],
            response=sample["response"],
            benchmark=sample["benchmark"],
            source_model=sample["source_model"],
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

        try:
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            logits = outputs.logits[0, -1, :]

            # Binary logit p_correct (baseline)
            probs_binary = torch.softmax(logits[[token_i, token_ii]], dim=0)
            p_correct = probs_binary[1].item()

            # Token entropy over full vocabulary
            full_probs = torch.softmax(logits, dim=0)
            # Clamp for numerical stability
            full_probs = torch.clamp(full_probs, min=1e-10)
            entropy = -torch.sum(full_probs * torch.log(full_probs)).item()

            # Hidden state (last layer, last token)
            hidden = outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()

        except Exception as e:
            if idx < 3:
                print(f"  WARNING: {e}")
            p_correct = 0.5
            entropy = 0.0
            hidden = np.zeros(model.config.hidden_size)

        all_logit_p.append(p_correct)
        all_entropy.append(entropy)
        all_hidden.append(hidden)
        all_labels.append(sample["correct"])
        all_benchmarks.append(sample["benchmark"])

    elapsed = time.time() - t0
    print(f"\nForward passes done in {elapsed/60:.1f} minutes")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    all_logit_p = np.array(all_logit_p)
    all_entropy = np.array(all_entropy)
    all_hidden = np.array(all_hidden)
    all_labels = np.array(all_labels)

    # ============================================================
    # TOKEN ENTROPY ANALYSIS
    # ============================================================
    print(f"\n{'='*60}")
    print("TOKEN ENTROPY RESULTS")
    print(f"{'='*60}")

    # Higher entropy = more uncertain = LOWER confidence
    # So for AUROC (higher score = more likely correct), use negative entropy
    neg_entropy = -all_entropy

    # Evaluate on test split
    test_labels = all_labels[test_idx]
    test_neg_entropy = neg_entropy[test_idx]
    test_logit_p = all_logit_p[test_idx]

    per_bench_entropy = defaultdict(lambda: {"preds": [], "labels": []})
    per_bench_logit = defaultdict(lambda: {"preds": [], "labels": []})
    for pos, idx in enumerate(test_idx):
        b = all_benchmarks[idx]
        per_bench_entropy[b]["preds"].append(test_neg_entropy[pos])
        per_bench_entropy[b]["labels"].append(test_labels[pos])
        per_bench_logit[b]["preds"].append(test_logit_p[pos])
        per_bench_logit[b]["labels"].append(test_labels[pos])

    entropy_metrics = compute_metrics(test_labels, test_neg_entropy, per_bench_entropy)
    logit_metrics = compute_metrics(test_labels, test_logit_p, per_bench_logit)

    print(f"  Logit baseline AUROC: {logit_metrics['auroc']:.4f}")
    print(f"  Token entropy AUROC:  {entropy_metrics['auroc']:.4f}")
    print(f"  Delta: {entropy_metrics['auroc'] - logit_metrics['auroc']:+.4f}")

    # Also try combining: logit + entropy as features with simple logistic regression
    from sklearn.linear_model import LogisticRegression

    train_X = np.column_stack([all_logit_p[train_idx], neg_entropy[train_idx]])
    train_y = all_labels[train_idx]
    test_X = np.column_stack([test_logit_p, test_neg_entropy])

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(train_X, train_y)
    combined_preds = lr.predict_proba(test_X)[:, 1]
    combined_metrics = compute_metrics(test_labels, combined_preds)

    print(f"  Combined (logit+entropy) AUROC: {combined_metrics['auroc']:.4f}")

    entropy_stats = {
        "mean": float(all_entropy.mean()),
        "std": float(all_entropy.std()),
        "median": float(np.median(all_entropy)),
        "correct_mean": float(all_entropy[all_labels == 1].mean()),
        "incorrect_mean": float(all_entropy[all_labels == 0].mean()),
    }
    print(f"\n  Entropy stats:")
    print(f"    Correct answers:   mean={entropy_stats['correct_mean']:.3f}")
    print(f"    Incorrect answers: mean={entropy_stats['incorrect_mean']:.3f}")

    entropy_results = {
        "strategy": "token_entropy",
        "logit_baseline": logit_metrics,
        "token_entropy": entropy_metrics,
        "combined_logit_entropy": combined_metrics,
        "entropy_stats": entropy_stats,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "inference_time_minutes": elapsed / 60,
    }

    out_dir = Path(args.output_dir) / "token_entropy"
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(entropy_results, f, indent=2)
    print(f"  Saved to {out_dir / 'results.json'}")

    # ============================================================
    # HIDDEN STATE PROBING
    # ============================================================
    print(f"\n{'='*60}")
    print("HIDDEN STATE PROBING RESULTS")
    print(f"{'='*60}")

    hidden_dim = all_hidden.shape[1]
    print(f"  Hidden dim: {hidden_dim}")

    # Train MLP: hidden_dim -> 256 -> 1
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    # Scale features
    scaler = StandardScaler()
    train_H = scaler.fit_transform(all_hidden[train_idx])
    test_H = scaler.transform(all_hidden[test_idx])

    # MLP probe
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 64),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        learning_rate_init=1e-3,
    )
    print("  Training MLP probe...")
    mlp.fit(train_H, all_labels[train_idx])
    probe_preds = mlp.predict_proba(test_H)[:, 1]

    per_bench_probe = defaultdict(lambda: {"preds": [], "labels": []})
    for pos, idx in enumerate(test_idx):
        b = all_benchmarks[idx]
        per_bench_probe[b]["preds"].append(probe_preds[pos])
        per_bench_probe[b]["labels"].append(test_labels[pos])

    probe_metrics = compute_metrics(test_labels, probe_preds, per_bench_probe)

    print(f"  Logit baseline AUROC: {logit_metrics['auroc']:.4f}")
    print(f"  MLP probe AUROC:      {probe_metrics['auroc']:.4f}")
    print(f"  Delta: {probe_metrics['auroc'] - logit_metrics['auroc']:+.4f}")

    # Also try: logistic regression on hidden states (simpler probe)
    from sklearn.linear_model import LogisticRegression as LR2
    lr_probe = LR2(max_iter=1000, random_state=42, C=1.0)
    lr_probe.fit(train_H, all_labels[train_idx])
    lr_probe_preds = lr_probe.predict_proba(test_H)[:, 1]
    lr_probe_metrics = compute_metrics(test_labels, lr_probe_preds)

    print(f"  Linear probe AUROC:   {lr_probe_metrics['auroc']:.4f}")

    # Combined: hidden state + logit
    train_HL = np.column_stack([train_H, all_logit_p[train_idx]])
    test_HL = np.column_stack([test_H, test_logit_p])
    mlp_combined = MLPClassifier(
        hidden_layer_sizes=(256, 64),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
    )
    mlp_combined.fit(train_HL, all_labels[train_idx])
    combined_probe_preds = mlp_combined.predict_proba(test_HL)[:, 1]
    combined_probe_metrics = compute_metrics(test_labels, combined_probe_preds)

    print(f"  MLP (hidden+logit) AUROC: {combined_probe_metrics['auroc']:.4f}")

    probe_results = {
        "strategy": "hidden_state_probing",
        "hidden_dim": hidden_dim,
        "logit_baseline": logit_metrics,
        "mlp_probe": probe_metrics,
        "linear_probe": lr_probe_metrics,
        "mlp_combined_logit": combined_probe_metrics,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "mlp_config": {"layers": [256, 64], "max_iter": 500, "early_stopping": True},
    }

    out_dir = Path(args.output_dir) / "hidden_state_probing"
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(probe_results, f, indent=2)
    print(f"  Saved to {out_dir / 'results.json'}")

    # Save hidden states for potential future use
    np.savez_compressed(
        out_dir / "hidden_states.npz",
        hidden=all_hidden, labels=all_labels,
        train_idx=train_idx, test_idx=test_idx,
    )
    print(f"  Hidden states saved to {out_dir / 'hidden_states.npz'}")

    return entropy_results, probe_results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Elicitation ablations v2")
    parser.add_argument("--strategy", required=True,
                        choices=["temperature_scaling", "token_entropy",
                                 "hidden_state_probing", "gpu_all", "all"])
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--scored_dir", default="data/use_cases/scored_test_only_v2",
                        help="Full scored data (with Q/A text)")
    parser.add_argument("--test_only_dir", default=SCORED_DIR,
                        help="Test-only IDs for filtering")
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}

    if args.strategy in ("temperature_scaling", "all"):
        r = run_temperature_scaling(args)
        results["temperature_scaling"] = r

    if args.strategy in ("token_entropy", "hidden_state_probing", "gpu_all", "all"):
        entropy_r, probe_r = run_gpu_strategies(args)
        results["token_entropy"] = entropy_r
        results["hidden_state_probing"] = probe_r

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if "temperature_scaling" in results:
        r = results["temperature_scaling"]
        print(f"  Temp scaling: T={r['temperature']:.3f}, "
              f"ECE {r['before']['ece']:.4f} -> {r['after']['ece']:.4f}, "
              f"Brier {r['before']['brier']:.4f} -> {r['after']['brier']:.4f}")
    if "token_entropy" in results:
        r = results["token_entropy"]
        print(f"  Token entropy: AUROC={r['token_entropy']['auroc']:.4f} "
              f"(baseline {r['logit_baseline']['auroc']:.4f}, "
              f"combined {r['combined_logit_entropy']['auroc']:.4f})")
    if "hidden_state_probing" in results:
        r = results["hidden_state_probing"]
        print(f"  MLP probe: AUROC={r['mlp_probe']['auroc']:.4f} "
              f"(baseline {r['logit_baseline']['auroc']:.4f}, "
              f"combined {r['mlp_combined_logit']['auroc']:.4f})")

    # Save combined summary
    summary_path = Path(args.output_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
