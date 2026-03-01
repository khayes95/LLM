#!/usr/bin/env python3
"""Novel elicitation strategy ablations for the UQ judge.

Three strategies:
  1. multi_sample  — Run inference N times with temperature>0, use agreement as confidence.
                     NO retraining needed — uses existing best_unified checkpoint.
  2. contrastive   — Include reference/gold answer in prompt: "Does the answer match?"
                     Requires retraining with new prompt template.
  3. verbalized    — Fine-tune the judge to output a calibrated probability (e.g. "0.73")
                     instead of binary (i)/(ii). Uses MSE loss on output probability.

Usage:
    # Multi-sample (inference only, no training)
    CUDA_VISIBLE_DEVICES=0 python scripts/elicitation_ablations.py \
        --strategy multi_sample --n_samples 5 \
        --checkpoint uq_models/best_unified \
        --output_dir data/ablations/elicitation/multi_sample_5

    # Contrastive (requires training)
    CUDA_VISIBLE_DEVICES=0 python scripts/elicitation_ablations.py \
        --strategy contrastive \
        --output_dir data/ablations/elicitation/contrastive

    # Verbalized judge (requires training)
    CUDA_VISIBLE_DEVICES=0 python scripts/elicitation_ablations.py \
        --strategy verbalized \
        --output_dir data/ablations/elicitation/verbalized

    # Smoke test
    CUDA_VISIBLE_DEVICES=0 python scripts/elicitation_ablations.py \
        --strategy multi_sample --n_samples 3 --smoke_test \
        --output_dir data/ablations/elicitation/smoke
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_best_uq import (
    MODEL_NAME, DATA_SOURCES, VLM_BENCHMARKS, EXCLUDED,
    IMAGE_CACHE_DIR, Sample, load_all_samples, prepare_all_images,
)


# ============================================================
# DATA LOADING WITH GOLD ANSWERS
# ============================================================

def load_samples_with_gold(max_per_benchmark=None):
    """Load samples including gold/reference answers where available."""
    from scripts.train_best_uq import load_combined_dir, load_prefixed_runs

    all_samples = []
    gold_answers = {}  # id -> gold answer string

    # Load from each source, also extract gold answers
    for model_name, config in DATA_SOURCES.items():
        if config["type"] == "combined":
            combined_path = Path(config["dir"])
            for bench_dir in sorted(combined_path.iterdir()):
                if not bench_dir.is_dir() or bench_dir.name in EXCLUDED:
                    continue
                pred_file = bench_dir / "predictions.jsonl"
                if not pred_file.exists():
                    continue
                with open(pred_file) as f:
                    for line in f:
                        try:
                            pred = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        sid = str(pred.get("id", ""))
                        # Extract gold answer
                        gold = pred.get("target", "")
                        if not gold:
                            score = pred.get("score", {})
                            if isinstance(score, dict):
                                gold = score.get("gold", "")
                        if not gold:
                            meta = pred.get("meta", {})
                            if isinstance(meta, dict):
                                gold = meta.get("correct_answer_text", "")
                        if gold and sid:
                            gold_answers[sid] = str(gold)[:500]
        else:
            runs_path = Path(config["dir"])
            prefix = config["prefix"]
            for run_dir in sorted(runs_path.iterdir()):
                if not run_dir.name.startswith(prefix):
                    continue
                benchmark = run_dir.name[len(prefix):]
                if benchmark in EXCLUDED:
                    continue
                pred_file = run_dir / "predictions.jsonl"
                if not pred_file.exists():
                    continue
                with open(pred_file) as f:
                    for line in f:
                        try:
                            pred = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        sid = str(pred.get("id", ""))
                        gold = pred.get("target", "")
                        if not gold:
                            score = pred.get("score", {})
                            if isinstance(score, dict):
                                gold = score.get("gold", "")
                        if gold and sid:
                            gold_answers[sid] = str(gold)[:500]

    # Load samples normally
    all_samples = load_all_samples(max_per_benchmark=max_per_benchmark)

    print(f"Gold answers available for {len(gold_answers)}/{len(all_samples)} samples "
          f"({100*len(gold_answers)/max(len(all_samples),1):.0f}%)")

    return all_samples, gold_answers


# ============================================================
# PROMPT TEMPLATES
# ============================================================

CONTRASTIVE_PROMPT = """Question: {question}

Model's answer: {response}

Reference answer: {gold}

Does the model's answer match the reference answer in meaning? (i) No (ii) Yes"""

VERBALIZED_PROMPT = """Question: {question}

Answer: {response}

What is the probability that this answer is correct? Output a number between 0.0 and 1.0."""


# ============================================================
# CANONICAL SPLIT
# ============================================================

def get_canonical_split(all_samples, test_fraction=0.15):
    strat_key = [s.benchmark for s in all_samples]
    strat_counts = defaultdict(int)
    for k in strat_key:
        strat_counts[k] += 1
    strat_key_safe = [k if strat_counts[k] >= 3 else "other" for k in strat_key]

    indices = list(range(len(all_samples)))
    try:
        train_idx, test_idx = train_test_split(
            indices, test_size=test_fraction, random_state=42, stratify=strat_key_safe
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            indices, test_size=test_fraction, random_state=42
        )
    return train_idx, test_idx


# ============================================================
# STRATEGY 1: MULTI-SAMPLE CONSISTENCY
# ============================================================

def enable_dropout(model):
    """Enable dropout layers for MC Dropout inference."""
    n_enabled = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()
            n_enabled += 1
    return n_enabled


def run_multi_sample(args, test_samples):
    """Run MC Dropout inference N times, use agreement and logit spread as confidence.

    Uses Monte Carlo Dropout (Gal & Ghahramani 2016) — keeps dropout active during
    inference so each forward pass produces different logits. This gives genuine
    epistemic uncertainty estimates from a single checkpoint.
    """
    print(f"\n{'='*60}")
    print(f"MC DROPOUT CONSISTENCY (N={args.n_samples}, temp={args.temperature})")
    print(f"Using checkpoint: {args.checkpoint}")
    print(f"{'='*60}")

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.checkpoint)
    model.eval()  # start in eval mode

    fallback = Image.new('RGB', (224, 224), color='gray')
    device = next(model.parameters()).device

    PROMPT_TEMPLATE = """Question: {question}

Answer: {response}

Is the answer correct? (i) No (ii) Yes"""

    all_preds_logit = []      # single-pass logit (eval mode, baseline)
    all_preds_mc_mean = []    # mean of N MC dropout logit probs
    all_preds_mc_std = []     # std of N MC dropout logit probs (uncertainty)
    all_preds_mc_vote = []    # fraction of N passes voting "correct" (p > 0.5)
    all_labels = []
    per_benchmark = defaultdict(lambda: {"logit": [], "mc_mean": [], "mc_vote": [], "labels": []})

    t0 = time.time()

    for i, sample in enumerate(test_samples):
        if i % 50 == 0:
            print(f"  Sample {i}/{len(test_samples)}...")

        # Load image
        if sample.has_image:
            cache_path = IMAGE_CACHE_DIR / sample.benchmark / f"{sample.id}.jpg"
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
            question=sample.question[:500],
            response=sample.response[:300]
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

        token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
        token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]

        # Single-pass baseline (eval mode — dropout OFF)
        model.eval()
        try:
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
            p_logit = probs[1].item()
        except Exception:
            p_logit = 0.5

        # MC Dropout: enable dropout, run N forward passes
        model.eval()
        n_dropout = enable_dropout(model)  # re-enable dropout layers only
        if i == 0:
            print(f"  MC Dropout: enabled {n_dropout} dropout layers")

        mc_probs = []
        for _ in range(args.n_samples):
            try:
                with torch.no_grad():
                    outputs = model(**inputs)
                logits = outputs.logits[0, -1, :]
                scaled_logits = logits[[token_i, token_ii]] / args.temperature
                probs = torch.softmax(scaled_logits, dim=0)
                mc_probs.append(probs[1].item())
            except Exception:
                mc_probs.append(0.5)

        mc_mean = np.mean(mc_probs)
        mc_std = np.std(mc_probs)
        mc_vote = np.mean([1.0 if p > 0.5 else 0.0 for p in mc_probs])

        all_preds_logit.append(p_logit)
        all_preds_mc_mean.append(mc_mean)
        all_preds_mc_std.append(mc_std)
        all_preds_mc_vote.append(mc_vote)
        all_labels.append(float(sample.is_correct))

        per_benchmark[sample.benchmark]["logit"].append(p_logit)
        per_benchmark[sample.benchmark]["mc_mean"].append(mc_mean)
        per_benchmark[sample.benchmark]["mc_vote"].append(mc_vote)
        per_benchmark[sample.benchmark]["labels"].append(float(sample.is_correct))

    elapsed = time.time() - t0

    # Compute metrics
    labels = np.array(all_labels)
    results = {"strategy": "multi_sample_mc_dropout", "n_samples": args.n_samples,
               "temperature": args.temperature}

    for method, preds_list in [("logit_baseline", all_preds_logit),
                                ("mc_mean", all_preds_mc_mean),
                                ("mc_vote", all_preds_mc_vote)]:
        preds = np.array(preds_list)
        r = {
            "auroc": float(roc_auc_score(labels, preds)) if len(set(labels)) > 1 else 0.5,
            "auprc": float(average_precision_score(labels, preds)) if len(set(labels)) > 1 else 0.5,
            "brier": float(brier_score_loss(labels, preds)),
            "n_samples": len(labels),
        }
        # VLM/text split
        vlm_p, vlm_l, txt_p, txt_l = [], [], [], []
        for bench, data in per_benchmark.items():
            if method == "logit_baseline":
                bp = data["logit"]
            elif method == "mc_mean":
                bp = data["mc_mean"]
            else:
                bp = data["mc_vote"]
            bl = data["labels"]
            if bench in VLM_BENCHMARKS:
                vlm_p.extend(bp)
                vlm_l.extend(bl)
            else:
                txt_p.extend(bp)
                txt_l.extend(bl)
        if vlm_l and len(set(vlm_l)) > 1:
            r["vlm_auroc"] = float(roc_auc_score(vlm_l, vlm_p))
        if txt_l and len(set(txt_l)) > 1:
            r["text_auroc"] = float(roc_auc_score(txt_l, txt_p))
        results[method] = r

    # Also compute MC std as an uncertainty signal (lower std = more confident = rank by -std)
    mc_std_arr = np.array(all_preds_mc_std)
    neg_std = -mc_std_arr  # negate so higher = more confident
    if len(set(labels)) > 1 and mc_std_arr.std() > 1e-8:
        results["mc_std_auroc"] = float(roc_auc_score(labels, neg_std))
    results["mc_std_mean"] = float(mc_std_arr.mean())
    results["mc_std_median"] = float(np.median(mc_std_arr))

    results["eval_time_minutes"] = round(elapsed / 60, 1)

    print(f"\n{'='*50}")
    print(f"RESULTS: MC Dropout Consistency (N={args.n_samples})")
    print(f"{'='*50}")
    for method in ["logit_baseline", "mc_mean", "mc_vote"]:
        r = results[method]
        vlm_s = f"{r['vlm_auroc']:.4f}" if 'vlm_auroc' in r else 'N/A'
        txt_s = f"{r['text_auroc']:.4f}" if 'text_auroc' in r else 'N/A'
        print(f"  {method:25s}: AUROC={r['auroc']:.4f}  VLM={vlm_s}  Text={txt_s}")
    if "mc_std_auroc" in results:
        print(f"  {'mc_std (neg)':25s}: AUROC={results['mc_std_auroc']:.4f}")
    print(f"  MC std mean={results['mc_std_mean']:.4f}, median={results['mc_std_median']:.4f}")
    print(f"  Eval time: {elapsed/60:.1f} min")

    return results


# ============================================================
# STRATEGY 2: CONTRASTIVE (with reference answer)
# ============================================================

class ContrastiveDataset(torch.utils.data.Dataset):
    """Dataset that includes gold/reference answers in the prompt."""

    def __init__(self, samples, gold_answers, processor, max_length=2048):
        self.samples = samples
        self.gold_answers = gold_answers
        self.processor = processor
        self.max_length = max_length
        self.fallback_image = Image.new('RGB', (336, 336), color='gray')
        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 512 * 28 * 28

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if sample.has_image:
            cache_path = IMAGE_CACHE_DIR / sample.benchmark / f"{sample.id}.jpg"
            if cache_path.exists():
                try:
                    image = Image.open(cache_path).convert("RGB")
                except Exception:
                    image = self.fallback_image
            else:
                image = self.fallback_image
            min_px, max_px = self.min_pixels, self.max_pixels
        else:
            image = self.fallback_image
            min_px = max_px = 256 * 28 * 28

        target = "ii" if sample.is_correct else "i"
        gold = self.gold_answers.get(sample.id, "Unknown")

        prompt = CONTRASTIVE_PROMPT.format(
            question=sample.question[:500],
            response=sample.response[:300],
            gold=gold[:300],
        )

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]},
            {"role": "assistant", "content": target},
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt", padding=True,
            min_pixels=min_px, max_pixels=max_px,
        )

        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()
        assistant_token = 77091
        positions = (input_ids == assistant_token).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            answer_pos = positions[-1].item() + 2
            labels[:] = -100
            if answer_pos < len(labels):
                labels[answer_pos] = input_ids[answer_pos]
        else:
            labels[:-3] = -100

        result = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }
        if "pixel_values" in inputs:
            pv = inputs["pixel_values"]
            result["pixel_values"] = pv[0] if isinstance(pv, list) and len(pv) > 0 else (pv.squeeze(0) if hasattr(pv, 'dim') and pv.dim() > 3 else pv)
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"]
        return result


def evaluate_contrastive(model, processor, test_samples, gold_answers, device):
    """Evaluate using contrastive prompt with gold answers."""
    model.eval()
    fallback = Image.new('RGB', (224, 224), color='gray')

    all_preds, all_labels = [], []
    per_benchmark = defaultdict(lambda: {"preds": [], "labels": []})

    for i, sample in enumerate(test_samples):
        if i % 100 == 0:
            print(f"  Eval {i}/{len(test_samples)}...")

        if sample.has_image:
            cache_path = IMAGE_CACHE_DIR / sample.benchmark / f"{sample.id}.jpg"
            if cache_path.exists():
                try:
                    image = Image.open(cache_path).convert("RGB")
                except Exception:
                    image = fallback
            else:
                image = fallback
        else:
            image = fallback

        gold = gold_answers.get(sample.id, "Unknown")
        prompt = CONTRASTIVE_PROMPT.format(
            question=sample.question[:500],
            response=sample.response[:300],
            gold=gold[:300],
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
                outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
            token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]
            probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
            p_correct = probs[1].item()
        except Exception as e:
            if i < 5:
                print(f"  Error: {e}")
            p_correct = 0.5

        all_preds.append(p_correct)
        all_labels.append(float(sample.is_correct))
        per_benchmark[sample.benchmark]["preds"].append(p_correct)
        per_benchmark[sample.benchmark]["labels"].append(float(sample.is_correct))

    preds, labels = np.array(all_preds), np.array(all_labels)
    results = {
        "auroc": float(roc_auc_score(labels, preds)) if len(set(labels)) > 1 else 0.5,
        "auprc": float(average_precision_score(labels, preds)) if len(set(labels)) > 1 else 0.5,
        "brier": float(brier_score_loss(labels, preds)),
        "n_samples": len(labels),
    }
    # VLM/text
    vlm_p, vlm_l, txt_p, txt_l = [], [], [], []
    for bench, data in per_benchmark.items():
        if bench in VLM_BENCHMARKS:
            vlm_p.extend(data["preds"]); vlm_l.extend(data["labels"])
        else:
            txt_p.extend(data["preds"]); txt_l.extend(data["labels"])
    if vlm_l and len(set(vlm_l)) > 1:
        results["vlm_auroc"] = float(roc_auc_score(vlm_l, vlm_p))
    if txt_l and len(set(txt_l)) > 1:
        results["text_auroc"] = float(roc_auc_score(txt_l, txt_p))

    # Per-benchmark
    results["per_benchmark"] = {}
    for bench, data in sorted(per_benchmark.items()):
        bp, bl = np.array(data["preds"]), np.array(data["labels"])
        entry = {"n_samples": len(bl), "is_vlm": bench in VLM_BENCHMARKS}
        if len(set(bl)) > 1:
            entry["auroc"] = float(roc_auc_score(bl, bp))
        results["per_benchmark"][bench] = entry

    return results


def run_contrastive(args, train_samples, test_samples, gold_answers):
    """Train and evaluate contrastive model."""
    print(f"\n{'='*60}")
    print("CONTRASTIVE PROMPTING (with reference answers)")
    print(f"{'='*60}")

    # Check gold coverage
    train_with_gold = sum(1 for s in train_samples if s.id in gold_answers)
    test_with_gold = sum(1 for s in test_samples if s.id in gold_answers)
    print(f"Gold answers: {train_with_gold}/{len(train_samples)} train, {test_with_gold}/{len(test_samples)} test")

    from transformers import (
        Qwen3VLForConditionalGeneration, AutoProcessor,
        TrainingArguments, Trainer, TrainerCallback,
    )
    from peft import LoraConfig, get_peft_model

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.1, bias="none",
        task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = ContrastiveDataset(train_samples, gold_answers, processor)

    def collate_fn(batch):
        max_len = max(x["input_ids"].size(0) for x in batch)
        pad_id = processor.tokenizer.pad_token_id or 0
        ids, masks, labs = [], [], []
        for x in batch:
            pad = max_len - x["input_ids"].size(0)
            ids.append(torch.cat([torch.full((pad,), pad_id, dtype=x["input_ids"].dtype), x["input_ids"]]))
            masks.append(torch.cat([torch.zeros(pad, dtype=x["attention_mask"].dtype), x["attention_mask"]]))
            labs.append(torch.cat([torch.full((pad,), -100, dtype=x["labels"].dtype), x["labels"]]))
        result = {"input_ids": torch.stack(ids), "attention_mask": torch.stack(masks), "labels": torch.stack(labs)}
        if "pixel_values" in batch[0]:
            result["pixel_values"] = torch.cat([x["pixel_values"] for x in batch], dim=0)
        if "image_grid_thw" in batch[0]:
            result["image_grid_thw"] = torch.cat([x["image_grid_thw"] for x in batch])
        return result

    eff_batch = args.batch_size * args.grad_accum
    steps = len(train_dataset) // eff_batch

    training_args = TrainingArguments(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate, weight_decay=0.01, warmup_ratio=0.1,
        logging_steps=max(10, steps // 10), save_strategy="epoch", save_total_limit=1,
        bf16=True, report_to="none", remove_unused_columns=False,
        dataloader_num_workers=0, dataloader_pin_memory=False,
        optim="adamw_torch_fused", max_grad_norm=1.0,
    )

    class MemClean(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 100 == 0:
                torch.cuda.empty_cache()
            return control

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
                      data_collator=collate_fn, callbacks=[MemClean()])

    print(f"\nTraining: {len(train_dataset)} samples, {args.epochs} epochs, {steps} steps/epoch")
    t0 = time.time()
    resume_ckpt = True if args.resume_from_checkpoint else None
    trainer.train(resume_from_checkpoint=resume_ckpt)
    train_time = time.time() - t0
    print(f"Training: {train_time/60:.1f} min")

    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))

    device = next(model.parameters()).device
    results = evaluate_contrastive(model, processor, test_samples, gold_answers, device)
    results["strategy"] = "contrastive"
    results["train_time_minutes"] = round(train_time / 60, 1)
    results["train_samples"] = len(train_samples)
    results["gold_coverage_train"] = train_with_gold / len(train_samples)
    results["gold_coverage_test"] = test_with_gold / len(test_samples)

    print(f"\n{'='*50}")
    print("RESULTS: Contrastive Prompting")
    print(f"{'='*50}")
    print(f"Overall AUROC: {results['auroc']:.4f}")
    print(f"VLM AUROC:     {results.get('vlm_auroc', 'N/A')}")
    print(f"Text AUROC:    {results.get('text_auroc', 'N/A')}")

    return results


# ============================================================
# STRATEGY 3: VERBALIZED JUDGE CONFIDENCE
# ============================================================

class VerbalizedDataset(torch.utils.data.Dataset):
    """Dataset where the target is a probability string like '0.85'."""

    def __init__(self, samples, processor, max_length=2048):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        self.fallback_image = Image.new('RGB', (336, 336), color='gray')
        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 512 * 28 * 28

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if sample.has_image:
            cache_path = IMAGE_CACHE_DIR / sample.benchmark / f"{sample.id}.jpg"
            if cache_path.exists():
                try:
                    image = Image.open(cache_path).convert("RGB")
                except Exception:
                    image = self.fallback_image
            else:
                image = self.fallback_image
            min_px, max_px = self.min_pixels, self.max_pixels
        else:
            image = self.fallback_image
            min_px = max_px = 256 * 28 * 28

        # Target: "0.95" for correct, "0.05" for incorrect
        # (using extreme but not 0/1 to avoid degenerate outputs)
        target = "0.95" if sample.is_correct else "0.05"

        prompt = VERBALIZED_PROMPT.format(
            question=sample.question[:500],
            response=sample.response[:300],
        )

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]},
            {"role": "assistant", "content": target},
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt", padding=True,
            min_pixels=min_px, max_pixels=max_px,
        )

        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()

        # Mask everything except the probability tokens
        assistant_token = 77091
        positions = (input_ids == assistant_token).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            answer_start = positions[-1].item() + 2
            labels[:answer_start] = -100
            # Keep the probability tokens (e.g., "0", ".", "9", "5")
        else:
            labels[:-6] = -100

        result = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }
        if "pixel_values" in inputs:
            pv = inputs["pixel_values"]
            result["pixel_values"] = pv[0] if isinstance(pv, list) and len(pv) > 0 else (pv.squeeze(0) if hasattr(pv, 'dim') and pv.dim() > 3 else pv)
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"]
        return result


def evaluate_verbalized(model, processor, test_samples, device):
    """Evaluate by generating probability text and parsing it."""
    model.eval()
    fallback = Image.new('RGB', (224, 224), color='gray')

    all_preds, all_labels = [], []
    per_benchmark = defaultdict(lambda: {"preds": [], "labels": []})
    parse_failures = 0

    for i, sample in enumerate(test_samples):
        if i % 100 == 0:
            print(f"  Eval {i}/{len(test_samples)}...")

        if sample.has_image:
            cache_path = IMAGE_CACHE_DIR / sample.benchmark / f"{sample.id}.jpg"
            if cache_path.exists():
                try:
                    image = Image.open(cache_path).convert("RGB")
                except Exception:
                    image = fallback
            else:
                image = fallback
        else:
            image = fallback

        prompt = VERBALIZED_PROMPT.format(
            question=sample.question[:500],
            response=sample.response[:300],
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
                generated = model.generate(
                    **inputs, max_new_tokens=10, do_sample=False,
                    temperature=1.0, pad_token_id=processor.tokenizer.pad_token_id,
                )
            # Decode only new tokens
            new_tokens = generated[0, inputs["input_ids"].shape[1]:]
            output_text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Parse probability
            match = re.search(r'(0\.\d+|1\.0|0|1)', output_text)
            if match:
                p_correct = float(match.group(1))
                p_correct = max(0.0, min(1.0, p_correct))
            else:
                p_correct = 0.5
                parse_failures += 1
        except Exception as e:
            if i < 5:
                print(f"  Error: {e}")
            p_correct = 0.5
            parse_failures += 1

        all_preds.append(p_correct)
        all_labels.append(float(sample.is_correct))
        per_benchmark[sample.benchmark]["preds"].append(p_correct)
        per_benchmark[sample.benchmark]["labels"].append(float(sample.is_correct))

    preds, labels = np.array(all_preds), np.array(all_labels)
    results = {
        "auroc": float(roc_auc_score(labels, preds)) if len(set(labels)) > 1 else 0.5,
        "auprc": float(average_precision_score(labels, preds)) if len(set(labels)) > 1 else 0.5,
        "brier": float(brier_score_loss(labels, preds)),
        "n_samples": len(labels),
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / len(labels),
    }
    vlm_p, vlm_l, txt_p, txt_l = [], [], [], []
    for bench, data in per_benchmark.items():
        if bench in VLM_BENCHMARKS:
            vlm_p.extend(data["preds"]); vlm_l.extend(data["labels"])
        else:
            txt_p.extend(data["preds"]); txt_l.extend(data["labels"])
    if vlm_l and len(set(vlm_l)) > 1:
        results["vlm_auroc"] = float(roc_auc_score(vlm_l, vlm_p))
    if txt_l and len(set(txt_l)) > 1:
        results["text_auroc"] = float(roc_auc_score(txt_l, txt_p))

    results["per_benchmark"] = {}
    for bench, data in sorted(per_benchmark.items()):
        bp, bl = np.array(data["preds"]), np.array(data["labels"])
        entry = {"n_samples": len(bl), "is_vlm": bench in VLM_BENCHMARKS}
        if len(set(bl)) > 1:
            entry["auroc"] = float(roc_auc_score(bl, bp))
        results["per_benchmark"][bench] = entry

    return results


def run_verbalized(args, train_samples, test_samples):
    """Train and evaluate verbalized judge."""
    print(f"\n{'='*60}")
    print("VERBALIZED JUDGE CONFIDENCE")
    print(f"{'='*60}")

    from transformers import (
        Qwen3VLForConditionalGeneration, AutoProcessor,
        TrainingArguments, Trainer, TrainerCallback,
    )
    from peft import LoraConfig, get_peft_model

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.1, bias="none",
        task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    train_dataset = VerbalizedDataset(train_samples, processor)

    def collate_fn(batch):
        max_len = max(x["input_ids"].size(0) for x in batch)
        pad_id = processor.tokenizer.pad_token_id or 0
        ids, masks, labs = [], [], []
        for x in batch:
            pad = max_len - x["input_ids"].size(0)
            ids.append(torch.cat([torch.full((pad,), pad_id, dtype=x["input_ids"].dtype), x["input_ids"]]))
            masks.append(torch.cat([torch.zeros(pad, dtype=x["attention_mask"].dtype), x["attention_mask"]]))
            labs.append(torch.cat([torch.full((pad,), -100, dtype=x["labels"].dtype), x["labels"]]))
        result = {"input_ids": torch.stack(ids), "attention_mask": torch.stack(masks), "labels": torch.stack(labs)}
        if "pixel_values" in batch[0]:
            result["pixel_values"] = torch.cat([x["pixel_values"] for x in batch], dim=0)
        if "image_grid_thw" in batch[0]:
            result["image_grid_thw"] = torch.cat([x["image_grid_thw"] for x in batch])
        return result

    eff_batch = args.batch_size * args.grad_accum
    steps = len(train_dataset) // eff_batch

    training_args = TrainingArguments(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate, weight_decay=0.01, warmup_ratio=0.1,
        logging_steps=max(10, steps // 10), save_strategy="epoch", save_total_limit=1,
        bf16=True, report_to="none", remove_unused_columns=False,
        dataloader_num_workers=0, dataloader_pin_memory=False,
        optim="adamw_torch_fused", max_grad_norm=1.0,
    )

    class MemClean(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 100 == 0:
                torch.cuda.empty_cache()
            return control

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
                      data_collator=collate_fn, callbacks=[MemClean()])

    print(f"\nTraining: {len(train_dataset)} samples, {args.epochs} epochs")
    t0 = time.time()
    resume_ckpt = True if args.resume_from_checkpoint else None
    trainer.train(resume_from_checkpoint=resume_ckpt)
    train_time = time.time() - t0
    print(f"Training: {train_time/60:.1f} min")

    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))

    device = next(model.parameters()).device
    results = evaluate_verbalized(model, processor, test_samples, device)
    results["strategy"] = "verbalized"
    results["train_time_minutes"] = round(train_time / 60, 1)
    results["train_samples"] = len(train_samples)

    print(f"\n{'='*50}")
    print("RESULTS: Verbalized Judge Confidence")
    print(f"{'='*50}")
    print(f"Overall AUROC: {results['auroc']:.4f}")
    print(f"VLM AUROC:     {results.get('vlm_auroc', 'N/A')}")
    print(f"Text AUROC:    {results.get('text_auroc', 'N/A')}")
    print(f"Parse failures: {results['parse_failures']} ({100*results['parse_failure_rate']:.1f}%)")

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Novel elicitation ablations")
    parser.add_argument("--strategy", type=str, required=True,
                        choices=["multi_sample", "contrastive", "verbalized"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--smoke_test", action="store_true")

    # Multi-sample args
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=str, default="uq_models/best_unified")

    # Training args (for contrastive and verbalized)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--resume_from_checkpoint", action="store_true",
                        help="Resume training from last checkpoint in output_dir")

    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    max_per_bench = 5 if args.smoke_test else None

    # Load data
    print("Loading data...")
    all_samples, gold_answers = load_samples_with_gold(max_per_benchmark=max_per_bench)

    # Canonical split
    train_idx, test_idx = get_canonical_split(all_samples)
    train_samples = [all_samples[i] for i in train_idx]
    test_samples = [all_samples[i] for i in test_idx]
    print(f"Split: {len(train_samples)} train / {len(test_samples)} test")

    # Prepare images
    prepare_all_images(train_samples + test_samples)

    # Run strategy
    if args.strategy == "multi_sample":
        results = run_multi_sample(args, test_samples)
    elif args.strategy == "contrastive":
        results = run_contrastive(args, train_samples, test_samples, gold_answers)
    elif args.strategy == "verbalized":
        results = run_verbalized(args, train_samples, test_samples)

    # Save
    config = vars(args).copy()
    config["output_dir"] = str(config["output_dir"])
    results["config"] = config
    with open(args.output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
