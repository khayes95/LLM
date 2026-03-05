#!/usr/bin/env python3
"""Training data size ablation for energetics UQ domain adaptation.

Trains the adapted UQ model at various fractions of the training data
and evaluates on the same held-out 230 questions. Shows how many
training samples are needed before performance plateaus.

Usage:
    # Full ablation (1 GPU, ~2-3 hours)
    CUDA_VISIBLE_DEVICES=0 python scripts/energetics_size_ablation.py

    # Smoke test
    CUDA_VISIBLE_DEVICES=0 python scripts/energetics_size_ablation.py --smoke_test
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "/scratch/khayes/energetics_bench/scoring/results/uq_input.jsonl"
BASE_CHECKPOINT = "uq_models/best_unified"
OUTPUT_DIR = Path("uq_models/energetics_ablation")

# Fractions of training questions to use
FRACTIONS = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0]
# Corresponding to roughly: 46, 92, 183, 366, 549, 733, 916 questions
#                            138, 276, 549, 1098, 1647, 2199, 2748 samples

TEST_FRACTION = 0.20  # Same 230 questions held out every time
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data and create the SAME test split as the full training run
    print("Loading data...")
    sys.path.insert(0, str(Path(__file__).parent))
    from train_energetics_uq import load_energetics_data, split_by_question

    all_samples = load_energetics_data(INPUT_FILE)
    train_samples, test_samples = split_by_question(all_samples, TEST_FRACTION, SEED)

    train_qids = sorted(set(s.question_id for s in train_samples))
    test_qids = set(s.question_id for s in test_samples)

    print(f"Total: {len(all_samples)} samples")
    print(f"Train pool: {len(train_samples)} samples ({len(train_qids)} questions)")
    print(f"Test (fixed): {len(test_samples)} samples ({len(test_qids)} questions)")

    if args.smoke_test:
        FRACTIONS_TO_RUN = [0.10, 0.50, 1.0]
        args.epochs = 1
    else:
        FRACTIONS_TO_RUN = FRACTIONS

    # Run each fraction as a subprocess to avoid CUDA memory issues
    results = []

    # Add zero-shot baseline
    results.append({
        "fraction": 0.0,
        "n_questions": 0,
        "n_samples": 0,
        "auroc": 0.663,  # From our earlier zero-shot evaluation
        "source": "zero-shot (pre-computed)"
    })

    for frac in FRACTIONS_TO_RUN:
        n_q = int(len(train_qids) * frac)
        if n_q < 5:
            n_q = 5  # minimum

        # Select subset of training questions
        np.random.seed(SEED)
        selected_qids = set(np.random.choice(train_qids, n_q, replace=False))
        subset_samples = [s for s in train_samples if s.question_id in selected_qids]

        print(f"\n{'='*60}")
        print(f"ABLATION: {frac:.0%} of training data")
        print(f"  Questions: {n_q}/{len(train_qids)}")
        print(f"  Samples: {len(subset_samples)}/{len(train_samples)}")
        print(f"{'='*60}")

        # Write subset to temp file
        subset_dir = OUTPUT_DIR / f"frac_{int(frac*100):03d}"
        subset_dir.mkdir(parents=True, exist_ok=True)

        subset_file = subset_dir / "train_subset.jsonl"
        with open(subset_file, "w") as f:
            for s in subset_samples:
                f.write(json.dumps({
                    "question_id": s.question_id,
                    "model": s.model,
                    "question": s.question,
                    "response": s.response,
                    "is_correct": s.is_correct,
                }) + "\n")

        # Run training as subprocess
        cmd = [
            sys.executable, "scripts/train_energetics_uq.py",
            "--input", INPUT_FILE,
            "--output_dir", str(subset_dir / "checkpoint"),
            "--base_checkpoint", BASE_CHECKPOINT,
            "--epochs", str(args.epochs),
            "--learning_rate", str(args.learning_rate),
            "--test_fraction", str(TEST_FRACTION),
        ]

        # We need to modify the training to use only our subset
        # Easier: write a wrapper that patches the data loading
        train_script = subset_dir / "run_train.py"
        with open(train_script, "w") as f:
            f.write(f'''#!/usr/bin/env python3
import json, os, sys, time, numpy as np, torch
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split

sys.path.insert(0, "{Path(__file__).parent.parent.parent.parent}")

@dataclass
class Sample:
    id: str
    question_id: str
    model: str
    question: str
    response: str
    is_correct: bool

# Load ALL data, split same way
INPUT = "{INPUT_FILE}"
samples = []
with open(INPUT) as fh:
    for i, line in enumerate(fh):
        row = json.loads(line.strip())
        is_correct = row.get("is_correct", False)
        if isinstance(is_correct, str):
            is_correct = is_correct.lower() in ("true", "1", "yes")
        samples.append(Sample(
            id=f"{{row.get('question_id', f'q{{i}}')}}__{{row.get('model', 'unk')}}",
            question_id=row.get("question_id", f"q{{i}}"),
            model=row.get("model", "unknown"),
            question=row["question"],
            response=row["response"],
            is_correct=bool(is_correct),
        ))

# Same split as main run
all_qids = sorted(set(s.question_id for s in samples))
train_qids_all, test_qids = train_test_split(all_qids, test_size={TEST_FRACTION}, random_state={SEED})
train_qids_all = sorted(train_qids_all)
test_qids = set(test_qids)

# Subset: take first {n_q} training questions (deterministic)
np.random.seed({SEED})
selected = set(np.random.choice(train_qids_all, {n_q}, replace=False))

train_samples = [s for s in samples if s.question_id in selected]
test_samples = [s for s in samples if s.question_id in test_qids]

print(f"Train: {{len(train_samples)}} samples ({{len(selected)}} questions)")
print(f"Test: {{len(test_samples)}} samples ({{len(test_qids)}} questions)")

# Import training components
PROMPT_TEMPLATE = """Question: {{question}}

Answer: {{response}}

Is the answer correct? (i) No (ii) Yes"""

GRAY_IMAGE = Image.new('RGB', (336, 336), color='gray')

# Dataset class
class EnergeticsUQDataset(torch.utils.data.Dataset):
    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor
        self.image = GRAY_IMAGE
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        target = "ii" if sample.is_correct else "i"
        prompt = PROMPT_TEMPLATE.format(question=sample.question[:500], response=sample.response[:300])
        messages = [
            {{"role": "user", "content": [{{"type": "image", "image": self.image}}, {{"type": "text", "text": prompt}}]}},
            {{"role": "assistant", "content": target}},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(text=[text], images=[self.image], return_tensors="pt", padding=True,
                                min_pixels=256*28*28, max_pixels=256*28*28)
        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()
        assistant_positions = (input_ids == 77091).nonzero(as_tuple=True)[0]
        if len(assistant_positions) > 0:
            answer_pos = assistant_positions[-1].item() + 2
            labels[:] = -100
            if answer_pos < len(labels):
                labels[answer_pos] = input_ids[answer_pos]
        else:
            labels[:-3] = -100
        result = {{"input_ids": inputs["input_ids"].squeeze(0), "attention_mask": inputs["attention_mask"].squeeze(0), "labels": labels.squeeze(0)}}
        if "pixel_values" in inputs:
            pv = inputs["pixel_values"]
            result["pixel_values"] = pv[0] if isinstance(pv, list) else (pv.squeeze(0) if pv.dim() > 3 else pv)
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"]
        return result

# Load model
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, TrainingArguments, Trainer, TrainerCallback
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

lora_path = "{BASE_CHECKPOINT}"
if not os.path.exists(os.path.join(lora_path, "adapter_config.json")):
    subdirs = sorted([d for d in os.listdir(lora_path) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(lora_path, d))])
    if subdirs:
        lora_path = os.path.join(lora_path, subdirs[-1])

model = PeftModel.from_pretrained(model, lora_path, is_trainable=True)
model.print_trainable_parameters()

# Dataset + collator
train_dataset = EnergeticsUQDataset(train_samples, processor)

def collate_fn(batch):
    max_len = max(x["input_ids"].size(0) for x in batch)
    pad_id = processor.tokenizer.pad_token_id or 0
    ids, masks, labs = [], [], []
    for x in batch:
        pad = max_len - x["input_ids"].size(0)
        ids.append(torch.cat([torch.full((pad,), pad_id, dtype=x["input_ids"].dtype), x["input_ids"]]))
        masks.append(torch.cat([torch.zeros(pad, dtype=x["attention_mask"].dtype), x["attention_mask"]]))
        labs.append(torch.cat([torch.full((pad,), -100, dtype=x["labels"].dtype), x["labels"]]))
    result = {{"input_ids": torch.stack(ids), "attention_mask": torch.stack(masks), "labels": torch.stack(labs)}}
    if "pixel_values" in batch[0]:
        result["pixel_values"] = torch.cat([x["pixel_values"] for x in batch], dim=0)
    if "image_grid_thw" in batch[0]:
        result["image_grid_thw"] = torch.cat([x["image_grid_thw"] for x in batch])
    return result

# Train
output_dir = "{str(subset_dir / 'checkpoint')}"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir, num_train_epochs={args.epochs},
    per_device_train_batch_size=1, gradient_accumulation_steps=16,
    learning_rate={args.learning_rate}, weight_decay=0.01, warmup_ratio=0.1,
    logging_steps=10, save_strategy="no", bf16=True, bf16_full_eval=True,
    report_to="none", remove_unused_columns=False, dataloader_num_workers=0,
    dataloader_pin_memory=False, optim="adamw_torch_fused", max_grad_norm=1.0)

class MemClean(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0: torch.cuda.empty_cache()
        return control

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
                  data_collator=collate_fn, callbacks=[MemClean()])

t0 = time.time()
trainer.train()
train_time = time.time() - t0
print(f"Training time: {{train_time:.0f}}s")

# Save
trainer.save_model(output_dir)
processor.save_pretrained(output_dir)

# Evaluate on fixed test set
model.eval()
device = next(model.parameters()).device
fallback = Image.new('RGB', (224, 224), color='gray')

all_preds, all_labels = [], []
per_model = defaultdict(lambda: {{"preds": [], "labels": []}})

for i, sample in enumerate(test_samples):
    if i % 100 == 0: print(f"  Eval {{i}}/{{len(test_samples)}}...")
    prompt = PROMPT_TEMPLATE.format(question=sample.question[:500], response=sample.response[:300])
    messages = [{{"role": "user", "content": [{{"type": "image", "image": fallback}}, {{"type": "text", "text": prompt}}]}}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[fallback], return_tensors="pt", padding=True,
                       min_pixels=256*28*28, max_pixels=256*28*28)
    inputs = {{k: v.to(device) for k, v in inputs.items()}}
    try:
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
        token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]
        probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
        p_correct = probs[1].item()
    except:
        p_correct = 0.5
    all_preds.append(p_correct)
    all_labels.append(float(sample.is_correct))
    per_model[sample.model]["preds"].append(p_correct)
    per_model[sample.model]["labels"].append(float(sample.is_correct))

preds = np.array(all_preds)
labels = np.array(all_labels)
auroc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
brier = brier_score_loss(labels, preds)

results = {{
    "fraction": {frac},
    "n_questions": {n_q},
    "n_samples": len(train_samples),
    "auroc": float(auroc),
    "brier": float(brier),
    "train_time_s": train_time,
    "per_model": {{}},
}}

for mname, data in per_model.items():
    mp, ml = np.array(data["preds"]), np.array(data["labels"])
    entry = {{"n": len(ml), "auroc": float(roc_auc_score(ml, mp)) if len(set(ml)) > 1 else 0.5}}
    results["per_model"][mname] = entry

print(f"\\nFraction: {frac:.0%}, AUROC: {{auroc:.4f}}, Brier: {{brier:.4f}}")
for m, d in results["per_model"].items():
    print(f"  {{m}}: AUROC={{d['auroc']:.4f}}")

with open("{str(subset_dir / 'results.json')}", "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {str(subset_dir / 'results.json')}")
''')

        # Run it
        print(f"Running training for {frac:.0%}...")
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, str(train_script)],
            capture_output=True, text=True, timeout=3600,
        )

        if result.returncode != 0:
            print(f"FAILED at {frac:.0%}!")
            print(result.stderr[-1000:] if result.stderr else "no stderr")
            continue

        # Print key output
        for line in result.stdout.split("\n"):
            if "AUROC" in line or "Fraction" in line or "Train:" in line or "Training time" in line:
                print(f"  {line.strip()}")

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.0f}s")

        # Load results
        results_file = subset_dir / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                res = json.load(f)
            results.append(res)

    # Save combined results
    combined_file = OUTPUT_DIR / "ablation_results.json"
    with open(combined_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print("ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Fraction':<10} {'Questions':<12} {'Samples':<10} {'AUROC':<10} {'Source'}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x["fraction"]):
        src = r.get("source", "trained")
        per_model = r.get("per_model", {})
        model_str = ""
        if per_model:
            model_str = " | ".join(f"{m}={d['auroc']:.3f}" for m, d in sorted(per_model.items()))
        print(f"{r['fraction']:<10.0%} {r.get('n_questions', 0):<12} {r.get('n_samples', 0):<10} "
              f"{r['auroc']:<10.4f} {src}")
        if model_str:
            print(f"{'':>10} {model_str}")

    print(f"\nResults saved to {combined_file}")


if __name__ == "__main__":
    main()
