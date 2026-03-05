# FineGRAIN x UQ: Test Plan & Integration Guide

**Goal:** Use our UQ calibrator as a meta-judge / evaluator for text-to-image model outputs, applied to the FineGRAIN benchmark.

**Status:** Initial evaluation running (see `data/finegrain_uq/results.json` for latest results).

---

## Background

FineGRAIN evaluates T2I models across 27 failure modes using a 3-step pipeline:
1. T2I model generates image from prompt
2. VLM (Molmo-72B) captions the image
3. LLM (Llama3-70B) judges whether a failure mode is present by comparing prompt vs caption

**FineGRAIN pipeline accuracy: 67.4%** (vs human ground truth)

Our UQ calibrator (Qwen3-VL-8B + LoRA) replaces steps 2 AND 3 with a single-pass evaluation:
1. T2I model generates image from prompt ← same
2. UQ model directly evaluates: image + prompt → P(compliant) ← replaces VLM+LLM

**Advantages:**
- Single-pass (no intermediate captioning step)
- Much smaller model (8B vs 72B+70B)
- Calibrated probability output (not just binary)
- No information loss from caption bottleneck

---

## Data Available

| Dataset | Location | Samples | Labels | Images |
|---------|----------|---------|--------|--------|
| Original 5 models | `/scratch/khayes/diff/t2i-finegrain/images/` | 3,800 | 0/1 (48% failure) | Yes (5.1GB) |
| New 12 models | HuggingFace `KevinDavidHayes/t2i-finegrain` | 24,155 | All 0 (compliant) | Yes |
| Metadata | `/scratch/khayes/diff/t2i-finegrain/metadata.csv` | 27,955 | Mixed | Partial |

**Original 5 models** (with human labels): flux, sd3.5_large, sd3.5_medium, sd3_m, sd3_xl
**New 12 models** (all labeled compliant): flux_kontext, gpt_image1, gpt_image15, hidream, nano_banana2, flux2_pro, qwen-image, gemini_image, wan22, seeDream3, sdv1.5, sd2.1

---

## Experiments

### Experiment 1: Direct Failure Detection (DONE / RUNNING)
**Script:** `scripts/finegrain_uq_eval.py`
**SLURM:** `slurm/finegrain_uq_eval.sh`
**Output:** `data/finegrain_uq/results.json`

Evaluates our UQ model on the 3,800 labeled samples from the original 5 models.

**How to run:**
```bash
# Smoke test (10 samples, ~30 seconds)
sbatch slurm/finegrain_smoke.sh

# Full evaluation (3800 samples, ~30 min)
sbatch slurm/finegrain_uq_eval.sh
```

**Metrics reported:**
- Overall AUROC, accuracy, F1
- Per-T2I-model breakdown (5 models)
- Per-failure-mode breakdown (27 categories)
- Selective prediction at 25/50/75/100% coverage
- Bootstrap CI on AUROC
- Comparison with FineGRAIN's 67.4% accuracy

**Smoke test results (10 samples):**
- AUROC: 0.920
- Accuracy: 80% (vs FineGRAIN's 67.4%)
- Selective acc @ 50% coverage: 100%

---

### Experiment 2: Prompt-Variant Ablation (TODO)
Compare different prompt framings to find the best approach.

```bash
# Baseline prompt (simple QA format)
CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_uq_eval.py \
    --prompt_variant baseline \
    --output_dir data/finegrain_uq/ablation_baseline

# Combined prompt (with metadata + CoT)
CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_uq_eval.py \
    --prompt_variant combined \
    --output_dir data/finegrain_uq/ablation_combined

# FineGRAIN-specific prompt (with failure mode description)
CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_uq_eval.py \
    --prompt_variant finegrain_direct \
    --output_dir data/finegrain_uq/ablation_finegrain_direct

# Standard QA mapping
CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_uq_eval.py \
    --prompt_variant finegrain_qa \
    --output_dir data/finegrain_uq/ablation_finegrain_qa
```

---

### Experiment 3: New Model Evaluation (TODO)
Score the 12 newer T2I models (all labeled compliant). Tests:
- Do our P(compliant) scores correctly identify these as compliant?
- Score distribution analysis (should be skewed toward high confidence)
- Can we rank T2I models by mean P(compliant)?

```bash
# Download images from HuggingFace first
python -c "
from datasets import load_dataset
ds = load_dataset('KevinDavidHayes/t2i-finegrain', split='benchmark')
# Images auto-download
"

# Then run evaluation (modify script to handle HuggingFace images)
CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_uq_eval.py \
    --models flux_kontext gpt_image1 hidream \
    --output_dir data/finegrain_uq/new_models
```

**Note:** Need to extend `finegrain_uq_eval.py` to handle new model images from HuggingFace (currently only supports local images for original 5 models).

---

### Experiment 4: Meta-Judge on VLM Captions (TODO)
Instead of direct image evaluation, score the VLM captions:
- Question: original T2I prompt
- Answer: Molmo's caption of the generated image
- Ground truth: human labels

**Requires:** Re-running the VLM captioning pipeline on the 3,800 labeled images (or using existing captions if available).

```bash
# Step 1: Generate Molmo captions for original 5 models
# (Uses the FineGRAIN pipeline: vlm_captioning_pil.py)
cd /scratch/khayes/diff/FineGRAIN_Eval
python vlm_captioning_pil.py --batch_size 8

# Step 2: Score captions with our UQ model
# (New script needed — framing caption as "answer" to prompt "question")
```

---

### Experiment 5: Selective Evaluation (TODO)
Demonstrate the key value proposition: abstain on uncertain cases to achieve higher accuracy.

Analysis (runs on output of Experiment 1):
```python
import json
import numpy as np
from sklearn.metrics import accuracy_score

# Load scored samples
with open("data/finegrain_uq/scored_samples.jsonl") as f:
    samples = [json.loads(l) for l in f]

labels = np.array([s["human_label"] for s in samples])
scores = np.array([1 - s["p_compliant"] for s in samples])  # failure scores

# Selective prediction curve
confidence = np.abs(scores - 0.5)
sorted_idx = np.argsort(-confidence)

for coverage in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    n = int(len(labels) * coverage)
    sel = sorted_idx[:n]
    preds = (scores[sel] >= 0.5).astype(int)
    acc = accuracy_score(labels[sel], preds)
    print(f"Coverage {coverage:.0%}: Accuracy {acc:.3f} (n={n})")
```

---

### Experiment 6: Cross-Pipeline Comparison (TODO)
Direct head-to-head with FineGRAIN's VLM+LLM pipeline on the same samples.

**Requires:**
1. Full FineGRAIN evaluations on all 3,800 images (run llm_judge.py without --test flag)
2. Align samples by (prompt_id, model)
3. Compare binary predictions and AUROC

---

### Experiment 7: Per-Model Difficulty Ranking (TODO)
Use mean P(compliant) to rank T2I models by quality, and compare with FineGRAIN's rankings.

Analysis:
```python
# From Experiment 1 results
results = json.load(open("data/finegrain_uq/results.json"))
for model, metrics in sorted(results["per_model"].items(),
                              key=lambda x: x[1].get("mean_failure_score", 0)):
    print(f"{model}: mean_failure_score={metrics['mean_failure_score']:.3f}")
```

---

## Key Files

| File | Purpose |
|------|---------|
| `scripts/finegrain_uq_eval.py` | Main evaluation script |
| `slurm/finegrain_uq_eval.sh` | Full SLURM job |
| `slurm/finegrain_smoke.sh` | Smoke test SLURM job |
| `data/finegrain_uq/results.json` | Full evaluation results |
| `data/finegrain_uq/scored_samples.jsonl` | Per-sample scores |
| `data/finegrain_uq/smoke/` | Smoke test outputs |
| `/scratch/khayes/diff/t2i-finegrain/` | FineGRAIN images + metadata |

---

## How to Add a New T2I Model

1. Generate images with the T2I model (one per prompt, named `{prompt_id:05d}.png`)
2. Add the model's images to `/scratch/khayes/diff/t2i-finegrain/images/{model_name}/`
3. Add human labels to `metadata.csv` (or use the UQ model's predictions as pseudo-labels)
4. Run:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_uq_eval.py \
    --models your_new_model \
    --output_dir data/finegrain_uq/new_model_eval
```

---

## What to Report

For the paper / discussion:

1. **AUROC table**: Our UQ model vs FineGRAIN pipeline, per failure mode
2. **Selective prediction curve**: Accuracy vs coverage (shows we can identify uncertain cases)
3. **Model efficiency**: 8B single-pass vs 72B+70B two-pass
4. **Per-failure-mode heatmap**: Which failure modes our model is best/worst at detecting
5. **T2I model ranking**: Do our rankings agree with FineGRAIN's?

---

## Dependencies

```bash
# Already installed in uq_eval conda env:
pip install torch transformers peft scikit-learn pandas pillow numpy
```

No additional dependencies needed — the evaluation uses the same model and pipeline as the UQ project.

---

## Questions / Open Issues

- [ ] Should we fine-tune the UQ model on FineGRAIN data? (Would improve results but loses the zero-shot transfer story)
- [ ] How to handle the 12 new models with all-compliant labels? (Synthetic negatives? Cross-prompt mismatches?)
- [ ] Should we release this as a FineGRAIN add-on / companion tool?
- [ ] Compare with other VLM judges (InternVL3, Pixtral) — do they benefit from UQ meta-judging?
