# MASTER PLAN: Cross-Model Uncertainty Estimation
## Target: ECCV 2026 Submission

**Created:** December 30, 2025
**Last major update:** February 21, 2026
**Status:** Active — Phases 1-3 largely complete, cross-model evals in progress, entering Phase 4

---

## Paper Contributions (3 Novel Claims)

1. **Cross-Model Transfer:** A calibrator trained on Model A's outputs can predict correctness for Model B's outputs — tested across a diverse matrix of open and closed-source models
2. **Multimodality:** The approach works for both text-only LLMs and vision-language models
3. **Closed-Source Application:** Enables uncertainty quantification for models where fine-tuning is impossible (GPT-5.2, Claude, etc.)

**Long-term vision:** Model-independent uncertainty quantification — a single lightweight judge that works across arbitrary models without retraining.

---

## What Was Done (Phase 0: Proof of Concept)

The previous work established feasibility but had limitations:

### Text Experiments
| Component | What Was Done | Limitations |
|-----------|---------------|-------------|
| Calibrator | Llama-3.1-8B-Instruct + LoRA | Old model (2023) |
| Training data | 2,698 samples, 15 benchmarks | GPT-4 responses only |
| Source model | Llama-3.1-8B | Single source |
| Target models | Qwen-2.5-7B | Single target, no closed-source |
| Results | 0.886 in-dist, 0.766 cross-model | Not verified for bugs |

### Vision Experiments
| Component | What Was Done | Limitations |
|-----------|---------------|-------------|
| Calibrator | Qwen3-VL-8B + LoRA | OK but could upgrade |
| Training data | 5,251 samples, 5 benchmarks | VSR bug (38% gray images), class imbalance (67/33) |
| Source model | InternVL3-78B | Single source |
| Target models | Qwen2.5-VL-72B | Single target, no closed-source |
| Results | 0.789 in-dist, 0.693 cross-model | Some results suspicious (MMMU) |

### Known Bugs Found
1. **VSR images:** 38% of training data used gray placeholders — **FIXED**
2. **MMMU options:** Stored as string literals, parsing fixed — **FIXED**
3. **HallusionBench grading:** '0'/'1' labels not handled — **FIXED**
4. **ERQA:** Multi-image benchmark — **excluded**

---

## Phase 1: Foundation & Bug Fixes (Days 1-2) — ✅ COMPLETE

### 1.1 Clean Codebase
- [x] Audit all data loading code for bugs
- [x] Fix VSR image loading (download from COCO URLs)
- [x] Verify MMMU grading is correct
- [x] Verify HallusionBench grading is correct
- [x] Create unified data loading pipeline with validation checks
- [x] Add image verification (shape, std > 0, not gray)

### 1.2 Documentation Structure
**Existing files (update, don't replace):**
```
MASTER_PLAN.md          # This file - high-level roadmap
CLAUDE.md               # Instructions for coding agent (extend existing)
RESEARCH_LOG.md         # 1660+ lines of experiment history - continue using
TECHNICAL_REPORT.md     # Paper-ready content (needs updating)
```

**Archive old versions before major changes:**
```bash
cp CLAUDE.md archive/CLAUDE_$(date +%Y%m%d).md
```

### 1.3 Directory Structure (Existing)
```
/scratch/khayes/LLM/
├── uq_eval/                    # Evaluation harness (32+ benchmarks)
│   ├── benchmarks/             # Benchmark implementations
│   └── models/                 # Model clients (OpenAI, vLLM, etc.)
├── scripts/                    # Training and evaluation scripts
│   ├── train_vlm_judge.py      # VLM judge training
│   ├── train_vlm_combined.py   # Combined vision+text training
│   ├── cross_model_transfer_all.py  # Cross-model evaluation
│   ├── generate_figures.py     # Figure generation
│   └── spurious_correlation_analysis.py
├── data/
│   ├── features/               # VLM benchmark features
│   │   ├── vsr/, mmmu/, charxiv/, hallusionbench/, erqa/
│   ├── finetune/               # Text training/test data
│   │   ├── train_v2.jsonl, test_v2.jsonl
│   ├── vlm_judge_combined/     # VLM judge checkpoints
│   │   └── checkpoint-788      # Previous best (pre-VSR-fix)
│   ├── vlm_judge_vsr_fixed/    # ★ CURRENT VLM calibrator (post-VSR-fix)
│   ├── ablations/              # Ablation experiment checkpoints
│   ├── cross_model/            # Cross-model evaluation results
│   └── results/                # Evaluation results
├── runs/                       # Benchmark run outputs (uq_eval)
│   └── gpt5_mini_combined/     # GPT-5-mini responses (4,443 samples, 21 benchmarks)
├── figures/                    # Generated figures
│   └── advisor_meeting/        # Presentation figures
├── logs/                       # SLURM logs
├── uq_models/                  # Text calibrator checkpoints
│   ├── llama-8b-uq-lora-v2/   # OLD Phase 0 calibrator (do not use)
│   └── text_calibrator_v3/     # ★ CURRENT text calibrator (Qwen2.5-7B)
├── MASTER_PLAN.md
├── RESEARCH_LOG.md             # MEMORY - update frequently
├── TECHNICAL_REPORT.md
└── CLAUDE.md                   # Instructions for coding agent
```

### 1.4 Compute Requirements
| Phase | GPUs | Time Estimate |
|-------|------|---------------|
| Data preparation | 1-2 | 2-4 hours |
| Text training (full) | 4 | ~15 min (LoRA) / ~2 hrs (full finetune) |
| VLM training (full) | 4 | 2-4 hours |
| Text eval (per model) | 1 | 30 min |
| VLM eval (per model) | 2 | 1-2 hours |
| Closed-source API | 0 | 1-2 hours (rate limited) |
| Qwen3.5-397B inference | 4-8 | 2-4 hours (FP8, vLLM) |
| Verbalized baseline (per model) | 0-2 | 30 min - 1 hour |

---

## Phase 2: Text Experiments (Days 2-5) — MOSTLY COMPLETE

### 2.1 Model Selection

**Text Calibrator (current):**
| Model | Size | Status |
|-------|------|--------|
| **Qwen2.5-7B-Instruct** | 7B | ★ Current — `uq_models/text_calibrator_v3/` |

> ⚠️ **IMPORTANT:** The old LLaMA calibrator at `uq_models/llama-8b-uq-lora-v2/checkpoint-198` is Phase 0 only. All agents must use `text_calibrator_v3`.

**Source Model (training data):**
| Model | Type | Notes |
|-------|------|-------|
| **GPT-5-mini** | Closed (API) | 4,443 samples across 21 benchmarks in `runs/gpt5_mini_combined/` |

**Target Models (cross-model evaluation):**
| Model | Type | Size | Status |
|-------|------|------|--------|
| GPT-5-mini | Closed | — | ✅ In-distribution baseline (AUROC 0.789) |
| Qwen3-VL-30B-A3B-Thinking | Open | 30B MoE | ✅ Cross-model done (AUROC 0.682) |
| **GPT-5.2** | Closed | — | 🔄 Responses generating (API running now) |
| **Claude** | Closed | — | ❌ Planned — needs API response generation (requires cost approval) |
| **Qwen3.5-397B-A17B-FP8** | Open | 397B MoE (17B active) | ❌ Planned — needs local inference on 4-8 GPUs |

### 2.2 Benchmark Selection

**Training Benchmarks (diverse, 40-70% accuracy range):**
| Category | Benchmarks | Samples (approx) |
|----------|------------|------------------|
| Math | GSM8K, MATH, OmniMath | ~2000 |
| Science | GPQA, ARC-Challenge | ~500 |
| Reasoning | BBEH, BIG-Bench Hard | ~1500 |
| Knowledge | MMLU (subset), TriviaQA | ~1000 |
| Factual | SimpleQA | ~500 |
| Total | | ~5500 |

**OOD Evaluation Benchmarks (not in training):**
- HellaSwag, WinoGrande, BoolQ (commonsense)
- DROP (reading comprehension)

### 2.3 Training Configuration
```yaml
model: Qwen2.5-7B-Instruct
lora:
  r: 16  # Increased from 8
  alpha: 32
  dropout: 0.1
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
training:
  epochs: 3
  batch_size: 4
  gradient_accumulation: 8
  learning_rate: 2e-5
  warmup_ratio: 0.1
  max_length: 2048
data:
  class_balance: 50/50
  train_samples: ~5000
  val_samples: ~500
  test_samples: ~1000
```

### 2.4 Experiments to Run

| Experiment | Description | Status |
|------------|-------------|--------|
| T1 | Train text calibrator on GPT-5-mini outputs (Qwen2.5-7B + LoRA) | ✅ Done (AUROC 0.789) |
| T2 | Eval in-distribution (GPT-5-mini) | ✅ Done (AUROC 0.789, ECE 0.040) |
| T3 | Eval cross-model (Qwen3-VL-30B) | ✅ Done (AUROC 0.682) |
| T4 | Eval cross-model (GPT-5.2) | 🔄 GPT-5.2 responses generating; eval next |
| T5 | Eval cross-model (Claude) | ❌ Needs Claude API responses first |
| T6 | Eval cross-model (Qwen3.5-397B) | ❌ Needs Qwen3.5-397B responses first |
| T7 | Training size ablation | ✅ Done (plateau at ~500 samples) |
| T8 | OOD benchmark evaluation | Partial |
| T9 | Verbalized confidence baseline (all target models) | ❌ NEW — see Phase 2B |

---

## Phase 2B: Verbalized Confidence Baselines — NEW

### Goal
Establish baselines by prompting models to self-report confidence, following Kapoor et al. Section 4. This is critical for the paper — reviewers will ask "why not just ask the model?"

### Methods to Implement
| Baseline | Description | Requirements |
|----------|-------------|--------------|
| **Verbalized** | Prompt model: "What is your confidence this answer is correct? Give a probability 0-1" → parse float | Works on any model with text output |
| **Zero-Shot Classifier** | Append "Is the answer correct? (a) Yes (b) No" → P("(a)") / (P("(a)") + P("(b)")) | Requires logprobs (open-source models only) |
| **Temperature Scaling** | Post-hoc scaling of logits on held-out set | Requires logprobs |

### Experiments
| Experiment | Model | Baseline | Status |
|------------|-------|----------|--------|
| B1 | GPT-5-mini (in-dist) | Verbalized | ❌ Planned |
| B2 | GPT-5.2 | Verbalized | ❌ Planned — can piggyback on response generation |
| B3 | Claude | Verbalized | ❌ Planned — can piggyback on response generation |
| B4 | Qwen3.5-397B | Verbalized + Zero-Shot Classifier | ❌ Planned — has logprobs |
| B5 | Qwen3-VL-30B | Verbalized + Zero-Shot Classifier | ❌ Planned — has logprobs |

### Implementation Notes
- For closed-source (GPT-5.2, Claude): Only verbalized is possible (no logprobs)
- For open-source (Qwen3.5-397B, Qwen3-VL-30B): Both verbalized AND zero-shot classifier
- **Efficiency:** Verbalized elicitation can be done in the same API call as response generation — add a second-stage prompt after the answer is generated. Do NOT make separate API calls.
- Parse failures default to 0.5 confidence (following Kapoor et al.)
- This is cheap: no training, no GPU for closed-source, minimal overhead

### Expected Result
Fine-tuned calibrator should substantially outperform verbalized baselines, especially on open-ended generation (per Kapoor et al. Figure 2). If verbalized is competitive on newer models, that's also an interesting finding.

---

## Phase 3: Vision Experiments (Days 5-9) — MOSTLY COMPLETE

### 3.1 Model Selection

**VLM Calibrator (current):**
| Model | Size | Status |
|-------|------|--------|
| **Qwen3-VL-8B-Instruct** | 8B | ★ Current — `data/vlm_judge_vsr_fixed/` (VSR-fixed retrain) |

**Source Model (training data):**
| Model | Type | Notes |
|-------|------|-------|
| **InternVL3-78B** | Open | 6,298 samples (4,200 vision + 2,098 text) |

**Target Models (cross-model evaluation):**
| Model | Type | Status |
|-------|------|--------|
| InternVL3-78B | Open | ✅ In-distribution (AUROC 0.804) |
| Qwen3-VL-30B-A3B-Thinking | Open | ✅ Cross-model done (AUROC 0.681) |
| GPT-5-mini | Closed | ✅ Cross-model eval was running (SLURM 7604) — check if complete |
| **GPT-5.2** | Closed | ❌ Planned — needs VLM benchmark responses via API |
| **Claude** | Closed | ❌ Planned — needs VLM benchmark responses via API |
| **Qwen3.5-397B-A17B-FP8** | Open | ❌ Planned — if model supports vision (verify) |

### 3.2 Benchmark Selection

**Training Benchmarks:**
| Benchmark | Samples | Task | Status |
|-----------|---------|------|--------|
| VSR | 2000 | True/False | ✅ Fixed (COCO images) |
| MMMU | 1000 | MCQ | ✅ Fixed |
| CharXiv | 1000 | Chart QA | ✅ |
| HallusionBench | 1000 | Yes/No | ✅ Fixed |
| MathVista | 1000 | Math + Vision | ✅ Added |
| RealWorldQA | 500 | Real-world | ✅ Added |
| **Total** | **6500** | | |

**OOD Evaluation:**
- AI2D (diagram understanding)
- ScienceQA (science + vision)
- TextVQA (text in images)

### 3.3 Training Configuration
```yaml
model: Qwen2.5-VL-7B-Instruct
lora:
  r: 16
  alpha: 32
  dropout: 0.1
  target_modules: [q_proj, k_proj, v_proj, o_proj]
training:
  epochs: 3
  batch_size: 1
  gradient_accumulation: 16
  learning_rate: 1e-4
  max_length: 2048
data:
  class_balance: 50/50 (downsample correct)
  train_samples: ~5000
  val_samples: ~500
  test_samples: ~1000
  image_validation: True  # Check images are real
```

### 3.4 Experiments to Run

| Experiment | Description | Status |
|------------|-------------|--------|
| V1 | Fix VSR data loading (COCO images) | ✅ Done |
| V2 | Retrain VLM judge with fixed VSR | ✅ Done (AUROC 0.804, VSR 0.834) |
| V3 | Eval in-distribution (InternVL) | ✅ Done (AUROC 0.804) |
| V4 | Eval cross-model (Qwen3-VL-30B) | ✅ Done (AUROC 0.681) |
| V5 | Eval cross-model (GPT-5-mini) | ✅/🔄 SLURM 7604 — check completion |
| V6 | Eval cross-model (GPT-5.2) | ❌ Needs GPT-5.2 VLM responses |
| V7 | Eval cross-model (Claude) | ❌ Needs Claude VLM responses |
| V8 | Training size ablation | ⚠️ **REDO NEEDED** — previous result (plateau at 250) was on buggy data (gray VSR images, broken ERQA). Must re-run with fixed data. |
| V9 | OOD benchmark evaluation | Partial |
| V10 | Verbalized confidence baselines (vision) | ❌ NEW |

---

## Phase 4: Multi-Model Evaluation Matrix (Days 9-11)

### Goal
Build a comprehensive cross-model transfer matrix showing that the calibrator generalizes across many target models. This is the paper's main results table.

### 4.1 Response Generation (must happen first)

**Text responses needed:**
| Model | Method | Est. Cost | Est. Time | Status |
|-------|--------|-----------|-----------|--------|
| GPT-5.2 | API | ~$5 | 1-2 hrs | 🔄 Running now |
| Claude | API | ~$5-10 | 1-2 hrs | ❌ **Needs cost approval** |
| Qwen3.5-397B-A17B-FP8 | Local (vLLM, FP8) | $0 (GPU) | 2-4 hrs on 4-8 GPUs | ❌ Planned |

**Vision responses needed:**
| Model | Method | Est. Cost | Est. Time | Status |
|-------|--------|-----------|-----------|--------|
| GPT-5.2 | API (with images) | ~$15-20 | 2-3 hrs | ❌ Planned |
| Claude | API (with images) | ~$10-15 | 2-3 hrs | ❌ **Needs cost approval** |

**Verbalized confidence (piggyback on response generation):**
- For each API call above, add a second-stage prompt to elicit verbalized confidence
- This gives us the baseline for free (no extra API cost if done in same call)

### 4.2 Cross-Model Evaluation Matrix (Text)

Run text calibrator (`uq_models/text_calibrator_v3/`) on each target model's responses:

| Target Model | Type | Expected AUROC | Status |
|--------------|------|----------------|--------|
| GPT-5-mini | Closed (source) | 0.789 (in-dist) | ✅ Done |
| Qwen3-VL-30B | Open | 0.682 | ✅ Done |
| GPT-5.2 | Closed | ~0.65-0.75? | ❌ Pending responses |
| Claude | Closed | ~0.65-0.75? | ❌ Pending responses |
| Qwen3.5-397B | Open | ~0.65-0.75? | ❌ Pending responses |

### 4.3 Cross-Model Evaluation Matrix (Vision)

Run VLM calibrator (`data/vlm_judge_vsr_fixed/`) on each target model's VLM responses:

| Target Model | Type | Expected AUROC | Status |
|--------------|------|----------------|--------|
| InternVL3-78B | Open (source) | 0.804 (in-dist) | ✅ Done |
| Qwen3-VL-30B | Open | 0.681 | ✅ Done |
| GPT-5-mini | Closed | TBD | ✅/🔄 Check SLURM 7604 |
| GPT-5.2 | Closed | TBD | ❌ Pending responses |
| Claude | Closed | TBD | ❌ Pending responses |

### 4.4 Baseline Comparison Table (per target model)

For each target model, compare:
| Method | Description | Who can use it |
|--------|-------------|---------------|
| Our calibrator (LoRA) | Fine-tuned judge | Any model |
| Verbalized confidence | "What's your confidence?" | Any model |
| Zero-Shot Classifier | logP("Yes")/logP("No") | Open-source only |
| Random baseline | Uniform [0,1] | Sanity check |

### 4.5 Multi-Source Training (stretch goal)

Test if training on multiple source models improves cross-model transfer:

| Experiment | Training Data | Eval | Status |
|------------|---------------|------|--------|
| M1 | GPT-5-mini only (current) | GPT-5.2 | ❌ Pending |
| M2 | GPT-5-mini + Qwen3-VL-30B | GPT-5.2 | ❌ Planned |
| M3 | GPT-5-mini + Qwen3-VL-30B + InternVL3 | GPT-5.2 | ❌ Planned |

**Hypothesis:** Multi-source training should improve transfer by preventing overfitting to single model's patterns.

> This is lower priority than getting the full eval matrix. Only do if core results are solid and time permits.

---

## Phase 5: Analysis & Validation (Days 11-12)

### 5.1 Spurious Correlation Checks
For both text and vision:
- [x] Within-benchmark AUROC (should be >> 0.5) — mean 0.757
- [x] Length-only baseline AUROC — 0.464 (anti-predictive)
- [x] Token-only baseline AUROC (for binary benchmarks)
- [x] Response pattern analysis

### 5.2 Calibration Analysis
- [ ] Reliability diagrams (all settings)
- [x] ECE computation — text 0.040, vision 0.073
- [ ] Confidence histograms

### 5.3 Selective Prediction
- [x] Coverage-accuracy curves — 90% acc at 70.9% coverage
- [ ] Comparison across all target models and settings

### 5.4 Statistical Significance
- [ ] Bootstrap confidence intervals (n=1000)
- [ ] Paired tests for key comparisons
- [ ] Multiple random seeds (if time)

### 5.5 Error Analysis
- [ ] Per-benchmark breakdown
- [ ] Failure case analysis
- [ ] Qualitative examples

### 5.6 Ablations
| Ablation | Description | Status |
|----------|-------------|--------|
| Training size (text) | 100/250/500/1000/2500/5000 samples | ✅ Done (plateau ~500) |
| Training size (vision) | Same sizes | ⚠️ **REDO** with fixed data |
| Full finetune vs LoRA | Compare full FT to LoRA on text calibrator | ❌ Low priority, do if time |
| LoRA rank | r=4/8/16/32 | ❌ Optional |
| JSD regularization | With/without (per Kapoor et al. Table 1) | ❌ Check if implemented |

---

## Phase 6: Paper Writing (Days 12-14)

### Figures Needed
1. Method diagram (pipeline overview)
2. **Main results table** — cross-model transfer matrix (text + vision × all target models)
3. Cross-model transfer bar chart (text): calibrator vs verbalized vs zero-shot
4. Cross-model transfer bar chart (vision): same comparison
5. Training efficiency curve (text — already done; vision — redo)
6. Selective prediction curve
7. Calibration reliability diagrams
8. Per-benchmark breakdown

### Paper Sections
1. Abstract
2. Introduction
3. Related Work
4. Method
5. Experiments
   - 5.1 Experimental Setup
   - 5.2 Text Experiments
   - 5.3 Vision Experiments
   - 5.4 Closed-Source Transfer
   - 5.5 Baselines (Verbalized, Zero-Shot Classifier)
   - 5.6 Ablations
   - 5.7 Analysis
6. Discussion & Limitations
7. Conclusion

---

## Budget Summary

| Category | Estimated Cost | Status |
|----------|----------------|--------|
| GPT-5-mini responses (text+vision) | ~$5 | ✅ Done |
| GPT-5.2 text responses | ~$5 | 🔄 Running |
| GPT-5.2 vision responses | ~$15-20 | ❌ Planned |
| Claude text responses | ~$5-10 | ❌ **Needs approval** |
| Claude vision responses | ~$10-15 | ❌ **Needs approval** |
| Verbalized confidence (closed-source) | ~$2-5 | Piggyback on above |
| Buffer/reruns | ~$15 | |
| **Total** | **~$60-80** | Within $100 budget |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Closed-source results don't transfer | Run small pilot (50 samples) first |
| Qwen3.5-397B doesn't fit on 8 GPUs | FP8 quantization; fall back to API if available |
| Training takes too long | Use 8 GPUs at night for large runs |
| Bug discovered late | Verify each step before proceeding |
| Results not reproducible | Set seeds, log everything, save checkpoints |
| ECCV deadline pressure | Prioritize core experiments, cut optional ones |
| Verbalized baseline is surprisingly good | Still publishable — shows when fine-tuning helps vs doesn't |

---

## Weekend Plan: February 21-23, 2026

### Currently Running
- GPT-5.2 text response generation (API) — check completion
- SLURM 7604: VLM judge cross-model eval on GPT-5-mini — check completion
- SLURM 7608, 7609: Qwen3-VL-30B missing benchmarks — check completion
- SLURM 7610, 7611: Follow-up cross-model evals (dependency on 7608+7609)

### Priority Order (do in this order, stop when time runs out)

**P0 — Headline results (today/tonight):**
1. Check all running SLURM jobs — get results from anything that's finished
2. Once GPT-5.2 text responses are in → run text calibrator cross-model eval (2 GPUs, ~1hr)
3. Run verbalized confidence baseline on GPT-5.2 responses (CPU/minimal, piggyback)

**P1 — Expand model coverage (Saturday):**
4. **Claude text responses** — generate via API (**needs cost approval, est. ~$5-10**)
5. **Qwen3.5-397B-A17B-FP8 text responses** — download model, run via vLLM FP8 on 4-8 GPUs
6. Cross-model eval on Claude + Qwen3.5 with text calibrator
7. Verbalized confidence baselines for Claude + Qwen3.5

**P2 — Vision closed-source (Saturday night/Sunday):**
8. **GPT-5.2 vision responses** — generate via API (**needs cost approval, est. ~$15-20**)
9. VLM cross-model eval on GPT-5.2 vision responses
10. Claude vision responses if budget allows

**P3 — Ablations & cleanup (Sunday):**
11. **Re-run vision training size ablation** with fixed VSR data
12. Full finetune ablation (one run, text calibrator) — only if time
13. Generate figures for completed results

### What to Cut if Time-Limited
- Multi-source training (Phase 4.5) — defer to next week
- Full finetune ablation — nice to have, not essential
- OOD benchmark evals — can do later
- Vision responses for Claude — text-only Claude results still valuable

---

## Agent Task Assignments

### For Coding Agents — Read This

**Active checkpoints (USE THESE):**
- Text calibrator: `uq_models/text_calibrator_v3/` (Qwen2.5-7B + LoRA, AUROC 0.789)
- VLM calibrator: `data/vlm_judge_vsr_fixed/` (Qwen3-VL-8B + LoRA, AUROC 0.804)

**DO NOT USE:**
- `uq_models/llama-8b-uq-lora-v2/checkpoint-198` — this is the old Phase 0 calibrator

**Before any paid API call:**
- Estimate cost and number of calls
- Report to user and wait for explicit approval
- Include verbalized confidence elicitation in the same call where possible

**Before any GPU job:**
- Check `squeue` for running jobs
- Check `nvidia-smi` for GPU availability
- Smoke test with 5-10 examples first
- Submit via SLURM, not direct execution

**Qwen3.5-397B-A17B-FP8 setup notes:**
- This is a MoE model: 397B total params, 17B active, FP8 quantized
- Will need vLLM with `--quantization fp8` flag
- Estimate 4-8 GPUs needed for inference (TP=4 or TP=8)
- Download from HuggingFace first, verify disk space (~200GB FP8)
- Smoke test on 5 examples before full benchmark run

---

## Next Steps (as of Feb 21, 2026 evening)

1. **Now:** Check all running SLURM jobs (7604, 7608, 7609, 7610, 7611)
2. **Now:** Check if GPT-5.2 API job has completed
3. **Tonight:** Run text cross-model eval on GPT-5.2 (the headline result)
4. **Tonight:** Start Qwen3.5-397B download + smoke test
5. **Tomorrow:** Generate Claude responses (after cost approval)
6. **Tomorrow:** Run Qwen3.5-397B on text benchmarks
7. **Sunday:** Vision closed-source evals, re-run vision ablation, figures

---

*Last updated: February 21, 2026 (evening)*
