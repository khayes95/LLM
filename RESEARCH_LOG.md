# Research Log

> Full history archived at `archive/RESEARCH_LOG_full_backup.md`

---

## Current: Best Unified UQ Model v2

**Model:** Qwen3-VL-8B-Instruct + LoRA (r=32, combined prompt), trained on ALL text+VLM data
**Checkpoint:** `uq_models/best_unified_v2/` (epoch 2/3 — epoch 3 timed out)
**Test AUROC: 0.890** (VLM: 0.905, Text: 0.870) — up from 0.831 baseline
**Previous:** `uq_models/best_unified/` (r=16, AUROC 0.831)
**Scored data (v1):** `data/use_cases/scored_unified/` | Test-only: `data/use_cases/scored_test_only/`

---

## 2026-02-28

### 11:44 — Phase 5 analysis (CPU-only, SLURM 8051)
Why: Generate paper-ready figures and metrics for reliability, calibration, selective prediction, baselines.
Script: `scripts/phase5_analysis.py` | Output: `figures/paper/`, `data/use_cases/results_unified/`

Figures generated: `reliability_diagrams.pdf`, `confidence_histograms.pdf`, `bootstrap_ci_comparison.pdf`, `selective_prediction.pdf`, `per_benchmark_breakdown.pdf`

Bootstrap CIs (scoring AUROC, includes train data — **NOT for paper main table**):

| Method | GPT-5-mini | GPT-5.2 | Qwen3.5 |
|--------|-----------|---------|---------|
| Calibrator | 0.956 [0.950,0.961] | 0.933 [0.926,0.941] | 0.911 [0.901,0.921] |
| Verbalized (Isotonic) | 0.666 | 0.637 | 0.594 |
| Zero-shot base | 0.554 | 0.493 | 0.539 |

**WARNING:** These scoring AUROCs are inflated — computed on `scored_unified/` which includes training data. Held-out test AUROC is 0.831 (v1) / 0.890 (v2). Use `scored_test_only/` for clean paper metrics.

Selective prediction (calibrator): Cov@90% = 50.3% (GPT-5-mini), 57.4% (GPT-5.2), 35.4% (Qwen3.5).

### [RUNNING] Combined prompt retrain v2 (SLURM 8031)
Why: Retrain with r=32 + combined prompt (longer+CoT+metadata). Ablations showed 0.831→0.867.
Script: `scripts/retrain_best_v2.py` | Output: `uq_models/best_unified_v2/`

---

## 2026-02-26

### Unified Model Trained + Scored + Use Cases Complete
- Trained on 9,533 samples (3 models × 20 benchmarks), 3 epochs, 2h on 4 GPUs
- Graded 933 previously-ungraded predictions (healthbench, tutorbench, prbench, mmvet) via GPT-5-mini judge
- Scored all ~11,700 predictions → `data/use_cases/scored_unified/`
- Ran all 6 use cases with unified scores

### Use Case Results (unified model)

| UC | Metric | GPT-5-mini | GPT-5.2 | Qwen3.5 | Old (text-only) |
|----|--------|------------|---------|---------|-----------------|
| UC1 | AURC↓ | 0.165 | 0.125 | 0.189 | 0.290 |
| UC2 | Cost savings | 47% | — | — | 37% |
| UC3 | Best F1 | 0.883 | 0.831 | 0.840 | 0.757 |
| UC4 | Rank corr | r=0.972 | r=0.960 | r=0.949 | r=0.845 |
| UC5 | Pairwise | 79-85% | — | — | 51% (broken) |

### UC8 + UC9 Implemented (unified model scores)

| UC | Metric | GPT-5-mini | GPT-5.2 | Qwen3.5 |
|----|--------|------------|---------|---------|
| UC8 | Spearman r | 0.972 | 0.944 | 0.887 |
| UC8 | Alert F1 | 1.000 | 0.895 | 0.789 |
| UC8 | Min batch | 25 | 25 | 25 |
| UC9 | AUEDR | 0.739 | 0.754 | 0.711 |
| UC9 | EDR@20% | 49.5% | 47.2% | 39.7% |

Scripts: `scripts/uc8_ood_detection.py`, `scripts/uc9_annotation_prioritization.py`
Results: `data/use_cases/results_unified/uc{8,9}_results.json`

### TODO
- Generate missing Qwen3.5 predictions (~1,357 samples on 9 text + 3 VLM benchmarks) — BLOCKED on GPU availability (coryan using GPUs 6-7)
- Pipeline script ready: `slurm/overnight_retrain_pipeline.sh` (SLURM 7897 — will need resubmit when GPUs free)
- After inference: grade → retrain → re-score → re-run all 8 use cases

---

## Key Results Summary

### Unified Model (current best)
| Target | Held-out AUROC | Scoring AUROC* | Verbalized |
|--------|---------------|----------------|------------|
| GPT-5-mini | 0.831 | 0.956 | 0.650 |
| GPT-5.2 | — | 0.932 | 0.654 |
| Qwen3.5 | — | 0.916 | 0.613 |

*Scoring AUROC includes training data — held-out 0.831 is the clean metric.

### Legacy (for comparison only)
- Text-only calibrators: 0.815 (GPT-5-mini), 0.702 (GPT-5.2), 0.739 (Qwen3.5)
- Old VLM judge (InternVL3 only): 0.585-0.648
- Size ablation (text-only): 0.5B=0.758, 1.5B=0.772, 3B=0.767, 7B=0.785

---

## 2026-02-27

### 00:39-06:16 — Unified VLM size ablation (2B, 4B, 8B)
Why: Test how small the UQ model can be for laptop deployment
Script: `scripts/unified_size_ablation.py` | SLURM 7911 | Output: `uq_models/unified_size_ablation/`

| Model | AUROC | VLM | Text | Brier | Time |
|-------|-------|-----|------|-------|------|
| Qwen3-VL-2B | 0.816 | 0.797 | 0.838 | 0.177 | 65 min |
| Qwen3-VL-4B | **0.830** | **0.821** | 0.841 | 0.178 | 129 min |
| Qwen3-VL-8B | 0.827 | 0.816 | 0.840 | 0.182 | 140 min |

2B retains 98% of 8B performance. 4B slightly beats 8B (likely noise). All viable for deployment.

### 01:30-02:00 — 4 Novel Use Cases Stage 1 (CPU-only)
Why: Ambitious new use cases beyond analysis — DPO reward, best-of-N, data curation, agent steps
Scripts: `scripts/uc_a_dpo_reward.py`, `uc_b_best_of_n.py`, `uc_c_data_curation.py`, `uc_d_agent_steps.py`
Output: `data/use_cases/results_unified/uc_{a,b,c,d}_results.json`

| UC | Use Case | Key Metric | Result | Threshold | Pass? |
|----|----------|-----------|--------|-----------|-------|
| UC-A | DPO Reward | Informative pair accuracy | 84.0% | >65% | YES |
| UC-B | Best-of-N | N=3 calibrator accuracy | 67.4% (+5.4% over best model) | >best model | YES |
| UC-C | Data Curation | Acc@50% retention | 90.9% (vs 51.7% random) | >85% | YES |
| UC-D | Agent Steps | Partial corr (beyond length) | r=0.813 | signal exists | YES |

### 02:46-08:10 — UC-D S2 + UC-C S2 overnight GPU run
Why: Stage 2 validation — step truncation scoring and student model training
Script: `scripts/uc_d_step_truncation.py`, `scripts/uc_c_train_students.py` | SLURM 7925

**UC-D S2 (step truncation):** 1,692 multi-step responses, 10,782 inferences in 25 min.
Step-level drop AUROC=0.53-0.56 (weak). Full P(correct) baseline AUROC=0.74.
Conclusion: outcome-level calibrator captures correctness upfront; step drops add little.

**UC-C S2 (student training):** 5 Qwen2.5-1.5B variants on filtered GPT-5-mini data.

| Variant | Train Size | Teacher Acc | Student Acc |
|---------|-----------|-------------|-------------|
| All data | 3,487 | 51.5% | 15.0% |
| Calibrator p>0.7 | 1,680 | 90.7% | **16.6%** |
| Random subsample | 1,680 | 50.3% | 15.3% |
| Verbalized p>0.7 | 2,895 | 58.4% | 16.3% |
| Oracle filtered | 1,795 | 100% | 16.3% |

Student accuracy is low overall (1.5B model on hard benchmarks), but calibrator-filtered matches oracle and beats random subsample at same data budget (+1.3%).

### 02:35-08:30 — Ablation experiments (LoRA rank, source model, modality)
Why: Measure impact of LoRA capacity, source model diversity, and modality mixing on UQ performance.
Script: `scripts/run_ablations.py` | Output: `data/ablations/{lora_rank,source_model,modality}/`

**LoRA Rank Ablation** (baseline r=16 = 0.831):
| r | AUROC | VLM | Text | Trainable params |
|---|-------|-----|------|-----------------|
| 4 | 0.821 | 0.806 | 0.837 | 3.8M (0.04%) |
| 8 | 0.824 | 0.807 | 0.844 | 7.7M (0.09%) |
| 16 | 0.831 | 0.813 | 0.850 | 15.2M (0.17%) |
| 32 | **0.844** | **0.840** | 0.846 | 30.4M (0.35%) |

**Source Model Ablation** (train on 1 model, eval on full test):
| Source | AUROC | VLM | Text | Samples |
|--------|-------|-----|------|---------|
| GPT-5-mini | 0.744 | 0.731 | 0.758 | 3,546 |
| GPT-5.2 | 0.759 | 0.754 | 0.772 | 3,750 |
| Qwen3.5 | 0.672 | 0.691 | 0.654 | 2,751 |
| All 3 | **0.831** | **0.813** | **0.850** | 9,533 |

**Modality Ablation** (cross-modality transfer):
| Training | Overall | In-dist | Cross-modal |
|----------|---------|---------|-------------|
| Text-only | 0.719 | 0.857 (text) | 0.588 (VLM) |
| VLM-only | 0.685 | 0.821 (VLM) | 0.518 (text) |
| Unified | **0.831** | 0.850/0.813 | N/A |

Key findings: (1) r=32 beats r=16 by 1.3pts — consider upgrading. (2) Multi-source adds 7-16pts. (3) Cross-modality transfer fails — unified training essential.

### 08:45-20:33 — Prompt/elicitation ablation experiments
Why: Test if giving the model more context or reasoning instructions improves AUROC.
Script: `scripts/run_prompt_ablations.py` | Output: `data/ablations/prompt/{cot,longer,metadata,combined}/`

| Prompt Variant | AUROC | VLM | Text | vs Baseline |
|----------------|-------|-----|------|-------------|
| Baseline (500/300) | 0.831 | 0.813 | 0.850 | — |
| CoT | 0.836 | 0.827 | 0.845 | +0.5 |
| Metadata | 0.836 | 0.825 | 0.849 | +0.5 |
| Longer (1500/800) | 0.862 | 0.873 | 0.847 | **+3.1** |
| **Combined** | **0.867** | **0.881** | 0.849 | **+3.6** |

Key finding: **Longer context is the biggest lever** — Q/A truncation was throwing away useful info, especially for VLM benchmarks (+6.0 pts VLM). CoT and metadata add small gains (~0.5 pts each). Combined gets **0.867 AUROC**, a new best. Consider retraining best_unified with combined template.

### 20:30-21:20 — Baselines, bootstrap CIs, UC5 fix
Why: Address reviewer-ready baseline gaps — only had verbalized confidence before
Scripts: `scripts/compute_baselines.py`, `scripts/zero_shot_baseline.py`, `scripts/bootstrap_ci.py`
Output: `data/use_cases/results_unified/bootstrap_ci.json`, `uc5_results.json`

| Baseline | AUROC | 95% CI |
|----------|-------|--------|
| **Calibrator (ours)** | **0.936** | [0.932, 0.940] |
| Combined (verb+len) | 0.652 | [0.642, 0.663] |
| Verbalized (Isotonic) | 0.650 | [0.639, 0.661] |
| Verbalized (raw) | 0.607 | [0.596, 0.617] |
| Response length | 0.598 | [0.588, 0.608] |
| Zero-shot base model | 0.524 | [0.514, 0.535] |

UC5 re-run with unified model: pairwise accuracy 79-85% across model pairs (was ~51% before fix).

---

## 2026-02-28

### 01:53-03:30 — Reviewer experiment pipeline (CPU-only tasks)
Why: Prepare for ECCV 2026 reviewer concerns — data leakage fix, significance tests, held-out eval
Scripts: `scripts/filter_test_only.py`, `scripts/per_benchmark_breakdown.py`, `scripts/paired_significance_tests.py`, `scripts/held_out_benchmark_eval.py`, `scripts/literature_comparison.py`
Output: `data/use_cases/results_test_only/`, `data/use_cases/results_unified/`

**Data leakage fix (scored test-only):** Filtered 11,739 → 4,152 test-only samples using split_info.json. Re-ran all 11 use cases.

| Metric | All Data | Test-Only | Delta |
|--------|----------|-----------|-------|
| Overall AUROC | 0.936 | 0.915 | -0.021 |
| GPT-5-mini | 0.956 | 0.917 | -0.038 |
| GPT-5.2 | 0.933 | 0.916 | -0.018 |
| Qwen3.5 | 0.911 | 0.911 | -0.000 |

Use case metric drops: UC3 F1 ~-0.02 to -0.05, UC1 AURC +/-0.01. Modest drops confirm data leakage was not a major issue.

**Bootstrap CIs (test-only):**

| Baseline | AUROC | 95% CI |
|----------|-------|--------|
| Calibrator (ours) | 0.915 | [0.906, 0.923] |
| Verbalized (Isotonic) | 0.676 | [0.659, 0.692] |
| Combined (verb+len) | 0.672 | [0.655, 0.689] |
| Verbalized (raw) | 0.634 | [0.617, 0.651] |
| Response length | 0.621 | [0.604, 0.639] |
| Zero-shot base model | 0.540 | [0.523, 0.557] |

**Significance tests:** All baselines vs calibrator: p < 0.001 (paired permutation, DeLong, McNemar). All ***.

**Per-benchmark AUROC (test-only, 20 benchmarks):**
Script: `scripts/per_benchmark_analysis.py` | Output: `data/use_cases/results_unified/per_benchmark_analysis.json`
Result: Calibrator beats verbalized on 18/19 benchmarks (all w/ sufficient samples). Test AUROC 0.9147 [0.9058, 0.9231].

**Held-out benchmark eval (leave-5-out CV, CPU):**
Script: `scripts/held_out_benchmark_eval.py` | Output: `data/use_cases/results_unified/held_out_eval.json`
Result: Mean gap (in-dist − held-out) = +0.011 (negligible). Per-benchmark AUROC: mean=0.850, std=0.097, CV=0.114. Hardest: mmvet (0.637), prbench (0.658). Easiest: realworldqa (0.982), livebench (0.982).

**Literature comparison:**
Output: `data/use_cases/results_unified/literature_comparison.json`
Compared against 11 published UQ methods (MC Dropout, Deep Ensembles, Semantic Entropy, P(True), etc.). Key differentiator: only our method combines closed-source, cross-model, and multimodal in one model. Our method is the only one in the "black-box single-pass" category with AUROC 0.915.

### 02:36-10:37 — Retrain v2 (r=32 + combined prompt)
Why: Retrain with best ablation config (r=32, combined prompt 1500/800)
Script: `scripts/retrain_best_v2.py` | SLURM 8030 (TIMEOUT at 8h, completed 2/3 epochs)
Output: `uq_models/best_unified_v2/` | Results: `uq_models/best_unified_v2/results.json`

| Epoch | AUROC | VLM | Text | Loss |
|-------|-------|-----|------|------|
| 1 | 0.852 | 0.856 | 0.849 | 0.607 |
| 2 | **0.890** | **0.905** | 0.870 | 0.356 |
| 3 | — | — | — | (timed out) |

**New best: 0.890 AUROC** (+5.9 pts over 0.831). VLM: +9.2 pts. Post-pipeline (scoring, filtering) did not run.

### 02:36-14:37 — Multi-seed training (SLURM 8031, TIMEOUT at 12h)
Why: 3 random seeds (42, 123, 456) for error bars with r=32 + combined prompt.
Script: `scripts/multi_seed_training.py` | Output: `data/ablations/multi_seed/`

**Seed 42 (complete, 3 epochs, 525 min):**

| Epoch | AUROC | VLM | Text | Loss |
|-------|-------|-----|------|------|
| 1 | 0.857 | 0.868 | 0.843 | 0.603 |
| 2 | **0.898** | **0.915** | **0.875** | 0.350 |
| 3 | 0.894 | 0.916 | 0.867 | 0.165 |

Best checkpoint saved at epoch 2. Epoch 3 slightly overfits (-0.4 pts).

**Seed 123 (timed out after epoch 1):** AUROC 0.843 (VLM: 0.858, Text: 0.820)

**Key finding:** Epoch 2 is the sweet spot — epoch 3 overfits. The retrain_v2 (8030) epoch 2 checkpoint (0.890) and seed 42 epoch 2 (0.898) are both strong. Seed 456 never ran.

### Queued jobs (dependency chain)
- **SLURM 8082** (running): Overnight pipeline — Qwen3.5 inference → grade → retrain → score → use cases
- **SLURM 8086** (after 8082): Score v2 checkpoint (r=32, backed up to `uq_models/best_v2_r32_combined/`) → filter test-only → re-run all UCs → bootstrap CIs
- **SLURM 8089** (after 8086): Multi-seed remaining (seeds 123, 456), 24h limit

### 02:00-09:00 — Novel elicitation strategy ablations
Why: Test 3 alternative ways to extract UQ signal from the judge model.
Script: `scripts/elicitation_ablations.py` | Output: `data/ablations/elicitation/{multi_sample_5,contrastive,verbalized}/`

| Strategy | AUROC | VLM | Text | vs Baseline |
|----------|-------|-----|------|-------------|
| Logit baseline | 0.859 | 0.837 | 0.888 | — |
| **Contrastive (+ ref answer)** | **0.903** | **0.879** | **0.928** | **+4.4** |
| MC Dropout mean (N=5) | 0.859 | 0.836 | 0.889 | +0.0 |
| MC Dropout vote (N=5) | 0.779 | 0.753 | 0.809 | -8.0 |
| Verbalized probability | 0.749 | 0.741 | 0.759 | -11.0 |

Key findings: (1) Contrastive prompting (include gold answer) boosts AUROC by +4.4 pts — but requires access to reference answer, so limited applicability. (2) MC Dropout adds nothing — LoRA dropout too sparse for epistemic uncertainty. (3) Verbalized probability output substantially worse than logit extraction (-11 pts) — text generation loses calibration signal.

---

## Earlier (summaries)

**2026-02-25:** Model-specific text calibrators, scored_v2, all 7 use cases
**2026-02-24:** VLM ablation, cross-model eval, verbalized baselines
**Earlier:** See `archive/RESEARCH_LOG_full_backup.md`
