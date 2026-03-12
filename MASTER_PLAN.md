# MASTER PLAN: Pinocchio — Uncertainty Quantification for Black-Box LLMs
## Target: ICML 2026 Submission

**Created:** December 30, 2025
**Last major update:** March 11, 2026
**Status:** Phase 5 (Paper Writing) — v3 model (clean question-level split) is current best, paper needs number updates

---

## Paper Title
**Pinocchio: Estimating the Uncertainty of Black-Box Language Models**

## Paper Contributions

1. **Cross-Model Transfer:** A single calibrator trained on responses from GPT-5-mini, GPT-5.2, and Qwen3.5-72B transfers across all three target models (0.873--0.882 AUROC, clean question-level split). Single-model calibrators achieve only 0.672--0.759.
2. **Unified Text + Vision:** A single Qwen3-VL-8B model handles text-only and vision-language benchmarks in a unified architecture (mean AUROC: text 0.867, VLM 0.887).
3. **Practical Efficiency:** Single forward pass through 8B model (or 0.8B with 95% retention). 5,000 training examples capture 96% of full performance.
4. **Production Workflows:** Three deployment use cases (adaptive clarification, confidence-gated actions, human escalation) where no baseline achieves meaningful performance.

---

## Key Results (v3 checkpoint, clean question-level split, test-only evaluation)

| Metric | Value |
|--------|-------|
| **Overall AUROC** | **0.878** [0.863, 0.892] |
| GPT-5-mini AUROC | 0.882 [0.857, 0.906] |
| GPT-5.2 AUROC | 0.877 [0.851, 0.900] |
| Qwen3.5 AUROC | 0.873 [0.844, 0.900] |
| Verbalized confidence | 0.610 |
| Per-benchmark mean AUROC | 0.821 (CV=0.128) |
| ECE | 0.087 |
| Brier score | 0.151 |
| All p-values vs baselines | < 0.001 |

**Note:** v2 scores (0.953 AUROC) were inflated due to question-level data leakage (82--96% question overlap in test set). The v3 split ensures 0% question overlap. See CONTAMINATED DATA WARNING below.

### Ablations (v3, clean split)
| Ablation | Key Finding |
|----------|-------------|
| No-metadata | With metadata: 0.878, without: 0.827 (−5.0 pts) |
| Truncation | r200=0.805, r400=0.846, r800=0.878, r1600=0.888, r3000=0.890 |
| Scramble | Word-scramble: 0.781 (−9.7 pts), sentence-scramble: 0.867 (−1.1 pts) |
| Training size | N=5000 → 96% of full; N=2000 → 91%; N=100 → 0.503 |
| Model size (Qwen3.5) | 0.8B: 0.852, 2B: 0.850, 4B: 0.663, 9B: 0.771 (needs re-run with per-size HP tuning) |
| Elicitation | Logit=0.859, MC Dropout=0.859, Hidden+logit=0.884, Verbalized=0.749 |
| Near-duplicate contamination | 20.7% near-dupes in test; deduped AUROC=0.872 (−0.006, not inflated) |

### Production Use Cases (v3 test-only — results in `data/use_cases/results_test_only_v3/`)
| Use Case | Key Metric |
|----------|------------|
| Adaptive Clarification | See results directory |
| Confidence-Gated Actions | See results directory |
| Human Escalation | See results directory |

---

## Phase Status

### Phase 1: Foundation & Bug Fixes — COMPLETE
- Fixed VSR images, MMMU options, HallusionBench grading
- Created unified evaluation harness (32+ benchmarks)
- Established data pipeline and directory structure

### Phase 2: Text Experiments — COMPLETE
- Trained text calibrator v3 (Qwen2.5-7B + LoRA)
- Cross-model eval on GPT-5-mini, GPT-5.2, Qwen3.5-72B
- Training size ablation, OOD benchmark eval

### Phase 3: Vision Experiments — COMPLETE
- Trained VLM judge (Qwen3-VL-8B + LoRA) with fixed VSR data
- Cross-model transfer to InternVL3, Qwen2.5-VL-72B
- Established that VLM judge transfers to text-only eval (0.907)

### Phase 4: Unified Model & Extensions — COMPLETE
- **Best Unified Model v3 (current):** Qwen3-VL-8B, r=32, alpha=64, combined prompt, 3 target models, clean question-level split
  - Checkpoint: `uq_models/best_v3_qsplit/`
  - Held-out AUROC: 0.878 | VLM: 0.887 | Text: 0.867 | ECE: 0.087
  - Test-only scoring: 0.878 [0.863, 0.892] (1,953 samples, 0% question overlap)
- **Question-level data leakage fix (v3):** v2 split by sample index caused 82--96% question overlap in test. v3 splits by question ID — FIXED.
- **v2 (superseded):** AUROC 0.953 was inflated by question-level leakage
- **Reviewer experiments:** Bootstrap CIs, significance tests, held-out benchmark eval, literature comparison
- **Baselines:** 8+ baselines (verbalized, Platt, isotonic, length, combined, zero-shot, proxy SE, self-consistency)
- **v3 ablations:** No-metadata (0.827), truncation, scramble, near-duplicate contamination check
- **Qwen3.5 model size ablation:** 0.8B, 2B, 4B, 9B (needs re-run with per-size HP tuning)
- **Elicitation ablations:** MC Dropout, hidden state probe, verbalized, temperature scaling, token entropy

### Phase 5: Paper Writing — IN PROGRESS
- Paper ~90% written in Overleaf (ICML 2026 format), but numbers need updating from v2 → v3
- Overleaf project: `https://git.overleaf.com/696e96cbc0f12ed169b15d1e`
- Local clone: `/scratch/khayes/LLM/overleaf/`
- All figures generated in `figures/paper/`
- Per-section `.tex` files in `overleaf/sections/`
- **TODO:** Update all paper tables/figures with v3 numbers (0.878 AUROC, not 0.953)

---

## Key Paths

| Resource | Path |
|----------|------|
| **Best checkpoint (v3, current)** | `uq_models/best_v3_qsplit/` |
| V2 checkpoint (superseded, leaked) | `uq_models/best_v2_r32_combined/` |
| V1 checkpoint (superseded) | `uq_models/best_unified/` |
| Test-only scored data (v3) | `data/use_cases/scored_test_only_v3/` (1,953 samples, 0% question overlap) |
| Test-only results (v3) | `data/use_cases/results_test_only_v3/` |
| All scored data (v3) | `data/use_cases/scored_v3_all/` |
| Bootstrap CIs (v3) | `data/use_cases/results_test_only_v3/bootstrap_ci_v3.json` |
| Contamination report (v3) | `data/use_cases/results_test_only_v3/contamination_report_v3.json` |
| Paper figures | `figures/paper/` |
| Overleaf repo | `overleaf/` |
| V3 ablations (no-metadata) | `data/ablations/no_metadata_v3/` |
| V3 ablations (truncation/scramble) | `data/ablations/truncation_scramble_v3/` |
| Model size ablation | `data/ablations/qwen35_model_size/` |
| Training size ablation | `data/ablations/training_size/` |
| Elicitation ablations | `data/ablations/elicitation_v2/` |

---

## Remaining Work

### Critical (before paper submission)
- [ ] Update all paper tables/figures with v3 numbers (currently show v2 0.953 AUROC)
- [ ] Multi-seed evaluation for v3 (job 8741 submitted, 5 seeds: 42, 123, 456, 789, 314)
- [ ] Re-run Qwen3.5 model size ablation with per-size HP tuning (current results are flawed)
- [ ] Regenerate paper figures with v3 data

### Paper Completion
- [ ] Fill in any remaining `\placeholder{}` or `\todo{}` items
- [ ] Verify all figure paths compile correctly in Overleaf
- [ ] Add missing references to `refs_corrected_com.bib`
- [ ] Write acknowledgements (camera-ready)
- [ ] Verify page count within ICML limits

### Optional Improvements (if time permits)
- [ ] True semantic entropy baseline on Qwen3.5-397B (legitimate head-to-head comparison)
- [ ] Claude/Gemini API evaluations (needs advisor approval for budget)
- [ ] DPO training with calibrator scores for UC-A
- [ ] Additional model families for cross-model matrix
- [ ] Investigate easy question calibration (potential new issue)

### Data Management
- [ ] Release model weights to HuggingFace
- [ ] Package training data for release
- [ ] Clean and release evaluation code

---

## CONTAMINATED DATA WARNING

### Question-Level Leakage (fixed in v3)
The v2 train/test split used sample index instead of question ID, causing 82--96% question overlap in test. This inflated v2 AUROC from ~0.878 to 0.953. **v3 fixes this with a clean question-level split (0% overlap).**

### Directory-Level Contamination (v1/v2)
The following directories contain training data mixed with test data. **NEVER use for paper metrics:**

| Directory | Status | Use Instead |
|-----------|--------|-------------|
| `data/use_cases/CONTAMINATED_scored_unified/` | QUARANTINED | `data/use_cases/scored_test_only_v3/` |
| `data/use_cases/CONTAMINATED_scored_v2/` | QUARANTINED | `data/use_cases/scored_test_only_v3/` |
| `data/use_cases/CONTAMINATED_results_unified/` | QUARANTINED | `data/use_cases/results_test_only_v3/` |
| `data/use_cases/scored_test_only_v2/` | SUSPECT (question-level leakage) | `data/use_cases/scored_test_only_v3/` |
| `data/use_cases/results_test_only_v2/` | SUSPECT (question-level leakage) | `data/use_cases/results_test_only_v3/` |

---

*Last updated: March 11, 2026*
