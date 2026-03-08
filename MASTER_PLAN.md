# MASTER PLAN: Pinocchio — Uncertainty Quantification for Black-Box LLMs
## Target: ICML 2026 Submission

**Created:** December 30, 2025
**Last major update:** March 5, 2026
**Status:** Phase 5 (Paper Writing) — experiments complete, paper ~90% written

---

## Paper Title
**Pinocchio: Estimating the Uncertainty of Black-Box Language Models**

## Paper Contributions

1. **Cross-Model Transfer:** A single calibrator trained on responses from GPT-5-mini, GPT-5.2, and Qwen3.5-72B transfers across all three target models (0.946--0.959 AUROC). Single-model calibrators achieve only 0.672--0.759.
2. **Unified Text + Vision:** A single Qwen3-VL-8B model handles text-only and vision-language benchmarks in a unified architecture (mean AUROC: text 0.897, VLM 0.942).
3. **Practical Efficiency:** Single forward pass through 8B model (or 0.8B with 95% retention). 5,000 training examples capture 96% of full performance.
4. **Production Workflows:** Three deployment use cases (adaptive clarification, confidence-gated actions, human escalation) where no baseline achieves meaningful performance.

---

## Key Results (v2 checkpoint, test-only evaluation)

| Metric | Value |
|--------|-------|
| **Overall AUROC** | **0.953** [0.947, 0.958] |
| GPT-5-mini AUROC | 0.951 [0.940, 0.961] |
| GPT-5.2 AUROC | 0.959 [0.950, 0.968] |
| Qwen3.5 AUROC | 0.946 [0.935, 0.956] |
| Best baseline (isotonic) | 0.653 |
| Verbalized confidence | 0.607 |
| Per-benchmark mean AUROC | 0.915 (CV=0.070) |
| Leave-K-out CV gap | +0.010 |
| All p-values vs baselines | < 0.001 |

### Ablations
| Ablation | Key Finding |
|----------|-------------|
| Training size | N=5000 → 96% of full; N=2000 → 91%; N=100 → 0.503 |
| Model size (Qwen3.5) | 0.8B: 0.852, 2B: 0.850, 4B: 0.663, 9B: 0.771 |
| Elicitation | Logit=0.859, MC Dropout=0.859, Hidden+logit=0.884, Verbalized=0.749 |
| Temperature scaling | T=1.34, AUROC unchanged, ECE 0.023→0.019 |

### Production Use Cases (v2 test-only)
| Use Case | Key Metric |
|----------|------------|
| Adaptive Clarification | AUPRC 0.947 (vs 0.597 verbalized) |
| Confidence-Gated Actions | 40% auto-execution at 95% accuracy |
| Human Escalation | 43% workload reduction to reach 95% accuracy |

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
- **Best Unified Model v2:** Qwen3-VL-8B, r=32, combined prompt, 3 target models
  - Checkpoint: `uq_models/best_v2_r32_combined/`
  - Held-out AUROC: 0.890 / 0.898 (multi-seed)
  - Test-only scoring: 0.953 [0.947, 0.958]
- **Reviewer experiments:** Bootstrap CIs, significance tests, held-out benchmark eval, literature comparison
- **Baselines:** 8+ baselines (verbalized, Platt, isotonic, length, combined, zero-shot, proxy SE, self-consistency)
- **Data leakage fix:** Identified and fixed train/test contamination; test-only splits (4,447 samples)
- **Qwen3.5 model size ablation:** 0.8B, 2B, 4B, 9B
- **Elicitation ablations:** MC Dropout, hidden state probe, verbalized, temperature scaling, token entropy

### Phase 5: Paper Writing — IN PROGRESS
- Paper ~90% written in Overleaf (ICML 2026 format)
- Overleaf project: `https://git.overleaf.com/696e96cbc0f12ed169b15d1e`
- Local clone: `/scratch/khayes/LLM/overleaf/`
- All figures generated in `figures/paper/`
- Per-section `.tex` files in `overleaf/sections/`

---

## Key Paths

| Resource | Path |
|----------|------|
| **Best checkpoint (v2)** | `uq_models/best_v2_r32_combined/` |
| V1 checkpoint | `uq_models/best_unified/` |
| Test-only scored data | `data/use_cases/scored_test_only_v2/` (4,447 samples) |
| Test-only results | `data/use_cases/results_test_only_v2/` |
| Paper figures | `figures/paper/` |
| Overleaf repo | `overleaf/` |
| Training data split | `uq_models/best_v2_r32_combined/split_info.json` |
| Model size ablation | `data/ablations/qwen35_model_size/` |
| Training size ablation | `data/ablations/training_size/` |
| Elicitation ablations | `data/ablations/elicitation_v2/` |

---

## Remaining Work

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

### Data Management
- [ ] Release model weights to HuggingFace
- [ ] Package training data for release
- [ ] Clean and release evaluation code

---

## CONTAMINATED DATA WARNING

The following directories contain training data mixed with test data (60-67% overlap). **NEVER use for paper metrics:**

| Directory | Use Instead |
|-----------|-------------|
| `data/use_cases/CONTAMINATED_scored_unified/` | `data/use_cases/scored_test_only/` |
| `data/use_cases/CONTAMINATED_scored_v2/` | `data/use_cases/scored_test_only_v2/` |
| `data/use_cases/CONTAMINATED_results_unified/` | `data/use_cases/results_test_only_v2/` |

---

*Last updated: March 5, 2026*
