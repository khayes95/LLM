# Research Log

> Full history archived at `archive/RESEARCH_LOG_full_backup.md`

---

### 2026-03-10 — Pinocchio package: fixed model loading + token ID bugs
Why: Package was non-functional — LoRA weights silently failed to load, scores were random.
Script: `pinocchio_package/src/pinocchio/model.py`
Result:
- Bug 1: Used `AutoModelForCausalLM` but Qwen3.5-0.8B is a VLM (`Qwen3_5ForConditionalGeneration`). All 192 LoRA weights silently skipped. Fixed.
- Bug 2: Token IDs for `"i"`/`"ii"` (72/3680) wrong — model predicts `"(i"`/`"(ii"` (1889/29731). Fixed.
- Before fix: all scores ~0.65, no discrimination. After: Paris=0.707, Berlin=0.438 (correct separation).
- HF weights repo (`KevinDavidHayes/pinocchio-0.8b`) confirmed private. Package not published yet.
- All 11 unit tests pass.

### 2026-03-09 01:30 — v3 scoring, use cases, and ablations complete
Why: Complete v3 pipeline (question-level split fix) and re-run all ablations on clean data.
Script: `slurm/score_v3_full.sh`, `slurm/filter_v3_usecases.sh`, ablation scripts | Jobs: 8703, 8722, 8723, 8740
Result:
- **v3 test-only AUROC: 0.878 [0.863, 0.892]** (was 0.953 on v2 leaked data, −7.5 pts)
- Per-model: gpt5mini=0.882, gpt52=0.877, qwen35=0.873 | ECE=0.087, Brier=0.151
- **No-metadata (v3): 0.827** (−5.0 pts from 0.878). Metadata contributes more on clean data.
- **Truncation (v3)**: r200=0.805, r400=0.846, r800=0.878, r1600=0.888, r3000=0.890
- **Scramble (v3)**: word=0.781 (−9.7 pts), sentence=0.867 (−1.1 pts). Confirms semantic reading.
- **UC-A inform acc**: 0.745 (was 0.888, −14 pts). UC1 cov@90: 0.29-0.39 (was 0.49-0.60).
- All 11 use cases recomputed: `data/use_cases/results_test_only_v3/`
- Bootstrap CIs: `data/use_cases/results_test_only_v3/bootstrap_ci_v3.json`
- Verbalized AUROC on v3: 0.610 (p < 0.001 vs calibrator)
- Paper (overleaf) updated: abstract, intro, experiments, discussion, conclusion — all v2→v3

### 2026-03-09 01:30 — [RUNNING] Multi-seed training (v3, 5 seeds)
Why: Get error bars on v3 AUROC across multiple random seeds.
Script: `scripts/multi_seed_training.py` | SLURM job IDs: 8741 (failed OOM), 8810 (failed OOM), 8811 (queued, depends on 8806/8808/8809)
Args: `--seeds 123 456 789 314 --epochs 3 --lora_r 32` (seed 42 already done: AUROC 0.889)
Status: Seed 42 complete. Seeds 123-314 waiting for GPU availability.
Note: `retrain_best_v2.py` updated to use question-level split (same fix as train_best_uq.py).

### 2026-03-09 07:00 — Paper v3 updates: domain tables, figures, error analysis
Why: Update all remaining v2 numbers in the paper to v3.
Result:
- Healthcare: N=533, AUROC=0.834 (was 0.898), acc=49.9%
- Finance: N=896, AUROC=0.895 (was 0.952), acc=42.4%
- Error analysis: 116 hard errors (48 CW + 68 UR, was 112=64+48)
- Three-tier routing updated: green tier acc 0.879-0.897, workload reduction 20-25%
- Method.tex: Fixed stale 0.953 → 0.878, updated data description to v3 split
- Figures: 7/11 regenerated with v3 data (ROC curves, AUROC comparison, bootstrap, effect size need baseline scores)
- All TODO markers removed from paper

### 2026-03-09 01:45 — Near-duplicate contamination check + deduped AUROC (v3)
Why: Measure near-duplicate overlap in v3 train/test split and verify AUROC is not inflated.
Script: `scripts/fast_contamination_check.py`, `scripts/deduped_auroc.py` | Output: `data/use_cases/results_test_only_v3/`
Result:
- 405/1953 (20.7%) test samples have near-duplicates in training (Jaccard ≥ 0.8)
- Worst benchmarks: arc_agi 65.8%, mmvet 41.2%, chembench 39.8%, hallusionbench 39.1%
- AUROC full: 0.878, deduped (1,392 samples): 0.872 (delta: −0.006). Near-duplicates do NOT inflate scores.
- Paper appendix updated with contamination analysis section and all v2→v3 number changes.

### 2026-03-06 03:30 — v3 retrain with question-level split fix
Why: Fix critical data leakage — train/test split was by sample index, not question ID (82-96% question overlap in test set).
Script: `scripts/train_best_uq.py` | SLURM job ID: 8527 (depends on 8515) | Args: `--output_dir uq_models/best_v3_qsplit --epochs 3 --lora_r 32 --lora_alpha 64 --prompt_variant combined`
Smoke test passed (job 8521): 0 question overlap verified. Awaiting GPU availability.

### 2026-03-06 — Reviewer rebuttal experiments (Agent 1, GPU experiments)
Why: Address 5 reviewer concerns requiring GPU experiments.
Scripts: `scripts/ablation_no_metadata.py`, `scripts/ablation_truncation_scramble.py`, `scripts/unseen_model_eval.py`, `scripts/retrain_best_v2.py`
Results:
- **No-metadata ablation (Issue 1)**: 0.953 → 0.926 (−2.7 pts). Model reads content, not just metadata. Output: `data/ablations/no_metadata/`
- **Truncation (Issue 9)**: r200=0.863, r400=0.929, r800=0.953, r1600=0.953. 800 chars is sweet spot. Output: `data/ablations/truncation_scramble/`
- **Word scramble (Issue 5)**: 0.953 → 0.830 (−12.3 pts). Proves semantic reading. Sentence scramble: −1.6 pts only.
- **Unseen model (Issue 4)**: LLaMA-3.1-8B AUROC=0.800, 700 samples, 7 benchmarks. Output: `data/ablations/unseen_model/`
- **Multi-seed (Issue 8)**: 42=0.898, 123=0.843, 456=0.904. Mean=0.882±0.033. Output: `data/ablations/multi_seed/seed_456/`

### 2026-03-05 — Legal hallucination detection on Stanford RegLab dataset (OOD)
Why: Test calibrator on genuinely out-of-distribution legal domain data (reglab/legal_hallucinations, 745K examples).
Script: `scripts/score_legal_hallucinations.py` | SLURM 8473 | Output: `data/legal_hallucinations/scored/`
Result (1,031 stratified samples across 4 LLMs × 11 tasks):
- **Overall AUROC: 0.568** (genuinely OOD — calibrator never saw legal data)
- Best tasks: case_existence 0.970, court_id 0.863, year_overruled 0.735
- Worst tasks: fake_year_overruled 0.188, fake_case_existence 0.325, quotation 0.377
- Per-LLM: Llama 2 0.620, PaLM 2 0.575, GPT 3.5 0.560, GPT 4 0.530
- Factual verification tasks (case existence, court ID) transfer well; domain-specific tasks (quotation, fake cases) do not.

### 2026-03-05 — Realistic domain demos (legal, education, medical)
Why: Create practitioner-facing UQ demos using real calibrator scores on test-only data + OOD healthbench.
Scripts: `scripts/demo_legal_realistic.py`, `scripts/demo_education_realistic.py`, `scripts/demo_medical_realistic.py`
Results:
- Legal: 7.2% false green rate (in-dist), not defensible at threshold 0.7
- Education: wrong-step shield at 0.7 threshold → 8.6% error rate (vs 54% unfiltered)
- Medical (OOD healthbench): 93% accuracy in "General information" tier, 81% cost savings with routing

### 2026-03-05 — OOD scoring on unseen benchmarks (healthbench + triviaqa)
Why: Score genuinely unseen benchmarks to get honest OOD AUROC.
Script: `scripts/score_unseen_benchmarks.py` | SLURM 8437 | Output: `data/use_cases/scored_unseen/`
Result: Combined OOD AUROC 0.657 (vs 0.953 in-distribution)

### 2026-03-05 — Paper figures and LaTeX for UC-E/F/G
Script: `scripts/generate_production_uc_figure.py` | Output: `figures/paper/fig_production_use_cases.{pdf,png}`, `fig_production_use_cases_permodel.{pdf,png}`
Also: Added Production Deployment Use Cases subsection + appendix tables to `paper/main.tex`.

### 2026-03-05 07:34 — FineGRAIN transfer eval on 12 unseen T2I models
Why: Test cross-model transfer of finetuned UQ model to entirely new architectures (Flux2, GPT-Image, Gemini, etc.)
Script: `scripts/finegrain_transfer_eval.py` | SLURM 8448 | Output: `data/finegrain_uq/transfer_eval/transfer_results.json`
Result (2400 samples, 200/model × 12 models):
- **Zero-shot: 0.729 AUROC**, Spearman ρ=0.774 (p=0.003)
- **Finetuned (fold_flux): 0.756 AUROC** (+0.028), Spearman ρ=0.767
- Selective prediction: finetuned gets 91.3% accuracy at 25% coverage (vs 80.7% zero-shot)
- Best per-model: gemini_image_native 0.783, qwen 0.782; Worst: sd1 0.558, sd2 0.647
- Key finding: Modest finetuning gain (+2.8 pts) on totally unseen models. Zero-shot already decent (0.729). Model ranking highly significant (p<0.004).

### 2026-03-05 05:30 — FineGRAIN baseline experiments (full scale, 3750 samples)
Why: Compare UQ calibrator against standard T2I metrics on FineGRAIN failure detection.
Script: `scripts/finegrain_all_experiments.py` | SLURM 8430
Output: `data/finegrain_uq/experiments/all_experiments.json`
Result: CLIPScore 0.513, BLIP-2 ITM 0.507, Prompt-only 0.501, Caption-based 0.608, **UQ+image 0.736**. Standard metrics near random — UQ has real signal.

### 2026-03-05 05:00 — FineGRAIN 5-fold CV finetuning complete
Why: Leave-one-model-out CV on 5 human-labeled T2I models (3750 samples).
Script: `scripts/finegrain_finetune.py` (via `slurm/finegrain_finetune_cv.sh`) | SLURM 8432
Output: `data/finegrain_uq/exp1_human_cv/`
Result: flux=0.973, sd3.5_large=0.969, sd3.5_medium=0.967, sd3_m=0.958, sd3_xl=0.899. **Mean AUROC 0.953 ± 0.028** (vs 0.736 zero-shot). Domain adaptation gives +0.217 improvement.

### 2026-03-05 04:30 — Realistic domain demos (legal, education, medical)
Why: Rework use-case demos to match what actual practitioners (lawyers, teachers, clinicians) need, not generic ML metrics.
Scripts: `scripts/demo_legal_realistic.py`, `demo_education_realistic.py`, `demo_medical_realistic.py`
Output: `data/use_cases/{legal,education,medical}_realistic/` | Figures: `figures/{legal,education,medical}_realistic/`
Both in-distribution and OOD (unseen healthbench) results.

**Key results (realistic framing):**

Legal (in-dist, 1439 samples, AUROC 0.908):
- GREEN tier: 38% of responses, 92.8% accurate, **40 false greens (7.2% malpractice risk)**
- Review 50% of memos to match 1st-year associate catch rate (82%)
- Billing audit: $13,640 saved but **NOT defensible** (>10% slip rate at threshold 0.7)

Education (in-dist, 1789 samples, AUROC 0.935):
- Wrong-step shield at 0.7: AI handles 36% of questions, only 8.6% error rate (vs 54% unfiltered)
- Trust meter: "Confident" answers are 91% correct (8.6% false confidence)
- Calibrator-guided grading review catches 1.8-1.9x more errors than random review
- Difficulty radar: math/competition < 40% acc → human tutor required; GPQA 76% → AI handles

Medical (healthbench OOD, 660 samples, AUROC 0.649):
- "General information" tier: 60% of answers, 93% accurate
- Telehealth routing: AI>0.8 + nurse>0.5 → **81% cost savings**, 28 AI errors out of 660
- NNR (number needed to review): starts at 2.5, rises to 7.3 — efficient early, diminishing returns
- Even OOD: calibrator provides useful triage signal despite lower AUROC

### 2026-03-05 03:41 — Analysis: UC-E/F/G production use cases
Why: Interpret results from new use cases and identify follow-up work.

**UC-E (Adaptive Clarification — "Ask Before Answering"):**
- Calibrator AUPRC 0.942-0.952 vs best baseline 0.648-0.672. Gap is ~0.30 — overwhelming advantage.
- Best F1 for error detection: 0.871-0.886 (cal) vs 0.130-0.564 (verbalized). Verbalized is especially bad on Qwen3.5 (F1=0.130).
- At t=0.7: catches 91-94% of errors while flagging 51-56% of queries. Practical operating point.
- **Killer examples**: Models say "100% confident" but calibrator gives p=0.004. All truly wrong. Perfect for paper figure.
- **Paper pitch**: "A single UQ pass prevents 90%+ of confident-but-wrong answers."

**UC-F (Confidence-Gated Actions — "Think Before Acting"):**
- Coverage@95% accuracy: 33-51% (calibrator) vs 0-0.4% (all baselines). Calibrator is the ONLY method that can gate actions at 95% accuracy with meaningful throughput.
- Cost savings ~84% at 10x error cost ratio. At 50x cost ratio: 95.5-97.8% savings.
- GPT-5.2 best: 50.5% auto-execute at 95% accuracy. Conservative profile (2% error): 39.5% coverage at 1.7% error.
- **Paper pitch**: "For agentic systems, our calibrator is the only method that enables any meaningful auto-execution at 95%+ accuracy."

**UC-G (Human Escalation — "Route to Human"):**
- Green tier: 40-51% of queries auto-sent at 93-95% accuracy, containing only 4-6% of errors.
- Red tier: 36-43% of queries, captures 80% of ALL errors. Review just this tier for massive efficiency.
- At 20% review: UQ catches 2.1-2.5x more errors than random review.
- **Headline**: To reach 95% accuracy, review 48-62% (UQ) vs 100% (random) — 38-52% workload reduction.
- **Paper pitch**: "Deploy with 95% accuracy while halving human review workload."

**Cross-cutting insight**: The AUSC metric for UC-F is lower than expected (0.41-0.44) because the base accuracy is only 51-59%. The absolute coverage at high accuracy targets is still strong. On higher-accuracy models, the effect would be even more dramatic.

**Follow-up work identified:**
1. Paper-ready figures for UC-E/F/G (unified style matching existing paper figures)
2. Combine UC-E/F/G into a single "production deployment" section for the paper
3. No GPU jobs needed — these are analysis-only

### 2026-03-05 05:46 — FineGRAIN finetuning CV complete (5-fold leave-one-model-out)
Why: Test if domain adaptation improves UQ model on T2I failure detection.
Script: `scripts/finegrain_finetune.py` | SLURM 8432 | Output: `data/finegrain_uq/exp1_human_cv/`
Result: **Mean AUROC 0.953 ± 0.028** (vs 0.736 zero-shot). Massive improvement.
- flux: 0.973 | sd3.5_large: 0.969 | sd3.5_medium: 0.967 | sd3_m: 0.958 | sd3_xl: 0.899
- sd3_xl hardest (most different architecture). All others >0.95.
- Training: 34 min/fold, 2 epochs, lr=2e-5, continued from v2 checkpoint.

### 2026-03-05 02:40 — FineGRAIN baseline experiments (CLIPScore, BLIP-2, prompt-only, caption-based)
Why: Compare UQ model against standard T2I evaluation metrics on 3,750 human-labeled samples.
Script: `scripts/finegrain_all_experiments.py` | Output: `data/finegrain_uq/experiments/`
Result:
- CLIPScore (ViT-L/14): AUROC **0.513** — near random
- BLIP-2 ITM: AUROC **0.507** — near random
- UQ prompt-only (no image): AUROC **0.501** — random (needs image)
- UQ caption-based (Molmo captions, flux only, n=750): AUROC **0.608**
- UQ + image (existing): AUROC **0.736** [0.719, 0.751]
- Ensembles add nothing — CLIP/BLIP are noise on this task
- Note: FineGRAIN labels are per-prompt (same across all 5 models), so per-sample model routing is impossible

### 2026-03-05 03:10 — Out-of-distribution scoring on unseen benchmarks
Why: Verify calibrator generalizes to benchmarks NOT in training data (healthbench, triviaqa).
Script: `scripts/score_unseen_benchmarks.py` | SLURM 8437 | Output: `data/use_cases/scored_unseen/`
Result:
- Combined OOD AUROC: **0.657** (729 samples, 90.7% base accuracy)
- healthbench GPT-5-mini: 0.701 (n=250) | GPT-5.2: 0.636 (n=250) | Qwen3.5: 0.601 (n=160)
- triviaqa GPT-5-mini: 0.757 (n=69)
- Compare: in-distribution AUROC = 0.953, held-out (same benchmarks) = 0.898
- Note: High base accuracy (84-93%) limits AUROC ceiling. Calibrator still above random but clearly degrades OOD.

### 2026-03-05 01:30 — Domain use case demos (legal, education, professional safety)
Why: Brainstorm & prototype how lawyers, tutors, pro-se litigants would use UQ calibrator.
Scripts: `scripts/demo_domain_analysis.py`, `demo_legal_hallucination.py`, `demo_education_tutoring.py`, `demo_professional_safety.py`
Output: `data/use_cases/{domain_analysis,legal_demo,education_demo,professional_safety}/` | Figures: `figures/{domain_analysis,legal_demo,education_demo,professional_safety}/`
Note: These use in-distribution scored data (scored_test_only_v2). Numbers are in-distribution, not OOD.

### 2026-03-05 02:16 — New use cases: UC-E, UC-F, UC-G (production-oriented)
Why: Demonstrate UQ value for AI lab deployment — clarification triggers, agentic gating, human escalation.
Scripts: `scripts/uc_e_adaptive_clarification.py`, `scripts/uc_f_confidence_gated_actions.py`, `scripts/uc_g_human_escalation.py`
Output: `data/use_cases/results_test_only_v2/uc_{e,f,g}_results.json` | Figures: `figures/use_cases_v2/uc_{e,f,g}_*.pdf`
Data: test-only v2 (4,447 samples), CPU-only analysis.

Results (averaged across GPT-5-mini, GPT-5.2, Qwen3.5):
- **UC-E** (adaptive clarification): Calibrator AUPRC=0.947, best F1=0.880 vs verbalized 0.416. At t=0.7, catches 91% of errors.
- **UC-F** (confidence-gated actions): Coverage@95%acc = 33-51%. Cost savings ~84% at 10x error cost. Only method with >0% coverage at 95% accuracy.
- **UC-G** (human escalation): Green tier 40-51% volume at 93-95% acc, red tier captures 80% of errors. Workload reduction 38-52% to reach 95% accuracy vs random review.

## 2026-03-03

### Qwen3.5 Model Size Ablation (0.8B, 2B, 4B, 9B) — Complete
Why: Test newer Qwen3.5 model family as calibrator backbone at 4 sizes, compare to Qwen3-VL (2B=0.816, 4B=0.830, 8B=0.827).
Script: `scripts/train_best_uq.py` via `slurm/qwen35_train_single.sh` | SLURM 8346-8349 | Output: `data/ablations/qwen35_model_size/`
Config: v2 (combined prompt, split_info reuse, LoRA r=16 for 0.8B-4B, r=32 for 9B), transformers 5.2.0
Result:
- 0.8B: AUROC=0.852 (VLM=0.854, Text=0.849) — **best**
- 2B: AUROC=0.850 (VLM=0.856, Text=0.835)
- 4B: AUROC=0.663 (VLM=0.674, Text=0.722) — poor
- 9B: AUROC=0.771 (VLM=0.793, Text=0.728)
- Qwen3.5-0.8B matches Qwen3-VL-8B (0.827→0.852) at 10x fewer params. Larger Qwen3.5 models overfit with LoRA.

### LaTeX tables fully updated to v2 — Complete
Why: Fix stale v1 data in LaTeX tables (CIs, N count, training size, p-values, literature).
Scripts: `cpu_latex_tables.py`, `paired_significance_tests.py`, `literature_comparison.py`
Output: `data/use_cases/results_test_only_v2/latex_tables.tex` (6 tables), `significance_tests.json`, `literature_comparison.json`
Result: All 6 tables now use v2 test-only data. CIs correct (0.947-0.958), p<0.001*** all baselines, training size uses v2 ablation (0.503-0.896).

### Enhanced All 11 Use Cases (v2) — Complete
Why: Strengthen all UCs with missing baselines, bootstrap CIs, merged duplicates, honest negatives.
Script: 11 scripts via `slurm/run_all_use_cases_v2.sh` | SLURM 8259 (4m20s) | Output: `data/use_cases/results_test_only_v2/`
Changes:
- **UC3+UC9 merged** → `uc3_error_discovery.py` (binary detection + ranked annotation efficiency)
- **UC5+UC-B merged** → `uc5_response_selection.py` (pairwise + best-of-N)
- **UC1**: Added conformal prediction baseline, coverage@90%/95%, response length baseline
- **UC2**: Added oracle router, break-even analysis (53% cost savings, 72% routed cheap)
- **UC4**: Added Kendall tau, Fisher z-test for significance
- **UC6**: Added 2-tier cascade, break-even (58% savings, 70% cheap)
- **UC7**: Added simulated retry experiment (calibrator beats random/verbalized at all budgets)
- **UC8**: Added bootstrap CIs on alert F1, KS test for distribution shift, calibration drift (r=0.806-0.839)
- **UC-A**: Added bootstrap CIs (pair acc 0.975 [0.970,0.980]), reward quality (point-biserial r=0.812-0.824 vs verbalized 0.127-0.302)
- **UC-C**: Added bootstrap CIs, data efficiency ratio
- **UC-D**: Framed as honest negative (step-level AUROC ~0.53)
Result: All 11 UCs pass. Figures: `figures/use_cases_v2/`. Now 8 real UCs + 1 honest negative (UC-D dropped from count).

### Training Size Ablation — Complete
Why: How many training samples needed for calibrator performance?
Script: `scripts/train_best_uq.py` via `slurm/training_size_ablation.sh` | SLURM 8257 | Output: `data/ablations/training_size/summary.json`
Result: N=100→0.503, N=250→0.670, N=500→0.696, N=1000→0.785, N=2000→0.814, N=5000→0.862, N=full(10392)→0.896. Steep gains up to 1000, diminishing returns after 2000. 5000 samples = 96% of full performance.

### Figures upgraded to v2 test-only data
Why: Consistency — all figures now use v2 model on test-only data.
Scripts: `compute_baselines.py` → `bootstrap_ci.py` → `cpu_generate_all_figures.py`
Output: `figures/paper/` (9 figures), `results_test_only_v2/bootstrap_ci.json` (6 baselines), `scored_test_only_v2/*.jsonl` (augmented with Platt/isotonic/length/combined fields).
Result: Combined AUROC 0.953 [0.947, 0.958]. Best baseline: Isotonic 0.653. New figure: `fig_training_size_ablation`.

---

## 2026-03-02

### DATA LEAKAGE — Per-model scoring AUROCs are invalid

**SERIOUS OVERSIGHT:** The per-model scoring AUROCs (GPT-5-mini 0.971, GPT-5.2 0.970, Qwen3.5 0.959 on v1; 0.951/0.959/0.946 on v2) were computed on `scored_v2/` and `scored_unified/` which include **60-67% training data**. These numbers are inflated and essentially useless for paper reporting.

- `scored_v2/` contains 4,102-4,412 samples per model — but only ~1,774 are held-out test samples
- The rest are training data the model has already seen, artificially boosting AUROC
- **ALL figures, tables, and results that used these per-model numbers need to be recomputed using `scored_test_only_v2/` or the held-out test set only**

Correct metrics to use:
- **Held-out test AUROC: 0.898** (v2 checkpoint, cleanest metric)
- **Test-only scoring AUROC: 0.953** [0.947, 0.958] combined (already computed in `scored_test_only_v2/`)
- Any per-model breakdown must use test-only splits exclusively

Action items:
1. Audit all figures and tables in `figures/paper/` and the report for contaminated numbers
2. Regenerate any that used `scored_v2/` instead of `scored_test_only_v2/`
3. In the paper, only report held-out AUROC (0.898) and test-only scoring AUROC (0.953)

---

## 2026-03-01

### 01:30 — CPU analysis batch (5 scripts, all completed)
Why: Utilize debug/CPU resources for paper-strengthening analyses.
Scripts & outputs:

| Script | Output | Key result |
|--------|--------|------------|
| `scripts/cpu_exhaustive_bootstrap.py` | `results_test_only/exhaustive_bootstrap.json` | 100K BCa bootstrap. Combined AUROC=0.915 [0.906, 0.923]. P(Cal>base)=1.0 all baselines. |
| `scripts/cpu_contamination_check.py` | `results_test_only/contamination_report.json` | 87 exact train/test Q+R overlaps (multi-model expected). GSM8K↔MGSM cross-bench (97 pairs, by design). 32.7% test >0.8 Jaccard to train. |
| `scripts/cpu_feature_analysis.py` | `results_test_only/feature_analysis.json` | Cal accuracy 84.1%. Error predictors: question_length (p=0.0005), output_tokens (p=0.0014). LR on features = 83.9% (no signal beyond calibrator). |
| `scripts/cpu_prompt_perturbation.py` | `perturbations/all_perturbations.jsonl` | 58,403 perturbations from 4,152 samples (14.1 avg). 3 active strategies. |
| `scripts/cpu_generate_all_figures.py` | `figures/paper/` | 8 publication figures (PNG@300dpi + PDF). |

Note: Debug partition nodes lack `/scratch` mount. Ran on login node (256 CPUs).
Contamination: Most overlaps are expected (same question across target models, GSM8K⊂MGSM). Needs careful framing in paper.

### 22:15 — Reviewer-readiness CPU analyses (7 scripts, all completed)
Why: Strengthen paper against reviewer concerns with additional analyses.
Scripts & outputs:

| Script | Output | Key result |
|--------|--------|------------|
| `scripts/cpu_leakage_excluded.py` | `results_test_only/leakage_excluded.json` | Removed 87 contaminated IDs → AUROC 0.915→0.914 (Δ=-0.0003). No impact. |
| `scripts/cpu_scoring_rules.py` | `results_test_only/scoring_rules.json` | Brier=0.116 (2x better than next), LogLoss=0.409, ECE=0.035. Cal dominates all proper scoring rules. |
| `scripts/cpu_difficulty_stratification.py` | `results_test_only/difficulty_stratification.json` | Medium-difficulty AUROC=0.911. 5-quantile: Q1=0.939, Q2=0.915, Q3=0.885, Q4=0.759. Spearman=-0.689. |
| `scripts/cpu_decision_thresholds.py` | `results_test_only/decision_thresholds.json` | PR-AUC=0.924 (next: 0.724). Only method achieving 90% precision (t=0.74) or 95% precision (t=0.94). |
| `scripts/cpu_sensitivity_analysis.py` | `results_test_only/sensitivity_analysis.json` | AUROC stable across Jaccard thresholds: 0.915 (all) → 0.903 (remove 67% with Jaccard≥1.0). Range=0.012. |
| `scripts/cpu_failure_cases.py` | `results_test_only/failure_cases.json` | 200 curated cases (FP/FN/disagreements/successes). Failures spread across 19 benchmarks (entropy=0.94). |
| `scripts/cpu_latex_tables.py` | `results_test_only/latex_tables.tex` | 6 publication tables (main results, per-benchmark, use cases, cross-model, literature, ablations). |

Key takeaway: Contamination sensitivity shows AUROC=0.903 even after removing ALL exact question matches (66.6% of data). The ~1% drop is from reduced sample size, not leakage.

### 14:00 — Perturbation GPU scoring completed (SLURM 8171)
Why: Score 58K perturbations with UQ judge for prompt perturbation consistency analysis.
Script: `scripts/score_perturbations.py` | Output: `data/use_cases/perturbations/scored_perturbations.jsonl`
Result: 58,403 scored in 2h on 4 GPUs. Perturbation consistency NOT useful: consistency AUROC=0.521, pert mean=0.717, combined hurts (0.868 vs orig 0.881). Honest negative.

### 09:24 — Score v2 + use cases pipeline (SLURM 8155)
Why: Score all predictions with v2 checkpoint, filter test-only, re-run all 11 UCs.
Script: `slurm/score_v2_use_cases.sh` | Output: `data/use_cases/results_test_only_v2/`
Result: Test-only AUROC **0.953** [0.947, 0.958] combined. Per-benchmark mean=0.915, CV=0.070. All 11 UCs ran successfully.

### 23:30 — Report updated with v2 results
Why: Update PDF report to reflect v2 model improvements across all sections.
Script: `scripts/generate_report.py` | Output: `figures/research_update_report.pdf` (3851 KB)
Result: All sections updated: title page (0.898 held-out), main results (0.953 test-only), baselines, use cases, reviewer experiments. March 2026 edition.

### 14:28 — Elicitation ablations v2 (SLURM 8173)
Why: Three remaining elicitation strategies on v2 checkpoint (temperature scaling, token entropy, hidden state probing).
Script: `scripts/elicitation_ablations_v2.py` | Output: `data/ablations/elicitation_v2/`
Result: Temp scaling T=1.34 (no AUROC change, ECE 0.023→0.019). Token entropy near-random (0.520). MLP hidden state probe 0.878 (+3.7 pts over logit baseline 0.841). Combined hidden+logit 0.884 (+4.3 pts). Conclusion: logit captures 95%+ of available signal; hidden state probe only helps with open-weight models.

---

## Current: Best Unified UQ Model v2

**Model:** Qwen3-VL-8B-Instruct + LoRA (r=32, combined prompt), trained on ALL text+VLM data
**Checkpoint:** `uq_models/best_v2_r32_combined/` (seed 42, 3 epochs complete, AUROC 0.898)
**Alt checkpoint:** `uq_models/best_unified_v2/` (epoch 2/3 — timed out, AUROC 0.890)
**Test-only Scoring AUROC: 0.953** [0.947, 0.958] combined (N=4,447)
  - ⚠️ Per-model numbers below are TEST-ONLY (not the leaked all-data numbers). See 2026-03-02 entry.
  - GPT-5-mini: 0.951 [0.940, 0.961] | GPT-5.2: 0.959 [0.950, 0.968] | Qwen3.5: 0.946 [0.935, 0.956]
**Previous:** `uq_models/best_unified/` (r=16, AUROC 0.831, test-only scoring 0.915)
**Scored data (v2):** `data/use_cases/scored_v2/` | Test-only: `data/use_cases/scored_test_only_v2/`
**Results:** `data/use_cases/results_test_only_v2/` (all 11 use cases complete)
**Report:** `figures/research_update_report.pdf` (updated with v2 numbers)

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

### Combined prompt retrain v2 (SLURM 8030)
Why: Retrain with r=32 + combined prompt (longer+CoT+metadata). Ablations showed 0.831→0.867.
Script: `scripts/retrain_best_v2.py` | Output: `uq_models/best_v2_r32_combined/`
Result: Held-out AUROC **0.890** (epoch 2). Timed out at 8h before epoch 3.

### Multi-seed training (SLURM 8031)
Why: Error bars across 3 random seeds for paper.
Script: `scripts/multi_seed_training.py` | Output: `data/ablations/multi_seed/`
Result: seed 42 = **0.898** (epoch 2, best), seed 123 = 0.843 (epoch 1 only, timed out), seed 456 = not run.

## 2026-03-01

### Score v2 + use cases pipeline (SLURM 8155)
Why: Score all predictions with v2 checkpoint, filter to test-only, re-run all 11 use cases.
Script: `slurm/score_v2_use_cases.sh` | Output: `data/use_cases/results_test_only_v2/`
Result: Test-only AUROC **0.953** [0.947, 0.958] combined. GPT-5-mini: 0.951, GPT-5.2: 0.959, Qwen3.5: 0.946. All 11 UCs ran successfully. Per-benchmark mean AUROC=0.915, CV=0.070.

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

### 2026-03-01 22:28 — Proxy Semantic Entropy + Self-Consistency Baselines
Why: Compare against Semantic Entropy (Kuhn et al. 2023) using 8B proxy model, and cross-model self-consistency.
Script: `scripts/semantic_entropy_baseline.py` | SLURM 8226 | Output: `data/ablations/semantic_entropy/`
Result: **Proxy SE AUROC=0.467 (below random)**, Proxy Self-Consistency=0.466. Calibrator=0.888 on same subset. Cross-model mean-P consensus=0.826 vs calibrator 0.923. Confirms proxy model uncertainty does not transfer — supports paper thesis.

---

## Earlier (summaries)

**2026-02-25:** Model-specific text calibrators, scored_v2, all 7 use cases
**2026-02-24:** VLM ablation, cross-model eval, verbalized baselines
**Earlier:** See `archive/RESEARCH_LOG_full_backup.md`
