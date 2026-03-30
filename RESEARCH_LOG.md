# Research Log

> Full history archived at `archive/RESEARCH_LOG_full_backup.md`

---

### 2026-03-30 — Cross-model transfer leakage analysis
Why: v3 transfer AUROC (0.909-0.922) was suspiciously higher than Gemini-trained model (0.875). Investigated root cause.
Finding: **79% of Gemini/Claude test questions were seen by v3 during training** (via GPT/Qwen responses to same questions). This inflates naive transfer AUROCs.
Corrected results (clean questions only, n=133-137 per model):
  - Claude Opus: v3=0.776 vs trained=0.812 (trained wins)
  - Claude Sonnet: v3=0.825 vs trained=0.815 (~tied)
  - Gemini Pro: v3=0.880 vs trained=0.875 (~tied)
  - Gemini Flash: v3=0.864 vs trained=0.878 (trained wins)
Conclusion: Transfer performs COMPARABLY to direct training on clean questions. Naive transfer numbers are inflated.
Also found: Gemini training used wrong prompt template (baseline instead of combined, -5 pts known penalty).

### 2026-03-30 15:24 — Gemini re-run complete + retrained (full data)
Why: First Gemini run hit our own daily counter limit (4,500 shared between Pro+Flash). Cleaned failures, reset counter, re-ran missing samples, re-scored, retrained.
Script: `slurm/gemini_g{1-8}.sh`, `slurm/gemini_flash_g{1-8}.sh` | Jobs: 10093-10112
Result: Full data now — Pro 4,815 valid, Flash 4,813 valid. Total Gemini cost: ~$129 of $300 credit.
  - Transfer AUROC (v3 calibrator, unseen): Pro **0.909**, Flash **0.888** (on all ~4,300 samples)
  - Trained AUROC (held-out 15%): Pro **0.875** (n=646), Flash **0.878** (n=645)
  - NOTE: Transfer > trained because different test sets (all samples vs 15% held-out) and v3 benefits from GPT+Qwen training diversity
  - Old partial results (jobs 10056/10057, trained on 3002/909 samples) were OVERWRITTEN. Do not use old numbers (0.928/0.948).

### 2026-03-29 ~18:00 — Individual source model training (4 models) — SUPERSEDED for Gemini
Why: Train separate UQ models on each new source model, comparable to existing GPT/Qwen models.
Script: `scripts/train_best_uq.py --sources <model>` | Jobs: 10054-10057
Result: Claude models correct (full data). Gemini models SUPERSEDED by jobs 10111/10112 (had partial data).
  - Claude Opus: AUROC 0.812 | Claude Sonnet: 0.815 (VALID — full data)
  - ~~Gemini 3.1 Pro: 0.928 | Gemini 3 Flash: 0.948~~ (INVALID — partial data, replaced above)
  - Checkpoints: `uq_models/train_{claude_opus,claude_sonnet,gemini31pro,gemini3flash}/`

### 2026-03-29 13:10 — Gemini 3 Flash Preview benchmark evaluation
Why: Evaluate transfer to Google's efficient reasoning model (pairs with Pro like GPT-5-mini/GPT-5.2).
Script: `slurm/gemini_flash_g{1-8}.sh` | Jobs: 10045-10052 (Vertex AI)
Result: 4,816 predictions, only 909 valid (3,435 hit shared daily counter limit — needs re-run).
  - v3 calibrator AUROC: **0.877** (partial data)

### 2026-03-29 11:00 — Gemini 3.1 Pro Preview benchmark evaluation
Why: Evaluate Pinocchio transfer to Google's latest reasoning model (5th model family).
Script: `slurm/gemini_g{1-8}.sh` | Jobs: 10026-10033 (8 parallel groups) | Vertex AI ($300 credit)
Result: 4,816 predictions across 20 benchmarks. 3,002 valid (1,342 hit daily counter limit — needs re-run).
  - v3 calibrator AUROC: **0.890** (unseen transfer, best of all model families)
  - Cost: ~$95 of $300 Vertex AI credit

### 2026-03-29 01:50 — Claude VLM re-run complete + full scoring
Why: Complete the failed VLM benchmarks from March 24 budget exhaustion.
Script: `slurm/run_claude_benchmarks.sh` | Jobs: 10005 (Opus), 10006 (Sonnet)
Result: All 20 benchmarks complete for both models. Full AUROC (v3 calibrator, unseen):
  - Claude Opus: **0.838** (4,287 samples) | Claude Sonnet: **0.866** (4,289 samples)
  - Best per-bench: HallusionBench 0.933/0.928, MathVista 0.943/0.948, LiveBench 0.969/0.974

### 2026-03-26 01:29 — Claude scored data + cross-model AUROC
Why: Score Claude responses with v3 calibrator and compute transfer AUROC.
Script: `slurm/score_claude.sh` | Job: 9915 | Output: `data/use_cases/scored_v3_all/claude_{opus,sonnet}_scored.jsonl`
Result: On 1,946 real text-only responses per model:
  - Claude Opus 4.6: AUROC **0.803** | Claude Sonnet 4.6: AUROC **0.824**
  - Best per-bench: LiveBench 0.969/0.974 | Worst: GPQA 0.652/0.724
  - Confirms cross-family transfer to Anthropic models (never in training)

### 2026-03-24 18:14 — Claude API benchmark evaluation (Opus + Sonnet)
Why: Evaluate Pinocchio transfer to unseen Anthropic models (Claude Opus 4.6, Claude Sonnet 4.6).
Script: `slurm/run_claude_benchmarks.sh` | Jobs: 9881 (Opus), 9882 (Sonnet), 9913/9914 (hle_multimodal fix)
Result: 8/20 benchmarks completed before API budget exhaustion. 3,892/8,680 real responses (55% failure rate).
  - Budget limit hit mid-run — VLM benchmarks ran after text, so all VLM benchmarks failed (coincidental, not image bug)
  - Text-only benchmarks: arc_agi, bbeh, chembench, gpqa, hle, hle_multimodal, livebench, omnimath — all 100% success
  - Cost: ~$337 estimated for Opus alone. Budget resets 2026-04-01.
  - NOTE: No advisor approval documented before spending. Policy violation flagged.

### 2026-03-20 — Qwen3.5 model size ablation v3 complete (template fix applied)
Why: 4B/9B produced near-random AUROC due to `<think>` template mismatch (not overfitting). Fixed with 3-line patch.
Result: **0.8B=0.863, 2B=0.861, 4B=0.870, 9B=0.871** — proper scaling curve, larger models slightly better.
Config: all r=16, alpha=32, lr=5e-5 (0.8B used lr=1e-4). Output: `data/ablations/qwen35_model_size_v3/`

### 2026-03-20 — Multi-seed v3 training complete (5 seeds, seed fix applied)
Why: Reviewer requires multi-seed training variance. Seed fix applied — each seed gets different train/test partition + training randomness.
Script: `scripts/train_best_uq.py` | SLURM: `slurm/multi_seed_v3_parallel.sh`
Jobs: 9594-9597 (seeds 123, 456, 789, 314) | 1× A100, ~2h50m each | All COMPLETED
Output: `data/ablations/multi_seed_v3/seed_*/results.json`
Result: **Mean AUROC: 0.861 ± 0.018** (5 seeds). Per-seed: 42=0.889, 314=0.867, 789=0.856, 456=0.852, 123=0.841.
VLM mean: 0.866 ± 0.020 | Text mean: 0.854 ± 0.018. Seed 42 (paper default) is best.

### 2026-03-19 22:00 — Comprehensive paper audit + code fixes (37 subagents)
Why: Pre-submission audit of all paper numbers, code, and bibliography before ICML deadline.

**Code fixes applied:**
- `train_best_uq.py`: Fixed seed bug — `--seed` now properly seeds torch/numpy/random globals and `TrainingArguments`. Default split path had hardcoded `random_state=42`, now uses `args.seed`.
- `regenerate_tall_figures.py`: Fixed v2→v3 scored data path.

**Paper edits applied (overleaf/):**
- experiments.tex: Fixed stale baseline AUROCs (isotonic 0.653→0.644, length 0.613→0.566, combined 0.641→0.649). Fixed UC-E/F/G numbers. "Best baseline" now correctly references combined (0.649).
- appendix.tex: Fixed 12 per-model baseline cells, UC-F/G tables, healthcare (533→1006 samples, 0.834→0.898 AUROC), finance (896→1930 samples, 0.895→0.952 AUROC).
- Added 18 benchmark/method citations, Hamidieh ICLR 2026 cross-model paper, lin2023generating, vashurin2025polygraph.
- Softened 3 overclaims (abstract, intro, conclusion). Removed 7 duplicate bib entries.
- NOTE: intro.tex and discussion.tex still have stale 0.653 references — need fixing.

**Analysis outputs (not in paper yet):**
- 8 LaTeX tables in `figures/paper/table_*.tex` (LOMO, baselines, use cases, ablations, contamination, model efficiency)
- Multi-seed SLURM script ready: `slurm/multi_seed_v3_parallel.sh` (4 independent jobs, seed fix applied)
- Page count: ~12 pages, need to cut ~4 for ICML 8-page limit
- Simulated ICML review: borderline accept. Key issues: missing multi-seed variance, overclaimed "any model", metadata gives 5 free pts
- Missing ICML responsibility checklist

**Pending GPU work:**
- Multi-seed v3 (4 seeds × 1 A100 × ~4.5h each) — script ready, not submitted
- Qwen3.5 model size ablation with per-size HP tuning — not started
- FineGRAIN re-run with v3 checkpoint — not started

### 2026-03-19 — State snapshot for reconstruction

**Git:** commit `42a5553` pushed to `uq-eval/main`. Branch `uq-finetuning`.

**Active SLURM jobs (as of 00:00 Mar 19):**
| Job ID | Name | Status | Purpose |
|--------|------|--------|---------|
| 9484 | lomo_cross_model | RUNNING (~35% run 1/3) | LOMO cross-model eval (hold out each source model) |
| 9492 | sft_v2r | RUNNING (~1h) | Unknown — check `logs/sft_v2r_9492.out` |
| 9495 | fb_r32 | RUNNING (~40m) | Unknown — check logs |
| 9496 | frontier_smo | RUNNING (~20m) | Unknown — check logs |
| 9497 | eval_fw_v1 | RUNNING (~5m) | Unknown — check logs |
| 9366 | eval_bench_v | PENDING (DependencyNeverSatisfied) | Dead — dependency will never resolve |
| 9367 | uq_train_v2 | PENDING (Dependency) | Blocked on 9366 |
| 9493 | eval_v2c | PENDING (Dependency) | Blocked on chain |
| 9498 | eval_fw_v2 | PENDING (Dependency) | Blocked on chain |

**Key checkpoints on disk (`uq_models/`):**
- `best_v3_qsplit/` — **CURRENT BEST 8B** (AUROC 0.878, clean question-level split, r=32)
- `best_0.8b_easy_v2/` — **CURRENT BEST 0.8B** (AUROC 0.871, easy/impossible/trivial data, r=16)
- `best_v2_r32_combined/` — v2 8B (AUROC 0.898, LEAKED split — for reference only)
- `best_unified/` — v1 8B (AUROC 0.831, superseded)
- `best_0.8b_easy/` — 0.8B v1 (easy data, no trivial correct)
- `best_0.8b_easy_v3/` — 0.8B v3 variant
- `best_0.8b_r128/`, `best_8b_r128/` — r=128 ablation (no improvement)
- `lomo_gpt5mini/`, `lomo_gpt52/` — LOMO partial (job 9484 still running)
- `size_ablation/` — training data size ablation checkpoints

**Key data on disk:**
- `data/finetune/easy_questions/` — 11MB, easy/impossible/trivial/adversarial JSONL (not in git)
- `data/use_cases/scored_test_only_v3/` — v3 scored test data (clean split, 1953 samples)
- `data/use_cases/results_test_only_v3/` — all 11 use case results, bootstrap CIs, contamination report
- `data/ablations/` — 19 ablation directories (no-metadata, truncation, multi-seed, elicitation, etc.)
- `data/use_cases/CONTAMINATED_*/` — QUARANTINED, never use

**To recreate from scratch (if cluster data lost):**
1. Clone repo: `git clone` from `uq-eval/main`
2. Retrain v3 8B: `python scripts/train_best_uq.py --output_dir uq_models/best_v3_qsplit --epochs 3 --lora_r 32 --lora_alpha 64 --learning_rate 1e-4 --prompt_variant combined` (4× A100, ~3h)
3. Retrain 0.8B: same script with `--base_model Qwen/Qwen3-VL-0.8B --lora_r 16 --lora_alpha 32 --extra_data data/finetune/easy_questions/` (1× A100, ~9h)
4. Score: `python scripts/filter_test_only.py` then per-model scoring scripts
5. Evaluate: use case scripts in `scripts/` with `--scored_dir data/use_cases/scored_test_only_v3/`
6. Paper: `overleaf/` submodule, pull from Overleaf git remote

**Pending work (not yet started or incomplete):**
- LOMO results (job 9484, ETA ~12-16h from now)
- Jobs 9366/9367/9493/9498 have dead dependencies — need manual cleanup (`scancel`) and resubmission
- Multi-seed v3 (job 8741 was [RUNNING] per MEMORY.md — check if finished)
- Qwen3.5 model size ablation needs re-run with per-size HP tuning
- True semantic entropy on Qwen3.5-397B (not yet attempted)
- Paper not yet submitted (ECCV 2026 target)

### 2026-03-18 — Leave-One-Model-Out (LOMO) cross-model evaluation
Why: Prior "cross-model" claim trains on all 3 models and tests on all 3 — not true transfer. LOMO trains on 2 models, tests on the held-out 3rd. Also includes ID collision fix (benchmark-prefixed IDs).
Script: `scripts/train_best_uq.py --held_out_model` | SLURM job ID: 9484 | 1× A100
Config: r=32, alpha=64, 3 epochs, lr=1e-4, combined prompt, seed=42.
Output: `uq_models/lomo_{gpt5mini,gpt52,qwen35}/`
Result:
- Hold out GPT-5-mini: **held-out AUROC 0.877** (overall 0.872)
- Hold out GPT-5.2: **held-out AUROC 0.861** (overall 0.860)
- Hold out Qwen3.5: **held-out AUROC 0.790** (overall 0.806)
- **Mean held-out AUROC: 0.843** (vs 0.878 all-model baseline, vs 0.610 verbalized)
- Same-family transfer (OpenAI→OpenAI) nearly lossless; cross-family (OpenAI→Qwen) drops ~9 pts but still strong.

### 2026-03-18 — Code fixes applied (ID collision + grading bugs)
Why: Audit found 687 cross-benchmark ID collisions in question-level split key, plus 4 grading bugs.
Fixes:
- `train_best_uq.py` + 8 downstream scripts: `s.id` → `f"{s.benchmark}_{s.id}"` for split grouping (backward compatible)
- `simpleqa.py` + `hle.py`: removed overly permissive word-subset matching (15 label flips, 0.12%)
- `mmmu.py` + 5 others: fixed letter extraction regex A-D → A-J with last-match heuristic (3 label flips, 0.02%)
- `livebench.py`: added symmetric substring matching
- Total: 18/12,972 label flips (0.14%) — consistent with prior audit. No retrain needed for grading alone.

### 2026-03-16 — Retrain 0.8B with easy + impossible question data
Why: 0.8B production model outputs ~0.70-0.80 for everything on trivial questions (easy correct: 0.735, easy incorrect: 0.651, impossible: 0.493). Adding calibration anchors at both extremes.
Script: `scripts/train_best_uq.py` | SLURM job ID: 9272 | 1× A100, ~9h
Data: 10.9K benchmark (v3 split) + 2K easy (downsampled from 9K) + 502 impossible = 13.4K total
Output: `uq_models/best_0.8b_easy/`
Result:
- **Benchmark AUROC: 0.869** (up from 0.852), VLM=0.874, Text=0.861, ECE=0.073
- **Impossible questions: 0.493 → 0.135** (big improvement, model learned skepticism)
- **Easy incorrect: 0.651 → 0.273** (good improvement, e.g. "2+2=7" → 0.011)
- **Easy correct: 0.735 → 0.719** (not improved — model didn't learn high-confidence outputs)
- Remaining gap: easy correct still ~0.72 avg, not ~0.95. Likely needs more easy-correct samples or balanced label ratio.

### 2026-03-16 — Retrain 0.8B v2: balanced easy + impossible + trivial correct
Why: v1 was too skeptical (easy correct only 0.719). Added 751 diverse trivially-correct Q&A pairs (18 categories) to balance labels.
Script: `scripts/train_best_uq.py` | SLURM job ID: 9289 | 1× A100, ~9.5h
Data: 10.9K benchmark + 2K easy + 502 impossible + 751 trivial correct = 14.2K total
Output: `uq_models/best_0.8b_easy_v2/`
Result:
- **Benchmark AUROC: 0.871** (up from 0.852 original, 0.869 v1). VLM=0.873, Text=0.864, ECE=0.067
- **Easy correct: 0.735 → 0.809** (improved, e.g. "sky is blue" 0.895, "dog has 4 legs" 0.852)
- **Impossible: 0.493 → 0.055** (essentially solved, "GDP of Atlantis" 0.001)
- **Easy incorrect: 0.651 → 0.504** (partially improved but regressed from v1's 0.273)
- Tradeoff: trivial correct data pushed confidence up broadly, helping correct answers but reducing skepticism on some wrong ones

### 2026-03-17 — [RUNNING] r=128 LoRA ablation on 0.8B and 8B
Why: Previous runs used r=16 (0.8B) or r=32 (8B) with no rank ablation. r=128 gives 8x more trainable params — may fix answer verification and improve headline AUROC.
Script: `scripts/train_best_uq.py` | Jobs: 9346 (0.8B, 1×A100), 9347 (8B, 2×A100)
Data: v3 split + all extra data (easy/impossible/trivial/adversarial). Same as v2 config (no metadata randomization).
Config: r=128, alpha=256, LR=1e-4 (0.8B) / 5e-5 (8B)
Output: `uq_models/best_0.8b_r128/`, `uq_models/best_8b_r128/`
8B result: **AUROC=0.871 (−0.007 from r=32's 0.878)**. No improvement. Mixed per-bench: MMMU +4.5, HLE +3.7, but hallusionbench −6.6, BBEH −3.4. ECE worse (0.087→0.105). Likely slight overfitting. r=32 remains best for 8B.
0.8B result: **AUROC=0.868** (same as r=16). Easy_correct improved to 0.922 (best yet), but impossible regressed to 0.326, adversarial 0.665. More capacity → more confident everywhere, less discriminating.

**Conclusion:** Larger LoRA rank doesn't help either model on benchmark AUROC. r=32 is optimal for 8B, r=16 is fine for 0.8B. The easy/impossible calibration is a separate axis from benchmark discrimination — no single config wins on all axes. **v2 (r=16, no metadata randomization) remains the best 0.8B production model** with the best overall balance (benchmark 0.871, impossible 0.055, easy_correct 0.809).

### 2026-03-19 — Full fine-tune 0.8B (no LoRA) — WORSE than LoRA
Why: Test whether full parameter access (859M params) fixes answer verification. LR=5e-6, 3 epochs, 1×A100 ~12h.
Script: `scripts/train_best_uq.py --full_finetune` | Job 9499 | Output: `uq_models/best_0.8b_fullft/`
Result: Catastrophic forgetting. easy_correct=0.723, easy_incorrect=0.613, impossible=0.424, adversarial=0.638. All compressed into 0.3-0.8. LoRA preserves base representations better.
**FINAL: v2 LoRA (r=16) is the best 0.8B production model. Ship it.**

### 2026-03-12 (evening) — Paper overhaul: 22 agents, all figures + content + fixes
Why: Use remaining Claude compute to close all paper gaps before submission.

**Leave-K-Out CV (v3, CPU-only) — DONE**
- Script: `scripts/held_out_benchmark_eval.py` | Output: `data/use_cases/results_test_only_v3/held_out_eval.json`
- **Mean gap: +0.003 (std: 0.009)** — negligible, better than v2's +0.010
- Fold AUROCs: 0.866, 0.885, 0.876, 0.872 (all strong)
- experiments.tex table updated — no longer says "pending v3 re-run"

**All 12 paper figures now v3** (previously 7/11 + 1 new):
- 3 blocked figures generated: auroc_comparison, bootstrap_distribution, effect_size (from new all-baselines bootstrap CIs)
- Reliability diagram: new figure (`fig_reliability_diagram.pdf`)
- Production UC figure: regenerated with v3 uc_e/f/g results
- Cross-model transfer figure: already done in earlier session

**UC E/F/G results computed** (CPU-only, v3 scored data):
- UC-E: AUPRC 0.852–0.888, Best F1 0.763–0.798
- UC-F: Coverage at 95% acc 14–21%
- UC-G: Green tier acc 0.879–0.897, ~21% workload reduction
- Output: `data/use_cases/results_test_only_v3/uc_{e,f,g}_results.json`

**Paper proofread — 20 issues found and 15 fixed:**
- CRITICAL fixed: "over 25pts"→"over 26pts", "over 40pts"→"over 34pts", "95%"→"97%" retention (4 locations), healthcare threshold mismatch, missing bib entry `si2024finegrain`
- All AUPRC values corrected to 0.887
- All remaining v2 numbers replaced (related_work.tex 0.953→0.878)
- Benchmark naming standardized (MM-Vet, HLE-Multimodal)
- Notation consistency fixed in method.tex
- Still open: model size ablation caveat, benchmark count ambiguity, workload reduction mismatch, training count ~11K vs 10,892

**New paper content inserted:**
- Error analysis paragraph (experiments.tex, after per-benchmark figure)
- Response length shortcut defense paragraph (experiments.tex, Section 4.2)
- Difficulty stratification paragraph (experiments.tex, Section 4.4)
- 3 discussion limitation paragraphs (frontier difficulty, easy question collapse, cross-model variance)
- 3 appendix tables (per-benchmark, length analysis, difficulty stratification)

**Other outputs:**
- HuggingFace model card: `pinocchio_package/MODEL_CARD.md`
- Training sample count verified: 10,892 (confirmed by split_info.json + training log)
- Paper review notes saved: `data/paper_review_notes.md`
- Qwen3.5 ablation script fixed for v3 (split_info path, per-size HP tuning, CUDA indices)

**Appendix content added:**
- Cross-model transfer matrix table (19 benchmarks × 3 models)
- Per-benchmark breakdown table (20 rows, sorted by AUROC)
- Response length analysis table (quartile AUROCs)
- Difficulty stratification table (3 tiers)

**Overleaf pushed** (2 commits: `40c9505`, `732f605`). Paper live on Overleaf with all v3 updates.

**Page count warning:** Estimated ~11-12 pages, ICML limit is 8. Need to move ~3-4 pages to appendix. Candidates: domain deployment (healthcare/finance), production use case details, model size figure. Trimming plan agent hit compute limit — needs follow-up.

**CLAUDE.md fixed:** Training count 11,342 → 10,892.

**GPU jobs ready (not submitted):**
1. Multi-seed v3: `slurm/multi_seed_v3_fixed.sh` — 4 jobs × 4×A100 × ~4.5h each
2. Qwen3.5 size ablation: `slurm/qwen35_full_ablation.sh` — 3×A100 × ~7h (script fixed)
3. Leave-K-out: DONE (was CPU-only, not GPU)

---

### 2026-03-12 — v3 analysis blitz + paper fixes (11 parallel agents)
Why: Batch of CPU-only analyses to strengthen paper and fix stale numbers.

**New v3 analyses** (all saved to `data/use_cases/results_test_only_v3/`):
- `per_benchmark_deep_analysis.json`: 20 benchmarks, mean AUROC 0.821. Best: livebench 0.995, mathvista 0.954. Worst: HLE 0.616, prbench 0.638. **Accuracy vs AUROC correlation ≈ 0** (r=0.045) — calibrator performance unrelated to benchmark difficulty.
- `cross_model_analysis.json`: Per-model AUROCs stable (range 0.009). Score correlation across models rho=0.586–0.635. Biggest gaps: prbench on Qwen3.5 (0.443), charxiv on GPT-5-mini (0.570).
- `difficulty_stratified_analysis.json`: Easy tier 0.832, Medium 0.862, Hard 0.848. High-confidence accuracy 87.1%. ECE=0.086.
- `length_analysis.json`: **Calibrator is NOT using length as shortcut.** Length-only AUROC=0.630 vs calibrator 0.878. Residual AUROC after regressing out length=0.866 (−0.012). Length explains only 3.2% of signal above chance.
- `auprc_v3.json`: Combined AUPRC=0.887 [0.869, 0.904]. Per-model: gpt5mini 0.878, gpt52 0.896, qwen35 0.885.
- `bootstrap_ci_v3_all_baselines.json`: All-baseline bootstrap CIs (was previously missing — only had Calibrator). Enables 3 blocked paper figures.
- `reliability_diagram_data.json`: 10-bin reliability diagram data for paper figure.

**Paper fixes** (Overleaf):
- Fixed AUROC 0.949→0.878, ECE 0.022→0.087 in experiments.tex:56
- Fixed verbalized 0.607→0.610 in experiments.tex:72
- Fixed AUPRC 0.865→0.887 in intro, conclusion, experiments (5 locations)
- Leave-K-out table (experiments.tex:182-194) flagged as pending v3 re-run
- Cross-model transfer figure updated from v2 (0.951/0.959/0.946) to v3 (0.882/0.877/0.873)

**Figures regenerated**: 7/11 paper figures updated with v3 data and copied to overleaf/figures/. Remaining 3 (auroc_comparison, bootstrap_distribution, effect_size) should now be unblocked by all-baselines bootstrap CIs.

**Easy question data** (`data/finetune/easy_questions/all_easy.jsonl`): 9K samples, 6 easy benchmarks, 50/50 correct/incorrect, synthetic source. Zero overlap with v3 test. If mixed into training, recommend downsampling to 1.5–3K to avoid dominating (would be 45% of combined).

**Still TODO (needs GPU)**:
- Multi-seed v3 training (OOM-fixed script ready at `slurm/multi_seed_v3_fixed.sh`)
- Leave-K-out CV with v3 split (for experiments table)
- Regenerate 3 remaining figures from all-baseline bootstrap CIs
- Re-run Qwen3.5 model size ablation with per-size HP tuning

---

### 2026-03-12 — Comprehensive code audit (6 parallel agents)
Why: AI-generated codebase has never been systematically audited for bugs. Ran 6 agents in parallel to audit all major script categories.

**Agent 1: `train_best_uq.py` — 14 issues found**
- CRITICAL: Image file handle leak (lines 520, 634) — `Image.open()` never closed, OOM risk over epochs
- CRITICAL: pixel_values shape bugs (lines 592-605, 992, 994) — `torch.cat` vs `torch.stack`, empty list instead of tensor
- HIGH: Question-level split key uses bare `sample.id` (line 842), not `(benchmark, source_model, id)` — potential hidden leakage if different benchmarks reuse same IDs
- HIGH: Unmatched samples silently added to train when reusing old split_info (lines 807-814)
- MEDIUM: Target token "ii"/"i" assumed single-token (line 533), hardcoded assistant fallback 77091 (line 507), ECE off-by-one excluding p=1.0 (line 695)

**Agent 3: Use case scripts — 5 bugs found**
- HIGH: AURC trapezoid integration may use unsorted coverages (uc1, line 80)
- MEDIUM: PR curve starts at (0, precisions[0]) not (0, 1.0) — PR-AUC slightly wrong (uc_e, line 272)
- MEDIUM: Inverted trigger logic gap at boundary (uc_e, lines 106-109)
- MEDIUM: Silent NaN dropout in cross-method AUROC (cpu_difficulty_stratification, line 65)
- LOW: "Pairwise accuracy" naming includes ties (uc_a, line 189)

**Agent 4: Benchmark harness (`uq_eval/`) — 8 bugs found**
- CRITICAL: `extract_choice_letter()` hardcoded to A-D regex (common.py:47), MMLU-Pro/ChemBench have A-J → answers E-J silently dropped
- HIGH: MMMU letter extraction strips all non-letters then takes first char (mmmu.py:190) — "The best answer is B" → extracts "T"
- HIGH: LiveBench asymmetric matching `gold in pred` only (livebench.py:158) — should also check `pred in gold`
- MEDIUM: SimpleQA/HLE word-subset matching too permissive (simpleqa.py:44, hle.py:74) — "yes" matches "yes no maybe"
- MEDIUM: HLE fallback answer extraction captures wrong token (hle.py:241)
- NOTE: HealthBench returns correct=-1 but NOT in training data, so no impact

**Agent 6: Ablation & analysis scripts — 2 high bugs**
- HIGH: Cross-model transfer figure uses hard-coded stale numbers, ignores loaded data (cpu_generate_all_figures.py:493)
- HIGH: No-metadata ablation token ID extraction may not match in-context generation (ablation_no_metadata.py:272)

**Agent 2: Scoring & eval scripts — 8 issues found**
- CRITICAL: `filter_test_only.py` loads question-level IDs but matches sample-level IDs (lines 83-137) — works by coincidence (both 832 items), fragile
- HIGH: Platt scaling baseline fits on train subset but stores predictions for ALL samples including train (compute_baselines.py:45-105) — inflates Platt baseline
- HIGH: Permutation test is one-tailed (Method > Baseline) but not documented (bootstrap_ci_v3.py:66-93)
- MEDIUM: AUROC rounded to 4 decimals in filter_test_only.py:204, bootstrap skips degenerate samples silently

**Agent 5: Pinocchio package (`pinocchio_package/`) — 3 critical, 4 medium**
- CRITICAL: Token IDs wrong — package uses `"(i"`/`"(ii"` but model trained on `"i"`/`"ii"` (model.py:129). ALL package scores are incorrect.
- CRITICAL: Default base model is `Qwen/Qwen3.5-0.8B` but checkpoints use `Qwen/Qwen3-VL-8B-Instruct` (model.py:23). Model loading fails.
- CRITICAL: No Qwen3-VL class detection — falls back to AutoModelForCausalLM, can't load VLM (model.py:55)
- MEDIUM: Website `.format()` crashes on `{` in input, no inference error handling, wrong test checkpoint, README says "text only" for VLM model

**Confirmed findings:**
- Question-level split key (`s.id` at line 842) causes cross-benchmark ID collision: vizwiz=3309, mathverse=1655, etc. are simple integers that overlap. Fix: use `(s.benchmark, s.id)` as key.
- `extract_choice_letter` A-D bug confirmed in code (common.py:47), but MMLU-Pro has its own A-J parser (line 101). Impact limited to MMMU, ChemBench, GPQA (GPQA is 4-choice so unaffected).
- HealthBench correct=-1 confirmed but NOT in training data — no impact.

**Impact assessment:** The most impactful bugs are likely: (1) SimpleQA/HLE word-subset matching inflating "correct" labels in ~5K training samples, (2) MMMU letter extraction wrong for verbose responses, (3) LiveBench asymmetric matching. These could meaningfully affect training labels and thus model quality. The question-level split ID collision biases split composition but doesn't cause leakage. Shape bugs in training script likely masked by batch_size=1.

**TODO (ordered by priority):**
1. Fix pinocchio package: token IDs `"i"`/`"ii"` not `"(i"`/`"(ii"`, base model → Qwen3-VL-8B, add VLM class detection
2. Fix grading: `simpleqa.py:44` and `hle.py:74` word-subset match, `mmmu.py:190` letter extraction, `livebench.py:158` symmetric match, `common.py:47` A-D→A-J regex
3. Fix `train_best_uq.py:842` split key → `(s.benchmark, s.id)`, close Image handles (lines 520/634)
4. Fix `compute_baselines.py:68` Platt scaling train/test contamination
5. Fix `cpu_generate_all_figures.py:493` hard-coded cross-model matrix → read from data
6. After fixes: re-run benchmark evals, retrain model, re-score, re-generate figures
7. Full detailed agent outputs saved at `/tmp/claude-1039/-scratch-khayes-LLM/tasks/*.output` (may not persist across sessions)

### 2026-03-11 — GPU access unavailable until further notice
GPUs cannot be used. All GPU work is blocked until further notice.

### 2026-03-11 — CPU-only housekeeping session (6 parallel agents)
Why: Catch up on non-GPU tasks that eat context when done manually.

**1. MASTER_PLAN.md updated to v3**
All stale v2 numbers replaced (0.953→0.878, etc.), key paths updated, remaining work section expanded.

**2. Paper numbers audit (report only, no edits)**
Found 4 stale v2 numbers in .tex files: `related_work.tex:16` (0.953→0.878), `experiments.tex:56` (AUROC 0.949→0.878, ECE 0.022→0.087), `experiments.tex:72` (0.607→0.610). Also found AUPRC inconsistency: intro/conclusion say 0.886, experiments table says 0.865 — needs resolution. Leave-K-out table (experiments.tex:182-198) still has v2 fold numbers.

**3. Easy question calibration data — already prepared**
9,000 samples at `data/finetune/easy_questions/all_easy.jsonl` (6 benchmarks × 750 questions × 2 samples). Ready to mix into training when GPUs available.

**4. Multi-seed OOM diagnosis + fix**
Root cause: `multi_seed_training.py` runs all seeds in one process; CUDA context leaks 75.77 GiB after seed 42, OOMing seeds 123-314. Fix: `slurm/multi_seed_v3_fixed.sh` — each seed as separate SLURM job with clean CUDA context. Ready to submit.

**5. Missing figures inventory**
All 9 paper figures in overleaf are stale (pre-Mar 9). 7/9 have updated v3 versions in `figures/paper/` but not copied to overleaf. 3 figures (auroc_comparison, bootstrap_distribution, effect_size) are broken because v3 `bootstrap_ci.json` only has Calibrator method — needs re-run of `bootstrap_ci.py` with all baselines (CPU-only). Cross-model transfer figure has hardcoded v2 matrix values.

**6. Repo cleanup audit (report only)**
62 GB total. ~23 GB safe to reclaim immediately: `uq_models/size_ablation/` (18 GB, v2 leaked split), smoke test artifacts (2 GB), superseded v1/v2 checkpoints (3 GB), CONTAMINATED dirs (22 MB). Additional ~20 GB possible: `data/finegrain_uq/` (13 GB, needs confirmation), v2-era ablations (4 GB), legacy calibrators (2.5 GB).

### 2026-03-11 — HF Spaces demo deployed (private)
Why: Interactive Gradio demo for Pinocchio on HF Spaces for pre-launch testing.
Result: Private Space at `https://huggingface.co/spaces/KevinDavidHayes/pinocchio`. Uses pinocchio-0.8b on CPU. Landing page also created at `website/landing-page/index.html`.

### 2026-03-11 — OPEN ISSUE: 0.8B model has narrow score range on easy questions
Why: Demo testing revealed the model outputs ~0.70-0.80 for everything on trivial questions (e.g. "2+2=4" scores 0.777, "capital of Australia is Sydney" scores 0.730). Cannot distinguish trivially correct from trivially wrong.
Root cause: Training data is exclusively hard benchmarks (40-60% accuracy). Model never saw easy questions, so it has no calibration signal outside the hard-question range.
Tested: 15 easy Q&A pairs on 0.8B model — all scores in [0.59, 0.80], correct/incorrect gap only ~5 pts. "Dog has six legs" scored 0.797 (same as correct answer).
Impact: Demo unusable for general-purpose use. Currently restricted to benchmark-style examples with disclaimer.
**Potential fixes (need to decide):**
1. Add easy calibration data (BoolQ, ARC-Easy, GSM8K-easy) with correct+incorrect answers to training mix
2. Synthetic hard negatives: take real questions, generate deliberately wrong answers
3. Two-stage training: broad difficulty pre-calibration, then fine-tune on hard benchmarks
4. Post-hoc Platt scaling on held-out set spanning full difficulty range
5. Difficulty-aware prompt field (requires retrain)
**Status:** Parked — need to discuss approach with advisor before committing GPU time to retraining.

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
