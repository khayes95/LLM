# Potential Issues, Errors, and Confounds

**Compiled:** 2026-03-08 | **Status:** Living document — update as issues are resolved

---

## CRITICAL — Blocks Paper Submission

### 1. Question-Level Data Leakage Confirmed: AUROC Drops 0.953 → 0.878

**Status:** ✅ FIXED — v3 scoring complete. Paper updated.

The headline result in the paper (AUROC 0.953) was computed on a test set where **82–96% of test questions also appeared in the training set** (across different source models). The v3 checkpoint, trained with a proper question-level split (0% question overlap), achieves:

- **v3 held-out AUROC: 0.878** (vs v2's 0.898 — drop of 0.020)
- **v3 test-only AUROC: TBD** (scoring running now, but expect ~0.88–0.91 range based on held-out)

This is a **7.5-point drop** from the paper's current headline number. Every table, figure, and claim in the paper must be updated once v3 scoring completes.

**What's affected:** ALL numbers in the paper — Tables 2–7, Figures 2–5, all use case metrics, all ablation comparisons. The relative ordering of methods may also change.

**Files:** `uq_models/best_v3_qsplit/split_info.json` confirms `"question_overlap": 0`.

---

### 2. Paper Currently Reports Invalid Numbers

The paper (Overleaf, last synced 2026-03-05) reports v2 metrics throughout:
- Abstract: "AUROC of 0.953 [0.947, 0.958]"
- Table 2: All baselines compared against 0.953
- Discussion: "nearly 30 percentage points above the strongest baseline"

**Status:** ✅ FIXED — All paper numbers updated to v3. Abstract, intro, method, experiments, discussion, conclusion, and appendix all use v3 metrics.

---

### 3. Near-Duplicate Contamination Persists Even After Question-Level Fix

**Status:** ✅ MEASURED AND ACKNOWLEDGED — deduped AUROC 0.872 (only −0.006 from 0.878). Not a concern.

The v3 contamination report (`data/use_cases/results_test_only_v3/contamination_report_v3.json`) found:
- **405/1,953 (20.7%) test samples** have >0.8 max Jaccard similarity to some training sample
- Worst: arc_agi 65.8%, mmvet 41.2%

Deduped AUROC (removing all near-duplicates): **0.872** vs full 0.878 — delta of only −0.006. Near-duplicate contamination does NOT inflate v3 scores. This is acknowledged in the paper appendix (Section "Near-Duplicate Contamination Analysis").

---

## MAJOR — Could Sink the Paper at Review

### 4. Metadata in Prompt Leaks Task Identity

The "combined" prompt (v2 config) includes:
```
Benchmark: {benchmark_name}
Source model: {source_model}
```

This gives the calibrator direct knowledge of which benchmark and which model generated the response. A reviewer will ask: **is the model just learning per-benchmark base rates?**

The no-metadata ablation shows only a −2.7 pt drop (0.953 → 0.926), but:
- This was measured on the **leaked v2 test set** — the gap may be larger on clean v3 data
- Even 0.926 could be partially driven by metadata
- The scramble test (word-level: −12.3 pts) helps defend semantic reading, but doesn't fully resolve the metadata concern
- **The unseen model test (LLaMA-3.1-8B) achieves 0.800** — but metadata still present

**Recommendation:** Re-run no-metadata ablation on v3 data. Consider training a no-metadata variant as the primary model and reporting metadata as an optional boost.

---

### 5. Model Size Ablation is Fundamentally Flawed

The Qwen3.5 model size ablation (0.8B, 2B, 4B, 9B) used **identical hyperparameters** (r=16, alpha=32, same LR) for all sizes. Results show inverted scaling:

| Size | AUROC |
|------|-------|
| 0.8B | 0.852 |
| 2B | 0.850 |
| 4B | 0.663 |
| 9B | 0.771 |

The 4B model has ECE=0.511 (predicts ~constant), confirming overfitting. **This is a hyperparameter artifact, not a finding about model capability.** The paper currently claims "0.8B retains 95% of 8B performance" — this claim is not supported by a fair comparison.

**Risk:** A reviewer familiar with scaling laws will immediately flag this. The paper either needs:
1. Per-size HP tuning (grid search over LR and LoRA rank for each size), or
2. Honest framing: "under a fixed HP budget, small models suffice" (weaker claim)

---

### 6. Multi-Seed Variance is High (σ = 0.033, N = 3)

Three seeds on v2 config:
- Seed 42: 0.898, Seed 123: 0.843, Seed 456: 0.904
- **Mean: 0.882 ± 0.033**

A 3.3-point standard deviation on only 3 seeds is concerning. Seed 123 is 5.5 points below the best. This suggests:
- The training is sensitive to initialization/data ordering
- Reporting the best seed (42) rather than the mean overestimates performance
- **The paper should report mean ± std, not the best seed**

With v3's lower baseline (~0.878), the variance could push some seeds below 0.85.

**Recommendation:** Run 5+ seeds on v3 and report mean ± std as the headline number.

---

### 7. Only 3 Source Models, All Auto-Regressive Transformers

Training data comes from GPT-5-mini, GPT-5.2, and Qwen3.5-397B. These are all:
- Auto-regressive transformers
- Instruction-tuned with RLHF/DPO
- Similar response styles (structured, hedged, etc.)

The paper claims "cross-model transfer" but hasn't tested on architecturally different models:
- No MoE vs dense comparison (Qwen3.5 is MoE but similar response style)
- No retrieval-augmented models
- No chain-of-thought specialist models (o3, etc.)
- The LLaMA-3.1-8B test (0.800) is encouraging but is also an auto-regressive transformer

**Risk:** Reviewers may argue the calibrator learns response style artifacts shared across similar models, not genuine correctness signals.

---

### 8. VSR Images Still 38% Gray Placeholders

Per CLAUDE.md, 38% of VSR benchmark images fail to load and get gray placeholders. This means:
- The VLM AUROC (0.887 on v3) is computed on partially degraded data
- The model may have learned that gray placeholder = VSR benchmark = specific base rate
- Cross-benchmark VLM comparisons are confounded

**Status:** Known bug, documented but not fixed.

---

### 9. Proxy Semantic Entropy Baseline is a Straw Man

The paper tests "proxy SE" using Qwen3-VL-8B (the calibrator's own base model) to generate N=5 responses, then measures agreement. This gets AUROC 0.467 (below random).

But this is not a fair SE comparison — the paper should note:
- True SE requires generations from the **target model** (GPT-5, Qwen3.5)
- The proxy approach is inherently invalid (small model agreement ≠ big model correctness)
- The paper's defense ("SE is too expensive for closed-source") is valid but the experimental comparison is misleading

A reviewer may demand true SE on Qwen3.5-397B (which is open-weight and can generate samples) as an honest head-to-head.

---

### 10. Test Set is Not Truly Held-Out for Ablations

The v2 ablations (truncation, scramble, no-metadata, elicitation) were all evaluated on the **same leaked v2 test set**. The ablation deltas (e.g., "metadata contributes 2.7 pts") may be inaccurate on clean data because:
- The leaked test set makes all methods look better
- Relative differences between methods could be amplified or dampened by leakage
- **All ablation claims need to be re-verified on v3 data**

---

## MODERATE — Reviewers May Flag

### 11. Gray Placeholder Images as Modality Shortcut

For text-only benchmarks, the model receives a 336×336 gray image. This means:
- The model can distinguish VLM vs text benchmarks from the image alone
- Combined with benchmark metadata, the model has strong task-identity signal
- **Confound:** The model might learn "gray image → text benchmark → different base rate" rather than reading the text

**Mitigation check needed:** What happens if you give random noise images instead of gray? Or real but irrelevant images?

---

### 12. Leave-K-Out CV Has Only 4 Folds

The generalization argument rests on leave-5-out CV with only 4 folds. With 20 benchmarks and 4 folds:
- Each fold holds out 5 benchmarks (25%)
- Mean gap of +0.010 is reassuring, but Fold 3 shows +0.062 gap
- **4 folds is very low for reliable CV estimates** — the standard error of the mean gap is large
- A reviewer could argue this is insufficient evidence of generalization

---

### 13. UC4 Benchmark Ranking Correlation May Be Spurious

UC4 reports rank correlation of 0.961–0.979 between predicted and actual per-benchmark accuracy. But:
- Only ~20 benchmarks (N=20 data points for correlation)
- Benchmarks have vastly different difficulty (19% to 95% accuracy)
- **Any method that roughly captures difficulty ordering will get high correlation**
- The verbalized baseline likely also achieves reasonable rank correlation (not reported)

---

### 14. Response Truncation at 800 Characters Loses Information

The truncation ablation shows AUROC plateaus at 800 characters. But:
- For math/code benchmarks, the answer is often in the first few characters
- For reasoning benchmarks, the critical logic may be deeper in the response
- **800 chars ≈ 200 tokens ≈ 3-4 sentences** — this is very short for CoT responses
- The model may be learning surface features (hedging language, confidence markers) rather than evaluating the actual answer

The truncation plateau at 800 chars is itself evidence that the model relies on early response features, not deep content analysis.

---

### 15. Class Balance is Nearly 50/50

The test set is 54.4% correct. This means:
- AUROC of 0.5 is achievable by random guessing
- The gap from random (0.5) to achieved (0.878 on v3) is 0.378
- A naive "always predict correct" baseline gets 54.4% accuracy
- **The paper should report precision-recall curves and AUPRC** in addition to AUROC to show the model works across different operating points

(AUPRC is reported for some use cases but not as a primary metric in Table 2.)

---

### 16. Token Extraction Assumes Specific Tokenization

The scoring extracts P(correct) from the logit difference between tokens "i" and "ii". This assumes:
- The Qwen3-VL tokenizer encodes "i" and "ii" as single tokens
- No other tokens compete for probability mass at the last position
- The model reliably outputs one of these two tokens

If the tokenizer changes (e.g., different Qwen version), the extraction breaks. This is fragile and should be documented as a limitation.

---

### 17. No Calibration on Unseen Domains

The paper shows strong in-distribution calibration (ECE=0.023 on v2, 0.087 on v3). But:
- Legal (PRBench): AUROC 0.637 — severe degradation
- HLE multimodal: AUROC 0.623 on v3 — near random
- Unseen benchmarks: AUROC 0.657 combined

**The calibration metrics are not reported for OOD domains.** A model can have low ECE in-distribution but catastrophically miscalibrated ECE out-of-distribution. If the paper claims the model is "well-calibrated," it needs to show calibration curves for OOD data too.

---

### 18. Verbalized Confidence Baseline May Be Unfairly Weak

The verbalized baseline asks the target model "How confident are you?" and extracts a number. This is known to be poorly calibrated for most LLMs. But:
- GPT-5 and Qwen3.5 may have improved verbalized calibration vs older models
- The baseline doesn't use chain-of-thought or structured self-evaluation
- More sophisticated prompting (e.g., "Rate your confidence on a scale of 1-10 after reconsidering your answer") might close the gap

The paper's gap (0.953 → 0.607, or ~0.878 → 0.607 on v3) is large enough that this probably doesn't matter, but a reviewer could still ask for a stronger verbalized baseline.

---

### 19. Training Data Quality Not Validated

The training labels (is_correct) come from automated grading of benchmark responses. But:
- Different benchmarks use different grading criteria (exact match, fuzzy match, LLM-as-judge)
- **Grading errors directly become label noise** in the training data
- The paper doesn't report inter-annotator agreement or grading accuracy
- Known grading issues (HallusionBench '0'/'1' labels, MMMU option formatting) were fixed but may not be the only ones

---

### 20. Single Calibrator Architecture Not Compared to Alternatives

The paper uses Qwen3-VL-8B + LoRA as the calibrator. But no comparison is made to:
- A simple MLP on response features (length, token entropy, perplexity)
- A smaller language model (e.g., BERT-base fine-tuned on the same data)
- A regression model on the combined prompt's text features
- A dedicated uncertainty estimation head (rather than repurposing a generative model)

The logit of a binary classification token from an 8B VLM is a very heavy-weight approach. If a simpler model achieves 90% of the performance, the contribution is weakened.

---

## MINOR — Worth Noting

### 21. Anonymization Placeholders
- Introduction line 34: `\texttt{[anonymized]}`
- Conclusion line 13: `\texttt{[anonymized]}`
- Need GitHub URL for camera-ready

### 22. HLE Multimodal: Only 35 Samples
Too small for any meaningful statistical conclusion. Should be excluded or merged with HLE text.

### 23. English-Only Training Data
All benchmarks are in English (except MGSM which has multilingual math). No evaluation of calibration on non-English responses.

### 24. v3 Split Changes Train/Test Ratio
- v2: ~10,047 train / ~4,447 test (69/31)
- v3: ~10,892 train / ~1,951 test (85/15)
- The v3 test set is **much smaller** (1,951 vs 4,447), which means wider confidence intervals on all metrics

### 25. No Error Bar on Use Case Metrics
Use cases (AURC, F1, coverage) report point estimates without confidence intervals. Given multi-seed variance of ±0.033 on AUROC, use case metrics likely have similar or larger variance.

### 26. Batch Effects from Sequential Scoring
Scoring runs sequentially across source models (gpt5mini → gpt52 → qwen35). If GPU state, caching, or numerical precision drifts between runs, it could introduce subtle batch effects.

### 27. Overleaf Paper Format
Currently set to ICML 2026 with `[review]` option. The CLAUDE.md says target is ECCV 2026. These are different venues with different formatting requirements.

### 28. Missing `\todo{}` and `\placeholder{}` Items
The paper likely contains unfilled TODO markers that need to be resolved before submission.

---

## Confounds Summary

| Confound | Severity | Mitigation Available? |
|----------|----------|----------------------|
| Question-level leakage | CRITICAL | v3 retrain fixes it |
| Near-duplicate leakage | MAJOR | Measure on deduped subset |
| Metadata in prompt | MAJOR | No-metadata ablation exists (needs v3 re-run) |
| Gray placeholder images | MODERATE | Test with random/irrelevant images |
| Truncation → surface features | MODERATE | Scramble test partially addresses |
| Same-family source models | MAJOR | LLaMA test partially addresses |
| Fixed HPs across model sizes | MAJOR | Per-size tuning needed |
| Small seed count (N=3) | MAJOR | Run 5+ seeds on v3 |
| Automated grading label noise | MODERATE | Manual audit of sample |

---

## Action Items (Priority Order)

1. ~~**Wait for v3 scoring to complete**~~ ✅ DONE — v3 AUROC: 0.878 [0.863, 0.892]
2. ~~**Update ALL paper numbers** to v3 metrics~~ ✅ DONE — abstract, intro, experiments, discussion, conclusion, appendix all updated
3. ~~**Re-run key ablations on v3**: no-metadata, truncation, scramble~~ ✅ DONE — no-metadata: −5.0 pts, scramble: −9.7 pts word / −1.1 pts sentence
4. **Run 5+ seeds on v3** and report mean ± std — 🔄 RUNNING (SLURM 8811, seed 42 done: 0.889, seeds 123-314 queued with dependency)
5. ~~**Measure AUROC on near-duplicate-free test subset**~~ ✅ DONE — deduped AUROC: 0.872 (−0.006 from 0.878)
6. **Fix model size ablation** with per-size HP tuning (or weaken claim) — ⏳ PENDING (needs GPU)
7. ~~**Update domain-specific tables** (healthcare, finance, three-tier routing, error analysis)~~ ✅ DONE — all recomputed on v3 data
8. ~~**Acknowledge near-duplicate contamination** in limitations~~ ✅ DONE — added to appendix + discussion
9. ~~**Regenerate paper figures** with v3 data~~ ✅ DONE — 7/11 figures regenerated (4 need baseline scores recomputed)
10. ~~**Remove all TODO markers** from paper~~ ✅ DONE — all resolved
11. **Fix VSR image loading** and retrain (lower priority) — ⏳ PENDING (needs GPU)
12. **Verify venue format** (ICML vs ECCV) — paper uses ICML 2026 format, CLAUDE.md says ECCV. Needs user clarification.
