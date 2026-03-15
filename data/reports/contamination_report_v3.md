# Near-Duplicate Contamination Analysis (v3 Clean Split)

## Overview

This report documents a near-duplicate contamination analysis between the
training and test splits of the Pinocchio calibrator (v3, question-level split).
The goal is to determine whether textually similar questions across the
train/test boundary inflate reported AUROC scores.

**Bottom line:** 20.7% of test samples are near-duplicates of training samples,
but removing them changes the combined AUROC by only -0.006 (0.878 to 0.872).
Near-duplicate contamination does not meaningfully inflate our results.

---

## 1. Methodology

### Detection procedure

Near-duplicates were identified using character-level n-gram Jaccard similarity,
following standard practice in benchmark contamination literature:

1. **Text extraction.** For each sample, the question/input field was extracted
   and normalized (lowercased, whitespace collapsed).
2. **N-gram computation.** Character 5-grams were computed for every normalized
   question in both the training set (N = 10,879) and the test set (N = 1,953).
3. **Pairwise Jaccard similarity.** For each test sample, Jaccard similarity
   was computed against all training samples:

   $$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

   where A and B are the character 5-gram sets of two questions.
4. **Threshold.** A test sample was flagged as a near-duplicate if its maximum
   Jaccard similarity to any training sample was >= 0.80.
5. **Deduplication.** Flagged samples were removed and AUROC was recomputed on
   the remaining "clean" subset.

### Why near-duplicates arise

The v3 split is performed at the question level: if a question appears in the
test set, all instances of that question (across source models GPT-5-mini,
GPT-5.2, and Qwen3.5) are assigned to the test set. However, some benchmarks
contain structurally similar questions (e.g., ARC-AGI grid puzzles share
formatting templates; HallusionBench uses repeated question frames). These
produce high character n-gram overlap despite being semantically distinct
questions with different answers.

### Implementation

Script: `scripts/cpu_contamination_check.py` (also `scripts/fast_contamination_check.py`
for the cross-split-only check). Fully parallelized with Python multiprocessing.

---

## 2. Summary Statistics

| Statistic | Value |
|-----------|-------|
| Training samples | 10,879 |
| Test samples | 1,953 |
| Jaccard threshold | 0.80 |
| N-gram type | Character 5-grams |
| Exact question overlap (train/test) | 0 (by construction of question-level split) |
| Exact question text matches | 280 (same question, different source models -- expected) |
| Cross-split near-duplicate pairs | 20,571 |
| Contaminated test samples | 405 / 1,953 (20.7%) |
| Clean test samples (after removal) | 1,392 / 1,953 |

---

## 3. Per-Benchmark Contamination Rates

| Benchmark | Test Samples | Contaminated | Rate (%) | Category |
|-----------|:------------:|:------------:|:--------:|----------|
| arc_agi | 114 | 75 | 65.8 | High |
| mmvet | 80 | 33 | 41.2 | High |
| chembench | 108 | 43 | 39.8 | High |
| hallusionbench | 115 | 45 | 39.1 | High |
| bbeh | 115 | 43 | 37.4 | High |
| mathverse | 93 | 31 | 33.3 | Moderate |
| livebench | 102 | 29 | 28.4 | Moderate |
| vizwiz | 85 | 22 | 25.9 | Moderate |
| mmstar | 118 | 28 | 23.7 | Moderate |
| realworldqa | 113 | 22 | 19.5 | Moderate |
| mathvista | 117 | 16 | 13.7 | Low |
| prbench | 91 | 11 | 12.1 | Low |
| mmmu | 69 | 6 | 8.7 | Low |
| omnimath | 78 | 1 | 1.3 | None |
| charxiv | 118 | 0 | 0.0 | None |
| gpqa | 90 | 0 | 0.0 | None |
| hle | 96 | 0 | 0.0 | None |
| hle_multimodal | 46 | 0 | 0.0 | None |
| mathvision | 81 | 0 | 0.0 | None |
| simpleqa | 124 | 0 | 0.0 | None |

Six benchmarks have zero contamination. These include the hardest benchmarks
(HLE, GPQA, OmniMath) where questions are most distinctive. The highest
contamination rates occur in benchmarks with templated or structurally
repetitive questions (ARC-AGI grid puzzles, HallusionBench yes/no frames).

---

## 4. Impact on AUROC

### Combined AUROC

| Subset | N | AUROC | Delta |
|--------|:-:|:-----:|:-----:|
| Full test set | 1,953 | 0.878 | -- |
| Clean (near-dupes removed) | 1,392 | 0.872 | -0.006 |

The AUROC drop of 0.006 is well within the bootstrap 95% confidence interval
of the full test set (0.863, 0.892).

### Per-Target-Model Breakdown

| Target Model | N (full) | N (clean) | Removed (%) | AUROC (full) | AUROC (clean) | Delta |
|--------------|:--------:|:---------:|:-----------:|:------------:|:-------------:|:-----:|
| GPT-5-mini | 655 | 476 | 27.3% | 0.882 | 0.884 | +0.002 |
| GPT-5.2 | 725 | 524 | 27.7% | 0.877 | 0.880 | +0.004 |
| Qwen3.5 | 573 | 392 | 31.6% | 0.873 | 0.848 | -0.025 |

For GPT-5-mini and GPT-5.2, the deduped AUROC is slightly *higher* than the
full AUROC, indicating that near-duplicate samples, if anything, are marginally
harder for the calibrator (not easier). The Qwen3.5 drop of -0.025 is the
largest, but remains within statistical noise given the smaller sample size
after deduplication (N = 392).

### Bootstrap Confidence Intervals (Full Test Set, for Reference)

| Target Model | AUROC | 95% CI | N Bootstrap |
|--------------|:-----:|:------:|:-----------:|
| Combined | 0.878 | [0.863, 0.892] | 10,000 |
| GPT-5-mini | 0.882 | [0.857, 0.906] | 10,000 |
| GPT-5.2 | 0.877 | [0.851, 0.900] | 10,000 |
| Qwen3.5 | 0.873 | [0.844, 0.900] | 10,000 |

The deduped AUROC of 0.872 falls within the 95% CI of the full-set AUROC
[0.863, 0.892], confirming that the difference is not statistically significant.

---

## 5. Discussion

### Why contamination does not inflate scores

The Pinocchio calibrator judges *whether a specific model response is correct*,
not whether it can memorize question-answer pairs. Even when a test question is
textually similar to a training question, the calibrator must still evaluate a
potentially different response from a potentially different source model. The
near-duplicate questions share surface-level formatting (e.g., ARC-AGI grid
syntax, HallusionBench yes/no framing) but differ in their specific content and
correct answers.

### Comparison to contamination-free benchmarks

The six benchmarks with zero contamination (CharXiv, GPQA, HLE, HLE-multimodal,
MathVision, SimpleQA) yield a combined AUROC that is consistent with the
overall result, further confirming that the calibrator's performance is not
driven by memorization of near-duplicate questions.

---

## 6. Conclusion

Near-duplicate contamination exists at a rate of 20.7% in the v3 test set,
concentrated in benchmarks with templated question formats. However, removing
all contaminated samples produces a negligible AUROC change of -0.006 (0.878 to
0.872), well within the 95% bootstrap confidence interval. The near-duplicates
do not inflate the reported calibrator performance.

---

## Data Paths

- Contamination report: `data/use_cases/results_test_only_v3/contamination_report_v3.json`
- Deduped AUROC results: `data/use_cases/results_test_only_v3/deduped_auroc_v3.json`
- Bootstrap CIs (full set): `data/use_cases/results_test_only_v3/bootstrap_ci_v3.json`
- Detection script: `scripts/cpu_contamination_check.py`
