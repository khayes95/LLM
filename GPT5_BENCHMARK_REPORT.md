# GPT-5 Benchmark Evaluation Planning Report

**Date:** December 2024
**Purpose:** Analysis of available benchmarks for UQ classifier training data collection

---

## Executive Summary

We have **24,588 examples** across 18 working hard benchmarks ready for GPT-5 evaluation. Based on GPT-5's expected accuracy, this will yield approximately:

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total Examples** | 24,588 | 100% |
| **Estimated Correct** | 11,972 | 48.7% |
| **Estimated Incorrect** | 12,616 | 51.3% |

**Balance Ratio:** 1.05:1 (incorrect:correct) — Near-ideal for UQ training

---

## Cost Estimates

| Scenario | Examples | Est. Cost |
|----------|----------|-----------|
| **All benchmarks** | 24,588 | ~$716 |
| **Without long-context** | 23,985 | ~$290 |
| **Auto-scorable only** | 17,970 | ~$200 |

*Pricing based on $5/1M input tokens, $20/1M output tokens (estimated GPT-5 rates)*

---

## Benchmarks by Scoring Type

### 1. Auto-Scorable (Immediate Use) — 17,970 examples

These benchmarks can be scored automatically and used directly for UQ training:

| Benchmark | Examples | GPT-5 Acc | Correct | Incorrect | Category |
|-----------|----------|-----------|---------|-----------|----------|
| bbeh | 4,520 | 50% | 2,260 | 2,260 | reasoning |
| simpleqa | 4,326 | 40% | 1,730 | 2,596 | knowledge |
| chembench | 2,785 | 40% | 1,114 | 1,671 | science |
| hle | 2,030 | 28% | 568 | 1,462 | knowledge |
| livebench | 1,436 | 79% | 1,134 | 302 | instruction |
| multinrc | 1,055 | 65% | 685 | 370 | reasoning |
| longbench_v2 | 503 | 63% | 316 | 187 | long_context |
| arc_agi | 419 | 10% | 41 | 378 | abstract |
| ether0 | 325 | 40% | 130 | 195 | science |
| multichallenge | 273 | 60% | 163 | 110 | reasoning |
| gpqa | 198 | 84% | 166 | 32 | knowledge |
| babilong | 100 | 70% | 70 | 30 | long_context |

**Subtotal:** 17,970 examples → 8,377 correct, 9,593 incorrect

### 2. Rubric-Based (Requires LLM Judge) — 4,123 examples

These require a judge model (e.g., GPT-4) to score responses:

| Benchmark | Examples | GPT-5 Acc | Category |
|-----------|----------|-----------|----------|
| prbench | 1,650 | 50% | professional |
| tutorbench | 1,473 | 55% | instruction |
| healthbench | 1,000 | 60% | science |

**Additional cost:** ~$50-100 for judge API calls

### 3. Execution-Based (Requires Code Runner) — 2,495 examples

These require code execution to verify correctness:

| Benchmark | Examples | GPT-5 Acc | Category |
|-----------|----------|-----------|----------|
| bigcodebench | 1,140 | 56% | coding |
| livecodebench | 1,055 | 50% | coding |
| swebench | 300 | 65% | coding |

**Requires:** Docker/sandbox execution environment

---

## Category Distribution

| Category | Examples | Correct | Incorrect | Balance |
|----------|----------|---------|-----------|---------|
| **Knowledge** | 6,554 | 2,464 | 4,090 | 1.66:1 ✓ |
| **Reasoning** | 5,848 | 3,108 | 2,740 | 0.88:1 |
| **Science** | 4,110 | 1,844 | 2,266 | 1.23:1 ✓ |
| **Instruction** | 2,909 | 1,944 | 965 | 0.50:1 |
| **Coding** | 2,495 | 1,360 | 1,135 | 0.83:1 |
| **Professional** | 1,650 | 825 | 825 | 1.00:1 ✓ |
| **Long Context** | 603 | 386 | 217 | 0.56:1 |
| **Abstract** | 419 | 41 | 378 | 9.22:1 ✓ |

*✓ = Good for incorrect sample collection*

---

## Context Length Analysis

### Short Context (≤4K tokens) — 23,293 examples
- **Benchmarks:** bbeh, simpleqa, tutorbench, healthbench, bigcodebench, multinrc, prbench, hle, ether0, chembench, gpqa, livebench, livecodebench, swebench
- **Avg tokens:** 200-1,500 per example
- **Compatible with:** All models

### Medium Context (4K-16K tokens) — 792 examples
- **Benchmarks:** multichallenge, arc_agi, babilong
- **Avg tokens:** 1,300-3,700 per example
- **Compatible with:** Most modern models

### Long Context (>16K tokens) — 503 examples
- **Benchmarks:** longbench_v2
- **Avg tokens:** 166,000 (max: 1.3M!)
- **Note:** Extremely long, will be expensive

---

## Issues & Recommendations

### Critical Issues

1. **omnimath (0 examples):** Dataset has HuggingFace compatibility issue
   - *Workaround:* Update `datasets` library or use alternative math benchmark
   - *Impact:* Loss of 4,428 olympiad math examples

2. **oolong (0 examples):** Dataset loading issue
   - *Workaround:* Check dataset availability on HuggingFace
   - *Impact:* Loss of ~13,000 long-context examples

3. **longbench_v2:** Max context 1.3M tokens
   - *Recommendation:* Sample subset or skip for initial run
   - *Cost impact:* ~$400 savings if skipped

### Benchmarks That May Not Work for UQ Classifier

| Benchmark | Issue | Recommendation |
|-----------|-------|----------------|
| **arc_agi** | Only 10% accuracy, almost all incorrect | Good for incorrect samples, but may skew distribution |
| **gpqa** | 84% accuracy, mostly correct | Good for correct samples, use with harder benchmarks |
| **livebench** | 79% accuracy | May need pass@k for more incorrect samples |
| **execution-based** | No ground truth labels | Run code execution before UQ training |
| **rubric-based** | Need judge scores | Run LLM judge before UQ training |

### UQ Classifier Training Considerations

1. **What the classifier needs:**
   - Question/prompt text
   - Model response text
   - Binary correct/incorrect label
   - (Optional) Model confidence score

2. **What we'll have after GPT-5 run:**
   - Auto-scorable: Full labels immediately (~18K examples)
   - Rubric-based: Labels after judge run (~4K examples)
   - Execution-based: Labels after code execution (~2.5K examples)

3. **Context length for classifier:**
   - Most examples: <4K tokens input + ~500 tokens output
   - Classifier input: prompt + response = typically <5K tokens
   - Some long-context: May need truncation or separate handling

---

## Recommended Evaluation Strategy

### Phase 1: Core Evaluation (~$200, ~18K examples)
Run auto-scorable benchmarks first:
```bash
# Priority benchmarks (ideal accuracy range)
bbeh, simpleqa, multichallenge, multinrc, hle, ether0, chembench

# Add if budget allows
gpqa, livebench, arc_agi, babilong
```

### Phase 2: Judged Benchmarks (~$150 + judge costs)
After Phase 1, run rubric-based benchmarks:
```bash
healthbench, tutorbench, prbench
```
Then run LLM judge (GPT-4) to score responses.

### Phase 3: Code Execution (~$100 + compute)
Run coding benchmarks and execute generated code:
```bash
bigcodebench, livecodebench, swebench
```

### Phase 4: Long Context (Optional, ~$400)
Only if needed for specific use case:
```bash
longbench_v2
```

---

## Sample Commands

```bash
# Run a single benchmark
python -m uq_eval.cli --bench bbeh --model_backend openai --model_name gpt-5 --max_examples 100

# Run multiple benchmarks
for bench in bbeh simpleqa hle ether0 chembench multinrc multichallenge; do
    python -m uq_eval.cli --bench $bench --model_backend openai --model_name gpt-5
done

# Run with specific options
python -m uq_eval.cli --bench prbench --domain finance_hard --model_backend openai
python -m uq_eval.cli --bench hle --text_only --model_backend openai
python -m uq_eval.cli --bench livebench --category_filter math --model_backend openai
```

---

## Expected Training Data Summary

After full evaluation run:

| Source | Examples | Correct | Incorrect | Notes |
|--------|----------|---------|-----------|-------|
| Auto-scored | 17,970 | 8,377 | 9,593 | Immediately usable |
| + Rubric-judged | 4,123 | 2,235 | 1,888 | After judge run |
| + Code-executed | 2,495 | 1,360 | 1,135 | After execution |
| **TOTAL** | **24,588** | **11,972** | **12,616** | |

**Final balance:** 51.3% incorrect, 48.7% correct — Excellent for UQ training

---

## Appendix: Full Benchmark Details

See `benchmark_analysis.json` for complete statistics including:
- Per-benchmark token counts
- Detailed category breakdown
- All identified issues
