# Text Benchmark Status Report

**Generated:** 2026-01-01
**Total Benchmarks Verified:** 24/25
**VLM Judge:** Qwen3-VL-8B + LoRA (checkpoint-788), 0.789 AUROC (vision), 0.907 AUROC (text)

---

## Summary by Priority Tier

### TIER 1: Ideal Accuracy Range (40-60% GPT-5) - BEST FOR UQ
| Benchmark | Samples | GPT-5 Acc | Grading | Status | Notes |
|-----------|---------|-----------|---------|--------|-------|
| **BBEH** | 4,520 | ~50% | Mixed | ✓ | 23 task types, local data |
| **SimpleQA** | 4,326 | 19-54% | Open-ended | ✓ | Factual questions |
| **PRBench** | 600 | ~50-51% | Rubric | ✓ | Finance domain, needs LLM grader |
| **BigCodeBench** | 1,140 | ~56% | Execution | ✓ | Code, needs execution |
| **TutorBench** | 1,473 | ~55% | Rubric | ✓ | Educational tutoring, PROMPT/RUBRICS fields |
| ~~MultiChallenge~~ | - | 58-64% | MCQ | ✗ | Dataset not on HuggingFace |
| **HealthBench** | 1,000 | ~60% | Rubric | ✓ | Load via `hard_2025-05-08-21-00-10.jsonl` |

### TIER 2: Lower Accuracy (<40% GPT-5) - MORE INCORRECT SAMPLES
| Benchmark | Samples | GPT-5 Acc | Grading | Status | Notes |
|-----------|---------|-----------|---------|--------|-------|
| **HLE (Text)** | 2,500 | 25-30% | Mixed | ✓ | Humanity's Last Exam, text-only |
| **ARC-AGI** | 400 | ~10% | Grid | ✓ | Abstract reasoning, very hard |
| **ChemBench** | 149 | 8-70% | MCQ | ✓ | Use `examples` field |

### TIER 3: Higher Accuracy (>70% GPT-5) - USE pass@k
| Benchmark | Samples | GPT-5 Acc | Grading | Status | Notes |
|-----------|---------|-----------|---------|--------|-------|
| **LiveBench** | 368 | ~79% | Mixed | ✓ | Math category verified |
| **GPQA** | 198 | 77-90% | MCQ | ✓ | Use `Correct Answer` field |
| **OmniMath** | 4,428 | ~72% | Open-ended | ✓ | Load via direct jsonl download |

### TIER 4: Coding Benchmarks
| Benchmark | Samples | GPT-5 Acc | Grading | Status | Notes |
|-----------|---------|-----------|---------|--------|-------|
| **LiveCodeBench** | 1,055 | 4-90% | Execution | ✓ | Needs code execution |
| **SWE-Bench Lite** | 300 | 52-75% | Execution | ✓ | Needs git + execution |

### TIER 5: Long Context
| Benchmark | Samples | GPT-5 Acc | Grading | Status | Notes |
|-----------|---------|-----------|---------|--------|-------|
| **Oolong** | 6,072 | 47-70% | Mixed | ✓ | D&D config verified |
| **BABILong** | 100 | varies | Open-ended | ✓ | 0k-128k context lengths |

### TIER 6: Standard Benchmarks (High Accuracy, use pass@k)
| Benchmark | Samples | GPT-5 Acc | Grading | Status | Notes |
|-----------|---------|-----------|---------|--------|-------|
| **MATH** | 1,187 | ~96% | Numeric | ✓ | Algebra subset |
| **MMLU-Pro** | 12,032 | ~87% | MCQ (A-J) | ✓ | 10 options |
| **GSM8K** | 1,319 | ~95% | Numeric | ✓ | Grade school math |
| **ARC-Challenge** | 1,172 | ~95% | MCQ | ✓ | Science reasoning |
| **DROP** | 9,535 | ~90% | Open-ended | ✓ | Reading comprehension |
| **TriviaQA** | 17,944 | ~90% | Open-ended | ✓ | Trivia facts |
| **HellaSwag** | 10,042 | ~95% | MCQ (0-3) | ✓ | Commonsense NLI |
| **WinoGrande** | 1,267 | ~95% | Binary | ✓ | Commonsense reasoning |

---

## Grading Format Summary

| Format | Auto-gradeable | Benchmarks |
|--------|----------------|------------|
| **MCQ (letter)** | YES | GPQA, MMLU-Pro, ARC, HellaSwag, ChemBench |
| **Binary** | YES | WinoGrande |
| **Numeric** | YES | MATH, GSM8K |
| **Open-ended** | NEEDS LLM | SimpleQA, BABILong, DROP, TriviaQA, OmniMath |
| **Mixed** | MOSTLY | BBEH, HLE, LiveBench, Oolong |
| **Rubric** | NEEDS LLM | PRBench, TutorBench, HealthBench (gpt-5-mini) |
| **Execution** | NEEDS SANDBOX | BigCodeBench, LiveCodeBench, SWE-Bench |
| **Grid** | EXACT MATCH | ARC-AGI |

---

## Failed Benchmarks - Root Causes

| Benchmark | Error | Workaround |
|-----------|-------|------------|
| MultiChallenge | Dataset not on HuggingFace | None - dataset unavailable |

---

## Recommended Training Data Composition

For balanced UQ training targeting ~50% accuracy:

| Priority | Benchmarks | Est. Samples | Notes |
|----------|------------|--------------| ------|
| **HIGH** | BBEH, SimpleQA, HLE | ~12,000 | 25-55% accuracy |
| **HIGH** | PRBench, Oolong, TutorBench, HealthBench | ~9,000 | 47-60% accuracy |
| **MEDIUM** | ARC-AGI, ChemBench | ~550 | Very hard (~10-70%) |
| **LOW** | Tier 6 with pass@k | varies | Use multiple samples |

**Total Priority Samples:** ~21,550 text samples

---

## VLM Judge Status

- **Current Model:** `data/vlm_judge_combined/checkpoint-788`
- **Base Model:** Qwen3-VL-8B-Instruct
- **Vision AUROC:** 0.789
- **Text AUROC:** 0.907
- **Known Issue:** VSR training data had 38% gray placeholder images
- **TODO:** Retrain with fixed VSR images

---

## Open-Source Models in Use

| Purpose | Model | Notes |
|---------|-------|-------|
| **VLM Judge** | Qwen3-VL-8B-Instruct + LoRA | checkpoint-788 |
| **Text Benchmark Inference** | Meta-Llama-3.1-8B-Instruct | Most runs |
| **Alternative** | Qwen2.5-7B-Instruct | Some runs |

---

*Last updated: 2026-01-01*
