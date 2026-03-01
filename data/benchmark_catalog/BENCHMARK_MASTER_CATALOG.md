# Benchmark Master Catalog

**Purpose:** Unified reference for all benchmarks with production-run-critical metadata.
**Last Updated:** 2025-12-30

---

## Quick Reference: Production-Ready Benchmarks

### VLM Benchmarks (Single Image, 20-80% GPT-5 Accuracy)

| Benchmark | HuggingFace ID | Size | GPT-5 Acc | Format | Class Balance | Status |
|-----------|----------------|------|-----------|--------|---------------|--------|
| **MathVerse** | `AI4Math/MathVerse` (testmini) | 3,940 | ~25% | MCQ/Open | B:30%,C:29%,A:25%,D:17% | ✅ Ready |
| **MathVision** | `MathLLMs/MathVision` | 3,040 | ~24% | MCQ/Open | 92 unique | ✅ Ready |
| **HallusionBench** | `lmms-lab/HallusionBench` | 951 | ~31% | Binary Yes/No | ~50/50 | ✅ Ready |
| **MathVista** | `AI4Math/MathVista` | 5,141 | ~50% | Mixed | - | ✅ Ready |
| **MMStar** | `Lin-Chen/MMStar` | 1,500 | ~55% | MCQ | B:32%,A:28%,D:21%,C:20% | ✅ Ready |
| **CharXiv** | `princeton-nlp/CharXiv` | 1,323 | ~60% | Open-ended | - | ✅ Ready |
| **A-OKVQA** | `HuggingFaceM4/A-OKVQA` | 6,702 | ~60% | MCQ | - | ✅ Ready |
| **VizWiz-VQA** | `lmms-lab/VizWiz-VQA` | 8,000 | ~60% | Open-ended | Many unanswerable | ✅ Ready |
| **VSR** | `cambridgeltl/vsr_random` | 2,195 | ~70% | Binary T/F | 53/47 | ⚠️ Needs COCO fix |
| **SEED-Bench** | `lmms-lab/SEED-Bench` | 17,990 | ~70% | MCQ | - | ✅ Ready |
| **OCRBench** | `echo840/OCRBench` | 1,000 | ~70% | Open-ended | 967 unique | ✅ Ready |
| **ChartQA** | `HuggingFaceM4/ChartQA` | 2,500 | ~75% | Open-ended | - | ✅ Ready |
| **NLVR2** | `TIGER-Lab/NLVR2` | 6,967 | ~80% | Binary | A:48%,B:52% | ✅ Ready |
| **AI2D** | `lmms-lab/ai2d` | 3,088 | ~85% | MCQ | 4-way | ✅ Ready |
| **ScienceQA** | `derek-thomas/ScienceQA` | 4,241 | ~85% | MCQ | - | ✅ Ready |

### VLM Benchmarks (Multi-Image - Requires Special Handling)

| Benchmark | HuggingFace ID | Size | GPT-5 Acc | Format | Notes |
|-----------|----------------|------|-----------|--------|-------|
| **ZeroBench** | `jonathan-roberts1/zerobench` | 100 | 0% | Multi-img | 3 images per Q |
| **Winoground** | `facebook/winoground` | 400 | ~50% | 2-image | Compositional |
| **MMVet** | `lmms-lab/MMVet` | 218 | ~55% | Multi-img | Integrated VL |
| **RealWorldQA** | `lmms-lab/RealWorldQA` | 765 | ~70% | Multi-img | Has image_2 col |
| **NLVR2** | `TIGER-Lab/NLVR2` | ~100K | ~80% | 2-image | Large download |

### VLM Benchmarks (Not Recommended for Production)

| Benchmark | Reason | GPT-5 Acc |
|-----------|--------|-----------|
| TextVQA | 7GB download, too large | ~75% |
| DocVQA | Large download | ~80% |
| ERQA | Multi-image, near-random perf | ~50% |
| COCO-based | Raw images only, no QA | - |

---

## Text Benchmarks (Already Implemented)

### Tier 1: Ideal Accuracy Range (~40-60%)

| Benchmark | CLI Name | Size | GPT-5 Acc | Format | Status |
|-----------|----------|------|-----------|--------|--------|
| BBEH | `bbeh` | 6,511 | ~50% | Mixed | ✅ Done |
| SimpleQA | `simpleqa` | 4,326 | 19-54% | Open | ✅ Done |
| TutorBench | `tutorbench` | 1,490 | ~55% | Rubric | ✅ Done |
| MultiChallenge | `multichallenge` | 273 | 58% | Multi-turn | ✅ Done |
| HealthBench Hard | `healthbench` | 5,000 | ~60% | Rubric | ✅ Done |
| BigCodeBench | `bigcodebench` | 1,140 | 56% | Code exec | ✅ Done |
| MultiNRC | `multinrc` | 1,055 | 65% | MCQ | ✅ Done |
| PRBench | `prbench` | 1,100 | ~50% | Rubric | ✅ Done |

### Tier 2: Lower Accuracy

| Benchmark | CLI Name | Size | GPT-5 Acc | Status |
|-----------|----------|------|-----------|--------|
| HLE | `hle` | 2,500 | 25-30% | ✅ Done |
| Ether0 | `ether0` | 325 | 20-60% | ✅ Done |
| ARC-AGI | `arc_agi` | 400 | ~10% | ✅ Done |
| ChemBench | `chembench` | - | 8-70% | ✅ Done |

### Tier 3: Higher Accuracy (Use pass@k)

| Benchmark | CLI Name | Size | GPT-5 Acc | Status |
|-----------|----------|------|-----------|--------|
| GPQA Diamond | `gpqa` | 198 | 77-90% | ✅ Done |
| Omni-Math | `omnimath` | 4,428 | 72% | ✅ Done |
| LiveBench | `livebench` | 1,200 | 79% | ✅ Done |

---

## NEW: Additional Benchmarks (Just Cataloged)

### Text/Reasoning Benchmarks (Open Access)

| Benchmark | HuggingFace ID | Size | GPT-5 Acc | Format | Open-Source Compatible | Notes |
|-----------|----------------|------|-----------|--------|------------------------|-------|
| **BrowseComp** | `Tevatron/browsecomp-plus` | 830 | 50-69% | Web browsing | ✅ Yes | Web comprehension |
| **LongBench v2** | `zai-org/LongBench-v2` | 503 | ~63% | Long context | ✅ Yes | Long docs |
| **MultiNRC** | `ScaleAI/MultiNRC` | 1,055 | ~65% | Multi-hop RC | ✅ Yes | Reading comprehension |
| **Arena-Hard** | `lmarena-ai/arena-hard-auto-v0.1` | 500 | varies | Chat | ✅ Yes | Conversation quality |
| **Oolong** | `oolongbench/oolong-real` (dnd) | 6,072 | 47% | Long context | ✅ Yes | Real-world LC |

### Coding Benchmarks (Require Execution Environment)

| Benchmark | HuggingFace ID | Size | GPT-5 Acc | Notes |
|-----------|----------------|------|-----------|-------|
| **SWE-Bench Verified** | `princeton-nlp/SWE-bench_Verified` | 500 | 52-75% | Software engineering |
| **LiveCodeBench** | `livecodebench/code_generation_lite` | 1,055 | 4-90% | Live coding problems |
| **BigCodeBench** | `bigcode/bigcodebench` | 1,140 | ~56% | Code generation |

### Gated Benchmarks (Require HuggingFace Access Request)

| Benchmark | HuggingFace ID | GPT-5 Acc | Notes |
|-----------|----------------|-----------|-------|
| EnigmaEval | `ScaleAI/EnigmaEval` | ~19% | Cryptic reasoning - **request access** |
| GAIA | `gaia-benchmark/GAIA` | ~30% | General assistant - **request access** |
| BFCL | `Salesforce/xlam-function-calling-60k` | varies | Function calling - **request access** |

### Not Recommended / Issues

| Benchmark | Issue |
|-----------|-------|
| tau2-bench | HuggingFace dataset has column mismatch errors |
| HELMET | Download errors, very slow |
| Omni-MATH | HuggingFace List feature type error |
| BFCL (gorilla-llm) | Column mismatch errors |

---

## Critical Metadata for Production Runs

### Image Format Requirements

```
SINGLE_IMAGE_BENCHMARKS = [
    "AI4Math/MathVerse",
    "MathLLMs/MathVision",
    "AI4Math/MathVista",
    "lmms-lab/HallusionBench",
    "Lin-Chen/MMStar",
    "princeton-nlp/CharXiv",
    "cambridgeltl/vsr_random",
    "lmms-lab/ai2d",
    "HuggingFaceM4/ChartQA",
]

MULTI_IMAGE_BENCHMARKS = [
    "jonathan-roberts1/zerobench",  # 3 images
    "facebook/winoground",          # 2 images
    "lmms-lab/MMVet",               # variable
    "lmms-lab/RealWorldQA",         # 2 images
    "TIGER-Lab/NLVR2",              # 2 images
]
```

### Class Balance Notes

| Benchmark | Correct % | Incorrect % | Notes |
|-----------|-----------|-------------|-------|
| VSR | 53% | 47% | Near-balanced binary |
| HallusionBench | ~50% | ~50% | Binary Yes/No |
| MathVerse | ~25% | ~75% | Hard benchmark |
| MathVision | ~25% | ~75% | Hard benchmark |
| MMMU | ~60% | ~40% | MCQ (multiple choice) |

### Known Issues (Lessons Learned)

1. **VSR Image Bug**: Dataset stores filenames, not PIL Images. Must download from `image_link` URLs.
2. **ERQA Multi-Image**: 28% of samples need multiple images. Excluded from evaluation.
3. **MathVerse Config**: Must specify `testmini` config, not default.
4. **MMMU Configs**: Has per-subject configs (Accounting, Art, Biology, etc.), not a single config.
5. **TextVQA Size**: 7GB download - skip for quick iterations.
6. **HallusionBench Grading**: Ground truth is '0'/'1', not 'yes'/'no'.

---

## Download Commands

```bash
# Quick download of priority benchmarks
python scripts/download_vlm_benchmarks.py

# Manual download example
from datasets import load_dataset
ds = load_dataset("AI4Math/MathVerse", "testmini", trust_remote_code=True)
```

---

## File Locations

- **Catalog JSON**: `data/benchmark_catalog/vlm_benchmarks_catalog.json`
- **Summary JSON**: `data/benchmark_catalog/vlm_benchmarks_summary.json`
- **Legacy vision catalog**: `vision_benchmarks_catalog.md` (deprecated)
- **Legacy JSON**: `benchmarks.json` (deprecated)
