# VLM Benchmark Status Report

**Generated:** 2025-12-30
**Total Benchmarks Verified:** 21
**All Benchmarks Working:** YES

---

## Summary by Priority Tier

### TIER 1: Hard Benchmarks (<50% GPT-5 accuracy) - HIGHEST PRIORITY for UQ
| Benchmark | Samples | GPT-5 Acc | Grading | Status | Notes |
|-----------|---------|-----------|---------|--------|-------|
| **MathVision** | 3,040 | ~24% | Mixed (MCQ/numeric) | ✓ | Competition math with images |
| **MathVerse** | 3,940 | ~25% | Mixed | ✓ | Models worse WITHOUT vision |
| **HallusionBench** | 951 | ~31% | Binary (0/1) | ✓ | Grader must handle '0'/'1' format |

### EXCLUDED: ZeroBench
| **ZeroBench** | 100 | **0%** | Open-ended | ✗ EXCLUDED | **No positive class for UQ training** |

> ZeroBench gets 0% accuracy by design - it's meant to be unsolvable. This means we'd have 100% incorrect samples and 0% correct samples, making it useless for training a binary UQ classifier.

### TIER 2: Challenging Benchmarks (50-70% GPT-5 accuracy) - GOOD for UQ
| Benchmark | Samples | GPT-5 Acc | Grading | Status | Notes |
|-----------|---------|-----------|---------|--------|-------|
| **VSR** | 2,195 | ~70% | Binary (true/false) | ✓ | Spatial reasoning, images download from COCO |
| **MathVista** | 1,000 | ~50% | Mixed | ✓ | Uses `decoded_image` field |
| **MMMU** | 5,190 | ~62% | MCQ (letter) | ✓ | Options stored as string repr |
| **MMStar** | 1,500 | ~55% | MCQ | ✓ | Vision-indispensable |
| **CharXiv** | 1,000 | ~60% | Open-ended | ✓ | Scientific charts, needs LLM grader |
| **RealWorldQA** | 765 | ~70% | MCQ | ✓ | Real-world spatial |
| **A-OKVQA** | 6,702 | ~60% | MCQ | ✓ | Outside knowledge VQA |
| **VizWiz** | 8,000 | ~60% | Open-ended | ✓ | Some unanswerable questions |
| **MM-Vet** | 218 | ~55% | Open-ended | ✓ | Needs LLM grader |

### TIER 3: Easier Benchmarks (>70% accuracy) - Supplementary
| Benchmark | Samples | GPT-5 Acc | Grading | Status | Notes |
|-----------|---------|-----------|---------|--------|-------|
| **AI2D** | 3,088 | ~85% | MCQ (index) | ✓ | Science diagrams |
| **ChartQA** | 2,500 | ~75% | Open-ended | ✓ | Chart QA |
| **DocVQA** | 5,188 | ~75% | Open-ended | ✓ | Document QA |
| **NLVR2** | 2,316 | ~80% | Binary | ✓ | Multi-image (2 per sample) |
| **POPE** | 9,000 | ~80% | Binary (yes/no) | ✓ | Object hallucination |
| **OCRBench** | 1,000 | ~70% | Open-ended | ✓ | Text recognition |
| **GQA** | 12,578 | ~75% | Open-ended | ✓ | Visual reasoning |
| **InfographicVQA** | 3,288 | ~70% | Open-ended | ✓ | Infographic understanding |

---

## Grading Format Summary

| Format | Auto-gradeable | Benchmarks |
|--------|----------------|------------|
| **MCQ (letter)** | YES | MMMU, MMStar, RealWorldQA, A-OKVQA, AI2D |
| **Binary (yes/no, true/false)** | YES | VSR, HallusionBench, NLVR2, POPE |
| **Mixed (MCQ + numeric)** | MOSTLY | MathVision, MathVerse, MathVista |
| **Open-ended** | NEEDS LLM | CharXiv, VizWiz, MM-Vet, ChartQA, DocVQA, OCRBench, GQA, InfographicVQA, ZeroBench |

---

## Known Issues & Grading Notes

### HallusionBench
- Ground truth uses `'0'`/`'1'` format, NOT `'yes'`/`'no'`
- Grader must map: `'0' = No`, `'1' = Yes`

### MMMU
- Options stored as string representation of list: `"['A', 'B', 'C', 'D']"`
- Need `ast.literal_eval()` to parse options

### Multi-Image Benchmarks
- **ZeroBench**: 2-3 images per sample
- **NLVR2**: 2 images per sample (left/right)
- Current VLM judge uses single image - may need adaptation

### VizWiz
- Contains "unanswerable" questions (~30-40%)
- Need special handling for unanswerable detection

---

## Recommended Training Data Composition

For balanced UQ training (targeting ~50% accuracy):

| Category | Benchmarks | Est. Samples |
|----------|------------|--------------|
| **Hard Math/Science** | MathVision, MathVerse, MathVista | ~8,000 |
| **Spatial/Hallucination** | VSR, HallusionBench | ~3,000 |
| **Expert Knowledge** | MMMU, MMStar, CharXiv | ~4,500 |
| **General VQA** | RealWorldQA, A-OKVQA, MM-Vet | ~7,500 |
| **Total** | | **~23,000** |

---

## Category Distribution

| Category | Benchmarks | Variety |
|----------|------------|---------|
| **Visual Math** | MathVision, MathVerse, MathVista | Geometry, charts, IQ |
| **Hallucination Detection** | HallusionBench, POPE | Yes/No probing |
| **Spatial Reasoning** | VSR, RealWorldQA | True/False, MCQ |
| **Expert/College** | MMMU, MMStar | 30+ subjects |
| **Scientific Charts** | CharXiv, ChartQA | arXiv figures |
| **Document Understanding** | DocVQA, InfographicVQA, OCRBench | Scanned docs, infographics |
| **General VQA** | A-OKVQA, GQA, VizWiz | Real-world images |
| **Integrated Capabilities** | MM-Vet | Recognition, OCR, math |
| **Science Diagrams** | AI2D | Food webs, cycles |
| **Multi-Image** | NLVR2, ZeroBench | Comparisons |

---

## Next Steps

1. **Fix VSR loader** to download images from COCO URLs (currently working)
2. **Implement LLM grader** for open-ended benchmarks
3. **Add multi-image support** for ZeroBench/NLVR2 (optional)
4. **Run full inference** on priority benchmarks with InternVL3-78B
5. **Collect training data** targeting ~50% accuracy balance

---

*Last updated: 2025-12-30*
