# Open Vision Reasoning Benchmarks for Frontier Model Training

A comprehensive catalog of challenging vision benchmarks where frontier models (GPT-5, Claude, Gemini) struggle, suitable for uncertainty quantification training data collection.

---

## 🎯 Quick Reference: Documented Frontier Model Weaknesses

| Category | Top Benchmarks | Why Models Struggle |
|----------|---------------|---------------------|
| **Spatial Reasoning** | VSR, OmniSpatial, SpatialRGPT-Bench | Models at chance level for orientation relations; ~25% gap vs humans |
| **Visual Math** | MathVerse, MATH-V, MathVista | GPT-4V: 23.98% vs Human: 70%; diagram understanding failures |
| **Expert Multimodal** | MMMU-Pro, ZeroBench | MMMU-Pro: 16-27% accuracy; ZeroBench: 0% pass@1 for all models |
| **Hallucination** | HallusionBench, POPE | GPT-4V: 31% question-pair accuracy |
| **Video Temporal** | EgoSchema, MVBench | 30-40% on high-level reasoning; single-frame bias issues |
| **Chart/Document** | CharXiv, ChartQA, DocVQA | Complex reasoning over scientific figures |

---

## 📊 SPATIAL REASONING BENCHMARKS

### 1. VSR (Visual Spatial Reasoning)
- **Size**: 10,972 data points, 66 spatial relation types
- **Task**: True/false on spatial relation captions
- **Gap**: Human ceiling >95%, models ~70%. Models at **chance level for orientation relations** (facing, parallel to, etc.)
- **Source**: [GitHub](https://github.com/cambridgeltl/visual-spatial-reasoning) | [Paper](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00566)

### 2. OmniSpatial
- **Size**: 8,400+ QA pairs, 50 fine-grained subcategories
- **Task**: Comprehensive spatial reasoning (dynamic, logic, interaction, perspective-taking)
- **Gap**: Both open- and closed-source VLMs exhibit significant limitations
- **Source**: [arXiv](https://arxiv.org/abs/2506.03135)

### 3. SpatialRGPT-Bench
- **Size**: 657 qualitative + 749 quantitative VQA pairs, 88 object classes
- **Task**: Metric distance/size estimation from images
- **Gap**: VLMs struggle with distance and length prediction without depth info
- **Source**: [Project](https://www.anjiecheng.me/SpatialRGPT)

### 4. SpatialVLM
- **Size**: 2 billion VQA examples on 10M images (training data)
- **Task**: 3D spatial reasoning in metric space
- **Gap**: VLMs lack 3D spatial knowledge; can't recognize quantitative relationships
- **Source**: [Project](https://spatial-vlm.github.io/) | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_SpatialVLM_Endowing_Vision-Language_Models_with_Spatial_Reasoning_Capabilities_CVPR_2024_paper.pdf)

### 5. VLM4D (Spatiotemporal)
- **Size**: Diverse real-world and synthetic videos
- **Task**: Spatiotemporal (4D) reasoning - translational/rotational motion, perspective awareness
- **Gap**: Significant performance gaps vs human baselines; struggle with temporal coherence
- **Source**: [Project](https://vlm4d.github.io/)

### 6. DSI-Bench (Dynamic Spatial Intelligence)
- **Size**: ~1,000 videos, 1,700+ VQA questions, 5 motion types
- **Task**: Dynamic observer-object spatial reasoning
- **Gap**: VLMs conflate observer/object motion; strong "forward bias" causes hallucinations
- **Source**: [arXiv](https://arxiv.org/abs/2510.18873)

### 7. SIBench (Spatial Intelligence Benchmark)
- **Size**: ~20 open-source datasets, 23 task settings
- **Task**: Basic perception → spatial understanding → spatial planning
- **Gap**: Models underperform significantly on planning and spatial imagination
- **Source**: [arXiv](https://arxiv.org/abs/2509.18905)

### 8. Open3D-VQA / Open3DVQA
- **Size**: Large-scale aerial/urban spatial reasoning
- **Task**: 4 types of spatial VQA (relative, absolute, situational, object-centric)
- **Gap**: MLLMs struggle with egocentric spatial relationships in 3D environments
- **Source**: [arXiv](https://arxiv.org/abs/2503.11094)

---

## 🔢 VISUAL MATHEMATICS BENCHMARKS

### 9. MathVerse
- **Size**: 2,612 problems × 6 versions = 15K test samples
- **Task**: Visual math (plane geometry, solid geometry, functions)
- **Gap**: Most MLLMs achieve **5%+ higher accuracy WITHOUT visual input**; diagram understanding is the bottleneck
- **Source**: [GitHub](https://github.com/ZrrSkywalker/MathVerse) | [Project](https://mathverse-cuhk.github.io/) | [ECCV 2024]

### 10. MATH-V (MATH-Vision)
- **Size**: 3,040 problems from real math competitions
- **Task**: Mathematical reasoning with visual contexts (19 subjects)
- **Gap**: GPT-4V: **23.98%** vs Human: **~70%**; GPT-4o: 30.39%
- **Source**: [GitHub](https://github.com/mathllm/MATH-V) | [NeurIPS 2024]

### 11. MathVista
- **Size**: 6,141 examples from 31 datasets
- **Task**: IQ tests, functional plots, scientific figures
- **Gap**: GPT-4o struggles with transformation geometry (<20% accuracy)
- **Source**: [Project](https://mathvista.github.io/)

### 12. OlympiadBench
- **Size**: 8,476 problems (Olympiad-level math + physics)
- **Task**: Bilingual multimodal scientific reasoning with step-by-step solutions
- **Source**: [HuggingFace](https://huggingface.co/datasets/olympiadbench)

### 13. We-Math
- **Size**: Newer benchmark for math reasoning
- **Task**: Visual mathematics with less data contamination
- **Source**: Referenced in multi-agent geometry papers

### 14. Geometry3K
- **Size**: 3,001 high-school geometry problems
- **Task**: Geometry with paired text and diagrams
- **Source**: Standard geometry benchmark

### 15. VisioMath
- **Size**: Multi-image math benchmark
- **Task**: All answer choices as images; fine-grained visual comparison
- **Gap**: Unique challenge requiring symbolic reasoning across similar images
- **Source**: [arXiv](https://arxiv.org/abs/2506.06727)

---

## 🎓 EXPERT-LEVEL MULTIMODAL BENCHMARKS

### 16. MMMU (Massive Multi-discipline Multimodal Understanding)
- **Size**: 11.5K questions, 30 subjects, 183 subfields
- **Task**: College-level expert reasoning (Art, Business, Science, Health, Humanities, Tech)
- **Gap**: GPT-4V: 56% accuracy; requires domain-specific knowledge + visual perception
- **Source**: [Project](https://mmmu-benchmark.github.io/) | [GitHub](https://github.com/MMMU-Benchmark/MMMU)

### 17. MMMU-Pro
- **Size**: Harder version with vision-only input settings
- **Task**: Questions embedded in images; augmented candidate options
- **Gap**: Accuracies ranging from **16.8% to 26.9%** across models
- **Source**: [GitHub](https://github.com/MMMU-Benchmark/MMMU)

### 18. ZeroBench
- **Size**: 100 high-quality, manually curated questions
- **Task**: Designed to be beyond current frontier model capabilities
- **Gap**: **None of the evaluated models achieves non-zero pass@1**
- **Source**: [HuggingFace](https://huggingface.co/datasets/zerobench)

### 19. EXAMS-V
- **Size**: 20,932 questions, 20 subjects, 11 languages
- **Task**: Multilingual multimodal school exams with integrated text+images
- **Gap**: Challenging even for GPT-4V and Gemini
- **Source**: [arXiv](https://arxiv.org/abs/2403.10378)

### 20. mmJEE-Eval
- **Size**: 1,460 questions from JEE Advanced (India, 2019-2025)
- **Task**: Bilingual (English/Hindi) pre-college Physics, Chemistry, Mathematics
- **Gap**: Open-source models plateau at 37-45% despite 400B parameters; closed models collapse on meta-cognitive reasoning
- **Source**: [arXiv](https://arxiv.org/abs/2511.09339)

### 21. MMIU (Multimodal Multi-image Understanding)
- **Size**: 77K images, 11K questions, 52 tasks
- **Task**: Multi-image relationships and reasoning
- **Gap**: GPT-4o achieves only **55.7%**; significant challenges in spatial understanding
- **Source**: [GitHub](https://github.com/HKUST-LongGroup/Awesome-MLLM-Benchmarks)

---

## 📈 CHART & DOCUMENT UNDERSTANDING

### 22. CharXiv
- **Size**: Charts from arXiv papers with reasoning questions
- **Task**: Scientific chart understanding with complex reasoning
- **Gap**: No clear correlation between domain and model performance; >20% gaps in some domains
- **Source**: [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/cdf6f8e9fd9aeaf79b6024caec24f15b-Paper-Datasets_and_Benchmarks_Track.pdf)

### 23. ChartQA
- **Size**: ~10,000 charts, 36,000+ questions
- **Task**: Visual parsing, data extraction, logical/mathematical reasoning
- **Gap**: Requires multi-step reasoning over chart elements
- **Source**: [GitHub](https://github.com/vis-nlp/ChartQA)

### 24. ChartQAPro
- **Size**: More diverse, challenging real-world charts
- **Task**: Extended chart question answering
- **Source**: [GitHub](https://github.com/vis-nlp/ChartQA) (new benchmark)

### 25. DocVQA
- **Size**: 12,000+ document images, 50,000+ questions
- **Task**: Document visual QA (scanned pages, forms, invoices)
- **Gap**: Requires OCR + layout understanding + NL reasoning
- **Source**: [HuggingFace](https://huggingface.co/datasets/docvqa)

### 26. InfographicVQA
- **Size**: 30,035 questions, 5,485 infographic images
- **Task**: Understanding diverse infographic layouts
- **Source**: Standard benchmark

### 27. OCRBench / OCRBench-v2
- **Task**: Text recognition in diverse visual contexts
- **Source**: Standard benchmark

---

## 👁️ HALLUCINATION BENCHMARKS

### 28. HallusionBench
- **Size**: 346 images, 1,129 questions
- **Task**: Image-context reasoning (language hallucination + visual illusion)
- **Gap**: GPT-4V: **31.42%** question-pair accuracy; all others <16%
- **Source**: [GitHub](https://github.com/tianyi-lab/hallusionbench) | [CVPR 2024]

### 29. POPE (Polling-based Object Probing Evaluation)
- **Size**: 3,000 questions, 500 images
- **Task**: Yes/no questions about object presence
- **Gap**: Tests object hallucination with random/popular/adversarial sampling
- **Source**: [arXiv](https://arxiv.org/abs/2305.10355)

### 30. HalluSegBench
- **Size**: 1,340 counterfactual instance pairs, 281 object classes
- **Task**: Counterfactual segmentation reasoning
- **Gap**: Vision-driven hallucinations more prevalent than label-driven
- **Source**: [arXiv](https://arxiv.org/abs/2506.21546)

### 31. AMBER
- **Task**: LLM-free multi-dimensional hallucination evaluation
- **Source**: Hallucination benchmark collection

---

## 🎬 VIDEO UNDERSTANDING BENCHMARKS

### 32. EgoSchema
- **Size**: 5,000+ questions, 250+ hours of video
- **Task**: Very long-form (3-minute clips) video QA
- **Gap**: Models achieve **<33%** (random: 20%); humans: 76%. Intrinsic temporal length 5.7x longer than other datasets
- **Source**: [Project](http://egoschema.github.io/)

### 33. MVBench
- **Size**: 4,000 samples, 20 temporal tasks
- **Task**: Static-to-dynamic task transformation (action sequence, prediction, localization)
- **Gap**: Models drop to **30-40%** on high-level reasoning vs 85-88% on low-level perception
- **Source**: [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_MVBench_A_Comprehensive_Multi-modal_Video_Understanding_Benchmark_CVPR_2024_paper.pdf)

### 34. Video-MME
- **Size**: 2,700 samples, 1,017s average duration
- **Task**: Full-spectrum video understanding (6 visual domains, 30 subfields)
- **Gap**: Tests temporal, spatial, emotional, plot, and long-form cognitive dimensions
- **Source**: [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Fu_Video-MME_The_First-Ever_Comprehensive_Evaluation_Benchmark_of_Multi-modal_LLMs_in_CVPR_2025_paper.pdf)

### 35. LVBench (Long Video Benchmark)
- **Size**: 1,549 samples, 4,101s average duration
- **Task**: Extreme long video understanding and information extraction
- **Gap**: Tests long-term memory and extended comprehension
- **Source**: [arXiv](https://arxiv.org/abs/2406.08035)

### 36. TemporalBench
- **Size**: Fine-grained temporal understanding
- **Task**: Tests temporal sequences vs spatial reasoning
- **Gap**: Existing benchmarks suffer from "single frame bias"
- **Source**: [arXiv](https://arxiv.org/abs/2410.10818)

### 37. MotionBench
- **Size**: Web videos + synthetic videos
- **Task**: Fine-grained motion-level perception
- **Gap**: New evaluation perspective on motion understanding
- **Source**: [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Hong_MotionBench_Benchmarking_and_Improving_Fine-grained_Video_Motion_Understanding_for_Vision_CVPR_2025_paper.pdf)

### 38. HERBench
- **Size**: VideoQA requiring multi-evidence integration
- **Task**: Questions requiring ≥3 non-overlapping evidential cues
- **Gap**: Neither language priors nor single snapshot can suffice
- **Source**: Recent benchmark

---

## 🔬 SCIENCE & DOMAIN-SPECIFIC BENCHMARKS

### 39. ScienceQA
- **Size**: 21,208 questions (elementary/high school curricula)
- **Task**: Multimodal science QA (natural science, social science, language science)
- **Gap**: 48.7% with image context
- **Source**: Standard benchmark

### 40. AI2D
- **Size**: 5,000+ diagrams, 15,000+ questions
- **Task**: Grade school science diagram understanding (food webs, life cycles, etc.)
- **Gap**: Requires joint understanding of layout, symbols, and text
- **Source**: [Leaderboard](https://llm-stats.com/benchmarks/ai2d)

### 41. RealWorldQA
- **Size**: 765 images with questions
- **Task**: Real-world spatial understanding (vehicle captures, etc.)
- **Gap**: Relatively easy for humans but challenging for frontier models
- **Source**: [HuggingFace](https://huggingface.co/datasets/visheratin/realworldqa) | X.ai

### 42. GQA
- **Size**: 22M questions, scene graphs
- **Task**: Real-world visual reasoning and compositional QA
- **Gap**: Blind LSTM: 42.1%, VQA models: 54.1%, Human: 89.3%
- **Source**: [Stanford](https://cs.stanford.edu/people/dorarad/gqa/about.html)

### 43. A-OKVQA
- **Task**: Commonsense reasoning and outside knowledge VQA
- **Gap**: Requires world knowledge beyond image content
- **Source**: Standard benchmark

### 44. OK-VQA
- **Task**: Outside knowledge visual question answering
- **Source**: Standard benchmark

---

## 🔧 LOW-LEVEL VISION TASKS (GPT-5 Specific Weaknesses)

### 45. RF100-VL (Roboflow)
- **Task**: Real-world object detection in multimodal setting
- **Gap**: GPT-5 object counting: **4/10 correct**; measurement: incorrect; small defect detection: fails
- **Source**: [Roboflow Blog](https://blog.roboflow.com/gpt-5-vision-multimodal-evaluation/)

### 46. BuilderBench
- **Task**: Spatially grounded, multi-step digital manipulation
- **Gap**: Foundation models struggle with grounded manipulation
- **Source**: Academic benchmark

### 47. Overcooked-AI
- **Task**: Cooperative multi-agent gameplay
- **Gap**: Even GPT-5 struggles with cooperative coordination
- **Source**: Standard RL benchmark

---

## 🌟 ADDITIONAL NOTABLE BENCHMARKS

### 48. VL-RewardBench
- **Size**: 1,250 examples
- **Task**: Evaluate vision-language reward models
- **Gap**: GPT-4o: **65.4%**; open-source models struggle to surpass random guessing
- **Source**: [Project](https://vl-rewardbench.github.io/)

### 49. MMStar
- **Size**: 1,500 samples
- **Task**: Vision-indispensable multimodal evaluation
- **Gap**: Ensures visual dependency and minimal data leakage
- **Source**: [Project](https://mmstar-benchmark.github.io/)

### 50. TextVQA
- **Task**: Text-in-image reasoning (scene text)
- **Source**: Standard benchmark

### 51. SEED-Bench-2-Plus
- **Task**: Multimodal LLM evaluation
- **Source**: Standard benchmark

### 52. MM-Vet
- **Size**: 200 images, 218 questions
- **Task**: Integrated VL capabilities (recognition, OCR, spatial awareness, math)
- **Source**: Standard benchmark

### 53. MMBench / MMBench-Video
- **Task**: Fine-grained vision-language skills
- **Source**: Standard benchmark

### 54. CMMMU (Chinese MMMU)
- **Size**: 30 subjects, 39 image types
- **Task**: Chinese college-level multimodal understanding
- **Source**: Standard benchmark

---

## 📚 Resources & Benchmark Collections

- **Awesome-MLLM-Benchmarks**: [GitHub](https://github.com/HKUST-LongGroup/Awesome-MLLM-Benchmarks)
- **Awesome-MLLM-Hallucination**: [GitHub](https://github.com/showlab/Awesome-MLLM-Hallucination)
- **VLMEvalKit**: Unified evaluation framework
- **EvalScope VLM Benchmarks**: [Docs](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/vlm.html)
- **lmms-eval**: Evaluation harness

---

## 🎯 Recommended Priority for UQ Training Data

### Tier 1: High Priority (Documented low accuracy + large dataset)
1. **MathVerse** - 15K samples, models worse without vision
2. **MATH-V** - 3K samples, GPT-4V at 24%
3. **VSR** - 11K samples, chance-level on orientation
4. **EgoSchema** - 5K samples, models <33%
5. **MMMU-Pro** - Harder version, 16-27% accuracy
6. **HallusionBench** - 1.1K samples, GPT-4V at 31%

### Tier 2: High Priority (Newer, less contaminated)
1. **OmniSpatial** - 8.4K samples, comprehensive spatial
2. **CharXiv** - Scientific charts from arXiv
3. **DSI-Bench** - Dynamic spatial, recent
4. **mmJEE-Eval** - 1.4K samples, 2019-2025 exams
5. **VLM4D** - Spatiotemporal reasoning

### Tier 3: Standard Benchmarks (Good coverage)
1. **ChartQA** - 36K questions
2. **DocVQA** - 50K questions
3. **ScienceQA** - 21K questions
4. **GQA** - 22M questions
5. **MVBench** - 4K samples (video)

---

*Last Updated: December 2025*
*Compiled for LLM Uncertainty Quantification Project*
