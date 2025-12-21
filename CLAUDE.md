# UQ Eval - Uncertainty Quantification Evaluation Harness

## Project Overview

This project finetunes an open-source LLM to act as an uncertainty quantification method for closed-source models. The evaluation harness runs inference on various benchmarks, collecting correct/incorrect responses as training data for the UQ model.

**Goal:** Run GPT-5.2 on difficult benchmarks targeting ~50% accuracy balance for UQ training data.

## Project Structure

```
/scratch/khayes/LLM/
├── uq_eval/                    # Main package
│   ├── __init__.py
│   ├── cli.py                  # CLI entry point (argparse)
│   ├── runner.py               # Core eval loop: run_eval(), aggregate_metrics()
│   ├── registry.py             # Benchmark & model registration
│   ├── types.py                # Core types: Example, Prediction, ModelRequest, ModelResponse
│   ├── io.py                   # JSONL read/write utilities
│   ├── benchmarks/
│   │   ├── base.py             # BaseBenchmark ABC
│   │   ├── common.py           # Shared utilities (JSON parsing, text normalization)
│   │   ├── gpqa.py             # GPQA Diamond ✅
│   │   ├── simpleqa.py         # SimpleQA ✅
│   │   ├── bbeh.py             # BIG-Bench Extra Hard ✅
│   │   ├── hle.py              # Humanity's Last Exam ✅
│   │   ├── healthbench.py      # HealthBench Hard ✅
│   │   └── [sanity benchmarks] # Test benchmarks
│   └── models/
│       ├── base.py             # BaseModelClient ABC
│       ├── openai_client.py    # OpenAI Responses API client
│       └── chat_completions_http_client.py  # Generic OpenAI-compatible HTTP client
├── data/
│   └── bbeh/                   # Cloned BBEH repo
├── scripts/
│   └── run_eval_single_gpu.sh  # SLURM job for vLLM + eval
├── runs/                       # Output directory for evaluations
└── logs/                       # SLURM job logs
```

## Key Commands

```bash
# Run evaluation with vLLM on SLURM
sbatch scripts/run_eval_single_gpu.sh <model> <benchmark> [max_examples]

# Run evaluation directly
python -m uq_eval.cli \
    --bench <benchmark_name> \
    --model_backend chat_http \
    --model_name <model_id> \
    --base_url <api_url> \
    --max_examples 10

# Resume a partial run
python -m uq_eval.cli ... --resume
```

## Output Format

Each run creates `runs/<timestamp>_<bench>_<model>/`:
- `predictions.jsonl` - One line per example with request/response/score
- `metrics.json` - Aggregated metrics (accuracy, brier score, etc.)

---

# BENCHMARK MASTER LIST

## Tier 1: Ideal Accuracy Range (~40-60%) — Priority

| Benchmark | CLI Name | GPT-5 Acc | Status |
|-----------|----------|-----------|--------|
| **BBEH** | `bbeh` | ~50% | ✅ Done |
| **SimpleQA** | `simpleqa` | 19-54% | ✅ Done |
| **TutorBench** | `tutorbench` | ~55% | ✅ Done |
| **MultiChallenge** | `multichallenge` | 58-64% | ✅ Done |
| **HealthBench Hard** | `healthbench` | ~60% | ✅ Done |
| **BigCodeBench** | `bigcodebench` | 56% | ✅ Done |
| **MultiNRC** | `multinrc` | 65% | ✅ Done |

## Tier 2: Lower Accuracy (More Incorrect Samples)

| Benchmark | CLI Name | GPT-5 Acc | Status |
|-----------|----------|-----------|--------|
| **HLE (Text Only)** | `hle` | 25-30% | ✅ Done |
| **Ether0-benchmark** | `ether0` | 20-60% | ✅ Done |
| **EnigmaEval** | - | 19% | ❌ TODO (needs access) |
| **FrontierMath** | - | 13-32% | ❌ TODO |

## Tier 3: Higher Accuracy (May Need pass@k)

| Benchmark | CLI Name | GPT-5 Acc | Status |
|-----------|----------|-----------|--------|
| **GPQA Diamond** | `gpqa` | 77-90% | ✅ Done |
| **omni-math** | `omnimath` | 72% | ✅ Done |
| **Livebench** | - | 79% | ❌ TODO |

## Tier 4: Coding Benchmarks

| Benchmark | CLI Name | GPT-5 Acc | Status |
|-----------|----------|-----------|--------|
| **LiveCodeBench** | `livecodebench` | 4-90% | ✅ Done |
| **SWE-Bench** | `swebench` | 52-75% | ✅ Done |
| **BigCodeBench** | `bigcodebench` | 56% | ✅ Done |

## Tier 5: Long Context

| Benchmark | CLI Name | GPT-5 Acc | Status |
|-----------|----------|-----------|--------|
| **Oolong** | `oolong` | 47-70% | ✅ Done |
| **LongBench v2** | - | ~63% | ❌ TODO |

## Tier 6: Standard Benchmarks (High Accuracy, use pass@k)

| Benchmark | CLI Name | GPT-5 Acc | Status |
|-----------|----------|-----------|--------|
| **MATH** | `math` | 96% | ✅ Done |
| **MMLU-Pro** | `mmlu_pro` | 87% | ✅ Done |
| **GSM8K** | `gsm8k` | ~95% | ✅ Done |
| **ARC-Challenge** | `arc` | ~95% | ✅ Done |
| **DROP** | `drop` | ~90% | ✅ Done |
| **TriviaQA** | `triviaqa` | ~90% | ✅ Done |
| **HellaSwag** | `hellaswag` | ~95% | ✅ Done |
| **WinoGrande** | `winogrande` | ~95% | ✅ Done |
| **MGSM** | `mgsm` | ~90% | ✅ Done |

---

# ALL IMPLEMENTED BENCHMARKS (29 total)

## Quick Reference
```bash
# Run any benchmark
python -m uq_eval.cli --bench <name> --model_backend chat_http --base_url <url>

# Available benchmarks:
arc, bbeh, bigcodebench, drop, dummy_qa, ether0, gpqa, gsm8k, healthbench,
hellaswag, hle, jsonl_qa, livecodebench, math, mgsm, mmlu_pro, multichallenge,
multinrc, omnimath, oolong, sanity_long_context, sanity_math, sanity_mcq,
sanity_unanswerable, simpleqa, swebench, triviaqa, tutorbench, winogrande
```

## Benchmark-Specific Options

| Benchmark | Options |
|-----------|---------|
| `gpqa` | `--subset gpqa_diamond\|gpqa_main\|gpqa_extended` |
| `bbeh` | `--bbeh_mini`, `--bbeh_tasks task1,task2` |
| `hle` | `--hle_with_images`, `--hle_answer_type mcq\|short_answer`, `--hle_category` |
| `healthbench` | `--subset hard\|consensus` |
| `swebench` | (uses `lite` subset by default) |
| `mgsm` | (default: English, modify `language` in code for others) |
| `arc` | (uses ARC-Challenge by default) |

## Scoring Notes

- **Execution-based benchmarks** (bigcodebench, livecodebench, swebench): Return `correct=-1`, need offline execution
- **Rubric-based benchmarks** (healthbench, tutorbench): Return `correct=-1`, need LLM judge
- **All others**: Automatic scoring with Brier score when confidence is available

---

## Notes

- Test with open-source models (Llama-3-70B via vLLM) before GPT-5.2
- Target ~50% accuracy for balanced correct/incorrect training data
- Use pass@k sampling for high-accuracy benchmarks (>80%)
- Multimodal benchmarks: filter to text-only by default
