# UQ Eval - Uncertainty Quantification for LLMs and VLMs

## Quick Links

| Document | Purpose |
|----------|---------|
| **[MASTER_PLAN.md](MASTER_PLAN.md)** | High-level roadmap, phases, timeline |
| **[RESEARCH_LOG.md](RESEARCH_LOG.md)** | Chronological log of experiments and results (MEMORY) |
| **[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)** | Paper-ready content |

---

## Environment Setup

```bash
conda activate uq_eval
```

---

## ⚠️⚠️⚠️ CRITICAL: QUESTION-LEVEL DATA LEAKAGE — ALL CURRENT SCORES ARE SUSPECT ⚠️⚠️⚠️

### Problem 1: CONTAMINATED Directories (previously known)

| Directory | Status | Use Instead |
|-----------|--------|-------------|
| `data/use_cases/CONTAMINATED_scored_unified/` | QUARANTINED | `data/use_cases/scored_test_only/` |
| `data/use_cases/CONTAMINATED_scored_v2/` | QUARANTINED | `data/use_cases/scored_test_only_v2/` |
| `data/use_cases/CONTAMINATED_results_unified/` | QUARANTINED | `data/use_cases/results_test_only_v2/` |

### Problem 2: QUESTION-LEVEL LEAKAGE IN TRAIN/TEST SPLIT (newly discovered 2026-03-05)

**The train/test split in `train_best_uq.py` splits by SAMPLE INDEX, not by QUESTION ID.**
Since each question appears up to 3 times (once per source model: gpt5mini, gpt52, qwen35),
the SAME QUESTION ends up in both train and test sets across different source models.

**Measured overlap in `scored_test_only_v2/`:**
- gpt5mini_scored.jsonl: **82.4%** of questions also in training set
- gpt52_scored.jsonl: **92.5%** of questions also in training set
- qwen35_scored.jsonl: **95.9%** of questions also in training set
- In `split_info.json`: **76.2%** of test IDs also appear in train IDs (1,169 of 1,535)

**Impact:** Both headline numbers are likely INFLATED:
- Held-out AUROC 0.898 — suspect
- Test-only scoring AUROC 0.953 — suspect

**Root cause (3 bugs):**
1. `train_best_uq.py` (line ~839): `train_test_split(indices, ...)` splits by sample index, not question ID
2. `split_info.json`: stores bare `id` values without `source_model`, making overlap invisible
3. `filter_test_only.py` (line ~115): matches on `id` alone, passing contaminated questions

**REQUIRED FIX before any paper submission:**
- Split must be done at the QUESTION ID level: if question X goes to test, ALL samples for X (across all source models) must go to test
- Retrain the model with the fixed split
- Re-score and re-run all evaluations
- Update all paper numbers

**Rules:**
- **NEVER read from `CONTAMINATED_*` directories** for any metric, figure, or table
- **ALL current `scored_test_only_v2/` numbers are SUSPECT** until the split is fixed and model retrained
- **DO NOT submit the paper** until the question-level split is fixed and numbers are verified
- If you see any script splitting by sample index instead of question ID, fix it immediately

---

## Multi-Agent Warning

**Multiple Claude Code agents may be working on this project simultaneously.** Before making changes:
- Check `squeue -u $USER` — other agents may have running SLURM jobs
- Check `git status` / `git log --oneline -5` — other agents may have uncommitted changes
- Do NOT cancel SLURM jobs you didn't submit without checking with the user
- Do NOT overwrite files other agents may be actively writing to (especially `RESEARCH_LOG.md`)
- Coordinate via the research log: mark entries `[RUNNING]` when you submit, update when done

---

## Project Overview

This project has two main components:

1. **Evaluation Harness (`uq_eval/`)** - Run benchmarks to generate training data
2. **UQ Judge Training** - Fine-tune models to predict correctness

**Primary Goal:** Train the BEST possible open-source UQ model that predicts whether a closed-source model's response is correct — for BOTH text and vision. This model is the foundation of the paper. Without a great UQ model, there is no paper.

**The UQ model must:**
- Handle both text benchmarks and VLM benchmarks (with real images)
- Work cross-model (train on model A, evaluate on model B)
- Work on closed-source models (GPT-5, Claude) where fine-tuning is impossible
- Be releasable to the community (open weights)

**Current best approach:** `scripts/train_best_uq.py` — trains Qwen3-VL-8B + LoRA on ALL benchmark data from GPT-5-mini + GPT-5.2 + Qwen3.5, using real images for VLM benchmarks and gray placeholders for text benchmarks. Output: `uq_models/best_unified/`

**Target:** ECCV 2026 submission

**Three Novel Contributions:**
1. Cross-model transfer (train on A, evaluate on B)
2. Multimodality (works for text LLMs and VLMs, with real images)
3. Closed-source application (GPT-5, Claude)

---

## Session Workflow

### Start of Session
```
1. Read last 5-10 entries in RESEARCH_LOG.md
2. Check current phase in MASTER_PLAN.md
3. Identify what needs to be done next
```

### During Work
```
1. Before running any experiment, log intent in RESEARCH_LOG.md
2. After each experiment, log results immediately
3. If you encounter a bug, document before fixing
```

### Research Log Format: Keep It Minimal
The research log will be read by many agents. Keep entries **extremely brief** — just the facts. Other agents (planning, writing) will interpret the results themselves.

**When submitting a job**, add a brief entry:
```
### YYYY-MM-DD HH:MM — [RUNNING] <short description>
Why: <1 sentence — what dataset/task and why we're running this>
Script: `scripts/<filename>.py` | SLURM job ID: <id> | Args: <key args>
```

**When the job finishes**, replace the `[RUNNING]` entry with results:
```
### YYYY-MM-DD HH:MM — <short description>
Why: <same 1 sentence from above>
Script: `scripts/<filename>.py` | Output: `<path to results>`
Result: <1-2 lines of key metrics or outcome>
```

**Rules:**
- Do NOT write analysis, interpretation, or commentary — just state what ran and what the numbers are.
- Do NOT duplicate results that are already saved in output files. Point to the file path instead.
- Each entry should be 2-4 lines max. If you need more, you're writing too much.
- Replace `[RUNNING]` entries in-place when done — do not leave stale running entries.
- Failed jobs get one line: script, error summary, and whether it was retried.

### End of Session
```
1. Verify all [RUNNING] entries in RESEARCH_LOG.md are resolved (completed or marked failed)
2. Note any issues for next session (1 line max)
3. Push code/configs/docs to the private GitHub repo (see Nightly Push rules)
```

---

## Directory Structure

```
/scratch/khayes/LLM/
├── uq_eval/                    # Evaluation harness (32+ benchmarks)
│   ├── benchmarks/             # Benchmark implementations
│   └── models/                 # Model clients
├── data/
│   ├── features/               # VLM benchmark features (VSR, MMMU, etc.)
│   ├── finetune/               # Text training/test data
│   ├── vlm_judge_combined/     # VLM judge checkpoints
│   ├── ablations/              # Ablation checkpoints
│   └── cross_model/            # Cross-model evaluation results
├── scripts/                    # Training and evaluation scripts
├── runs/                       # Benchmark run outputs
├── figures/                    # Generated figures
│   └── advisor_meeting/        # Presentation figures
├── logs/                       # SLURM logs
├── uq_models/                  # Text calibrator checkpoints
├── MASTER_PLAN.md              # Roadmap
├── RESEARCH_LOG.md             # Experiment log (MEMORY)
└── CLAUDE.md                   # This file
```

---

## Part 1: Evaluation Harness (`uq_eval`)

### Quick Commands

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
```

### Available Benchmarks (32+)

```bash
arc, arc_agi, babilong, bbeh, bigcodebench, boolq, chembench, drop, dummy_qa,
ether0, gpqa, gsm8k, healthbench, hellaswag, hle, humaneval, jsonl_qa,
livebench, livecodebench, longbench, longbench_v2, math, mbpp, mgsm, mmlu, mmlu_pro,
multichallenge, multinrc, naturalqa, omnimath, oolong, prbench, sanity_long_context,
sanity_math, sanity_mcq, sanity_unanswerable, simpleqa, swebench, triviaqa,
tutorbench, winogrande
```

### Benchmark Options

| Benchmark | Options |
|-----------|---------|
| `gpqa` | `--subset gpqa_diamond\|gpqa_main\|gpqa_extended` |
| `bbeh` | `--bbeh_mini`, `--bbeh_tasks task1,task2` |
| `hle` | `--text_only`, `--answer_type_filter mcq\|exact_match` |
| `healthbench` | `--subset hard\|consensus` |

### Output Format

Each run creates `runs/<timestamp>_<bench>_<model>/`:
- `predictions.jsonl` - One line per example
- `metrics.json` - Aggregated metrics

---

## Part 2: UQ Judge Training

### Best Unified Model (PRIMARY — this is what matters)

**Script:** `scripts/train_best_uq.py`
**Base model:** Qwen3-VL-8B-Instruct + LoRA
**Training data:** ALL benchmarks × ALL target models (GPT-5-mini + GPT-5.2 + Qwen3.5), 10,892 samples
**Images:** Real images for VLM benchmarks (cached in `data/training_images/`), gray placeholder for text benchmarks

**Training command:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_best_uq.py \
    --output_dir uq_models/best_unified \
    --epochs 3 --lora_r 16 --learning_rate 1e-4
```

**Configuration:**
```yaml
# Unified (Qwen3-VL-8B-Instruct)
lora_r: 16
lora_alpha: 32
learning_rate: 1e-4
epochs: 3
batch_size: 1
gradient_accumulation: 16
```

### Legacy Checkpoints (for comparison only)

| Checkpoint | Model | Data | AUROC |
|-----------|-------|------|-------|
| `uq_models/text_calibrator_v3/checkpoint-198` | Qwen2.5-7B (text-only) | GPT-5-mini text | 0.815 in-dist |
| `data/vlm_judge_combined/checkpoint-788` | Qwen3-VL-8B (VLM) | InternVL3 vision | 0.804 in-dist |
| `uq_models/text_calibrator_gpt52/` | Qwen2.5-7B (text-only) | GPT-5.2 text | 0.702 |
| `uq_models/text_calibrator_qwen35/` | Qwen2.5-7B (text-only) | Qwen3.5 text | 0.739 |

### Prompt Template

```
Question: {question}

Answer: {response}

Is the answer correct? (i) No (ii) Yes
```

---

## Part 3: Cross-Model Evaluation

### Text Cross-Model

```bash
python scripts/cross_model_text_eval.py \
    --model uq_models/text_calibrator_v3 \
    --target_model Qwen/Qwen2.5-7B-Instruct \
    --output data/cross_model/text_qwen.json
```

### VLM Cross-Model

```bash
python scripts/cross_model_transfer_all.py \
    --checkpoint data/vlm_judge_combined/checkpoint-788 \
    --benchmarks vsr,charxiv,mmmu,hallusionbench \
    --output data/cross_model/vlm_results.json
```

### Closed-Source Evaluation

```bash
# Generate GPT-5 responses
python scripts/generate_closed_source_responses.py \
    --model gpt-5 \
    --input data/finetune/test_v2.jsonl \
    --output data/cross_model/gpt5_responses.jsonl

# Run calibrator on GPT-5 responses
python scripts/eval_on_responses.py \
    --calibrator uq_models/text_calibrator_v3 \
    --responses data/cross_model/gpt5_responses.jsonl \
    --output data/results/gpt5_eval.json
```

---

## Known Bugs and Fixes

| Bug | Symptom | Fix | Status |
|-----|---------|-----|--------|
| VSR images | 38% gray placeholders | Download from `image_link` | 🔄 Needs retrain |
| MMMU options | Options as string literal | `ast.literal_eval()` | ✅ Fixed |
| HallusionBench grading | AUROC 0.352 | Check '0'/'1' labels | ✅ Fixed |
| ERQA multi-image | Random AUROC | Excluded | ⏸️ Known limitation |

### VSR Fix

```python
# Problem: row['image'] is a filename string, not a PIL Image
# Solution: Download from image_link
import requests
from PIL import Image
from io import BytesIO

def get_vsr_image(row, cache_dir="data/vsr_images"):
    image_link = row.get('image_link')
    if not image_link:
        return None
    filename = image_link.split('/')[-1]
    cache_path = f"{cache_dir}/{filename}"
    
    if os.path.exists(cache_path):
        return Image.open(cache_path).convert("RGB")
    
    response = requests.get(image_link, timeout=30)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image.save(cache_path)
    return image
```

### HallusionBench Fix

```python
# WRONG: gt_yes = ground_truth_lower == "yes"
# CORRECT:
gt_yes = ground_truth_lower in ["yes", "1", "true"]
```

---

## Key Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| AUROC | > 0.70 | Primary metric |
| ECE | < 0.10 | Calibration |
| Within-benchmark AUROC | > 0.60 | Proves not spurious |
| Coverage @ 90% acc | > 60% | Selective prediction |

---

## GPU Usage

| Task | GPUs | Time |
|------|------|------|
| Text training | 4 | 1-2 hours |
| VLM training | 4 | 2-4 hours |
| Text eval | 1 | 30 min |
| VLM eval | 2 | 1-2 hours |
| API calls | 0 | 1-2 hours (rate limited) |

---

## HPC Environment Notes

- **No GRES**: This HPC does not use `--gres=gpu:N` in SLURM. GPUs are allocated automatically based on partition.
- **SLURM Required**: Always submit GPU jobs via SLURM (`sbatch`). Do not run GPU jobs directly on login/compute nodes outside of SLURM.
- **GPU Allocation**: Use `CUDA_VISIBLE_DEVICES=0,1,2,3` to control which GPUs are used.
- **No nohup**: `/usr/bin/nohup` is permission denied. Use `bash script.sh &` instead of `nohup script.sh &`.
- **vLLM Multi-GPU Issues**: vLLM 0.13.0 has WorkerProc initialization failures with tensor_parallel_size > 1 on this system. Use TP=1 and run models that fit on single GPU, or use alternative inference backends.
- **Not root**: The user (`khayes`) is not the root user. Do not attempt `sudo`, `apt install`, or any other commands requiring root privileges. Use `pip install --user`, conda environments, or request the admin to install system packages.
- **NFS CPU Environment**: `/scratch-nfs/khayes/uq_cpu/` is a lightweight conda env (Python 3.11 + numpy, scipy, sklearn, pandas, transformers, tqdm, matplotlib, Pillow, jsonlines) on the shared NFS filesystem, visible from **all nodes** including debug partition. Use this for CPU-only SLURM jobs on debug nodes. In SLURM scripts, use the full Python path: `/scratch-nfs/khayes/uq_cpu/bin/python`. Note: `/scratch/` is local to gpunode00 and NOT visible from debug nodes — CPU jobs must read/write data via `/scratch-nfs/` or copy data there first.
- **Two-filesystem layout**:
  - `/scratch/khayes/LLM/` (gpunode00 local) — main workspace, GPU training, large data
  - `/scratch-nfs/khayes/LLM/` — mirror for CPU-only debug node jobs (sync data as needed)

---

## Agent & Development Rules

### GPU Discipline
- **Daytime (~6am-11pm ET, including weekends):** Default to **GPUs 0-4** (`CUDA_VISIBLE_DEVICES=0,1,2,3,4`). Others use this system during the day, so be considerate. If a job genuinely needs more, you may use **up to 6 GPUs** (0-5), but prefer 5.
- **Nighttime only (11pm-6am ET):** You may use **all 8 GPUs** (`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`). Verify no major jobs from other users are running first (check `squeue` and `nvidia-smi`). Jobs that use 8 GPUs must be timed to finish before 6am, or accept that they may overlap into daytime.
- When in doubt about the time, check with `date` and default to the conservative daytime policy.
- Do not spread work across arbitrary GPU indices. Always start from GPU 0 upward.

### CPU-Only Jobs: Use the Debug Partition Aggressively
- **For CPU-only tasks** (data preprocessing, feature extraction, grading, dataset assembly, JSON/JSONL processing, figure generation, analysis scripts), submit to the **`debug` partition** via SLURM.
- Debug nodes have **80 CPUs each**. Use them fully — always set `--cpus-per-task` to the maximum available (up to 80) unless the job genuinely doesn't benefit from parallelism.
- **Maximize parallelism in CPU scripts:**
  - Use `multiprocessing.Pool` or `concurrent.futures.ProcessPoolExecutor` with `max_workers=os.cpu_count()` (or the SLURM-allocated CPU count via `int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))`).
  - For pandas operations, consider `pandarallel` or chunked `multiprocessing`.
  - For I/O-bound tasks (API calls, downloads), use `concurrent.futures.ThreadPoolExecutor` or `asyncio` with high concurrency.
  - For embarrassingly parallel tasks across files/datasets, split work across workers — do not process sequentially.
  - Set `num_workers` in PyTorch `DataLoader` to match available CPUs.
- **SLURM template for CPU jobs (debug nodes):**
  ```bash
  #!/bin/bash
  #SBATCH --partition=debug
  #SBATCH --nodes=1
  #SBATCH --cpus-per-task=80
  #SBATCH --mem=64G
  #SBATCH --time=04:00:00
  #SBATCH --output=/scratch-nfs/khayes/LLM/logs/%x_%j.out

  cd /scratch-nfs/khayes/LLM
  export PYTHONUNBUFFERED=1

  /scratch-nfs/khayes/uq_cpu/bin/python scripts/your_script.py
  ```
  **Rules for debug node jobs:**
  - Debug nodes have **no GPUs** and **cannot see `/scratch/`**
  - All input data, scripts, and output paths must be on `/scratch-nfs/khayes/LLM/`
  - Sync data before submitting: `rsync -a /scratch/khayes/LLM/data/needed_dir/ /scratch-nfs/khayes/LLM/data/needed_dir/`
  - Copy results back after: `rsync -a /scratch-nfs/khayes/LLM/results/ /scratch/khayes/LLM/results/`
  - NFS is slower for heavy I/O — best for compute-bound CPU tasks (grading, analysis, contamination checks)
- **Multi-node CPU jobs:** For very large tasks (processing millions of examples), request multiple debug nodes and split work across them.
- CPU jobs do not require GPU discipline checks, but still check `squeue` to avoid overloading the scheduler.

### Check Before You Act: Protect Other Users' Work
- **Before submitting any SLURM job**, run `squeue -u $USER` and `squeue` (all users) to see what is already running. Do not submit jobs that would conflict with or starve other users' jobs.
- **Before killing any process**, verify it belongs to `khayes` and is one you launched. Never kill another user's process. Use `ps aux | grep <pattern>` or `squeue` to confirm ownership.
- **Before cancelling any SLURM job** (`scancel`), double-check the job ID belongs to `khayes` with `squeue -u $USER`. Never cancel another user's job.
- **Only cancel exactly what the user asks.** If the user says "cancel job X", cancel ONLY job X — do NOT cancel dependent jobs, related jobs, or anything else. Dependent jobs will remain pending with unsatisfied dependencies, which is fine. Never expand a cancel request to a chain of cancellations without explicit approval.
- **Never cancel jobs submitted by other agents** unless the user explicitly asks. Other agents' queued jobs represent planned work — destroying them wastes their effort.
- **Check GPU utilization** with `nvidia-smi` before submitting GPU jobs. If GPUs are already heavily loaded by other users, wait or use fewer GPUs.
- **Never run `kill -9` or `scancel` on a PID/job ID without first confirming it is yours.** When in doubt, ask the user.

### No Spending Without Approval
- **Never run any job that costs money without explicit user approval.** This includes API calls to paid services (OpenAI, Anthropic, etc.), cloud compute, or any other billed resource.
- Before launching API-based evaluations or inference (e.g., GPT-5, Claude), tell the user the estimated cost and number of calls, and wait for their go-ahead.
- This applies even if the user previously approved a similar job — each new run that incurs cost requires fresh approval.
- Smoke tests on paid APIs also require approval, since they still cost money.
- **BLOCKED: Claude and Gemini API calls require advisor approval before spending.** Do NOT generate Claude or Gemini responses (text or vision) until the user explicitly confirms advisor has approved the budget. This applies to all Claude/Gemini-related tasks in MASTER_PLAN.md (T5, V7, B3, Phase 4 Claude rows).
- **GPT-5 API calls: Use Micah's API key.** When making any GPT-5 / OpenAI API calls (inference, grading, etc.), use Micah's API key, not any other key.

### Smoke Test First, Then Scale
- **Every new script must include a smoke test mode.** Add a `--smoke_test` or `--max_examples N` flag that runs on a tiny subset (5-10 examples) in under 2 minutes.
- **The agent must always run a smoke test before submitting the full job.** This is mandatory, not optional. The workflow is:
  1. Write or prepare the script.
  2. Submit a smoke test via SLURM (e.g., `--max_examples 5`) and wait for it to complete.
  3. Check the smoke test output for errors. If it fails, fix and re-test.
  4. Only after a clean smoke test, submit the full-scale job.
- If writing a SLURM job script, include a commented-out smoke test invocation at the top.
- Do not skip the smoke test to save time. A failed full-scale job wastes far more time than a 2-minute smoke test.
- **Always report GPU count and estimated walltime to the user.** Before submitting the smoke test, state: how many GPUs, expected walltime. After the smoke test passes, before submitting the full job, state again: how many GPUs, expected walltime for the full run. This applies to every SLURM job (training, eval, inference). Example: "Smoke test: 2× A100, ~2 min. Full job: 4× A100, ~90 min."

### Long-Running Jobs: Monitor with Periodic Checks
- **After submitting a long-running SLURM job, the agent must periodically check on it.** Do not submit and forget. The workflow is:
  1. Submit the job with `sbatch` and note the job ID.
  2. Use `sleep 60 && squeue -u $USER` (or similar polling) to check status every 1-5 minutes.
  3. Once the job finishes or disappears from `squeue`, check the output log (`tail -50 logs/<logfile>`) and `sacct -j <jobid>` for exit status.
  4. If the job failed, examine the log, diagnose the error, and report to the user.
  5. If the job succeeded, report the results.
- **Polling pattern:** Use `sleep <seconds>` between checks. Start with 60-second intervals for short jobs, 2-5 minute intervals for longer jobs (training).
- For very long jobs (multi-hour training), check every 5 minutes. Tail the log to verify progress (e.g., loss decreasing, steps advancing).
- If a job is stuck (no log output for 10+ minutes, or GPU utilization at 0%), flag it to the user.

### SLURM Always
- **Always use SLURM (`sbatch`) for all GPU jobs.** SLURM handles scheduling, resource allocation, and logging. Do not run GPU jobs directly outside of SLURM.
- Write a SLURM batch script (`.sh`) for every GPU job — training, evaluation, inference — and submit it with `sbatch`.
- Never fall back to running GPU jobs directly with `CUDA_VISIBLE_DEVICES=... python ...` on login or shared nodes. If SLURM is down, wait or ask the user before proceeding.
- Always test SLURM availability at the start of a session (`squeue -u $USER` or `sinfo`).

### Performance: Maximize Compute Efficiency
GPU hours are expensive and shared. Always use the fastest, most efficient approach available. Do not write naive or unoptimized code when better tools exist.

**Inference:**
- **Always use vLLM** for local model inference. Never fall back to raw HuggingFace `model.generate()` unless vLLM is genuinely incompatible with the model.
- Use vLLM's batched inference and continuous batching — do not send one prompt at a time.
- For API-based models (OpenAI, Anthropic), use **async/concurrent requests** (`asyncio`, `aiohttp`, `concurrent.futures`) up to the rate limit. Never make sequential API calls when parallel is possible.
- Use the **batch API** (e.g., OpenAI Batch API) for large-scale jobs where latency is not critical.

**Training:**
- **Always use DDP (DistributedDataParallel)** or **FSDP** for multi-GPU training. Never use naive `DataParallel`.
- **Use DeepSpeed** (ZeRO Stage 2/3) or **HuggingFace Accelerate** for memory-efficient training of large models.
- Use **mixed precision** (`bf16` or `fp16`) by default. Do not train in full `fp32` unless there is a specific numerical stability reason.
- Use **gradient accumulation** to simulate larger batch sizes without increasing memory.
- Use **LoRA/QLoRA** for fine-tuning large models instead of full fine-tuning when possible.
- Use **Flash Attention 2** when available (`attn_implementation="flash_attention_2"` in HuggingFace).

**Data & I/O:**
- Leverage **batch processing** and **multiprocessing** (`num_workers > 0` in DataLoader) wherever possible.
- Pre-tokenize and cache datasets rather than tokenizing on-the-fly during training.
- Use **streaming** for large datasets that don't fit in memory.
- Write results incrementally to disk (JSONL append) rather than holding everything in memory until the end.

### Robustness: Checkpointing & Resumability
- **Always save model checkpoints** during training (e.g., every N steps or every epoch). Use `save_steps` / `save_strategy` in HuggingFace Trainer or equivalent in custom loops.
- **Support resuming from checkpoint.** Training scripts should accept a `--resume_from_checkpoint` flag so crashed or preempted jobs can pick up where they left off.
- **Save intermediate results** for long-running evaluations (e.g., flush predictions to disk periodically, not just at the end). If a job dies at 90%, we should not lose the first 90%.
- **Log progress** so it's easy to tell how far along a job is (e.g., tqdm, periodic print statements with step/total).

### Nightly Push
- **Every night, push current work to the private GitHub remote** so nothing is lost if the cluster has issues.
- **Push:** scripts, configs, notebooks, docs, paper source, small result summaries (RESEARCH_LOG.md, CSV/JSON metrics files).
- **Do NOT push:** model checkpoints, datasets, large generated outputs (STEP/STL files, rendered images), anything in `models/`, `data/`, `runs/`, or files over ~10 MB. These stay on the cluster only.
- Ensure `.gitignore` covers the large directories. If unsure whether something is too big, check file size first.
- Commit with a descriptive message. Do not force-push.

---

## Common Issues

### OOM During Training
- Reduce batch size to 1
- Use gradient accumulation
- Don't use gradient checkpointing with VLM (causes issues on backward pass)

### CUDA Context Corruption
- When running ablations, use subprocess mode
- Don't load multiple models in same process

### API Rate Limits
- GPT-5: ~100 RPM
- Add delays between calls
- Use batch API if available

### Dataset Caching
- Hugging Face datasets are cached in `/scratch/khayes/.cache/huggingface/datasets/`
- If a dataset update fails to download, the latest cached version is used
- MMMU uses subject-specific configs (Accounting, Art, Biology, etc.) - ensure all are downloaded

---

## Tier 1: Priority Benchmarks for UQ Training (~40-60% accuracy)

| Benchmark | CLI Name | GPT-5 Acc | Samples |
|-----------|----------|-----------|---------|
| BBEH | `bbeh` | ~50% | 4,500 |
| SimpleQA | `simpleqa` | 19-54% | 4,300 |
| GPQA Diamond | `gpqa` | 77-90% | 198 |
| HLE | `hle` | 25-30% | 2,000 |
| HealthBench | `healthbench` | ~60% | 1,000 |

---

*See [RESEARCH_LOG.md](RESEARCH_LOG.md) for detailed experiment history.*
*See [MASTER_PLAN.md](MASTER_PLAN.md) for current project roadmap.*
