# Pinocchio Project — Full Reconstruction Guide

**Purpose:** This document tells you (or a future Claude agent) how to reconstruct the entire Pinocchio project from its distributed backups, and what's stored where. If you're reading this, you probably lost access to the HPC and need to get everything set up on a new machine.

---

## 1. Where Everything Lives

### GitHub Repositories

| Repo | URL | Contains |
|------|-----|----------|
| **Main research repo** | `https://github.com/khayes95/LLM.git` | All code, scripts, configs, docs, small figures, website. Branch: `uq-finetuning` |
| **Pinocchio pip package** | `https://github.com/khayes95/pinocchio.git` | Installable package (`pinocchio-uq`), tests, CI/CD. Branch: `main` |
| **Upstream (Haosong)** | `https://github.com/Haosong-Zhang/LLM` | Shared repo with collaborator. Don't push here without coordinating. |

### HuggingFace (all owned by `KevinDavidHayes`)

| Repo | Type | Private? | Contains |
|------|------|----------|----------|
| **`pinocchio-0.8b`** | Model | Public | Release model — Qwen3.5-0.8B LoRA adapter for pip package |
| **`pinocchio-checkpoints`** | Model | **Private** | ALL model checkpoints: best_v3_qsplit (2.3GB), best_v2, best_unified, all ablation checkpoints (elicitation, lora_rank, multi_seed, modality, prompt, source_model, etc.) |
| **`pinocchio-data`** | Dataset | **Private** | ALL data: training images, scored data (v3_all, test_only_v3), results (test_only_v3), finetune splits (train/test_v3.jsonl), VLM features, finegrain UQ data |
| **`uq-eval-data`** | Dataset | **Private** | V1/V2 finetune splits and test-only results (legacy) |
| **`pinocchio-v2`** | Model | Public | Old v2 release model (superseded by pinocchio-0.8b) |
| **`pinocchio-qwen3-vl-8b-uq-v2`** | Model | Public | Old v2 VLM checkpoint |
| **`pinocchio`** | Space | Public | Gradio demo (sleeps when inactive, auto-wakes) |

### Overleaf (Paper)

| Item | URL | Auth |
|------|-----|------|
| **Paper project** | `https://www.overleaf.com/project/696e96cbc0f12ed169b15d1e` | — |
| **Git clone** | `https://git.overleaf.com/696e96cbc0f12ed169b15d1e` | Token: `olp_1Ddd7MIV1Ogl82tO26nefOjgxS3PgT20O0X3` |
| **Format** | ICML 2026 | Title: "Pinocchio: Estimating the Uncertainty of Black-Box Language Models" |

Clone command:
```bash
git clone https://git:olp_1Ddd7MIV1Ogl82tO26nefOjgxS3PgT20O0X3@git.overleaf.com/696e96cbc0f12ed169b15d1e overleaf
```

---

## 2. Reconstruction Steps

### Step 1: Clone the Code

```bash
# Main research repo
git clone https://github.com/khayes95/LLM.git
cd LLM
git checkout uq-finetuning

# Pinocchio package (as subdir or separate)
git clone https://github.com/khayes95/pinocchio.git pinocchio_package

# Paper
git clone https://git:olp_1Ddd7MIV1Ogl82tO26nefOjgxS3PgT20O0X3@git.overleaf.com/696e96cbc0f12ed169b15d1e overleaf
```

### Step 2: Download Model Checkpoints from HuggingFace

```bash
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download

# Best v3 model (current best — MOST IMPORTANT)
snapshot_download(
    'KevinDavidHayes/pinocchio-checkpoints',
    repo_type='model',
    local_dir='hf_checkpoints',
    allow_patterns=['uq_models_best_v3_qsplit/*'],
)

# To download ALL checkpoints (~28GB total):
# snapshot_download('KevinDavidHayes/pinocchio-checkpoints', repo_type='model', local_dir='hf_checkpoints')
"
```

**Checkpoint naming on HF:** Directories use underscores instead of slashes. Map back:
- `uq_models_best_v3_qsplit/` → `uq_models/best_v3_qsplit/`
- `data_ablations/*/` → `data/ablations/*/`

### Step 3: Download Data from HuggingFace

```bash
python -c "
from huggingface_hub import snapshot_download

# All pinocchio data (scored results, features, training images, finetune splits)
snapshot_download(
    'KevinDavidHayes/pinocchio-data',
    repo_type='dataset',
    local_dir='hf_data',
)

# Legacy v1/v2 data
snapshot_download(
    'KevinDavidHayes/uq-eval-data',
    repo_type='dataset',
    local_dir='hf_data_legacy',
)
"
```

**Data naming on HF:** Same underscore convention:
- `data_use_cases/scored_v3_all/` → `data/use_cases/scored_v3_all/`
- `data_finetune/train_v3.jsonl` → `data/finetune/train_v3.jsonl`
- `data_training_images/` → `data/training_images/`
- `data_features/` → `data/features/`

### Step 4: Restructure Downloaded Files

```bash
# Move checkpoints into expected structure
mv hf_checkpoints/uq_models_best_v3_qsplit uq_models/best_v3_qsplit
# ... repeat for other checkpoints

# Move data into expected structure
mv hf_data/data_use_cases/* data/use_cases/
mv hf_data/data_finetune/* data/finetune/
mv hf_data/data_training_images data/training_images
mv hf_data/data_features data/features
# ... etc
```

### Step 5: Set Up Python Environment

```bash
conda create -n uq_eval python=3.12
conda activate uq_eval
pip install -e .  # installs uq_eval package from pyproject.toml

# Key dependencies (check pyproject.toml for full list):
pip install torch transformers peft accelerate vllm
pip install scikit-learn pandas matplotlib seaborn tqdm
pip install openai  # for GPT-5 API calls
```

---

## 3. Key Results (For Reference)

### Best Model: v3 (clean question-level split)
- **Checkpoint:** `uq_models/best_v3_qsplit/`
- **Base model:** Qwen3-VL-8B-Instruct + LoRA (r=32, alpha=64)
- **Held-out AUROC:** 0.878 | VLM: 0.887 | Text: 0.867
- **Test-only AUROC:** 0.878 [0.863, 0.892] (1,953 samples, 0% question overlap)
- **Per-source:** GPT-5-mini 0.882, GPT-5.2 0.877, Qwen3.5 0.873
- **Training script:** `scripts/train_best_uq.py`

### Release Model: 0.8B
- **HF:** `KevinDavidHayes/pinocchio-0.8b`
- **Base:** Qwen3.5-0.8B + LoRA
- **Purpose:** Lightweight model for pip package / laptop deployment

### Key Ablation Results
- No-metadata: −5.0 pts (metadata matters)
- Truncation: 800 chars is sweet spot
- Word scramble: −9.7 pts (model reads semantics)
- Near-duplicate contamination: only −0.006 (not inflated)
- Training size: 5000 samples = 96% of full performance

---

## 4. Publishing Checklist (Not Yet Done)

### PyPI Package (`pinocchio-uq`)
- **Status:** Built locally, NOT on PyPI
- **How to publish:** Create a git tag on `khayes95/pinocchio` → GitHub Actions auto-publishes
  ```bash
  cd pinocchio_package
  git tag v0.1.0
  git push origin v0.1.0
  # GitHub Actions workflow (.github/workflows/) triggers pypa/gh-action-pypi-publish
  ```
- **Pre-requisite:** Set PyPI API token as GitHub secret (`PYPI_API_TOKEN`)
- **To store without publishing:** The wheel is in `pinocchio_package/dist/`. Just don't create the tag.

### HuggingFace Space
- **Status:** DEPLOYED and working (sleeps after 48h inactivity, auto-wakes)
- **URL:** `https://huggingface.co/spaces/KevinDavidHayes/pinocchio`
- **Code:** `website/huggingface-space/` (app.py, pinocchio_local.py, requirements.txt)
- **To update:** Push changes to the HF Space repo via `huggingface-cli` or the web UI

### Landing Page / Website
- **Status:** Single HTML file at `website/landing-page/index.html` — NOT deployed anywhere
- **Options to deploy:**
  1. GitHub Pages on `khayes95/pinocchio` (free, easiest)
  2. Any static hosting (Vercel, Netlify, etc.)
  3. HuggingFace Space (already has a Gradio app, could add static page)

### ArXiv
- **Paper:** In Overleaf, ICML 2026 format
- **Status:** Not submitted. All figures reference `arxiv.org/abs/TODO`
- **Before submission:** Update all numbers to v3, fix TODO links

---

## 5. What's NOT Backed Up (Only on HPC)

These are generally reproducible or not critical:

| Item | Size | Reproducible? |
|------|------|---------------|
| `runs/` | 9.7 GB | Yes — re-run benchmarks |
| `uq_models/size_ablation/` | 18 GB | Yes — re-run `train_best_uq.py` with size flags |
| Smoke test checkpoints | ~2 GB | Not needed |
| `data/use_cases/CONTAMINATED_*` | Various | Intentionally quarantined, do not need |
| HuggingFace cache | Large | Auto-downloads |

---

## 6. API Keys & Credentials

| Service | Key Location | Notes |
|---------|-------------|-------|
| **OpenAI (GPT-5)** | Use Micah's API key | Do NOT use personal key |
| **HuggingFace** | `huggingface-cli login` / `HF_TOKEN` env var | Account: `KevinDavidHayes` |
| **Overleaf** | Token in clone URL above | Read/write access |
| **GitHub** | SSH key or personal access token | Account: `khayes95` |

---

## 7. Critical Warnings

1. **Data leakage (FIXED in v3):** V1 and V2 results are INFLATED due to question-level leakage. Only v3 (`best_v3_qsplit`, `scored_test_only_v3`) uses clean splits. See CLAUDE.md for details.

2. **CONTAMINATED directories:** `data/use_cases/CONTAMINATED_*` — NEVER use for metrics.

3. **Model class:** The v3 model is `Qwen3VLForConditionalGeneration` (NOT `Qwen2_5_VL`). Don't use `attn_implementation="flash_attention_2"` unless you install flash-attn.

4. **Prompt template:** V3 uses the "combined" prompt variant. Always pass `--prompt_variant combined` when scoring with v2/v3 checkpoints.
