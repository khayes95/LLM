#!/usr/bin/env python3
"""Score perturbations with the unified UQ model (Qwen3-VL-8B + LoRA).

Loads perturbations from JSONL, scores each with the UQ judge, and writes
per-perturbation P(correct) to output JSONL. Supports multi-GPU via
separate subprocesses (one model replica per GPU).

Usage:
    # Smoke test (100 perturbations, 1 GPU)
    CUDA_VISIBLE_DEVICES=0 python scripts/score_perturbations.py --smoke_test

    # Full run, 4 GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/score_perturbations.py --num_gpus 4

    # Resume from partial run
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/score_perturbations.py --num_gpus 4 --resume
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp

# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_CHECKPOINT = "uq_models/best_unified"
DEFAULT_INPUT = "data/use_cases/perturbations/all_perturbations.jsonl"
DEFAULT_OUTPUT = "data/use_cases/perturbations/scored_perturbations.jsonl"

PROMPT_TEMPLATE = """Question: {question}

Answer: {response}

Is the answer correct? (i) No (ii) Yes"""

Q_TRUNCATION = 500
R_TRUNCATION = 300


# ============================================================
# DATA LOADING
# ============================================================

def load_perturbations(input_path: str, smoke_test: bool = False) -> list:
    """Load perturbations from JSONL file."""
    perturbations = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            perturbations.append(record)
            if smoke_test and len(perturbations) >= 100:
                break
    return perturbations


def load_already_scored(output_path: str) -> set:
    """Load perturbation_ids that have already been scored (for --resume)."""
    scored_ids = set()
    if not os.path.exists(output_path):
        return scored_ids
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                pid = record.get("perturbation_id")
                if pid:
                    scored_ids.add(pid)
            except json.JSONDecodeError:
                continue
    return scored_ids


# ============================================================
# INFERENCE (per-GPU worker)
# ============================================================

def score_chunk(gpu_id: int, chunk: list, checkpoint_path: str,
                temp_dir: str, worker_id: int):
    """Score a chunk of perturbations on a single GPU.

    Each worker loads its own model replica, scores its chunk,
    and writes results to a temp file.
    """
    # Set CUDA device for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    from PIL import Image
    from peft import PeftModel
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    device = torch.device("cuda:0")
    temp_path = os.path.join(temp_dir, f"worker_{worker_id}.jsonl")

    print(f"[Worker {worker_id}] GPU {gpu_id}: Loading model...")
    t0 = time.time()

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()
    processor = AutoProcessor.from_pretrained(BASE_MODEL)

    print(f"[Worker {worker_id}] Model loaded in {time.time() - t0:.1f}s. "
          f"Scoring {len(chunk)} perturbations...")

    # Gray placeholder image (all perturbations are text-only)
    gray_image = Image.new('RGB', (224, 224), color='gray')

    # Get token IDs for "i" (No) and "ii" (Yes)
    token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
    token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]

    start_time = time.time()
    scored = 0

    with open(temp_path, "w") as f_out:
        for idx, record in enumerate(chunk):
            question = record.get("question", "")[:Q_TRUNCATION]
            response = record.get("response", "")[:R_TRUNCATION]

            prompt = PROMPT_TEMPLATE.format(
                question=question,
                response=response,
            )
            messages = [{"role": "user", "content": [
                {"type": "image", "image": gray_image},
                {"type": "text", "text": prompt},
            ]}]

            try:
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(
                    text=[text], images=[gray_image],
                    return_tensors="pt", padding=True,
                    min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)

                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
                p_correct = probs[1].item()
            except Exception as e:
                if idx < 5:
                    print(f"[Worker {worker_id}] Error on sample {idx} "
                          f"({record.get('perturbation_id', '?')}): {e}")
                p_correct = 0.5

            out_record = {
                "original_id": record.get("original_id", ""),
                "perturbation_type": record.get("perturbation_type", ""),
                "variant_idx": record.get("variant_idx", 0),
                "perturbation_id": record.get("perturbation_id", ""),
                "p_correct": round(p_correct, 6),
                "benchmark": record.get("benchmark", ""),
                "target_model": record.get("target_model", ""),
                "is_correct": record.get("is_correct", -1),
            }
            f_out.write(json.dumps(out_record) + "\n")
            scored += 1

            # Progress every 100 samples
            if scored % 100 == 0:
                elapsed = time.time() - start_time
                rate = scored / elapsed if elapsed > 0 else 0
                remaining = len(chunk) - scored
                eta = remaining / rate if rate > 0 else 0
                print(f"[Worker {worker_id}] [{scored}/{len(chunk)}] "
                      f"{rate:.1f} samples/s, ETA {eta/60:.1f}m")
                f_out.flush()

            # Intermediate flush every 500 samples
            if scored % 500 == 0:
                f_out.flush()

    elapsed = time.time() - start_time
    rate = scored / elapsed if elapsed > 0 else 0
    print(f"[Worker {worker_id}] Done: {scored} samples in {elapsed:.0f}s "
          f"({rate:.1f}/s). Output: {temp_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Score perturbations with unified UQ model"
    )
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Input JSONL of perturbations")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output JSONL with scores")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Path to LoRA checkpoint")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs to use (parallel workers)")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Score first 100 perturbations only")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-scored perturbation_ids")
    args = parser.parse_args()

    print("=" * 70)
    print("Score Perturbations with Unified UQ Model")
    print("=" * 70)
    print(f"Input:      {args.input}")
    print(f"Output:     {args.output}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Num GPUs:   {args.num_gpus}")
    print(f"Smoke test: {args.smoke_test}")
    print(f"Resume:     {args.resume}")
    print()

    # Resolve checkpoint path (find best checkpoint subdir if available)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint {args.checkpoint} does not exist!")
        sys.exit(1)

    checkpoint_subdirs = sorted(checkpoint_path.glob("checkpoint-*"))
    if checkpoint_subdirs:
        lora_path = str(checkpoint_subdirs[-1])
        print(f"Using checkpoint: {lora_path}")
    else:
        lora_path = str(checkpoint_path)
        print(f"Using adapter dir: {lora_path}")

    # Load perturbations
    print(f"\nLoading perturbations from {args.input}...")
    perturbations = load_perturbations(args.input, smoke_test=args.smoke_test)
    print(f"Loaded {len(perturbations)} perturbations")

    if not perturbations:
        print("ERROR: No perturbations loaded!")
        sys.exit(1)

    # Resume: filter out already-scored
    if args.resume:
        scored_ids = load_already_scored(args.output)
        if scored_ids:
            before = len(perturbations)
            perturbations = [
                p for p in perturbations
                if p.get("perturbation_id") not in scored_ids
            ]
            print(f"Resume: {len(scored_ids)} already scored, "
                  f"{before} -> {len(perturbations)} remaining")
        else:
            print("Resume: No existing output found, starting fresh")

    if not perturbations:
        print("All perturbations already scored. Nothing to do.")
        return

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine available GPU IDs from CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        available_gpus = [int(g) for g in cuda_visible.strip().split(",")]
    else:
        available_gpus = list(range(torch.cuda.device_count()))

    num_gpus = min(args.num_gpus, len(available_gpus))
    if num_gpus < 1:
        print("ERROR: No GPUs available!")
        sys.exit(1)
    print(f"\nUsing {num_gpus} GPU(s): {available_gpus[:num_gpus]}")

    # Split perturbations into chunks
    chunk_size = (len(perturbations) + num_gpus - 1) // num_gpus
    chunks = []
    for i in range(num_gpus):
        start = i * chunk_size
        end = min(start + chunk_size, len(perturbations))
        if start < len(perturbations):
            chunks.append(perturbations[start:end])
    print(f"Split into {len(chunks)} chunks: "
          f"{[len(c) for c in chunks]}")

    # Create temp directory for worker outputs
    temp_dir = tempfile.mkdtemp(
        prefix="score_perturbations_",
        dir=str(output_path.parent)
    )
    print(f"Temp dir: {temp_dir}")

    # Launch workers
    start_time = time.time()

    if num_gpus == 1:
        # Single GPU: run in current process (simpler debugging)
        score_chunk(
            gpu_id=available_gpus[0],
            chunk=chunks[0],
            checkpoint_path=lora_path,
            temp_dir=temp_dir,
            worker_id=0,
        )
    else:
        # Multi-GPU: spawn separate processes
        mp.set_start_method("spawn", force=True)
        processes = []
        for i, chunk in enumerate(chunks):
            p = mp.Process(
                target=score_chunk,
                args=(available_gpus[i], chunk, lora_path, temp_dir, i),
            )
            p.start()
            processes.append(p)
            print(f"Launched worker {i} (PID {p.pid}) on GPU {available_gpus[i]} "
                  f"with {len(chunk)} samples")

        # Wait for all workers
        for i, p in enumerate(processes):
            p.join()
            if p.exitcode != 0:
                print(f"WARNING: Worker {i} exited with code {p.exitcode}")

    elapsed = time.time() - start_time
    print(f"\nAll workers finished in {elapsed:.0f}s")

    # Merge temp files into final output
    print(f"\nMerging worker outputs...")
    mode = "a" if args.resume else "w"
    total_merged = 0

    with open(args.output, mode) as f_out:
        for i in range(len(chunks)):
            temp_path = os.path.join(temp_dir, f"worker_{i}.jsonl")
            if not os.path.exists(temp_path):
                print(f"WARNING: Missing temp file for worker {i}: {temp_path}")
                continue
            with open(temp_path) as f_in:
                for line in f_in:
                    f_out.write(line)
                    total_merged += 1
            # Clean up temp file
            os.remove(temp_path)

    # Clean up temp dir
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass

    total_time = time.time() - start_time
    rate = total_merged / total_time if total_time > 0 else 0
    print(f"\n{'=' * 70}")
    print(f"SCORING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Scored:  {total_merged} perturbations")
    print(f"Time:    {total_time:.0f}s ({rate:.1f} samples/s)")
    print(f"Output:  {args.output}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
