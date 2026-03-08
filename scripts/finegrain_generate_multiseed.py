#!/usr/bin/env python3
"""Generate N images per prompt from SDXL with different seeds.

Produces the dataset needed for best-of-N reward model evaluation.
Each prompt gets N images (default 8), saved as prompt_{pid}_seed_{s}.png.

Usage:
    # Smoke test (5 prompts, 4 seeds)
    CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_generate_multiseed.py \
        --smoke_test --n_seeds 4

    # Full run (760 prompts, 8 seeds = 6,080 images)
    CUDA_VISIBLE_DEVICES=0 python scripts/finegrain_generate_multiseed.py \
        --n_seeds 8 --output_dir data/finegrain_uq/bestofn_images
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Block xformers to avoid version mismatch crashes
sys.modules['xformers'] = None
sys.modules['xformers.ops'] = None

import torch
from PIL import Image
from tqdm import tqdm

METADATA_CSV = Path("/scratch/khayes/diff/t2i-finegrain/metadata.csv")
DEFAULT_OUTPUT = Path("data/finegrain_uq/bestofn_images")

# SDXL config
BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
TARGET_WIDTH = 1024
TARGET_HEIGHT = 1024
NUM_INFERENCE_STEPS = 30  # Faster than 50, still good quality


def load_unique_prompts(max_prompts=None):
    """Load unique prompts from FineGRAIN metadata."""
    prompts = {}
    with open(METADATA_CSV) as f:
        for row in csv.DictReader(f):
            pid = int(row["prompt_id"])
            if pid not in prompts:
                prompts[pid] = {
                    "prompt_id": pid,
                    "prompt_text": row["prompt_text"],
                    "failure_mode": row["failure_mode"],
                }
    # Sort by prompt_id
    prompt_list = sorted(prompts.values(), key=lambda x: x["prompt_id"])
    if max_prompts:
        prompt_list = prompt_list[:max_prompts]
    return prompt_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--n_seeds", type=int, default=8)
    parser.add_argument("--seed_start", type=int, default=1000,
                        help="First seed value (seeds = seed_start .. seed_start+n_seeds-1)")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--no_refiner", action="store_true",
                        help="Skip refiner for faster generation")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_prompts = 5 if args.smoke_test else None
    prompts = load_unique_prompts(max_prompts)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    total = len(prompts) * len(seeds)
    print(f"Generating {total} images ({len(prompts)} prompts x {len(seeds)} seeds)")
    print(f"Seeds: {seeds}")
    print(f"Output: {output_dir}")

    # Check what already exists (for resume)
    existing = set()
    for f in output_dir.glob("prompt_*_seed_*.png"):
        existing.add(f.stem)
    if existing:
        print(f"Found {len(existing)} existing images, will skip them")

    # Load SDXL
    from diffusers import DiffusionPipeline

    print(f"Loading SDXL base model...")
    pipe = DiffusionPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")
    pipe.enable_attention_slicing()
    print("SDXL loaded")

    # Generate
    metadata = []
    t0 = time.time()
    done = 0

    for prompt_info in tqdm(prompts, desc="Prompts"):
        pid = prompt_info["prompt_id"]
        text = prompt_info["prompt_text"]
        fm = prompt_info["failure_mode"]

        for seed in seeds:
            fname = f"prompt_{pid:05d}_seed_{seed}"
            if fname in existing:
                done += 1
                continue

            generator = torch.Generator(device="cuda").manual_seed(seed)
            try:
                image = pipe(
                    prompt=text,
                    height=TARGET_HEIGHT,
                    width=TARGET_WIDTH,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                ).images[0]

                out_path = output_dir / f"{fname}.png"
                image.save(out_path)

                metadata.append({
                    "prompt_id": pid,
                    "prompt_text": text,
                    "failure_mode": fm,
                    "seed": seed,
                    "image_path": str(out_path.resolve()),
                    "filename": f"{fname}.png",
                })

            except Exception as e:
                print(f"  Error generating pid={pid} seed={seed}: {e}")
                continue

            done += 1
            if done % 100 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                remaining = (total - done) / rate if rate > 0 else 0
                print(f"  {done}/{total} done, {rate:.1f} img/s, "
                      f"~{remaining/60:.0f} min remaining")

    # Save metadata
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "n_prompts": len(prompts),
            "n_seeds": len(seeds),
            "seeds": seeds,
            "n_images": len(metadata),
            "model": "sdxl",
            "guidance_scale": args.guidance_scale,
            "steps": NUM_INFERENCE_STEPS,
            "resolution": f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
            "images": metadata,
        }, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone! Generated {len(metadata)} images in {elapsed/60:.1f} min")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
