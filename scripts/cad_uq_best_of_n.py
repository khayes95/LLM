#!/usr/bin/env python3
"""Best-of-N selection for CadQuery generation using the CAD UQ model.

Given a JSONL where each prompt has N candidate CadQuery programs,
score each candidate with P(correct) and select the highest-confidence one.

Input format (JSONL, one line per prompt):
    {
        "prompt": "Create a 10x10x10 cube with a 5mm hole through the center",
        "candidates": [
            {"code": "import cadquery as cq\nresult = cq.Workplane('XY')...", "source": "rollout_0"},
            {"code": "import cadquery as cq\nresult = cq.Workplane('XY')...", "source": "rollout_1"},
            ...
        ]
    }

Optional fields per candidate:
    - "label": ground-truth 0/1 (for evaluation, not needed for inference)
    - "source": identifier for the candidate

Output format (JSONL):
    {
        "prompt": "...",
        "selected_idx": 2,
        "selected_code": "...",
        "selected_score": 0.87,
        "all_scores": [0.23, 0.45, 0.87, 0.12],
        "oracle_correct": true  // only if labels provided
    }

Usage:
    # Score and select best candidates
    python scripts/cad_uq_best_of_n.py \
        --checkpoint uq_models/cad_uq_v1 \
        --input data/cad_candidates.jsonl \
        --output data/cad_best_of_n_results.jsonl

    # With ground truth labels for evaluation
    python scripts/cad_uq_best_of_n.py \
        --checkpoint uq_models/cad_uq_v1 \
        --input data/cad_candidates.jsonl \
        --output data/cad_best_of_n_results.jsonl \
        --evaluate
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

PROMPT_TEMPLATE = """Description: {prompt}

CadQuery Code:
```python
{code}
```

Is the code correct? (i) No (ii) Yes"""


def load_model(checkpoint_path: str):
    """Load the CAD UQ model from checkpoint."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)

    num_gpus = torch.cuda.device_count()
    max_memory = {i: "78GiB" for i in range(num_gpus)}

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        max_memory=max_memory, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    return model, processor


def score_single(model, processor, prompt: str, code: str, device) -> float:
    """Score a single (prompt, code) pair. Returns P(correct)."""
    placeholder = Image.new('RGB', (336, 336), color='gray')

    text = PROMPT_TEMPLATE.format(prompt=prompt[:800], code=code[:1200])
    messages = [{"role": "user", "content": [
        {"type": "image", "image": placeholder},
        {"type": "text", "text": text},
    ]}]

    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[chat_text], images=[placeholder], return_tensors="pt",
        padding=True, min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
    token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]
    probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
    return probs[1].item()


def main():
    parser = argparse.ArgumentParser(description="Best-of-N selection with CAD UQ model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained CAD UQ checkpoint")
    parser.add_argument("--input", type=str, required=True,
                        help="JSONL with prompts and candidate codes")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL with selection results")
    parser.add_argument("--evaluate", action="store_true",
                        help="Compute metrics if ground-truth labels are available")
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Max prompts to process (for smoke testing)")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model, processor = load_model(args.checkpoint)
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    # Load input data
    rows = []
    with open(args.input) as f:
        for line in f:
            rows.append(json.loads(line))
    if args.max_prompts:
        rows = rows[:args.max_prompts]
    print(f"Processing {len(rows)} prompts...")

    results = []
    total_candidates = 0

    # Tracking for evaluation
    uq_correct = 0
    random_correct = 0
    oracle_correct = 0
    n_with_labels = 0

    for i, row in enumerate(rows):
        prompt = row["prompt"]
        candidates = row["candidates"]
        total_candidates += len(candidates)

        if i % 10 == 0:
            print(f"  Prompt {i}/{len(rows)} ({len(candidates)} candidates)...")

        # Score all candidates
        scores = []
        for cand in candidates:
            try:
                s = score_single(model, processor, prompt, cand["code"], device)
            except Exception as e:
                print(f"  Error scoring candidate: {e}")
                s = 0.5
            scores.append(s)

        # Select best
        best_idx = int(np.argmax(scores))
        result = {
            "prompt": prompt,
            "selected_idx": best_idx,
            "selected_code": candidates[best_idx]["code"],
            "selected_score": scores[best_idx],
            "all_scores": scores,
            "n_candidates": len(candidates),
        }

        # Evaluation (if labels available)
        if args.evaluate:
            labels = [cand.get("label") for cand in candidates]
            if all(l is not None for l in labels):
                n_with_labels += 1
                result["labels"] = labels
                result["selected_label"] = labels[best_idx]
                result["oracle_has_correct"] = any(l == 1 for l in labels)

                if labels[best_idx] == 1:
                    uq_correct += 1
                if any(l == 1 for l in labels):
                    oracle_correct += 1
                    # Random baseline: P(correct) = n_correct / n_total
                    random_correct += sum(labels) / len(labels)

        results.append(result)

    # Write results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nResults written to {args.output}")
    print(f"Total: {len(results)} prompts, {total_candidates} candidates scored")

    if args.evaluate and n_with_labels > 0:
        print(f"\n{'=' * 50}")
        print("BEST-OF-N EVALUATION")
        print(f"{'=' * 50}")
        print(f"Prompts with labels: {n_with_labels}")
        print(f"Oracle (any correct):    {oracle_correct}/{n_with_labels} "
              f"({100 * oracle_correct / n_with_labels:.1f}%)")
        print(f"UQ selection (best):     {uq_correct}/{n_with_labels} "
              f"({100 * uq_correct / n_with_labels:.1f}%)")
        print(f"Random baseline (E[]):   {random_correct:.1f}/{n_with_labels} "
              f"({100 * random_correct / n_with_labels:.1f}%)")

        # Save summary
        summary = {
            "n_prompts": n_with_labels,
            "oracle_accuracy": oracle_correct / n_with_labels,
            "uq_accuracy": uq_correct / n_with_labels,
            "random_accuracy": random_correct / n_with_labels,
            "uq_lift_over_random": (uq_correct - random_correct) / n_with_labels,
        }
        summary_path = output_path.with_suffix(".summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
