#!/usr/bin/env python3
"""
Test VLM judge on VSR with properly loaded images.
This tests if the existing checkpoint improves when given real images.
"""
import sys
import json
import requests
from io import BytesIO
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, average_precision_score

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent))


PROMPT_TEMPLATE = """Question: {question}
Answer: {response}
Is the answer correct? (i) No (ii) Yes"""


def load_vlm_judge(checkpoint_path: str = "data/vlm_judge_combined/checkpoint-788"):
    """Load the trained VLM judge."""
    base_model = "Qwen/Qwen3-VL-8B-Instruct"
    print(f"Loading VLM judge from {checkpoint_path}...")

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()
    return model, processor


def get_vsr_image(row) -> Optional[Image.Image]:
    """Get image from VSR row, downloading if needed."""
    img = row.get("image")
    if isinstance(img, Image.Image):
        return img.convert("RGB")

    # Download from COCO URL
    url = row.get("image_link")
    if url:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception:
            pass
    return None


def get_p_correct(model, processor, image: Image.Image,
                  question: str, response: str) -> float:
    """Get P(correct) from VLM judge."""
    prompt = PROMPT_TEMPLATE.format(question=question, response=response)

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]
    token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
    token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]
    probs = torch.softmax(logits[[token_i, token_ii]], dim=0)

    return probs[1].item()


def main():
    print("=" * 70)
    print("VSR EVALUATION WITH REAL IMAGES")
    print("=" * 70)

    # Load judge
    model, processor = load_vlm_judge()
    device = next(model.parameters()).device
    fallback_image = Image.new("RGB", (336, 336), color="gray")

    # Load VSR test data (from feature files)
    feature_dir = Path("data/features/vsr")
    test_ids_path = Path("data/probe_results/test_ids.json")

    with open(test_ids_path) as f:
        test_ids = {item["question_id"] for item in json.load(f)}

    # Load VSR dataset for images
    print("\nLoading VSR dataset...")
    ds = load_dataset("cambridgeltl/vsr_random", split="test")

    # Load test samples from feature files
    test_samples = []
    for pt_file in feature_dir.glob("*.pt"):
        data = torch.load(pt_file, weights_only=False)
        qid = data.get("question_id", "")
        if qid in test_ids:
            # Extract index from question_id (e.g., "vsr_vsr_test_42" -> 42)
            import re
            match = re.search(r'_(\d+)$', qid)
            if match:
                idx = int(match.group(1))
                test_samples.append({
                    "question_id": qid,
                    "prompt": data.get("prompt", ""),
                    "response": data.get("response", ""),
                    "is_correct": data.get("is_correct", False),
                    "dataset_index": idx,
                })

    print(f"Found {len(test_samples)} VSR test samples")

    # Evaluate with real images vs fallback
    print("\n--- Evaluating with REAL images ---")

    predictions_real = []
    predictions_gray = []
    labels = []
    n_real_loaded = 0

    for sample in tqdm(test_samples[:100], desc="Evaluating"):  # Limit for speed
        idx = sample["dataset_index"]
        row = ds[idx]

        # Get real image
        real_image = get_vsr_image(row)
        if real_image is not None:
            n_real_loaded += 1
        else:
            real_image = fallback_image

        try:
            # With real image
            p_real = get_p_correct(
                model, processor, real_image,
                sample["prompt"][:500], sample["response"][:300]
            )
            # With gray fallback
            p_gray = get_p_correct(
                model, processor, fallback_image,
                sample["prompt"][:500], sample["response"][:300]
            )
        except Exception as e:
            print(f"Error: {e}")
            p_real = 0.5
            p_gray = 0.5

        predictions_real.append(p_real)
        predictions_gray.append(p_gray)
        labels.append(float(sample["is_correct"]))

    predictions_real = np.array(predictions_real)
    predictions_gray = np.array(predictions_gray)
    labels = np.array(labels)

    # Compute metrics
    auroc_real = roc_auc_score(labels, predictions_real)
    auroc_gray = roc_auc_score(labels, predictions_gray)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nImages loaded: {n_real_loaded}/{len(test_samples[:100])}")
    print(f"\nAUROC with REAL images: {auroc_real:.4f}")
    print(f"AUROC with GRAY images: {auroc_gray:.4f}")
    print(f"Improvement: {auroc_real - auroc_gray:+.4f}")

    # P(correct) distribution
    print(f"\nP(correct) with REAL images: mean={predictions_real.mean():.3f}, std={predictions_real.std():.3f}")
    print(f"P(correct) with GRAY images: mean={predictions_gray.mean():.3f}, std={predictions_gray.std():.3f}")

    # Compare to training baseline
    print("\n" + "-" * 50)
    print("Comparison to training (which used gray images):")
    print(f"  Training VSR AUROC: 0.637 (with gray images)")
    print(f"  Current VSR AUROC:  {auroc_real:.3f} (with real images)")

    if auroc_real > 0.70:
        print("\n✓ Real images significantly improve VSR performance!")
    elif auroc_real > auroc_gray:
        print("\n⚠ Real images help somewhat, but model may need retraining")
    else:
        print("\n✗ Model doesn't benefit from real images - needs retraining")

    # Save results
    results = {
        "n_samples": len(labels),
        "n_real_images": n_real_loaded,
        "auroc_real": float(auroc_real),
        "auroc_gray": float(auroc_gray),
        "improvement": float(auroc_real - auroc_gray),
        "p_correct_real_mean": float(predictions_real.mean()),
        "p_correct_real_std": float(predictions_real.std()),
        "p_correct_gray_mean": float(predictions_gray.mean()),
        "p_correct_gray_std": float(predictions_gray.std()),
    }

    save_path = "data/vlm_judge_combined/vsr_real_images_test.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
