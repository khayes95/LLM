#!/usr/bin/env python3
"""
Cross-model transfer test with FIXED VSR image loading.

Tests if VLM judge trained on InternVL3-78B responses can evaluate
responses from Qwen2.5-VL-72B using REAL images.
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

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TransferSample:
    """Sample for cross-model transfer test."""
    question_id: str
    benchmark: str
    image: Optional[Image.Image]
    prompt: str
    response: str
    is_correct: bool


def get_vsr_image(row) -> Optional[Image.Image]:
    """Get image from VSR row, downloading from COCO if needed."""
    img = row.get("image")
    if isinstance(img, Image.Image):
        return img.convert("RGB")

    # Download from COCO URL
    url = row.get("image_link")
    if url:
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception:
            pass
    return None


def load_qwen_vl(model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct"):
    """Load Qwen2.5-VL for inference."""
    from transformers import Qwen2_5_VLForConditionalGeneration

    print(f"Loading {model_name}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    return model, processor


def load_vlm_judge(checkpoint_path: str = "data/vlm_judge_combined/checkpoint-788"):
    """Load the trained VLM judge."""
    from transformers import Qwen3VLForConditionalGeneration
    from peft import PeftModel

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


def generate_response_qwen(model, processor, image: Image.Image, prompt: str) -> str:
    """Generate response from Qwen2.5-VL."""
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return response.strip()


JUDGE_PROMPT = """Question: {question}
Answer: {response}
Is the answer correct? (i) No (ii) Yes"""


def get_p_correct(judge_model, judge_processor, image: Image.Image,
                  question: str, response: str) -> float:
    """Get P(correct) from VLM judge."""
    prompt = JUDGE_PROMPT.format(question=question, response=response)

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}
    ]

    text = judge_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = judge_processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    inputs = {k: v.to(judge_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = judge_model(**inputs)

    logits = outputs.logits[0, -1, :]
    token_i = judge_processor.tokenizer.encode("i", add_special_tokens=False)[-1]
    token_ii = judge_processor.tokenizer.encode("ii", add_special_tokens=False)[-1]
    probs = torch.softmax(logits[[token_i, token_ii]], dim=0)

    return probs[1].item()


def check_correctness_vsr(response: str, label: bool) -> bool:
    """Check VSR response correctness."""
    response_lower = response.lower().strip()
    if label:
        return "true" in response_lower and "false" not in response_lower
    else:
        return "false" in response_lower and "true" not in response_lower


def main():
    print("=" * 70)
    print("CROSS-MODEL TRANSFER TEST (VSR with FIXED images)")
    print("=" * 70)
    print("Source model (training): InternVL3-78B")
    print("Target model (testing):  Qwen2.5-VL-72B")
    print("=" * 70)

    max_samples = 200

    # Step 1: Verify image loading
    print("\n" + "=" * 50)
    print("STEP 1: Verify VSR image loading fix")
    print("=" * 50)

    ds = load_dataset("cambridgeltl/vsr_random", split="test")
    print(f"Loaded VSR dataset: {len(ds)} samples")

    # Test first 3 images
    n_real = 0
    n_gray = 0
    for i in range(min(5, len(ds))):
        img = get_vsr_image(ds[i])
        if img is not None:
            arr = np.array(img)
            print(f"  Sample {i}: shape={arr.shape}, mean={arr.mean():.1f}, std={arr.std():.1f} ✓")
            n_real += 1
        else:
            print(f"  Sample {i}: FAILED to load ✗")
            n_gray += 1

    if n_real < 3:
        print("\nERROR: Image loading still broken!")
        return
    print(f"\n✓ Image loading verified: {n_real}/5 real images loaded")

    # Step 2: Load models
    print("\n" + "=" * 50)
    print("STEP 2: Loading models")
    print("=" * 50)

    qwen_model, qwen_processor = load_qwen_vl()
    judge_model, judge_processor = load_vlm_judge()

    fallback_image = Image.new("RGB", (336, 336), color="gray")

    # Step 3: Run Qwen inference
    print("\n" + "=" * 50)
    print(f"STEP 3: Running Qwen2.5-VL-72B on {max_samples} VSR samples")
    print("=" * 50)

    samples = []
    n_images_loaded = 0
    n_images_fallback = 0

    for idx in tqdm(range(max_samples), desc="Generating"):
        row = ds[idx]

        # Get REAL image
        image = get_vsr_image(row)
        if image is not None:
            n_images_loaded += 1
        else:
            image = fallback_image
            n_images_fallback += 1

        caption = row.get("caption", "")
        label = row.get("label", 0)

        prompt = f'Is this statement about the image true or false? "{caption}"\nAnswer with just "True" or "False".'

        try:
            response = generate_response_qwen(qwen_model, qwen_processor, image, prompt)
            is_correct = check_correctness_vsr(response, label == 1)
        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            response = "Error"
            is_correct = False

        samples.append(TransferSample(
            question_id=f"vsr_{idx}",
            benchmark="vsr",
            image=image,
            prompt=prompt,
            response=response,
            is_correct=is_correct,
        ))

    print(f"\nImages: {n_images_loaded} real, {n_images_fallback} fallback")

    # Calculate Qwen accuracy
    n_correct = sum(1 for s in samples if s.is_correct)
    qwen_accuracy = n_correct / len(samples)
    print(f"Qwen2.5-VL-72B accuracy: {qwen_accuracy:.1%} ({n_correct}/{len(samples)})")

    # Show some sample responses
    print("\nSample responses:")
    for i in [0, 1, 2]:
        s = samples[i]
        label = "True" if ds[i]["label"] == 1 else "False"
        print(f"  [{i}] Caption: {ds[i]['caption'][:50]}...")
        print(f"      Response: {s.response[:30]}... | Correct: {s.is_correct} | Label: {label}")

    # Step 4: Evaluate with judge
    print("\n" + "=" * 50)
    print("STEP 4: Evaluating with VLM judge")
    print("=" * 50)

    predictions = []
    labels = []

    for sample in tqdm(samples, desc="Judging"):
        try:
            p_correct = get_p_correct(
                judge_model, judge_processor, sample.image,
                sample.prompt[:500], sample.response[:300]
            )
        except Exception as e:
            print(f"Judge error: {e}")
            p_correct = 0.5

        predictions.append(p_correct)
        labels.append(float(sample.is_correct))

    predictions = np.array(predictions)
    labels = np.array(labels)

    # Compute metrics
    auroc = roc_auc_score(labels, predictions)
    auprc = average_precision_score(labels, predictions)

    pred_correct = predictions > 0.5
    actual_correct = labels > 0.5
    accuracy = (pred_correct == actual_correct).mean()

    brier = np.mean((predictions - labels) ** 2)

    # Step 5: Report
    print("\n" + "=" * 70)
    print("CROSS-MODEL TRANSFER RESULTS")
    print("=" * 70)

    print(f"\n1. Image loading:")
    print(f"   Real images: {n_images_loaded}/{max_samples} ({100*n_images_loaded/max_samples:.1f}%)")

    print(f"\n2. Qwen2.5-VL-72B on VSR:")
    print(f"   Accuracy: {qwen_accuracy:.1%}")
    if qwen_accuracy < 0.50:
        print(f"   ⚠ Below 50% - may still have issues")
    else:
        print(f"   ✓ Above 50% - inference working")

    print(f"\n3. VLM Judge on Qwen responses:")
    print(f"   AUROC:    {auroc:.4f}")
    print(f"   AUPRC:    {auprc:.4f}")
    print(f"   Accuracy: {accuracy:.1%}")
    print(f"   Brier:    {brier:.4f}")

    print(f"\n4. P(correct) distribution:")
    print(f"   Mean: {predictions.mean():.3f}")
    print(f"   Std:  {predictions.std():.3f}")
    print(f"   Range: [{predictions.min():.3f}, {predictions.max():.3f}]")

    print("\n" + "-" * 50)
    print("Comparison:")
    print(f"  In-distribution VSR AUROC:  0.770 (judge on InternVL responses)")
    print(f"  Cross-model VSR AUROC:      {auroc:.3f} (judge on Qwen responses)")

    if auroc > 0.65:
        print("\n✓ Cross-model transfer works! Judge generalizes to different VLM.")
    elif auroc > 0.55:
        print("\n⚠ Partial transfer - judge has some generalization")
    else:
        print("\n✗ Poor transfer - judge may be overfitting to InternVL patterns")

    # Save results
    results = {
        "benchmark": "vsr",
        "source_model": "InternVL3-78B",
        "target_model": "Qwen2.5-VL-72B",
        "n_samples": len(samples),
        "n_real_images": n_images_loaded,
        "qwen_accuracy": float(qwen_accuracy),
        "judge_auroc": float(auroc),
        "judge_auprc": float(auprc),
        "judge_accuracy": float(accuracy),
        "judge_brier": float(brier),
        "p_correct_mean": float(predictions.mean()),
        "p_correct_std": float(predictions.std()),
    }

    save_path = "data/vlm_judge_combined/cross_model_transfer_fixed.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
