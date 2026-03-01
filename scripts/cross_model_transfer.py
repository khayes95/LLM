#!/usr/bin/env python3
"""
Cross-model transfer test for VLM Judge.

Tests if VLM judge trained on InternVL3-78B responses can evaluate
responses from a different model (Qwen2.5-VL-72B).

This tests generalization: can the judge evaluate correctness regardless
of which model generated the response?
"""
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, average_precision_score

# VLM Judge model
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
            max_new_tokens=256,
            do_sample=False,
        )

    # Decode only new tokens
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
        max_pixels=256 * 28 * 28,
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


def check_correctness_mcq(response: str, correct_answer: str) -> bool:
    """Check MCQ response correctness."""
    response_upper = response.upper().strip()
    correct_upper = correct_answer.upper().strip()

    # Check if correct answer letter appears prominently
    if correct_upper in ["A", "B", "C", "D", "E"]:
        # Look for answer letter at start or after common patterns
        import re
        patterns = [
            rf"^{correct_upper}\b",
            rf"answer[:\s]+{correct_upper}\b",
            rf"\({correct_upper}\)",
            rf"option {correct_upper}\b",
        ]
        for pattern in patterns:
            if re.search(pattern, response_upper, re.IGNORECASE):
                return True

    return correct_upper in response_upper


def run_transfer_test(
    benchmark: str = "vsr",
    max_samples: int = 100,
    save_path: Optional[str] = None,
):
    """Run cross-model transfer test."""
    print("=" * 70)
    print(f"CROSS-MODEL TRANSFER TEST: {benchmark.upper()}")
    print("=" * 70)
    print("Source model (training): InternVL3-78B")
    print("Target model (testing):  Qwen2.5-VL-72B")
    print("=" * 70)

    # Load models
    print("\n1. Loading models...")
    qwen_model, qwen_processor = load_qwen_vl()
    judge_model, judge_processor = load_vlm_judge()

    fallback_image = Image.new("RGB", (224, 224), color=(128, 128, 128))

    # Load benchmark dataset
    print(f"\n2. Loading {benchmark} dataset...")
    if benchmark == "vsr":
        ds = load_dataset("cambridgeltl/vsr_random", split="test")

        # Sample
        indices = list(range(min(max_samples, len(ds))))
        samples = []

        print(f"\n3. Running Qwen2.5-VL-72B inference on {len(indices)} samples...")
        import requests
        from io import BytesIO

        for idx in tqdm(indices, desc="Generating"):
            row = ds[idx]
            # Handle image - VSR stores filename + image_link URL
            img = row.get("image")
            image = None
            if isinstance(img, Image.Image):
                image = img.convert("RGB")
            else:
                # Try downloading from image_link (COCO URL)
                url = row.get("image_link")
                if url:
                    try:
                        resp = requests.get(url, timeout=10)
                        if resp.status_code == 200:
                            image = Image.open(BytesIO(resp.content)).convert("RGB")
                    except Exception as e:
                        pass
            if image is None:
                image = fallback_image
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

    elif benchmark == "mmmu":
        ds = load_dataset("MMMU/MMMU", "Art", split="validation")

        indices = list(range(min(max_samples, len(ds))))
        samples = []

        print(f"\n3. Running Qwen2.5-VL-72B inference on {len(indices)} samples...")
        for idx in tqdm(indices, desc="Generating"):
            row = ds[idx]

            # Get first image
            image = None
            for i in range(1, 8):
                img = row.get(f"image_{i}")
                if img is not None:
                    image = img.convert("RGB")
                    break
            if image is None:
                image = fallback_image

            question = row.get("question", "")
            options = row.get("options", [])
            answer = row.get("answer", "")

            # Format MCQ prompt
            options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
            prompt = f"{question}\n\n{options_text}\n\nAnswer with just the letter (A, B, C, or D)."

            try:
                response = generate_response_qwen(qwen_model, qwen_processor, image, prompt)
                is_correct = check_correctness_mcq(response, answer)
            except Exception as e:
                print(f"Error on sample {idx}: {e}")
                response = "Error"
                is_correct = False

            samples.append(TransferSample(
                question_id=f"mmmu_{idx}",
                benchmark="mmmu",
                image=image,
                prompt=prompt,
                response=response,
                is_correct=is_correct,
            ))

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    # Calculate Qwen accuracy
    n_correct = sum(1 for s in samples if s.is_correct)
    qwen_accuracy = n_correct / len(samples)
    print(f"\nQwen2.5-VL-72B accuracy: {qwen_accuracy:.1%} ({n_correct}/{len(samples)})")

    # Now evaluate with VLM judge
    print(f"\n4. Evaluating Qwen responses with VLM judge...")
    predictions = []
    labels = []

    for sample in tqdm(samples, desc="Judging"):
        try:
            p_correct = get_p_correct(
                judge_model, judge_processor, sample.image,
                sample.prompt[:500], sample.response[:300]
            )
        except Exception as e:
            print(f"Judge error on {sample.question_id}: {e}")
            p_correct = 0.5

        predictions.append(p_correct)
        labels.append(float(sample.is_correct))

    predictions = np.array(predictions)
    labels = np.array(labels)

    # Compute metrics
    auroc = roc_auc_score(labels, predictions)
    auprc = average_precision_score(labels, predictions)

    # Accuracy at threshold 0.5
    pred_correct = predictions > 0.5
    actual_correct = labels > 0.5
    accuracy = (pred_correct == actual_correct).mean()

    # Brier score
    brier = np.mean((predictions - labels) ** 2)

    print("\n" + "=" * 70)
    print("CROSS-MODEL TRANSFER RESULTS")
    print("=" * 70)
    print(f"\nQwen2.5-VL-72B accuracy on {benchmark}: {qwen_accuracy:.1%}")
    print(f"\nVLM Judge performance on Qwen responses:")
    print(f"  AUROC:    {auroc:.4f}")
    print(f"  AUPRC:    {auprc:.4f}")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Brier:    {brier:.4f}")

    # Compare to in-distribution (InternVL) if we have it
    print("\n" + "-" * 40)
    print("Comparison to in-distribution (InternVL3-78B):")
    print("  VSR AUROC:  0.637 (from training data model)")
    print(f"  {benchmark.upper()} AUROC: {auroc:.3f} (cross-model transfer)")

    if auroc > 0.6:
        print("\n✓ VLM judge generalizes to different source model!")
    else:
        print("\n⚠ Judge may be overfitting to InternVL response patterns")

    # Save results
    if save_path:
        results = {
            "benchmark": benchmark,
            "source_model": "InternVL3-78B",
            "target_model": "Qwen2.5-VL-72B",
            "n_samples": len(samples),
            "qwen_accuracy": qwen_accuracy,
            "judge_auroc": auroc,
            "judge_auprc": auprc,
            "judge_accuracy": accuracy,
            "judge_brier": brier,
            "predictions": predictions.tolist(),
            "labels": labels.tolist(),
        }
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_path}")

    return {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": accuracy,
        "brier": brier,
        "qwen_accuracy": qwen_accuracy,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="vsr", choices=["vsr", "mmmu"])
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--save_path", default="data/vlm_judge_combined/cross_model_transfer.json")
    args = parser.parse_args()

    run_transfer_test(
        benchmark=args.benchmark,
        max_samples=args.max_samples,
        save_path=args.save_path,
    )
