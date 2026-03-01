#!/usr/bin/env python3
"""
Cross-model transfer test across ALL vision benchmarks.
Tests if VLM judge trained on InternVL3-78B generalizes to Qwen2.5-VL-72B.

Benchmarks: VSR, CharXiv, MMMU, HallusionBench
(ERQA excluded - multi-image benchmark, 28% samples need multiple images)
"""
import sys
import json
import requests
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from peft import PeftModel

# VLM Judge model
from transformers import Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent.parent))


PROMPT_TEMPLATE = """Question: {question}
Answer: {response}
Is the answer correct? (i) No (ii) Yes"""


@dataclass
class Sample:
    question_id: str
    benchmark: str
    image: Image.Image
    prompt: str
    response: str
    is_correct: bool


def load_qwen_model():
    """Load Qwen2.5-VL-72B for inference."""
    model_name = "Qwen/Qwen2.5-VL-72B-Instruct"
    print(f"Loading {model_name}...")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


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


def generate_response_qwen(model, processor, image: Image.Image, prompt: str) -> str:
    """Generate response from Qwen model."""
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

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


def get_p_correct(model, processor, image: Image.Image, question: str, response: str) -> float:
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


def load_benchmark_samples(benchmark: str, max_samples: int = 200) -> List[Dict]:
    """Load samples from a benchmark."""
    samples = []
    fallback_image = Image.new("RGB", (336, 336), color="gray")

    if benchmark == "vsr":
        ds = load_dataset("cambridgeltl/vsr_random", split="test")
        for idx in range(min(max_samples, len(ds))):
            row = ds[idx]
            image = get_vsr_image(row)
            if image is None:
                image = fallback_image
            caption = row.get("caption", "")
            label = row.get("label", 0)
            samples.append({
                "question_id": f"vsr_{idx}",
                "benchmark": "vsr",
                "image": image,
                "prompt": f'Is this statement about the image true or false? "{caption}"\nAnswer with just "True" or "False".',
                "ground_truth": "True" if label == 1 else "False",
                "label": label == 1,
            })

    elif benchmark == "charxiv":
        ds = load_dataset("princeton-nlp/CharXiv", split="validation")
        sample_idx = 0
        for idx in range(len(ds)):
            if sample_idx >= max_samples:
                break
            row = ds[idx]
            image = row.get("image")
            if isinstance(image, Image.Image):
                image = image.convert("RGB")
            else:
                image = fallback_image

            # CharXiv has reasoning_q/reasoning_a fields, not question/answer
            question = row.get("reasoning_q", "")
            answer = row.get("reasoning_a", "")

            # Skip if question or answer is empty
            if not question or not answer:
                continue

            samples.append({
                "question_id": f"charxiv_{idx}",
                "benchmark": "charxiv",
                "image": image,
                "prompt": question,
                "ground_truth": answer,
                "label": None,
            })
            sample_idx += 1

    elif benchmark == "mmmu":
        # Use multiple subjects to get a representative sample
        # validation splits have 30 samples each, so combine several subjects
        subjects = ["Art", "Math", "Physics", "Computer_Science", "Biology", "Chemistry", "Economics"]
        all_samples = []
        import ast

        for subject in subjects:
            try:
                ds = load_dataset("MMMU/MMMU", subject, split="validation")
                for idx in range(len(ds)):
                    row = ds[idx]
                    # MMMU has image_1, image_2, etc.
                    image = row.get("image_1")
                    if isinstance(image, Image.Image):
                        image = image.convert("RGB")
                    else:
                        image = fallback_image
                    question = row.get("question", "")

                    # Options is stored as a string representation of a list
                    options_raw = row.get("options", "[]")
                    if isinstance(options_raw, str):
                        try:
                            options = ast.literal_eval(options_raw)
                        except:
                            options = []
                    else:
                        options = options_raw if options_raw else []

                    answer = row.get("answer", "")

                    # Skip if no valid answer (test split has "?")
                    if not answer or answer == "?":
                        continue

                    # Check if this is a multiple choice question (answer is a letter)
                    is_mcq = answer.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" and len(answer) == 1

                    # Format as MCQ
                    if options and is_mcq:
                        # Generate letter labels for all options
                        opts_text = "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)])
                        full_prompt = f"{question}\n\n{opts_text}\n\nAnswer with the option's letter from the given choices directly."
                    else:
                        # Open-ended question
                        full_prompt = f"{question}\n\nAnswer the question using a single word or phrase."

                    all_samples.append({
                        "question_id": f"mmmu_{subject}_{idx}",
                        "benchmark": "mmmu",
                        "image": image,
                        "prompt": full_prompt,
                        "ground_truth": answer,
                        "subject": subject,
                        "is_mcq": is_mcq,
                        "n_options": len(options) if options else 0,
                        "label": None,
                    })
            except Exception as e:
                print(f"Warning: Could not load MMMU/{subject}: {e}")

        # Take up to max_samples, shuffled for diversity
        import random
        random.seed(42)
        random.shuffle(all_samples)
        samples.extend(all_samples[:max_samples])

    elif benchmark == "hallusionbench":
        ds = load_dataset("lmms-lab/HallusionBench", split="image")
        for idx in range(min(max_samples, len(ds))):
            row = ds[idx]
            image = row.get("image")
            if isinstance(image, Image.Image):
                image = image.convert("RGB")
            else:
                image = fallback_image
            question = row.get("question", "")
            answer = row.get("gt_answer", "")
            samples.append({
                "question_id": f"hallusionbench_{idx}",
                "benchmark": "hallusionbench",
                "image": image,
                "prompt": f"{question}\nAnswer with Yes or No.",
                "ground_truth": answer,
                "label": None,
            })

    elif benchmark == "mathvista":
        ds = load_dataset("AI4Math/MathVista", split="testmini")
        for idx in range(min(max_samples, len(ds))):
            row = ds[idx]
            image = row.get("decoded_image") or row.get("image")
            if isinstance(image, Image.Image):
                image = image.convert("RGB")
            else:
                image = fallback_image
            question = row.get("question", "")
            choices = row.get("choices", [])
            answer = row.get("answer", "")
            question_type = row.get("question_type", "")

            # Format MCQ if choices exist
            if choices:
                opts_text = "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(choices)])
                full_prompt = f"{question}\n\n{opts_text}\n\nAnswer with just the letter."
            else:
                full_prompt = f"{question}\n\nProvide your answer."

            samples.append({
                "question_id": f"mathvista_{idx}",
                "benchmark": "mathvista",
                "image": image,
                "prompt": full_prompt,
                "ground_truth": answer,
                "choices": choices,  # Store choices for grading
                "question_type": question_type,
                "label": None,
            })

    elif benchmark == "realworldqa":
        ds = load_dataset("xai-org/RealworldQA", split="test")
        for idx in range(min(max_samples, len(ds))):
            row = ds[idx]
            image = row.get("image")
            if isinstance(image, Image.Image):
                image = image.convert("RGB")
            else:
                image = fallback_image
            question = row.get("question", "")
            answer = row.get("answer", "")
            samples.append({
                "question_id": f"realworldqa_{idx}",
                "benchmark": "realworldqa",
                "image": image,
                "prompt": question,
                "ground_truth": answer,
                "label": None,
            })

    return samples


def check_correctness(benchmark: str, response: str, ground_truth: str, choices: list = None) -> bool:
    """Check if response is correct for a benchmark."""
    response = response.strip().lower()
    ground_truth_lower = ground_truth.strip().lower()

    if benchmark == "vsr":
        resp_true = "true" in response and "false" not in response
        resp_false = "false" in response and "true" not in response
        gt_true = ground_truth_lower == "true"
        if resp_true:
            return gt_true
        elif resp_false:
            return not gt_true
        return False

    elif benchmark == "charxiv":
        # Exact match or contains
        return ground_truth_lower in response or response in ground_truth_lower

    elif benchmark == "mmmu":
        # Extract letter answer - MMMU uses letter answers (A-Z for MCQ, or free-form)
        # Must use regex to find standalone letters, not letters within words
        import re

        # Check if ground truth is a single letter (MCQ) or free-form
        if len(ground_truth_lower) == 1 and ground_truth_lower in "abcdefghijklmnopqrstuvwxyz":
            # MCQ: Extract letter from response
            # Patterns to match (in order of priority):
            patterns = [
                r'^[^a-z]*([a-z])\b',           # Letter at start
                r'answer[:\s]+([a-z])\b',       # answer is/: X
                r'(?:should|would|must)\s+(?:be|say)\s+([a-z])\b',  # should be X
                r'\b([a-z])\s+is\s+(?:correct|right|the)',  # X is correct/right/the
                r'\bis\s+([a-z])\b',            # is X (at word boundary)
            ]
            for pattern in patterns:
                match = re.search(pattern, response[:150], re.IGNORECASE)
                if match:
                    letter = match.group(1).lower()
                    return letter == ground_truth_lower
            # Fallback: check if just a single letter
            if len(response.strip()) == 1 and response.strip() in "abcdefghijklmnopqrstuvwxyz":
                return response.strip() == ground_truth_lower
            return False
        else:
            # Free-form answer: check for containment or numeric match
            # Try numeric comparison first
            gt_nums = re.findall(r'[\d.]+', ground_truth_lower)
            resp_nums = re.findall(r'[\d.]+', response)
            if gt_nums and resp_nums:
                try:
                    return abs(float(gt_nums[0]) - float(resp_nums[0])) < 0.01
                except:
                    pass
            # Text containment
            return ground_truth_lower in response or response in ground_truth_lower

    elif benchmark == "hallusionbench":
        resp_yes = "yes" in response and "no" not in response
        resp_no = "no" in response and "yes" not in response
        # Handle different ground truth formats: "yes", "1", "true" all mean yes
        gt_yes = ground_truth_lower in ["yes", "1", "true"]
        if resp_yes:
            return gt_yes
        elif resp_no:
            return not gt_yes
        return False

    elif benchmark == "mathvista":
        # MathVista has mixed format:
        # - MCQ: ground_truth is the TEXT of correct choice (e.g., "145°"), not the letter
        # - Free-form: ground_truth is numeric/text answer

        if choices:
            # MCQ: Extract letter from response and map to choice text
            import re
            # Look for letter at start or after common patterns
            match = re.search(r'^[^a-z]*([a-e])\b', response[:50])
            if match:
                letter_idx = ord(match.group(1)) - ord('a')
                if letter_idx < len(choices):
                    model_choice = choices[letter_idx].strip().lower()
                    return model_choice == ground_truth_lower
            # Also check if model directly stated the answer text
            return ground_truth_lower in response
        else:
            # Free-form: numeric/text match
            # Try to extract numbers for comparison
            import re
            gt_nums = re.findall(r'[\d.]+', ground_truth_lower)
            resp_nums = re.findall(r'[\d.]+', response)
            if gt_nums and resp_nums:
                # Compare first number found
                try:
                    return abs(float(gt_nums[0]) - float(resp_nums[0])) < 0.01
                except:
                    pass
            return ground_truth_lower in response or response in ground_truth_lower

    elif benchmark == "realworldqa":
        # RealWorldQA is MCQ, check letter match
        for letter in ["a", "b", "c", "d"]:
            if letter in response[:20]:
                return letter == ground_truth_lower
        # Fallback to text match
        return ground_truth_lower in response

    return False


def run_cross_model_transfer(benchmarks: List[str], max_samples_per_benchmark: int = 200):
    """Run cross-model transfer test across all benchmarks."""
    print("=" * 70)
    print("CROSS-MODEL TRANSFER TEST (ALL VISION BENCHMARKS)")
    print("=" * 70)
    print("Source model (training): InternVL3-78B")
    print("Target model (testing):  Qwen2.5-VL-72B")
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print("=" * 70)

    # Load models
    print("\n1. Loading models...")
    qwen_model, qwen_processor = load_qwen_model()
    judge_model, judge_processor = load_vlm_judge()

    results = {}
    all_predictions = []
    all_labels = []

    for benchmark in benchmarks:
        print(f"\n{'='*50}")
        print(f"BENCHMARK: {benchmark.upper()}")
        print("=" * 50)

        # Load samples
        print(f"Loading {benchmark} samples...")
        samples = load_benchmark_samples(benchmark, max_samples_per_benchmark)
        print(f"Loaded {len(samples)} samples")

        # Count real vs fallback images
        n_real = sum(1 for s in samples if s["image"].size != (336, 336))
        print(f"Images: {n_real} real, {len(samples) - n_real} fallback")

        # Run Qwen inference
        print(f"\nRunning Qwen2.5-VL-72B inference...")
        predictions = []
        labels = []

        for sample in tqdm(samples, desc=f"Generating ({benchmark})"):
            try:
                response = generate_response_qwen(
                    qwen_model, qwen_processor,
                    sample["image"], sample["prompt"]
                )
                is_correct = check_correctness(
                    benchmark, response, sample["ground_truth"],
                    choices=sample.get("choices")
                )

                # Get judge prediction
                p_correct = get_p_correct(
                    judge_model, judge_processor,
                    sample["image"],
                    sample["prompt"][:500],
                    response[:300]
                )

                predictions.append(p_correct)
                labels.append(float(is_correct))
                all_predictions.append(p_correct)
                all_labels.append(float(is_correct))

            except Exception as e:
                print(f"Error on {sample['question_id']}: {e}")
                continue

        predictions = np.array(predictions)
        labels = np.array(labels)

        # Compute metrics
        if len(np.unique(labels)) > 1:
            auroc = roc_auc_score(labels, predictions)
            auprc = average_precision_score(labels, predictions)
        else:
            auroc = 0.5
            auprc = labels.mean()

        accuracy = labels.mean()
        brier = brier_score_loss(labels, predictions)

        results[benchmark] = {
            "n_samples": len(labels),
            "qwen_accuracy": float(accuracy),
            "judge_auroc": float(auroc),
            "judge_auprc": float(auprc),
            "judge_brier": float(brier),
            "p_correct_mean": float(predictions.mean()),
            "p_correct_std": float(predictions.std()),
        }

        print(f"\n{benchmark} Results:")
        print(f"  Qwen accuracy: {accuracy:.1%}")
        print(f"  Judge AUROC:   {auroc:.3f}")
        print(f"  Judge AUPRC:   {auprc:.3f}")
        print(f"  P(correct):    {predictions.mean():.3f} ± {predictions.std():.3f}")

    # Overall metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    if len(np.unique(all_labels)) > 1:
        overall_auroc = roc_auc_score(all_labels, all_predictions)
        overall_auprc = average_precision_score(all_labels, all_predictions)
    else:
        overall_auroc = 0.5
        overall_auprc = all_labels.mean()

    results["overall"] = {
        "n_samples": len(all_labels),
        "qwen_accuracy": float(all_labels.mean()),
        "judge_auroc": float(overall_auroc),
        "judge_auprc": float(overall_auprc),
        "judge_brier": float(brier_score_loss(all_labels, all_predictions)),
    }

    # Print summary
    print("\n" + "=" * 70)
    print("CROSS-MODEL TRANSFER SUMMARY")
    print("=" * 70)
    print(f"\n{'Benchmark':<20} {'Samples':>8} {'Qwen Acc':>10} {'Judge AUROC':>12}")
    print("-" * 50)
    for bench in benchmarks:
        r = results[bench]
        print(f"{bench:<20} {r['n_samples']:>8} {r['qwen_accuracy']:>10.1%} {r['judge_auroc']:>12.3f}")
    print("-" * 50)
    r = results["overall"]
    print(f"{'OVERALL':<20} {r['n_samples']:>8} {r['qwen_accuracy']:>10.1%} {r['judge_auroc']:>12.3f}")

    # Compare to in-distribution
    print("\n" + "=" * 70)
    print("COMPARISON TO IN-DISTRIBUTION (InternVL3-78B)")
    print("=" * 70)
    in_dist = {
        "vsr": 0.770,  # With real images
        "charxiv": 0.720,
        "mmmu": 0.651,
        "hallusionbench": 0.809,
    }

    print(f"\n{'Benchmark':<20} {'In-Dist AUROC':>14} {'Cross-Model':>12} {'Delta':>10}")
    print("-" * 56)
    for bench in benchmarks:
        if bench in in_dist:
            cross = results[bench]["judge_auroc"]
            ind = in_dist[bench]
            delta = cross - ind
            print(f"{bench:<20} {ind:>14.3f} {cross:>12.3f} {delta:>+10.3f}")

    # Save results
    output_path = Path("data/vlm_judge_combined/cross_model_transfer_all.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    # Full benchmark suite (excluding ERQA - multi-image benchmark)
    # CharXiv: Fixed to use reasoning_q/reasoning_a fields
    # MMMU: Fixed to use multiple subjects (Art, Math, Physics, CS, Bio, Chem, Econ)
    benchmarks = ["vsr", "charxiv", "mmmu", "hallusionbench", "mathvista", "realworldqa"]
    run_cross_model_transfer(benchmarks, max_samples_per_benchmark=200)
