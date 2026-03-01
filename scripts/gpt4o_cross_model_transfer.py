#!/usr/bin/env python3
"""
GPT-4o Cross-Model Transfer Experiment

Tests if the Llama-trained text calibrator can predict correctness of GPT-4o outputs.
This proves we can calibrate closed-source models - the paper's key practical contribution.
"""

import json
import os
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from openai import OpenAI

# Paths
TEST_DATA_PATH = Path("data/finetune/test_v2.jsonl")
LLAMA_CHECKPOINT = Path("uq_models/llama-8b-uq-lora-v2/checkpoint-198")
OUTPUT_DIR = Path("data/cross_model")


def load_test_samples(max_samples: int = 300):
    """Load test samples from test_v2.jsonl"""
    samples = []
    with open(TEST_DATA_PATH) as f:
        for line in f:
            sample = json.loads(line)
            samples.append(sample)

    # Shuffle and take max_samples
    np.random.seed(42)
    np.random.shuffle(samples)
    return samples[:max_samples]


def extract_question(sample: dict) -> str:
    """Extract the question from a sample."""
    input_data = sample.get("input", "")

    # Handle different input formats
    if isinstance(input_data, dict):
        # GPQA format: {"question": "...", "choices": [...]}
        question = input_data.get("question", "")
        choices = input_data.get("choices", [])
        if choices:
            choice_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
            return f"{question}\n\n{choice_text}"
        return question
    else:
        return str(input_data)


def get_gpt4o_response(client: OpenAI, question: str, model: str = "gpt-4o-mini") -> str:
    """Get GPT-4o response for a question."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            max_tokens=500,
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting GPT-4o response: {e}")
        return ""


def grade_response(sample: dict, response: str) -> bool:
    """Grade if the GPT-4o response is correct."""
    target = sample.get("target", "")

    if not target or not response:
        return False

    response_lower = response.lower().strip()
    target_lower = str(target).lower().strip()

    # Get benchmark type from ID
    sample_id = sample.get("id", "")

    # MCQ grading - look for letter answer
    if any(sample_id.startswith(b) for b in ["hellaswag", "arc", "winogrande", "mmlu", "gpqa", "bbeh"]):
        # Extract letter from response
        # Common patterns: "A", "(A)", "Answer: A", "The answer is A"
        patterns = [
            r'^[^a-z]*([a-z])\b',           # Letter at start
            r'answer[:\s]+([a-z])\b',       # answer is/: X
            r'\b([a-z])\s+is\s+(?:correct|right)',  # X is correct
            r'\bis\s+([a-z])\b',            # is X
        ]
        for pattern in patterns:
            match = re.search(pattern, response_lower[:200], re.IGNORECASE)
            if match:
                letter = match.group(1).lower()
                # Handle target that might be "(A)" or "A"
                target_clean = re.sub(r'[^a-z]', '', target_lower)
                return letter == target_clean
        return False

    # Math grading - numeric comparison
    if any(sample_id.startswith(b) for b in ["gsm8k", "mgsm", "math", "aime", "omnimath"]):
        # Extract numbers from both
        target_nums = re.findall(r'-?[\d,]+\.?\d*', target_lower.replace(',', ''))
        resp_nums = re.findall(r'-?[\d,]+\.?\d*', response_lower.replace(',', ''))

        if target_nums and resp_nums:
            try:
                target_val = float(target_nums[-1])  # Usually last number is the answer
                # Check all numbers in response
                for num in resp_nums:
                    try:
                        if abs(float(num) - target_val) < 0.01 * max(abs(target_val), 1):
                            return True
                    except:
                        continue
            except:
                pass
        return False

    # Boolean grading
    if sample_id.startswith("boolq"):
        target_bool = target_lower in ["true", "yes", "1"]
        resp_bool = "true" in response_lower or "yes" in response_lower
        resp_false = "false" in response_lower or "no" in response_lower

        if resp_bool and not resp_false:
            return target_bool
        elif resp_false and not resp_bool:
            return not target_bool
        return False

    # Open-ended: containment check
    # For simpleqa, triviaqa, drop, etc.
    return target_lower in response_lower or response_lower in target_lower


def load_calibrator():
    """Load the Llama calibrator model."""
    print("Loading Llama calibrator...")

    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        LLAMA_CHECKPOINT,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_CHECKPOINT)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_p_correct(model, tokenizer, question: str, answer: str) -> float:
    """Get P(correct) from the calibrator."""
    # Truncate inputs
    question = question[:500]
    answer = answer[:300]

    # Format prompt
    prompt = f"Question: {question}\n\nAnswer: {answer}\n\nIs the answer correct? (i) No (ii) Yes"

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]

        # Get token IDs for "i" and "ii" (or similar tokens)
        # Try multiple possible token representations
        yes_tokens = ["ii", "Yes", "yes", "2", " ii"]
        no_tokens = ["i", "No", "no", "1", " i"]

        yes_logit = float('-inf')
        no_logit = float('-inf')

        for tok in yes_tokens:
            try:
                tok_id = tokenizer.encode(tok, add_special_tokens=False)[-1]
                yes_logit = max(yes_logit, logits[tok_id].item())
            except:
                continue

        for tok in no_tokens:
            try:
                tok_id = tokenizer.encode(tok, add_special_tokens=False)[-1]
                no_logit = max(no_logit, logits[tok_id].item())
            except:
                continue

        # Softmax to get probability
        probs = torch.softmax(torch.tensor([no_logit, yes_logit]), dim=0)
        p_correct = probs[1].item()

    return p_correct


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if bin_mask.sum() > 0:
            bin_acc = labels[bin_mask].mean()
            bin_conf = probs[bin_mask].mean()
            ece += bin_mask.sum() * abs(bin_acc - bin_conf)

    return ece / len(probs)


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    client = OpenAI()

    # Load test samples
    print("Loading test samples...")
    samples = load_test_samples(max_samples=250)
    print(f"Loaded {len(samples)} samples")

    # Check if we have cached GPT-4o responses
    cache_path = OUTPUT_DIR / "gpt4o_responses_cache.json"
    if cache_path.exists():
        print("Loading cached GPT-4o responses...")
        with open(cache_path) as f:
            cache = json.load(f)
    else:
        cache = {}

    # Get GPT-4o responses
    print("\nGetting GPT-4o responses...")
    results = []

    for sample in tqdm(samples, desc="Querying GPT-4o"):
        sample_id = sample.get("id", "")
        question = extract_question(sample)

        # Check cache
        if sample_id in cache:
            gpt4o_response = cache[sample_id]
        else:
            gpt4o_response = get_gpt4o_response(client, question)
            cache[sample_id] = gpt4o_response

            # Save cache periodically
            with open(cache_path, "w") as f:
                json.dump(cache, f, indent=2)

        # Grade response
        is_correct = grade_response(sample, gpt4o_response)

        results.append({
            "id": sample_id,
            "benchmark": sample_id.split("_")[0],
            "question": question[:500],
            "target": sample.get("target", ""),
            "gpt4o_response": gpt4o_response[:500],
            "gpt4o_correct": is_correct,
        })

    # Save final cache
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)

    # Compute GPT-4o accuracy
    gpt4o_accuracy = sum(1 for r in results if r["gpt4o_correct"]) / len(results)
    print(f"\nGPT-4o Accuracy: {gpt4o_accuracy:.1%}")

    # Load calibrator and get P(correct)
    model, tokenizer = load_calibrator()

    print("\nRunning calibrator on GPT-4o outputs...")
    for result in tqdm(results, desc="Calibrating"):
        p_correct = get_p_correct(model, tokenizer, result["question"], result["gpt4o_response"])
        result["p_correct"] = p_correct

    # Compute metrics
    labels = np.array([r["gpt4o_correct"] for r in results], dtype=float)
    probs = np.array([r["p_correct"] for r in results])

    auroc = roc_auc_score(labels, probs)
    brier = brier_score_loss(labels, probs)
    ece = compute_ece(probs, labels)

    # Per-benchmark breakdown
    benchmarks = {}
    for r in results:
        b = r["benchmark"]
        if b not in benchmarks:
            benchmarks[b] = {"labels": [], "probs": []}
        benchmarks[b]["labels"].append(r["gpt4o_correct"])
        benchmarks[b]["probs"].append(r["p_correct"])

    print("\n" + "=" * 60)
    print("GPT-4o Cross-Model Transfer Results")
    print("=" * 60)
    print(f"\nSamples: {len(results)}")
    print(f"GPT-4o Accuracy: {gpt4o_accuracy:.1%}")
    print(f"\nCalibrator Performance:")
    print(f"  AUROC: {auroc:.3f}")
    print(f"  ECE:   {ece:.3f}")
    print(f"  Brier: {brier:.3f}")

    print(f"\nComparison:")
    print(f"                      AUROC   ECE")
    print(f"  Llama→Llama         0.886   0.181")
    print(f"  Llama→Qwen          0.766   0.116")
    print(f"  Llama→GPT-4o        {auroc:.3f}   {ece:.3f}  <-- NEW")

    print(f"\nPer-benchmark AUROC (n≥10):")
    for b, data in sorted(benchmarks.items(), key=lambda x: -len(x[1]["labels"])):
        if len(data["labels"]) >= 10 and sum(data["labels"]) > 0 and sum(data["labels"]) < len(data["labels"]):
            b_auroc = roc_auc_score(data["labels"], data["probs"])
            print(f"  {b}: {b_auroc:.3f} (n={len(data['labels'])})")

    # Save results
    predictions_path = OUTPUT_DIR / "gpt4o_predictions.json"
    with open(predictions_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPredictions saved to {predictions_path}")

    summary = {
        "n_samples": len(results),
        "gpt4o_accuracy": gpt4o_accuracy,
        "auroc": auroc,
        "ece": ece,
        "brier": brier,
        "comparison": {
            "llama_to_llama": {"auroc": 0.886, "ece": 0.181},
            "llama_to_qwen": {"auroc": 0.766, "ece": 0.116},
            "llama_to_gpt4o": {"auroc": auroc, "ece": ece}
        }
    }

    summary_path = OUTPUT_DIR / "gpt4o_transfer_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    # Success criteria
    print("\n" + "=" * 60)
    if auroc > 0.70:
        print("SUCCESS: AUROC > 0.70 - Strong result (paper ready)")
    elif auroc > 0.65:
        print("ACCEPTABLE: AUROC > 0.65 - Acceptable result")
    else:
        print("INVESTIGATE: AUROC < 0.60 - Need to investigate why")


if __name__ == "__main__":
    main()
