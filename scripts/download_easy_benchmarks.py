#!/usr/bin/env python3
"""Download and format easy benchmark datasets for UQ training.

Creates synthetic correct/incorrect samples from easy benchmarks to complement
the hard-benchmark training data. Each question produces two samples:
one correct (correct=1) and one incorrect (correct=0).

Output: data/finetune/easy_questions/{benchmark}.jsonl + all_easy.jsonl
"""

import json
import os
import random
import hashlib
from pathlib import Path

# Ensure reproducibility
random.seed(42)

OUTPUT_DIR = "/scratch/khayes/LLM/data/finetune/easy_questions"
SAMPLES_PER_DATASET = 750  # Target ~750 questions -> ~1500 samples (correct + incorrect)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_response(answer: str, confidence: float = None) -> str:
    """Format answer as a realistic LLM JSON response."""
    if confidence is None:
        confidence = round(random.uniform(0.6, 0.99), 2)
    return json.dumps({"answer": str(answer), "confidence": confidence})


def make_sample(id_str: str, benchmark: str, question: str, response: str, correct: int) -> dict:
    return {
        "id": id_str,
        "benchmark": benchmark,
        "input": question,
        "target": "",
        "model_response": response,
        "prediction": "",
        "correct": correct,
        "source_model": "synthetic",
    }


def write_samples(benchmark: str, samples: list[dict]):
    """Write samples to per-benchmark file and return them."""
    path = os.path.join(OUTPUT_DIR, f"{benchmark}.jsonl")
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"  {benchmark}: {len(samples)} samples written to {path}")
    return samples


# ============================================================
# 1. BoolQ
# ============================================================
def process_boolq():
    print("\n[1/6] BoolQ — yes/no reading comprehension")
    from datasets import load_dataset
    ds = load_dataset("google/boolq", split="train")
    indices = random.sample(range(len(ds)), min(SAMPLES_PER_DATASET, len(ds)))

    samples = []
    for idx in indices:
        row = ds[idx]
        question = row["question"]
        passage = row["passage"]
        answer = "Yes" if row["answer"] else "No"
        wrong = "No" if row["answer"] else "Yes"

        full_q = f"Passage: {passage}\n\nQuestion: {question}"

        # Correct sample
        samples.append(make_sample(
            f"boolq_{idx}", "boolq", full_q,
            make_response(answer, round(random.uniform(0.85, 0.99), 2)), 1
        ))
        # Incorrect sample
        samples.append(make_sample(
            f"boolq_{idx}_wrong", "boolq", full_q,
            make_response(wrong, round(random.uniform(0.5, 0.85), 2)), 0
        ))

    return write_samples("boolq", samples)


# ============================================================
# 2. ARC-Easy
# ============================================================
def process_arc_easy():
    print("\n[2/6] ARC-Easy — elementary science MCQ")
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
    indices = random.sample(range(len(ds)), min(SAMPLES_PER_DATASET, len(ds)))

    samples = []
    for idx in indices:
        row = ds[idx]
        question = row["question"]
        choices = row["choices"]
        labels = choices["label"]
        texts = choices["text"]
        answer_key = row["answerKey"]

        # Build MCQ string
        options_str = "\n".join(f"({l}) {t}" for l, t in zip(labels, texts))
        full_q = f"{question}\n\n{options_str}"

        # Find correct and wrong answers
        correct_idx_pos = labels.index(answer_key)
        correct_text = texts[correct_idx_pos]
        wrong_indices = [i for i in range(len(labels)) if i != correct_idx_pos]
        wrong_idx = random.choice(wrong_indices)
        wrong_label = labels[wrong_idx]

        # Correct
        samples.append(make_sample(
            f"arc_easy_{idx}", "arc_easy", full_q,
            make_response(answer_key, round(random.uniform(0.85, 0.99), 2)), 1
        ))
        # Incorrect
        samples.append(make_sample(
            f"arc_easy_{idx}_wrong", "arc_easy", full_q,
            make_response(wrong_label, round(random.uniform(0.4, 0.8), 2)), 0
        ))

    return write_samples("arc_easy", samples)


# ============================================================
# 3. GSM8K
# ============================================================
def process_gsm8k():
    print("\n[3/6] GSM8K — grade school math")
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    indices = random.sample(range(len(ds)), min(SAMPLES_PER_DATASET, len(ds)))

    samples = []
    for idx in indices:
        row = ds[idx]
        question = row["question"]
        # Answer is after "####" in the answer field
        answer_text = row["answer"]
        final_answer = answer_text.split("####")[-1].strip()

        # Generate wrong answer by perturbing the number
        try:
            correct_num = float(final_answer.replace(",", ""))
            # Various perturbation strategies
            strategy = random.choice(["add", "subtract", "multiply", "off_by_one"])
            if strategy == "add":
                wrong_num = correct_num + random.choice([1, 2, 5, 10, 20])
            elif strategy == "subtract":
                wrong_num = correct_num - random.choice([1, 2, 5, 10])
                if wrong_num < 0 and correct_num >= 0:
                    wrong_num = correct_num + random.choice([3, 7, 11])
            elif strategy == "multiply":
                wrong_num = correct_num * random.choice([2, 3, 0.5])
            else:  # off_by_one
                wrong_num = correct_num + random.choice([-1, 1])

            if wrong_num == correct_num:
                wrong_num = correct_num + 5

            # Format wrong answer similarly
            if correct_num == int(correct_num):
                wrong_answer = str(int(wrong_num))
            else:
                wrong_answer = f"{wrong_num:.2f}"
        except ValueError:
            wrong_answer = "42"  # fallback

        # Correct — include reasoning steps
        correct_response = json.dumps({
            "answer": final_answer,
            "confidence": round(random.uniform(0.8, 0.99), 2)
        })
        samples.append(make_sample(
            f"gsm8k_{idx}", "gsm8k", question, correct_response, 1
        ))

        # Incorrect
        wrong_response = json.dumps({
            "answer": wrong_answer,
            "confidence": round(random.uniform(0.4, 0.85), 2)
        })
        samples.append(make_sample(
            f"gsm8k_{idx}_wrong", "gsm8k", question, wrong_response, 0
        ))

    return write_samples("gsm8k", samples)


# ============================================================
# 4. TriviaQA
# ============================================================
def process_triviaqa():
    print("\n[4/6] TriviaQA — factual trivia")
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.nocontext", split="train")
    indices = random.sample(range(len(ds)), min(SAMPLES_PER_DATASET, len(ds)))

    # Collect all answers for wrong-answer generation
    all_answers = set()
    for i in range(min(5000, len(ds))):
        ans = ds[i]["answer"]
        if ans and ans.get("value"):
            all_answers.add(ans["value"])
    all_answers = list(all_answers)

    samples = []
    for idx in indices:
        row = ds[idx]
        question = row["question"]
        answer = row["answer"]["value"]
        aliases = row["answer"].get("aliases", [])

        # Pick a wrong answer from other questions' answers
        wrong_answer = answer
        for _ in range(20):
            candidate = random.choice(all_answers)
            if candidate != answer and candidate not in aliases:
                wrong_answer = candidate
                break

        if wrong_answer == answer:
            wrong_answer = "I don't know the answer to this question."

        # Correct
        samples.append(make_sample(
            f"triviaqa_{idx}", "triviaqa", question,
            make_response(answer, round(random.uniform(0.8, 0.99), 2)), 1
        ))
        # Incorrect
        samples.append(make_sample(
            f"triviaqa_{idx}_wrong", "triviaqa", question,
            make_response(wrong_answer, round(random.uniform(0.4, 0.85), 2)), 0
        ))

    return write_samples("triviaqa", samples)


# ============================================================
# 5. HellaSwag
# ============================================================
def process_hellaswag():
    print("\n[5/6] HellaSwag — sentence completion")
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="train")
    indices = random.sample(range(len(ds)), min(SAMPLES_PER_DATASET, len(ds)))

    samples = []
    for idx in indices:
        row = ds[idx]
        context = row["ctx"]
        endings = row["endings"]
        correct_idx_val = int(row["label"])

        correct_ending = endings[correct_idx_val]
        wrong_indices = [i for i in range(len(endings)) if i != correct_idx_val]
        wrong_ending = endings[random.choice(wrong_indices)]

        # Format as MCQ
        options_str = "\n".join(f"({i}) {e}" for i, e in enumerate(endings))
        full_q = f"Context: {context}\n\nWhich is the most plausible continuation?\n{options_str}"

        # Correct
        samples.append(make_sample(
            f"hellaswag_{idx}", "hellaswag", full_q,
            make_response(str(correct_idx_val), round(random.uniform(0.8, 0.99), 2)), 1
        ))
        # Incorrect
        wrong_idx_val = random.choice(wrong_indices)
        samples.append(make_sample(
            f"hellaswag_{idx}_wrong", "hellaswag", full_q,
            make_response(str(wrong_idx_val), round(random.uniform(0.4, 0.8), 2)), 0
        ))

    return write_samples("hellaswag", samples)


# ============================================================
# 6. WinoGrande
# ============================================================
def process_winogrande():
    print("\n[6/6] WinoGrande — commonsense reasoning")
    from datasets import load_dataset
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="train")
    indices = random.sample(range(len(ds)), min(SAMPLES_PER_DATASET, len(ds)))

    samples = []
    for idx in indices:
        row = ds[idx]
        sentence = row["sentence"]
        option1 = row["option1"]
        option2 = row["option2"]
        answer = row["answer"]  # "1" or "2"

        full_q = f"{sentence}\n\n(1) {option1}\n(2) {option2}"

        correct_answer = answer
        wrong_answer = "2" if answer == "1" else "1"

        correct_text = option1 if answer == "1" else option2
        wrong_text = option2 if answer == "1" else option1

        # Correct
        samples.append(make_sample(
            f"winogrande_{idx}", "winogrande", full_q,
            make_response(correct_answer, round(random.uniform(0.8, 0.99), 2)), 1
        ))
        # Incorrect
        samples.append(make_sample(
            f"winogrande_{idx}_wrong", "winogrande", full_q,
            make_response(wrong_answer, round(random.uniform(0.4, 0.8), 2)), 0
        ))

    return write_samples("winogrande", samples)


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Downloading and formatting easy benchmark datasets")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Target: ~{SAMPLES_PER_DATASET} questions per dataset")
    print("=" * 60)

    all_samples = []

    processors = [
        process_boolq,
        process_arc_easy,
        process_gsm8k,
        process_triviaqa,
        process_hellaswag,
        process_winogrande,
    ]

    for proc in processors:
        try:
            samples = proc()
            all_samples.extend(samples)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Write combined file
    random.shuffle(all_samples)
    combined_path = os.path.join(OUTPUT_DIR, "all_easy.jsonl")
    with open(combined_path, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s) + "\n")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Count per benchmark
    from collections import Counter
    bench_counts = Counter(s["benchmark"] for s in all_samples)
    correct_counts = Counter((s["benchmark"], s["correct"]) for s in all_samples)

    total = 0
    for bench in sorted(bench_counts.keys()):
        n = bench_counts[bench]
        n_correct = correct_counts.get((bench, 1), 0)
        n_wrong = correct_counts.get((bench, 0), 0)
        print(f"  {bench:15s}: {n:5d} samples ({n_correct} correct, {n_wrong} incorrect)")
        total += n

    print(f"  {'TOTAL':15s}: {total:5d} samples")
    print(f"\nCombined file: {combined_path}")
    print("Done!")


if __name__ == "__main__":
    main()
