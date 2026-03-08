#!/usr/bin/env python3
"""Evaluate the calibrator on a truly unseen target model (LLaMA-3.1-8B).

Addresses reviewer Issue 4: "No truly unseen target model evaluation."

Pipeline:
1. Load LLaMA-3.1-8B-Instruct and run inference on text benchmarks
2. Grade responses using benchmark graders
3. Score with the calibrator
4. Report AUROC on unseen model

Uses HuggingFace transformers directly (not vLLM) to avoid known startup issues.

Usage:
    # Smoke test (2 benchmarks, 10 examples each)
    CUDA_VISIBLE_DEVICES=0,1 python scripts/unseen_model_eval.py --smoke_test

    # Full run
    CUDA_VISIBLE_DEVICES=0,1 python scripts/unseen_model_eval.py

    # Skip generation (re-score existing responses)
    CUDA_VISIBLE_DEVICES=1 python scripts/unseen_model_eval.py --skip_generation
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Qwen3VLForConditionalGeneration, AutoProcessor,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# CONFIG
# ============================================================

LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
CALIBRATOR_CHECKPOINT = "uq_models/best_v2_r32_combined"
BASE_CALIBRATOR = "Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR = Path("data/ablations/unseen_model")

# Text-only benchmarks with automated grading
TEXT_BENCHMARKS = ["bbeh", "gpqa", "simpleqa", "omnimath", "chembench",
                   "prbench", "livebench", "hle"]
SMOKE_BENCHMARKS = ["bbeh", "gpqa"]

PROMPT_COMBINED = """Benchmark: {benchmark}
Source model: {source_model}

Question: {question}

Answer: {response}

Analyze whether the answer above is correct. Consider:
- Does the answer address the question?
- Are there factual errors or logical flaws?
- Is the answer complete?

Based on your analysis, is the answer correct? (i) No (ii) Yes"""


# ============================================================
# STEP 1: Generate LLaMA responses
# ============================================================

def load_benchmark_data(benchmark: str, max_examples=None):
    """Load benchmark examples using the eval harness API."""
    try:
        from uq_eval.registry import load_benchmark
        kwargs = {}
        if benchmark == "gpqa":
            kwargs["subset"] = "gpqa_diamond"
        elif benchmark == "hle":
            kwargs["text_only"] = True
        bench = load_benchmark(benchmark, **kwargs)
        examples = list(bench.iter_examples(split="test"))
        if max_examples and len(examples) > max_examples:
            np.random.seed(42)
            indices = np.random.choice(len(examples), max_examples, replace=False)
            examples = [examples[i] for i in sorted(indices)]
        return examples, bench
    except Exception as e:
        print(f"  Could not load benchmark {benchmark}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def extract_question_text(input_data) -> str:
    """Extract question text from Example.input field."""
    if isinstance(input_data, str):
        return input_data
    if isinstance(input_data, dict):
        for key in ["question", "query", "query_cot", "prompt", "text"]:
            if key in input_data and input_data[key]:
                val = input_data[key]
                if isinstance(val, str):
                    return val
        if "messages" in input_data:
            for msg in input_data["messages"]:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return content
        clean = {k: v for k, v in input_data.items() if k != "images"}
        return json.dumps(clean)[:2000]
    return str(input_data)[:2000]


def run_benchmark_with_llama(llama_model, tokenizer, bench, examples,
                              benchmark, device="cuda:0"):
    """Run a benchmark end-to-end: build requests, generate, parse, score."""
    from uq_eval.types import ModelResponse

    graded = []
    for i, example in enumerate(examples):
        try:
            # Build the request using the benchmark's formatter
            request = bench.build_request(example)
            messages = request.messages
            max_tokens = min(request.max_output_tokens, 2048)

            # Send to LLaMA
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=4096).to(device)

            with torch.no_grad():
                output = llama_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            gen_tokens = output[0][inputs["input_ids"].shape[1]:]
            response_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

            # Parse prediction using benchmark's parser
            model_resp = ModelResponse(text=response_text)
            pred = bench.parse_prediction(example, model_resp)

            # Score
            score_result = bench.score(example, pred)
            correct = score_result.get("correct", -1)
            if correct not in (0, 1):
                continue

            # Extract question text for the calibrator
            question = extract_question_text(example.input)

            graded.append({
                "id": example.id,
                "benchmark": benchmark,
                "question": question[:2000],
                "response": response_text[:1000],
                "is_correct": int(correct),
            })

        except Exception as e:
            if i < 3:
                print(f"    Error on example {i}: {e}")
            continue

        if (i + 1) % 20 == 0:
            n_correct = sum(g["is_correct"] for g in graded)
            acc = n_correct / len(graded) if graded else 0
            print(f"    [{i+1}/{len(examples)}] {len(graded)} graded, acc={acc:.3f}")

    return graded


# ============================================================
# STEP 2: Score with calibrator
# ============================================================

def score_with_calibrator(model, processor, device, samples, source_model="llama31_8b"):
    fallback_image = Image.new('RGB', (224, 224), color='gray')
    labels, scores = [], []
    per_benchmark = defaultdict(lambda: {"labels": [], "scores": []})

    for i, sample in enumerate(samples):
        prompt = PROMPT_COMBINED.format(
            question=sample["question"][:1500],
            response=sample["response"][:800],
            benchmark=sample["benchmark"],
            source_model=source_model,
        )
        messages = [{"role": "user", "content": [
            {"type": "image", "image": fallback_image},
            {"type": "text", "text": prompt},
        ]}]

        text = processor.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
        inputs = processor(
            text=[text], images=[fallback_image], return_tensors="pt", padding=True,
            min_pixels=256*28*28, max_pixels=256*28*28,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[0, -1, :]
        token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
        token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]
        probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
        p = probs[1].item()

        labels.append(sample["is_correct"])
        scores.append(p)
        per_benchmark[sample["benchmark"]]["labels"].append(sample["is_correct"])
        per_benchmark[sample["benchmark"]]["scores"].append(p)

        if (i + 1) % 100 == 0 or (i + 1) == len(samples):
            interim = ""
            if len(set(labels)) > 1:
                interim = f", AUROC={roc_auc_score(labels, scores):.4f}"
            print(f"  [{i+1}/{len(samples)}]{interim}")

    return labels, scores, per_benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CALIBRATOR_CHECKPOINT)
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--max_per_benchmark", type=int, default=200,
                        help="Max examples per benchmark (default 200)")
    parser.add_argument("--llama_gpu", type=int, default=0)
    parser.add_argument("--calibrator_gpu", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = SMOKE_BENCHMARKS if args.smoke_test else TEXT_BENCHMARKS
    max_examples = 10 if args.smoke_test else args.max_per_benchmark

    all_graded = []
    responses_path = output_dir / "llama_responses.jsonl"

    if not args.skip_generation:
        # Step 1: Generate and grade
        print("="*70)
        print("STEP 1: Generate LLaMA-3.1-8B responses")
        print("="*70)

        llama_device = f"cuda:{args.llama_gpu}"
        print(f"Loading LLaMA-3.1-8B on {llama_device}...")
        tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
        llama_model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL, torch_dtype=torch.bfloat16,
            device_map={"": llama_device},
        )
        llama_model.eval()
        print("LLaMA loaded.")

        # Flush stdout for SLURM log visibility
        sys.stdout.reconfigure(line_buffering=True)

        with open(responses_path, "w") as f_out:
            for bench_name in benchmarks:
                print(f"\n--- Benchmark: {bench_name} ---")
                examples, bench = load_benchmark_data(bench_name, max_examples)
                if not examples:
                    print(f"  Skipping {bench_name} (no data)")
                    continue

                print(f"  Running {len(examples)} examples...")
                t0 = time.time()
                graded = run_benchmark_with_llama(
                    llama_model, tokenizer, bench, examples, bench_name,
                    device=llama_device,
                )
                elapsed = time.time() - t0
                n_correct = sum(g["is_correct"] for g in graded)
                acc = n_correct / len(graded) if graded else 0
                print(f"  {bench_name}: {len(graded)} graded, acc={acc:.3f} ({elapsed:.0f}s)")

                for g in graded:
                    f_out.write(json.dumps(g) + "\n")
                all_graded.extend(graded)

        # Free LLaMA memory
        del llama_model
        torch.cuda.empty_cache()
        time.sleep(3)

    else:
        # Load existing responses
        print("Loading existing responses...")
        with open(responses_path) as f:
            for line in f:
                all_graded.append(json.loads(line))
        print(f"Loaded {len(all_graded)} graded samples")

    if not all_graded:
        print("ERROR: No graded samples")
        sys.exit(1)

    # Step 2: Score with calibrator
    print("\n" + "="*70)
    print("STEP 2: Score with calibrator")
    print("="*70)

    cal_device = torch.device(f"cuda:{args.calibrator_gpu}")
    print(f"Loading calibrator on {cal_device}...")
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_CALIBRATOR, torch_dtype=torch.bfloat16,
        device_map={"": cal_device}, trust_remote_code=True,
    )
    cal_model = PeftModel.from_pretrained(base_model, args.checkpoint)
    cal_model.eval()

    # Score with metadata
    print("\nScoring with metadata (source_model='llama31_8b')...")
    labels, scores, per_bench = score_with_calibrator(
        cal_model, processor, cal_device, all_graded, source_model="llama31_8b",
    )

    # Score without metadata
    print("\nScoring without metadata...")
    labels2, scores2, _ = score_with_calibrator(
        cal_model, processor, cal_device, all_graded, source_model="",
    )

    # Results
    print("\n" + "="*70)
    print("RESULTS: Unseen Model (LLaMA-3.1-8B-Instruct)")
    print("="*70)

    results = {
        "unseen_model": LLAMA_MODEL,
        "calibrator": args.checkpoint,
        "n_total": len(all_graded),
        "n_correct": int(sum(labels)),
        "overall_acc": float(np.mean(labels)),
    }

    if len(np.unique(labels)) > 1:
        auroc = roc_auc_score(labels, scores)
        auroc_no_meta = roc_auc_score(labels2, scores2)
        results["auroc_with_metadata"] = float(auroc)
        results["auroc_without_metadata"] = float(auroc_no_meta)
        print(f"AUROC (with metadata):    {auroc:.4f}")
        print(f"AUROC (without metadata): {auroc_no_meta:.4f}")

    results["per_benchmark"] = {}
    for bench in sorted(per_bench):
        data = per_bench[bench]
        bl, bs = np.array(data["labels"]), np.array(data["scores"])
        entry = {"n": len(bl), "acc": float(np.mean(bl))}
        if len(np.unique(bl)) > 1:
            entry["auroc"] = float(roc_auc_score(bl, bs))
            print(f"  {bench:<20} AUROC={entry['auroc']:.3f} (n={len(bl)}, acc={entry['acc']:.3f})")
        else:
            print(f"  {bench:<20} single class (n={len(bl)}, acc={entry['acc']:.3f})")
        results["per_benchmark"][bench] = entry

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save predictions
    with open(output_dir / "predictions.jsonl", "w") as f:
        for sample, p in zip(all_graded, scores):
            f.write(json.dumps({**sample, "p_correct": round(p, 6)}) + "\n")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
