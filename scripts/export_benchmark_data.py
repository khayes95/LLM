"""Export all benchmark Q&A data with UQ scores to a single JSON file.

Extracts questions, model answers, ground truth, correctness, and self-reported
confidence from prediction JSONL files across all models and benchmarks.

Output: data/benchmark_export.json
"""

import json
import os
import glob
import sys
from pathlib import Path

RUNS_DIR = "/scratch/khayes/LLM/runs"
OUTPUT_PATH = "/scratch/khayes/LLM/data/benchmark_export.json"

# Define which run directories to use for each model
MODEL_RUNS = {
    "gpt-5-2": {
        "pattern": "gpt52_high_*",
        "display_name": "GPT-5-2",
    },
    "qwen3.5-397b": {
        "pattern": "qwen35_397b_*",
        "display_name": "Qwen3.5-397B-A17B-FP8",
        "min_samples": 25,  # skip smoke tests
    },
    # GPT-5-mini final runs (170504 batch = most complete)
    "gpt-5-mini": {
        "pattern": "20260102_170504_*_gpt-5-mini",
        "display_name": "GPT-5-mini",
    },
}

# Benchmarks to skip (backup dirs, not real benchmarks)
SKIP_DIRS = {"nothinking_backup"}


def extract_question(record: dict) -> str:
    """Extract the question text from a prediction record."""
    # Try input.question first
    inp = record.get("input", {})
    if isinstance(inp, dict):
        q = inp.get("question", "")
        if q:
            return q
    if isinstance(inp, str):
        return inp

    # Try extracting from request messages
    req = record.get("request", {})
    if isinstance(req, dict):
        msgs = req.get("messages", [])
        for msg in msgs:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                # Handle list-format content (multimodal)
                if isinstance(content, list):
                    texts = [p.get("text", "") for p in content
                             if isinstance(p, dict) and p.get("type") == "text"]
                    return "\n".join(texts)
    return ""


def extract_confidence(record: dict) -> float | None:
    """Extract self-reported confidence score."""
    pred = record.get("prediction", {})
    if isinstance(pred, dict):
        conf = pred.get("confidence")
        if conf is not None:
            return float(conf)
    # Sometimes stored in response_text as JSON
    resp = record.get("response_text", "")
    if isinstance(resp, str):
        try:
            parsed = json.loads(resp)
            if isinstance(parsed, dict) and "confidence" in parsed:
                return float(parsed["confidence"])
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def extract_answer(record: dict) -> str:
    """Extract the model's parsed answer."""
    pred = record.get("prediction", {})
    if isinstance(pred, dict):
        ans = pred.get("answer", "")
        if ans:
            return str(ans)
        # For some benchmarks, prediction is the full text
        return json.dumps(pred) if pred else ""
    if isinstance(pred, str):
        return pred
    return str(pred) if pred is not None else ""


def extract_correct(record: dict) -> int | None:
    """Extract correctness label."""
    score = record.get("score", {})
    if isinstance(score, dict):
        c = score.get("correct")
        if c is not None:
            return int(c)
    if isinstance(score, (int, float)):
        return int(score)
    return None


def process_predictions_file(filepath: str, benchmark: str, model: str) -> list[dict]:
    """Process a single predictions.jsonl file."""
    records = []
    with open(filepath) as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = extract_question(raw)
            answer = extract_answer(raw)
            confidence = extract_confidence(raw)
            correct = extract_correct(raw)
            target = raw.get("target", "")

            # Get response text (full model output)
            response_text = raw.get("response_text", "")
            if isinstance(response_text, dict):
                response_text = json.dumps(response_text)

            # Get thinking/reasoning if present
            thinking = raw.get("thinking", None)

            # Get metadata
            meta = raw.get("meta", {})
            if not isinstance(meta, dict):
                meta = {}

            record = {
                "id": raw.get("id", f"{benchmark}_{line_num}"),
                "benchmark": benchmark,
                "model": model,
                "question": question,
                "answer": answer,
                "response_text": response_text,
                "target": str(target) if target is not None else "",
                "correct": correct,
                "confidence": confidence,
            }

            # Add optional fields only if present
            if thinking:
                record["thinking"] = thinking
            if meta:
                # Include subset info if available
                subset = meta.get("subset", "")
                if subset:
                    record["subset"] = subset

            records.append(record)

    return records


def main():
    all_records = []
    summary = {}

    for model_key, model_info in MODEL_RUNS.items():
        pattern = model_info["pattern"]
        display_name = model_info["display_name"]
        min_samples = model_info.get("min_samples", 1)

        run_dirs = sorted(glob.glob(os.path.join(RUNS_DIR, pattern)))

        model_total = 0
        model_benchmarks = {}

        for run_dir in run_dirs:
            dirname = os.path.basename(run_dir)

            # Extract benchmark name from directory
            if model_key == "gpt-5-mini":
                # Format: 20260102_170504_<bench>_gpt-5-mini
                parts = dirname.split("_")
                # Remove timestamp (first 2) and model suffix (last 2: gpt-5-mini)
                bench = "_".join(parts[2:-2])
                if not bench:
                    # Try: gpt-5 is one token split
                    bench = "_".join(parts[2:-1]).replace("_gpt-5-mini", "")
            elif model_key == "gpt-5-2":
                bench = dirname.replace("gpt52_high_", "")
            elif model_key == "qwen3.5-397b":
                bench = dirname.replace("qwen35_397b_", "")
            else:
                bench = dirname

            if bench in SKIP_DIRS:
                continue

            pred_file = os.path.join(run_dir, "predictions.jsonl")
            if not os.path.exists(pred_file):
                continue

            records = process_predictions_file(pred_file, bench, display_name)

            if len(records) < min_samples:
                continue

            all_records.extend(records)
            model_total += len(records)
            model_benchmarks[bench] = len(records)

        summary[display_name] = {
            "total_samples": model_total,
            "benchmarks": model_benchmarks,
        }

        print(f"{display_name}: {model_total} samples across "
              f"{len(model_benchmarks)} benchmarks")

    # Build output
    output = {
        "description": "Benchmark Q&A data with model responses and UQ scores",
        "exported_from": "UQ Eval project (/scratch/khayes/LLM)",
        "fields": {
            "id": "Unique sample identifier",
            "benchmark": "Benchmark name",
            "model": "Model that generated the response",
            "question": "Input question/prompt",
            "answer": "Model's parsed answer (extracted from response)",
            "response_text": "Full model response text",
            "target": "Ground truth answer",
            "correct": "1 if correct, 0 if incorrect (null if unknown)",
            "confidence": "Model's self-reported confidence (0-1, null if not available)",
            "thinking": "(optional) Model's chain-of-thought reasoning",
            "subset": "(optional) Benchmark subset/split",
        },
        "summary": summary,
        "total_samples": len(all_records),
        "data": all_records,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nTotal: {len(all_records)} samples")
    print(f"Output: {OUTPUT_PATH}")
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
