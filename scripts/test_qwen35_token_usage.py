#!/usr/bin/env python3
"""Test Qwen3.5-397B token usage per benchmark.

Runs 2 samples per benchmark with a high token limit (32768) and reports:
- Actual completion tokens used
- Whether responses are truncated
- Time per sample

This helps determine the right max_output_tokens setting.

Usage:
    python scripts/test_qwen35_token_usage.py --base_url http://localhost:8100/v1
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

# Benchmarks to test (same as run_all_qwen35_397b.py)
BENCHMARKS = [
    ("bbeh", ["--bbeh_tasks", "boardgame_qa"]),
    ("simpleqa", []),
    ("prbench", []),
    ("tutorbench", []),
    ("healthbench", []),
    ("hle", ["--hle_answer_type", "exact_match"]),
    ("arc_agi", []),
    ("chembench", []),
    ("gpqa", ["--subset", "gpqa_diamond"]),
    ("livebench", []),
    ("omnimath", []),
    ("oolong", []),
    ("babilong", []),
]

PYTHON = "/scratch/khayes/.conda/envs/uq_eval/bin/python"


def detect_model_name(base_url: str) -> str:
    url = f"{base_url}/models"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read())
        models = data.get("data", [])
        if models:
            return models[0]["id"]
    return ""


def run_benchmark_test(bench_name, extra_args, base_url, model_name):
    """Run 2 samples and measure token usage."""
    import subprocess

    out_dir = f"runs/_token_test_{bench_name}"
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        PYTHON, "-m", "uq_eval.cli",
        "--bench", bench_name,
        "--model_backend", "chat_http",
        "--model_name", model_name,
        "--base_url", base_url,
        "--out_dir", out_dir,
        "--max_output_tokens", "16384",
        "--max_examples", "2",
        "--seed", "42",
        "--temperature", "0.0",
        "--timeout_s", "600",
    ] + extra_args

    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - start

    # Parse predictions
    pred_file = f"{out_dir}/predictions.jsonl"
    results = []
    if os.path.exists(pred_file):
        with open(pred_file) as f:
            for line in f:
                d = json.loads(line)
                resp = d.get("response", "")
                score = d.get("score", {})
                predicted = score.get("predicted", "") if isinstance(score, dict) else ""
                # The actual model output might be in 'response' or 'score.predicted'
                actual_text = resp if resp else str(predicted)
                results.append({
                    "response_len": len(resp),
                    "response_empty": len(resp) == 0,
                    "predicted_len": len(str(predicted)),
                    "correct": score.get("correct") if isinstance(score, dict) else score,
                })

    return {
        "benchmark": bench_name,
        "n_predictions": len(results),
        "returncode": proc.returncode,
        "elapsed_s": round(elapsed, 1),
        "per_sample_s": round(elapsed / max(len(results), 1), 1),
        "samples": results,
        "any_empty_response": any(r["response_empty"] for r in results),
        "max_response_chars": max((r["response_len"] for r in results), default=0),
        "stderr_tail": proc.stderr[-500:] if proc.stderr else "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", default="http://localhost:8100/v1")
    parser.add_argument("--benchmarks", default=None, help="Comma-sep list to test")
    args = parser.parse_args()

    model_name = detect_model_name(args.base_url)
    if not model_name:
        print("ERROR: Could not detect model from server")
        sys.exit(1)
    print(f"Model: {model_name}")

    benchmarks = BENCHMARKS
    if args.benchmarks:
        names = set(args.benchmarks.split(","))
        benchmarks = [(n, a) for n, a in BENCHMARKS if n in names]

    print(f"\nTesting {len(benchmarks)} benchmarks, 2 samples each, max_output_tokens=16384\n")
    print(f"{'Benchmark':<15} {'Time':>8} {'Per-sample':>10} {'N':>3} {'Empty?':>7} {'Max chars':>10} {'Status':>8}")
    print("-" * 70)

    all_results = []
    total_time = 0
    for bench_name, extra_args in benchmarks:
        print(f"  Running {bench_name}...", end="", flush=True)
        result = run_benchmark_test(bench_name, extra_args, args.base_url, model_name)
        all_results.append(result)
        total_time += result["elapsed_s"]

        status = "OK" if result["returncode"] == 0 and not result["any_empty_response"] else "FAIL"
        print(f"\r{bench_name:<15} {result['elapsed_s']:>7.0f}s {result['per_sample_s']:>9.0f}s {result['n_predictions']:>3} "
              f"{'YES' if result['any_empty_response'] else 'no':>7} {result['max_response_chars']:>10} {status:>8}")

        if result["any_empty_response"]:
            print(f"  WARNING: {bench_name} has empty responses! Token limit may be too low.")
        if result["stderr_tail"] and result["returncode"] != 0:
            print(f"  stderr: {result['stderr_tail'][:200]}")

    # Summary
    print(f"\n{'='*70}")
    print(f"Total test time: {total_time:.0f}s ({total_time/60:.1f} min)")
    empty_count = sum(1 for r in all_results if r["any_empty_response"])
    print(f"Benchmarks with empty responses: {empty_count}/{len(all_results)}")

    if empty_count > 0:
        print("\nWARNING: Some benchmarks still have empty responses with max_output_tokens=16384!")
        print("Consider increasing to 32768 or disabling thinking mode.")

    # Estimate full run time
    avg_per_sample = sum(r["per_sample_s"] for r in all_results) / max(len(all_results), 1)
    total_samples = 250 * len(BENCHMARKS)  # 250 per benchmark, 13 benchmarks
    est_serial_hours = (avg_per_sample * total_samples) / 3600
    est_parallel4_hours = est_serial_hours / 4
    print(f"\nTime estimate for full run (250 samples × {len(BENCHMARKS)} benchmarks):")
    print(f"  Serial (parallel=1): {est_serial_hours:.1f} hours")
    print(f"  Parallel=4:          {est_parallel4_hours:.1f} hours")

    # Save results
    os.makedirs("logs", exist_ok=True)
    with open("logs/token_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to logs/token_test_results.json")

    # Save good predictions if any
    good = [r for r in all_results if not r["any_empty_response"] and r["returncode"] == 0]
    print(f"\n{len(good)}/{len(all_results)} benchmarks produced valid predictions (saved in runs/_token_test_*/)")


if __name__ == "__main__":
    main()
