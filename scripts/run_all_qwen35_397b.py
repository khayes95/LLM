#!/usr/bin/env python3
"""Run all text benchmarks on Qwen3.5-397B-A17B-FP8 via vLLM server.

Uses --include_ids to match exact samples from GPT-5.2 runs for cross-model comparison.

Usage:
    python scripts/run_all_qwen35_397b.py --base_url http://localhost:8100/v1
    python scripts/run_all_qwen35_397b.py --smoke_test  # 3 samples per benchmark
    python scripts/run_all_qwen35_397b.py --benchmarks bbeh,gpqa
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Text-only benchmarks (Qwen3.5-397B is text-only MoE, no vision)
# Ordered: fast (short response) benchmarks first, slow (long response) last
BENCHMARKS = [
    ("simpleqa", []),
    ("chembench", []),
    ("gpqa", ["--subset", "gpqa_diamond"]),
    ("hle", ["--hle_answer_type", "exact_match"]),
    ("arc_agi", []),
    ("omnimath", []),
    ("livebench", []),
    ("babilong", []),
    ("oolong", []),
    # Slow benchmarks (long responses even without thinking)
    ("healthbench", []),
    ("bbeh", []),
    ("prbench", []),
    ("tutorbench", []),
]

GPT52_RUNS = "runs"
OUTPUT_BASE = "runs"


def run_benchmark(bench_name, extra_args, base_url, model_name, smoke_test=False):
    """Run a single benchmark via CLI subprocess."""
    out_dir = f"{OUTPUT_BASE}/qwen35_397b_{bench_name}"
    include_ids_path = f"{GPT52_RUNS}/gpt52_high_{bench_name}/sampled_ids.json"

    cmd = [
        sys.executable, "-m", "uq_eval.cli",
        "--bench", bench_name,
        "--model_backend", "chat_http",
        "--model_name", model_name,
        "--base_url", base_url,
        "--out_dir", out_dir,
        "--max_output_tokens", "2048",
        "--temperature", "0.0",
        "--timeout_s", "900",
        "--disable_thinking",
    ]

    # Use same sample IDs as GPT-5.2 for direct comparison
    if os.path.exists(include_ids_path):
        cmd.extend(["--include_ids", include_ids_path])
    elif smoke_test:
        cmd.extend(["--max_examples", "3", "--seed", "42"])
    else:
        cmd.extend(["--max_examples", "500", "--seed", "42"])

    if smoke_test and os.path.exists(include_ids_path):
        cmd.extend(["--max_examples", "3"])

    cmd.extend(extra_args)

    log_file = f"logs/qwen35_{bench_name}.log"
    os.makedirs("logs", exist_ok=True)
    print(f"  [{bench_name}] Starting: {' '.join(cmd)}")

    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=14400)

    return bench_name, proc.returncode


def detect_model_name(base_url: str) -> str:
    """Query /v1/models to get the actual model name from the server."""
    import urllib.request
    try:
        url = f"{base_url}/models"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
            models = data.get("data", [])
            if models:
                return models[0]["id"]
    except Exception:
        pass
    return ""


def main():
    parser = argparse.ArgumentParser(description="Run text benchmarks on Qwen3.5-397B")
    parser.add_argument("--base_url", default="http://localhost:8100/v1")
    parser.add_argument("--model_name", default=None,
                        help="Model name as reported by vLLM server (auto-detected if omitted)")
    parser.add_argument("--smoke_test", action="store_true", help="3 samples per benchmark")
    parser.add_argument("--parallel", type=int, default=13, help="Number of parallel benchmarks")
    parser.add_argument("--benchmarks", default=None, help="Comma-separated list of benchmarks")
    args = parser.parse_args()

    # Auto-detect model name from server
    if args.model_name is None:
        args.model_name = detect_model_name(args.base_url)
        if not args.model_name:
            print("ERROR: Could not detect model name from server. Is it running?")
            sys.exit(1)
        print(f"Detected model: {args.model_name}")

    benchmarks = BENCHMARKS
    if args.benchmarks:
        names = set(args.benchmarks.split(","))
        benchmarks = [(n, a) for n, a in BENCHMARKS if n in names]

    print(f"Running {len(benchmarks)} benchmarks against {args.base_url}")
    print(f"  Parallel: {args.parallel}, Smoke test: {args.smoke_test}")
    print()

    results = {}
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(
                run_benchmark, name, extra, args.base_url, args.model_name, args.smoke_test
            ): name
            for name, extra in benchmarks
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                name, rc = future.result()
                results[name] = rc
                status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            except Exception as e:
                results[name] = -1
                status = f"EXCEPTION: {e}"
            print(f"  [{name}] {status}")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    ok = sum(1 for rc in results.values() if rc == 0)
    fail = len(results) - ok
    for name, rc in sorted(results.items()):
        pred_file = Path(f"runs/qwen35_397b_{name}/predictions.jsonl")
        n = sum(1 for _ in open(pred_file)) if pred_file.exists() else 0
        print(f"  {name}: {'OK' if rc == 0 else 'FAILED'} ({n} predictions)")
    print(f"\n{ok}/{len(results)} succeeded, {fail} failed")


if __name__ == "__main__":
    main()
