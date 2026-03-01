from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .registry import load_benchmark, load_model_client
from .runner import run_eval


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="uq_eval", description="Minimal UQ eval runner (closed-source API supported).")
    p.add_argument("--model_backend", default="chat_http", help='e.g. "chat_http" or "openai"')
    p.add_argument("--model_name", default="gpt-4o", help='model id, e.g. "gpt-4o"')
    p.add_argument("--bench", default="dummy_qa", help='benchmark name, e.g. "dummy_qa"')
    p.add_argument("--split", default="dev", help="dataset split name (bench-dependent)")
    p.add_argument("--out_dir", default=None, help="output directory (default: runs/<timestamp>_<bench>_<model>)")

    # Benchmark data (optional; used by jsonl_qa)
    p.add_argument("--bench_data", default=None, help="Path to benchmark data file (e.g., JSONL).")
    # GPQA subset selection
    p.add_argument("--subset", default=None, help="Dataset subset (e.g., gpqa_diamond, gpqa_main, gpqa_extended)")
    # BBEH options
    p.add_argument("--bbeh_mini", action="store_true", help="Use BBEH mini version (460 examples) instead of full (4520)")
    p.add_argument("--bbeh_tasks", default=None, help="Comma-separated list of BBEH tasks to run (default: all 23)")
    # HLE options
    p.add_argument("--hle_with_images", action="store_true", help="Include HLE questions with images (requires vision model)")
    p.add_argument("--hle_answer_type", default=None, help="Filter HLE by answer type: 'mcq' or 'short_answer'")
    p.add_argument("--hle_category", default=None, help="Filter HLE by category (e.g., 'Mathematics', 'Physics')")

    # API config
    p.add_argument("--api_key", default=None, help="API key (or set env UQ_API_KEY / OPENAI_API_KEY)")
    p.add_argument("--base_url", default=None, help="Base URL like https://.../v1 (or full .../v1/chat/completions).")

    p.add_argument("--max_examples", type=int, default=None, help="cap number of examples (for smoke tests)")
    p.add_argument("--seed", type=int, default=None, help="random seed for reproducible sampling (requires --max_examples)")
    p.add_argument("--exclude_ids", default=None, help="path to JSON file with IDs to exclude (from previous runs)")
    p.add_argument("--include_ids", default=None, help="path to JSON file with IDs to include (for cross-model comparison)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_output_tokens", type=int, default=256)
    p.add_argument("--timeout_s", type=float, default=600.0, help="HTTP timeout in seconds (default 600s for long-context)")
    p.add_argument("--resume", action="store_true", help="resume if predictions.jsonl already exists")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.set_defaults(resume=True)
    p.add_argument("--logprobs", action="store_true", help="request logprobs (if backend/model supports it)")
    p.add_argument("--reasoning_effort", default=None, choices=["low", "medium", "high"],
                   help="Reasoning effort for reasoning models like gpt-5.2 (low/medium/high)")
    p.add_argument("--disable_thinking", action="store_true",
                   help="Disable thinking mode for Qwen3-style models (faster inference)")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = args.model_name.replace("/", "_")
        out_dir = Path("runs") / f"{ts}_{args.bench}_{safe_model}"

    model_kwargs = dict(
        model_name=args.model_name,
        api_key=args.api_key,
        base_url=args.base_url,
        timeout_s=args.timeout_s,
    )
    if args.disable_thinking:
        model_kwargs["disable_thinking"] = True
    model = load_model_client(args.model_backend, **model_kwargs)

    bench_kwargs = {}
    if args.bench_data:
        bench_kwargs["data_path"] = args.bench_data
    if args.subset:
        bench_kwargs["subset"] = args.subset
    # BBEH-specific options
    if args.bbeh_mini:
        bench_kwargs["use_mini"] = True
    if args.bbeh_tasks:
        bench_kwargs["tasks"] = [t.strip() for t in args.bbeh_tasks.split(",")]
    # HLE-specific options
    if args.hle_with_images:
        bench_kwargs["include_images"] = True
        bench_kwargs["text_only"] = False
    if args.hle_answer_type:
        bench_kwargs["answer_type_filter"] = args.hle_answer_type
    if args.hle_category:
        bench_kwargs["category_filter"] = args.hle_category
    bench = load_benchmark(args.bench, **bench_kwargs)

    # Load exclude IDs if provided
    exclude_ids = None
    if args.exclude_ids:
        import json
        exclude_ids = set(json.loads(Path(args.exclude_ids).read_text()))

    # Load include IDs if provided (for cross-model comparison)
    include_ids = None
    if args.include_ids:
        import json
        include_ids = set(json.loads(Path(args.include_ids).read_text()))

    metrics = run_eval(
        model=model,
        bench=bench,
        split=args.split,
        out_dir=out_dir,
        max_examples=args.max_examples,
        seed=args.seed,
        exclude_ids=exclude_ids,
        include_ids=include_ids,
        resume=args.resume,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        logprobs=args.logprobs,
        reasoning_effort=args.reasoning_effort,
    )

    print(f"[done] wrote: {out_dir}")
    print(metrics)


if __name__ == "__main__":
    main()
