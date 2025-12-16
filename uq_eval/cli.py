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

    # API config
    p.add_argument("--api_key", default=None, help="API key (or set env UQ_API_KEY / OPENAI_API_KEY)")
    p.add_argument("--base_url", default=None, help="Base URL like https://.../v1 (or full .../v1/chat/completions).")

    p.add_argument("--max_examples", type=int, default=None, help="cap number of examples (for smoke tests)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_output_tokens", type=int, default=256)
    p.add_argument("--timeout_s", type=float, default=30.0, help="HTTP timeout in seconds")
    p.add_argument("--resume", action="store_true", help="resume if predictions.jsonl already exists")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.set_defaults(resume=True)
    p.add_argument("--logprobs", action="store_true", help="request logprobs (if backend/model supports it)")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = args.model_name.replace("/", "_")
        out_dir = Path("runs") / f"{ts}_{args.bench}_{safe_model}"

    model = load_model_client(
        args.model_backend,
        model_name=args.model_name,
        api_key=args.api_key,
        base_url=args.base_url,
        timeout_s=args.timeout_s,
    )

    bench_kwargs = {}
    if args.bench_data:
        bench_kwargs["data_path"] = args.bench_data
    bench = load_benchmark(args.bench, **bench_kwargs)

    metrics = run_eval(
        model=model,
        bench=bench,
        split=args.split,
        out_dir=out_dir,
        max_examples=args.max_examples,
        resume=args.resume,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        logprobs=args.logprobs,
    )

    print(f"[done] wrote: {out_dir}")
    print(metrics)


if __name__ == "__main__":
    main()
