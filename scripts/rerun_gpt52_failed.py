#!/usr/bin/env python3
"""
Rerun GPT-5.2 samples that failed due to reasoning token exhaustion.

72 samples returned empty (all 16384 tokens used for reasoning, 0 visible)
+ 5 samples returned truncated (hit limit mid-response, no valid JSON)
= 77 total to rerun with max_output_tokens=65536.

Usage:
    python scripts/rerun_gpt52_failed.py                    # full run
    python scripts/rerun_gpt52_failed.py --dry_run           # just list what would be rerun
    python scripts/rerun_gpt52_failed.py --max_output_tokens 32768  # custom limit
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def find_failed_samples(runs_dir: str = "runs") -> dict[str, list[str]]:
    """Find all GPT-5.2 samples that need rerunning."""
    failed = {}

    for bench_dir in sorted(os.listdir(runs_dir)):
        if not bench_dir.startswith("gpt52_high_"):
            continue
        bench = bench_dir.replace("gpt52_high_", "")
        pred_path = os.path.join(runs_dir, bench_dir, "predictions.jsonl")
        if not os.path.exists(pred_path):
            continue

        with open(pred_path) as f:
            lines = [json.loads(l) for l in f]

        bench_failed = []
        for l in lines:
            usage = l.get("usage", {})
            ot = usage.get("output_tokens", 0)
            resp = l.get("response_text", "")
            parsed = (
                isinstance(l.get("prediction", {}), dict)
                and l.get("prediction", {}).get("extra", {}).get("parsed_json", False)
            )

            # Skip if already rerun with higher limit (output_tokens > 16384)
            if ot > 16384:
                continue

            # Empty response (all tokens used for reasoning)
            if not resp.strip():
                bench_failed.append(l["id"])
            # Non-empty but truncated at limit (no valid JSON parse)
            elif ot >= 16384 and not parsed:
                bench_failed.append(l["id"])

        if bench_failed:
            failed[bench] = bench_failed

    return failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_output_tokens", type=int, default=65536,
                        help="Max output tokens for rerun (default: 65536)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Just list failed samples, don't rerun")
    parser.add_argument("--timeout_s", type=float, default=900,
                        help="Timeout per request in seconds (default: 900)")
    args = parser.parse_args()

    failed = find_failed_samples()
    total = sum(len(v) for v in failed.values())

    print("=" * 60)
    print(f"GPT-5.2 FAILED SAMPLE RERUN")
    print(f"=" * 60)
    print(f"Total failed samples: {total}")
    print(f"Max output tokens: {args.max_output_tokens} (was 16384)")
    print(f"Timeout: {args.timeout_s}s")
    print()

    for bench, ids in failed.items():
        print(f"  {bench}: {len(ids)} samples")

    if args.dry_run:
        print("\n[DRY RUN] Would rerun the above samples. Exiting.")
        return

    print()

    # Import after dry_run check to avoid slow imports unnecessarily
    from uq_eval.registry import load_benchmark
    from uq_eval.models.openai_client import OpenAIResponsesClient
    from uq_eval.types import ModelRequest, ModelResponse

    # Create client with higher timeout for extended reasoning
    client = OpenAIResponsesClient(
        model_name="gpt-5.2",
        timeout_s=args.timeout_s,
    )

    total_input_tokens = 0
    total_output_tokens = 0
    total_rerun = 0
    total_fixed = 0
    start_time = time.time()

    for bench, failed_ids in failed.items():
        print(f"\n{'=' * 60}")
        print(f"Rerunning {bench}: {len(failed_ids)} samples")
        print(f"{'=' * 60}")

        # Load benchmark
        try:
            benchmark = load_benchmark(bench)
        except Exception as e:
            print(f"  ERROR loading benchmark {bench}: {e}")
            continue

        # Load existing predictions
        pred_path = f"runs/gpt52_high_{bench}/predictions.jsonl"
        with open(pred_path) as f:
            all_preds = [json.loads(l) for l in f]

        # Build ID -> index map
        id_to_idx = {p["id"]: i for i, p in enumerate(all_preds)}

        # Load examples and filter to failed IDs
        failed_set = set(failed_ids)
        examples = {}

        # Get benchmark-specific args
        extra_kwargs = {}
        if bench == "gpqa":
            extra_kwargs["subset"] = "gpqa_diamond"
        elif bench == "hle":
            extra_kwargs["hle_answer_type"] = "exact_match"

        # iter_examples to find the failed ones
        split = "dev" if bench in ("gpqa", "hle", "mmmu", "mmstar", "healthbench",
                                     "charxiv", "hallusionbench", "mathverse",
                                     "mathvision", "mathvista", "realworldqa",
                                     "aokvqa", "vizwiz", "mmvet", "vsr",
                                     "arc_agi") else "test"

        for ex in benchmark.iter_examples(split):
            if ex.id in failed_set:
                examples[ex.id] = ex

        print(f"  Found {len(examples)}/{len(failed_ids)} examples in dataset")

        # Helper to run a single API call with a hard timeout
        # NOTE: Do NOT use `with` context manager — its __exit__ calls
        # shutdown(wait=True), which blocks until the thread finishes,
        # defeating the timeout.
        def _call_with_timeout(req, timeout_s):
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(client.generate, req)
            try:
                result = future.result(timeout=timeout_s)
                executor.shutdown(wait=False)
                return result
            except FuturesTimeoutError:
                future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise

        # Rerun each failed sample
        for sample_id in failed_ids:
            if sample_id not in examples:
                print(f"  SKIP {sample_id}: not found in dataset")
                continue

            ex = examples[sample_id]
            req = benchmark.build_request(ex)

            # Override max_output_tokens
            req = ModelRequest(
                messages=req.messages,
                temperature=req.temperature,
                max_output_tokens=args.max_output_tokens,
                reasoning_effort="high",
                tools=req.tools,
                tool_choice=req.tool_choice,
                logprobs=req.logprobs,
                metadata=req.metadata,
            )

            total_rerun += 1
            print(f"  [{total_rerun}/{total}] {sample_id}...", end=" ", flush=True)

            try:
                resp = _call_with_timeout(req, args.timeout_s)
                response_text = resp.text or ""

                # Parse and score
                try:
                    pred = benchmark.parse_prediction(ex, resp)
                    score_dict = benchmark.score(ex, pred)
                    score = score_dict
                    prediction = {
                        "answer": pred.answer,
                        "confidence": pred.confidence,
                        "extra": pred.extra or {},
                    }
                except Exception as e:
                    prediction = {"answer": response_text[:200], "extra": {"parse_error": str(e)}}
                    score = {"correct": -1, "error": str(e)}

                usage = resp.usage or {}

                # Build new prediction entry
                new_pred = {
                    "id": ex.id,
                    "input": all_preds[id_to_idx[sample_id]].get("input", ""),
                    "target": all_preds[id_to_idx[sample_id]].get("target", ""),
                    "meta": all_preds[id_to_idx[sample_id]].get("meta", {}),
                    "request": all_preds[id_to_idx[sample_id]].get("request", {}),
                    "response_text": response_text,
                    "prediction": prediction,
                    "score": score,
                    "usage": usage,
                }

                # Replace in predictions list
                all_preds[id_to_idx[sample_id]] = new_pred

                # Track tokens
                inp_tok = usage.get("input_tokens", 0)
                out_tok = usage.get("output_tokens", 0)
                reason_tok = usage.get("output_tokens_details", {}).get("reasoning_tokens", 0)
                total_input_tokens += inp_tok
                total_output_tokens += out_tok

                correct = score.get("correct", "?")
                has_resp = bool(response_text.strip())

                if has_resp:
                    total_fixed += 1
                    print(f"OK (out={out_tok}, reason={reason_tok}, correct={correct})")
                else:
                    print(f"STILL EMPTY (out={out_tok}, reason={reason_tok})")

            except FuturesTimeoutError:
                print(f"TIMEOUT ({args.timeout_s}s)")
                continue
            except Exception as e:
                print(f"ERROR: {e}")
                continue

            # Save after each sample to avoid losing progress
            with open(pred_path, "w") as f:
                for p in all_preds:
                    f.write(json.dumps(p) + "\n")

        # Recompute metrics
        scored = [p for p in all_preds if isinstance(p.get("score"), dict) and p["score"].get("correct", -1) >= 0]
        n_correct = sum(1 for p in scored if p["score"]["correct"] == 1)
        accuracy = n_correct / len(scored) if scored else 0

        metrics = {
            "n": len(all_preds),
            "n_scored": len(scored),
            "accuracy": accuracy,
            "n_correct": n_correct,
        }
        metrics_path = f"runs/gpt52_high_{bench}/metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"  Updated {bench}: {n_correct}/{len(scored)} = {accuracy:.1%}")

    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("RERUN SUMMARY")
    print("=" * 60)
    print(f"Total rerun: {total_rerun}")
    print(f"Fixed (got response): {total_fixed}")
    print(f"Still failed: {total_rerun - total_fixed}")
    print(f"Input tokens: {total_input_tokens:,}")
    print(f"Output tokens: {total_output_tokens:,}")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    # Rough cost estimate
    cost = total_input_tokens * 0.000002 + total_output_tokens * 0.000008
    print(f"Estimated cost: ${cost:.2f}")


if __name__ == "__main__":
    main()
