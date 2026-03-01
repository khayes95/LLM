from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional, Set

from .benchmarks.base import BaseBenchmark
from .io import append_jsonl, iter_jsonl, load_existing_ids
from .models.base import BaseModelClient
from .types import ModelRequest


def aggregate_metrics(predictions_path: Path) -> dict[str, Any]:
    n = 0
    correct_sum = 0

    conf_sum = 0.0
    conf_n = 0
    brier_sum = 0.0
    brier_n = 0

    rows = iter_jsonl(predictions_path)
    if rows is None:
        return {"n": 0}

    for row in rows:
        n += 1
        score = row.get("score", {}) or {}
        correct = score.get("correct", None)
        if isinstance(correct, (int, float)):
            correct_sum += int(correct)

        pred = row.get("prediction", {}) or {}
        conf = pred.get("confidence", None)
        if isinstance(conf, (int, float)):
            conf_sum += float(conf)
            conf_n += 1

        brier = score.get("brier", None)
        if isinstance(brier, (int, float)):
            brier_sum += float(brier)
            brier_n += 1

    metrics: dict[str, Any] = {"n": n}
    if n > 0:
        metrics["accuracy"] = correct_sum / n
    if conf_n > 0:
        metrics["avg_confidence"] = conf_sum / conf_n
    if brier_n > 0:
        metrics["brier"] = brier_sum / brier_n
    return metrics


def run_eval(
    *,
    model: BaseModelClient,
    bench: BaseBenchmark,
    split: str,
    out_dir: Path,
    max_examples: Optional[int] = None,
    seed: Optional[int] = None,
    exclude_ids: Optional[Set[str]] = None,
    include_ids: Optional[Set[str]] = None,
    resume: bool = True,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    logprobs: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = out_dir / "predictions.jsonl"
    metrics_path = out_dir / "metrics.json"
    sampled_ids_path = out_dir / "sampled_ids.json"

    seen = load_existing_ids(predictions_path) if resume else set()
    exclude_ids = exclude_ids or set()
    include_ids = include_ids or set()

    # If include_ids is provided, only run on those specific IDs (for cross-model comparison)
    if include_ids:
        all_examples = list(bench.iter_examples(split))
        examples = [ex for ex in all_examples if ex.id in include_ids]
        # Save sampled IDs for reference
        sampled_ids = [ex.id for ex in examples]
        sampled_ids_path.write_text(json.dumps(sampled_ids, indent=2))
        print(f"[include_ids] running on {len(examples)}/{len(include_ids)} matching examples")
        print(f"[include_ids] saved IDs to {sampled_ids_path}")
    # Collect all examples for random sampling if seed is provided
    elif seed is not None and max_examples is not None:
        all_examples = list(bench.iter_examples(split))
        # Filter out excluded IDs
        all_examples = [ex for ex in all_examples if ex.id not in exclude_ids]
        # Random sample
        random.seed(seed)
        if len(all_examples) > max_examples:
            examples = random.sample(all_examples, max_examples)
        else:
            examples = all_examples
        # Save sampled IDs for reproducibility and future exclusion
        sampled_ids = [ex.id for ex in examples]
        sampled_ids_path.write_text(json.dumps(sampled_ids, indent=2))
        print(f"[sampling] seed={seed}, sampled {len(examples)}/{len(all_examples)} examples")
        print(f"[sampling] saved IDs to {sampled_ids_path}")
    else:
        examples = None  # Use iterator directly

    processed = 0
    example_iter = iter(examples) if examples is not None else bench.iter_examples(split)
    for ex in example_iter:
        if ex.id in exclude_ids:
            continue
        if resume and ex.id in seen:
            continue

        req: ModelRequest = bench.build_request(ex)

        if temperature is not None:
            req = replace(req, temperature=float(temperature))
        if max_output_tokens is not None:
            req = replace(req, max_output_tokens=int(max_output_tokens))
        if logprobs is not None:
            req = replace(req, logprobs=bool(logprobs))
        if reasoning_effort is not None:
            req = replace(req, reasoning_effort=reasoning_effort)

        resp = model.generate(req)
        pred = bench.parse_prediction(ex, resp)
        score = bench.score(ex, pred)

        record = {
            "id": ex.id,
            "input": ex.input,
            "target": ex.target,
            "meta": ex.meta,
            "request": {
                "messages": req.messages,
                "temperature": req.temperature,
                "max_output_tokens": req.max_output_tokens,
                "logprobs": req.logprobs,
            },
            "response_text": resp.text,
            "thinking": resp.thinking,
            "prediction": {
                "answer": pred.answer,
                "confidence": pred.confidence,
                "extra": pred.extra,
            },
            "score": score,
            "usage": resp.usage,
        }
        append_jsonl(predictions_path, record)

        processed += 1
        # Only apply max_examples limit if not using random sampling (seed handles limit)
        if seed is None and max_examples is not None and processed >= max_examples:
            break

    metrics = aggregate_metrics(predictions_path)
    metrics.update({"benchmark": getattr(bench, "name", None), "split": split, "seed": seed})

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics
