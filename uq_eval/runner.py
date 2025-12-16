from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

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
    resume: bool = True,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    logprobs: Optional[bool] = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = out_dir / "predictions.jsonl"
    metrics_path = out_dir / "metrics.json"

    seen = load_existing_ids(predictions_path) if resume else set()

    processed = 0
    for ex in bench.iter_examples(split):
        if resume and ex.id in seen:
            continue

        req: ModelRequest = bench.build_request(ex)

        if temperature is not None:
            req = replace(req, temperature=float(temperature))
        if max_output_tokens is not None:
            req = replace(req, max_output_tokens=int(max_output_tokens))
        if logprobs is not None:
            req = replace(req, logprobs=bool(logprobs))

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
        if max_examples is not None and processed >= max_examples:
            break

    metrics = aggregate_metrics(predictions_path)
    metrics.update({"benchmark": getattr(bench, "name", None), "split": split})

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics
