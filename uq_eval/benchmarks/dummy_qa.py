from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _extract_first_json_obj(text: str) -> dict | None:
    """Best-effort JSON extraction: find the first {...} block and parse it."""
    if not text:
        return None

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None

    candidate = m.group(0)
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


@dataclass(slots=True)
class DummyQABenchmark(BaseBenchmark):
    """A tiny smoke-test benchmark to validate the framework end-to-end."""

    name: str = "dummy_qa"
    mode: str = "text"

    def iter_examples(self, split: str) -> Iterable[Example]:
        data = [
            ("q1", "What is the capital of France?", "Paris"),
            ("q2", "What is 2 + 2?", "4"),
            ("q3", "Which planet is known as the Red Planet?", "Mars"),
        ]
        for ex_id, q, a in data:
            yield Example(id=ex_id, input=q, target=a, meta={"split": split})

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are running in an evaluation harness.\n"
            "Answer the user's question.\n"
            'Return ONLY a JSON object with keys: "answer" (string) and "confidence" (number 0..1).\n'
            "No extra text, no markdown."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(ex.input)},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=256)

    def parse_prediction(self, ex: Example, resp: ModelResponse) -> Prediction:
        raw_text = resp.text or ""
        obj = _extract_first_json_obj(raw_text)

        answer = None
        confidence: float | None = None

        if obj:
            answer = obj.get("answer", None)
            conf = obj.get("confidence", None)
            try:
                if conf is not None:
                    confidence = _clamp01(float(conf))
            except Exception:
                confidence = None

        if answer is None:
            answer = raw_text.strip()

        return Prediction(
            example_id=ex.id,
            answer=answer,
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = _normalize_text(str(ex.target))
        got = _normalize_text(str(pred.answer))

        correct = int(gold == got)
        out = {"correct": correct}

        if pred.confidence is not None:
            out["brier"] = (pred.confidence - correct) ** 2

        return out
