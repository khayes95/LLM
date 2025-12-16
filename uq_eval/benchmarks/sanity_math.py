from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj, extract_last_number


@dataclass(slots=True)
class SanityMathBenchmark(BaseBenchmark):
    """Small numeric benchmark; score compares numeric values (tolerant)."""

    name: str = "sanity_math"
    mode: str = "text"

    def iter_examples(self, split: str) -> Iterable[Example]:
        data = [
            ("math1", "Compute 17 + 25.", 42),
            ("math2", "Compute 13 * 7.", 91),
            ("math3", "Compute 144 / 12.", 12),
            ("math4", "Compute 2^10 (2 to the power of 10).", 1024),
            ("math5", "Compute 0.25 * 0.8.", 0.2),
        ]
        for ex_id, q, ans in data:
            yield Example(id=ex_id, input=q, target=ans, meta={"split": split})

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are running in an evaluation harness.\n"
            "Answer the user's math question.\n"
            'Return ONLY a JSON object with keys: "answer" (number) and "confidence" (0..1).\n'
            "No extra text."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(ex.input)},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=128)

    def parse_prediction(self, ex: Example, resp: ModelResponse) -> Prediction:
        raw_text = resp.text or ""
        obj = extract_first_json_obj(raw_text)

        answer = None
        confidence = None

        if obj:
            answer = obj.get("answer", None)
            conf = obj.get("confidence", None)
            try:
                if conf is not None:
                    confidence = clamp01(float(conf))
            except Exception:
                confidence = None

        if answer is None:
            # fallback: extract last number
            answer = extract_last_number(raw_text) or raw_text.strip()

        return Prediction(
            example_id=ex.id,
            answer=answer,
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = float(ex.target)
        got = None
        try:
            got = float(str(pred.answer).strip())
        except Exception:
            got = None

        correct = int(got is not None and abs(got - gold) <= 1e-6)
        out = {"correct": correct}
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
