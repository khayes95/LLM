from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_choice_letter, extract_first_json_obj


@dataclass(slots=True)
class SanityMCQBenchmark(BaseBenchmark):
    """Small multiple-choice benchmark to validate MCQ parsing + confidence."""

    name: str = "sanity_mcq"
    mode: str = "text"

    def iter_examples(self, split: str) -> Iterable[Example]:
        data = [
            (
                "mcq1",
                "Which planet is closest to the Sun?\nA) Earth\nB) Venus\nC) Mercury\nD) Mars",
                "C",
            ),
            (
                "mcq2",
                "What is 2 + 2?\nA) 3\nB) 4\nC) 5\nD) 6",
                "B",
            ),
            (
                "mcq3",
                "Capital of Italy?\nA) Milan\nB) Rome\nC) Venice\nD) Turin",
                "B",
            ),
            (
                "mcq4",
                "Which is a prime number?\nA) 9\nB) 21\nC) 29\nD) 39",
                "C",
            ),
            (
                "mcq5",
                "Which is NOT a programming language?\nA) Python\nB) Java\nC) HTML\nD) Banana",
                "D",
            ),
        ]
        for ex_id, q, ans in data:
            yield Example(id=ex_id, input=q, target=ans, meta={"split": split})

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are running in an evaluation harness.\n"
            "This is a multiple-choice question.\n"
            'Return ONLY a JSON object with keys: "answer" (one of A/B/C/D) and "confidence" (0..1).\n'
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
            # fallback: extract last A/B/C/D in the text
            answer = extract_choice_letter(raw_text) or raw_text.strip()

        answer = str(answer).strip().upper()

        return Prediction(
            example_id=ex.id,
            answer=answer,
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = str(ex.target).strip().upper()
        got = str(pred.answer).strip().upper()

        correct = int(gold == got)
        out = {"correct": correct}
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
