from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj, normalize_text


@dataclass(slots=True)
class SanityLongContextBenchmark(BaseBenchmark):
    """Tiny long-context retrieval test (not huge tokens, but checks prompt plumbing)."""

    name: str = "sanity_long_context"
    mode: str = "longtext"

    def iter_examples(self, split: str) -> Iterable[Example]:
        ctx1 = (
            "You are given a document.\n\n"
            "Section 1:\n"
            "This is filler text. " * 50
            + "\nSection 2:\n"
            + "More filler text. " * 50
            + "\nSection 3:\n"
            + "IMPORTANT: The secret codeword is BLUEBIRD.\n"
            + "Even more filler text. " * 30
        )
        ctx2 = (
            "Notes:\n"
            + "blah " * 120
            + "\nIMPORTANT: The passphrase is STARLIGHT.\n"
            + "end " * 80
        )
        data = [
            ("lc1", f"{ctx1}\n\nQuestion: What is the secret codeword?", "BLUEBIRD"),
            ("lc2", f"{ctx2}\n\nQuestion: What is the passphrase?", "STARLIGHT"),
        ]
        for ex_id, q, ans in data:
            yield Example(id=ex_id, input=q, target=ans, meta={"split": split})

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are running in an evaluation harness.\n"
            "Read the provided text carefully.\n"
            'Return ONLY a JSON object with keys: "answer" (string) and "confidence" (0..1).\n'
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
            answer = raw_text.strip()

        return Prediction(
            example_id=ex.id,
            answer=str(answer).strip(),
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = normalize_text(ex.target)
        got = normalize_text(pred.answer)

        correct = int(gold == got)
        out = {"correct": correct}
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
