from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


# Languages in MGSM
MGSM_LANGUAGES = [
    "bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"
]


@dataclass(slots=True)
class MGSMBenchmark(BaseBenchmark):
    """MGSM: Multilingual Grade School Math.

    250 math problems per language across 11 languages.
    Dataset: https://huggingface.co/datasets/juletxara/mgsm
    """

    name: str = "mgsm"
    mode: str = "text"

    # Filter by language (e.g., "en", "zh", "es")
    language: str = "en"

    def iter_examples(self, split: str) -> Iterable[Example]:
        # MGSM only has test split
        ds = load_dataset("juletxara/mgsm", self.language, split="test")

        for idx, row in enumerate(ds):
            # Extract numeric answer
            answer = row.get("answer_number", row.get("answer", ""))

            yield Example(
                id=f"mgsm_{self.language}_{idx}",
                input=row["question"],
                target=str(answer),
                meta={
                    "split": split,
                    "language": self.language,
                    "equation": row.get("equation_solution", ""),
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are a helpful assistant solving math problems.\n"
            "Think step by step, then provide your final numeric answer.\n"
            'Return a JSON object with keys: "reasoning" (your steps), "answer" (final number), and "confidence" (0..1).'
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(ex.input)},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=1024)

    def parse_prediction(self, ex: Example, resp: ModelResponse) -> Prediction:
        raw_text = resp.text or ""
        obj = extract_first_json_obj(raw_text)

        answer = obj.get("answer", raw_text) if obj else raw_text
        confidence = None
        if obj and obj.get("confidence") is not None:
            try:
                confidence = clamp01(float(obj["confidence"]))
            except:
                pass

        # Extract numeric answer
        if isinstance(answer, str):
            # Try to find a number in the answer
            numbers = re.findall(r"-?\d+\.?\d*", str(answer))
            if numbers:
                answer = numbers[-1]  # Take last number

        return Prediction(
            example_id=ex.id,
            answer=str(answer).strip(),
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        try:
            gold = float(str(ex.target).replace(",", ""))
            got = float(str(pred.answer).replace(",", ""))
            correct = int(abs(gold - got) < 1e-3)
        except:
            correct = int(str(ex.target).strip() == str(pred.answer).strip())

        out = {
            "correct": correct,
            "gold": str(ex.target),
            "predicted": pred.answer,
            "language": ex.meta.get("language", ""),
        }
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
