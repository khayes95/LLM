from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@dataclass(slots=True)
class MultiNRCBenchmark(BaseBenchmark):
    """MultiNRC: Multilingual reasoning benchmark.

    1,055 examples testing multilingual reasoning capabilities.
    Dataset: https://huggingface.co/datasets/ScaleAI/MultiNRC
    """

    name: str = "multinrc"
    mode: str = "text"

    # Filter by language (None = all languages)
    language_filter: str | None = None

    def iter_examples(self, split: str) -> Iterable[Example]:
        ds = load_dataset("ScaleAI/MultiNRC", split="test")

        for idx, row in enumerate(ds):
            lang = row.get("language", "")
            if self.language_filter and lang != self.language_filter:
                continue

            yield Example(
                id=f"multinrc_{idx}",
                input=row.get("i18n_prompt", row.get("english_prompt", "")),
                target=row.get("i18n_gtfa", row.get("english_gtfa", "")),
                meta={
                    "split": split,
                    "language": lang,
                    "category": row.get("category", ""),
                    "task_id": row.get("task_id", ""),
                    "english_prompt": row.get("english_prompt", ""),
                    "english_answer": row.get("english_gtfa", ""),
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are a helpful assistant answering questions in any language.\n"
            "Answer in the same language as the question.\n"
            'Return a JSON object with keys: "answer" (your response) and "confidence" (0..1).'
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(ex.input)},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=512)

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

        return Prediction(
            example_id=ex.id,
            answer=str(answer).strip(),
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = normalize_answer(str(ex.target))
        got = normalize_answer(pred.answer)

        # Simple containment check
        correct = int(gold in got or got in gold or gold == got)

        out = {
            "correct": correct,
            "gold": str(ex.target),
            "predicted": pred.answer,
            "language": ex.meta.get("language", ""),
        }
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
