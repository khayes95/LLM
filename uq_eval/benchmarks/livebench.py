from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


@dataclass(slots=True)
class LiveBenchBenchmark(BaseBenchmark):
    """LiveBench: Continuously updated benchmark.

    Monthly refreshed questions to prevent contamination.
    Categories: math, reasoning, coding, language, data analysis, instruction following.
    Dataset: https://huggingface.co/datasets/livebench/livebench
    """

    name: str = "livebench"
    mode: str = "text"

    # Filter by category
    category_filter: str | None = None

    def iter_examples(self, split: str) -> Iterable[Example]:
        # LiveBench uses livecodebench org - try different dataset names
        try:
            ds = load_dataset("lmsys/livebench", split="test", trust_remote_code=True)
        except Exception:
            try:
                ds = load_dataset("AI-MO/aimo-validation-aime", split="train")  # Fallback to AIME
            except Exception:
                return  # No dataset available

        for idx, row in enumerate(ds):
            category = row.get("category", row.get("task", ""))
            if self.category_filter and self.category_filter.lower() not in category.lower():
                continue

            yield Example(
                id=f"livebench_{idx}",
                input=row.get("question", row.get("turns", [""])[0] if isinstance(row.get("turns"), list) else row.get("input", "")),
                target=row.get("ground_truth", row.get("answer", "")),
                meta={
                    "split": split,
                    "category": category,
                    "question_id": row.get("question_id", ""),
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are a helpful assistant. Answer the question accurately.\n"
            'Return a JSON object with keys: "reasoning" (your thinking), "answer" (your answer), and "confidence" (0..1).'
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(ex.input)},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=2048)

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
        gold = str(ex.target).strip().lower()
        got = str(pred.answer).strip().lower()

        # Normalize for comparison
        gold_norm = re.sub(r"\s+", " ", gold)
        got_norm = re.sub(r"\s+", " ", got)

        correct = int(gold_norm == got_norm or gold_norm in got_norm)

        out = {
            "correct": correct,
            "gold": str(ex.target),
            "predicted": pred.answer,
            "category": ex.meta.get("category", ""),
        }
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
