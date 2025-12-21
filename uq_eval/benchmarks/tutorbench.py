from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


@dataclass(slots=True)
class TutorBenchBenchmark(BaseBenchmark):
    """TutorBench: Tutoring evaluation benchmark.

    Evaluates LLM tutoring capabilities with 1,490 examples.
    Dataset: https://huggingface.co/datasets/ScaleAI/TutorBench

    Note: Has visual component - this implementation is text-only.
    """

    name: str = "tutorbench"
    mode: str = "text"

    def iter_examples(self, split: str) -> Iterable[Example]:
        # TutorBench only has train split
        ds = load_dataset("ScaleAI/TutorBench", split="train")

        for idx, row in enumerate(ds):
            # Skip examples that require images
            if row.get("image") is not None:
                continue

            yield Example(
                id=f"tutorbench_{idx}",
                input=row.get("question", row.get("prompt", "")),
                target=row.get("answer", row.get("response", "")),
                meta={
                    "split": split,
                    "subject": row.get("subject", ""),
                    "difficulty": row.get("difficulty", ""),
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are a helpful tutor. Answer the student's question clearly and educationally.\n"
            'Return a JSON object with keys: "answer" (your response) and "confidence" (0..1).'
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

        return Prediction(
            example_id=ex.id,
            answer=str(answer).strip(),
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        # TutorBench typically uses LLM-as-judge for scoring
        # For now, return placeholder for offline grading
        out = {
            "correct": -1,  # Needs LLM judge
            "needs_grading": True,
            "response_length": len(pred.answer),
        }
        if pred.confidence is not None:
            out["confidence"] = pred.confidence
        return out
