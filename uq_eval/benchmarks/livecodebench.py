from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


@dataclass(slots=True)
class LiveCodeBenchBenchmark(BaseBenchmark):
    """LiveCodeBench: Continuously updated coding benchmark.

    Fresh coding problems from competitive programming platforms.
    Dataset: https://huggingface.co/datasets/livecodebench/code_generation_lite
    """

    name: str = "livecodebench"
    mode: str = "text"

    # Filter by difficulty
    difficulty_filter: str | None = None

    def iter_examples(self, split: str) -> Iterable[Example]:
        ds = load_dataset(
            "livecodebench/code_generation_lite",
            split="test",
            trust_remote_code=True
        )

        for idx, row in enumerate(ds):
            difficulty = row.get("difficulty", "")
            if self.difficulty_filter and difficulty != self.difficulty_filter:
                continue

            yield Example(
                id=f"livecodebench_{idx}",
                input=row.get("question_content", row.get("prompt", "")),
                target=row.get("solution", ""),
                meta={
                    "split": split,
                    "difficulty": difficulty,
                    "question_id": row.get("question_id", ""),
                    "platform": row.get("platform", ""),
                    "contest_date": row.get("contest_date", ""),
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are an expert competitive programmer.\n"
            "Solve the given problem with clean, efficient code.\n"
            'Return a JSON object with keys: "reasoning" (your approach), "code" (your solution), and "confidence" (0..1).'
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(ex.input)},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=2048)

    def parse_prediction(self, ex: Example, resp: ModelResponse) -> Prediction:
        raw_text = resp.text or ""
        obj = extract_first_json_obj(raw_text)

        code = obj.get("code", raw_text) if obj else raw_text
        confidence = None
        if obj and obj.get("confidence") is not None:
            try:
                confidence = clamp01(float(obj["confidence"]))
            except:
                pass

        # Extract code from markdown blocks
        if "```python" in code:
            match = re.search(r"```python\s*(.*?)\s*```", code, re.DOTALL)
            if match:
                code = match.group(1)
        elif "```" in code:
            match = re.search(r"```\s*(.*?)\s*```", code, re.DOTALL)
            if match:
                code = match.group(1)

        return Prediction(
            example_id=ex.id,
            answer=str(code).strip(),
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        # Requires execution for accurate scoring
        out = {
            "correct": -1,  # Needs execution
            "needs_execution": True,
            "code_length": len(pred.answer),
            "difficulty": ex.meta.get("difficulty", ""),
            "platform": ex.meta.get("platform", ""),
        }
        if pred.confidence is not None:
            out["confidence"] = pred.confidence
        return out
