from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


def normalize_chemistry_answer(s: str) -> str:
    """Normalize chemistry answers for comparison."""
    s = s.lower().strip()
    # Remove common units and formatting
    s = re.sub(r"\s+", " ", s)
    return s


@dataclass(slots=True)
class Ether0Benchmark(BaseBenchmark):
    """Ether0: Hard chemistry reasoning benchmark.

    325 expert-level chemistry problems.
    GPT-5 accuracy: 20-60%
    Dataset: https://huggingface.co/datasets/futurehouse/ether0-benchmark
    """

    name: str = "ether0"
    mode: str = "text"

    def iter_examples(self, split: str) -> Iterable[Example]:
        ds = load_dataset("futurehouse/ether0-benchmark", split="test")

        for idx, row in enumerate(ds):
            yield Example(
                id=f"ether0_{idx}",
                input=row["problem"],
                target=row.get("ideal", row.get("solution", "")),
                meta={
                    "split": split,
                    "problem_type": row.get("problem_type", ""),
                    "problem_id": row.get("id", ""),
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are an expert chemist solving chemistry problems.\n"
            "Think step by step, then provide your final answer.\n"
            'Return a JSON object with keys: "reasoning" (your work), "answer" (final answer), and "confidence" (0..1).'
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
        gold = normalize_chemistry_answer(str(ex.target))
        got = normalize_chemistry_answer(pred.answer)

        # Chemistry answers often need expert/LLM grading
        # Simple check: exact match or containment
        correct = int(gold == got or gold in got)

        out = {
            "correct": correct,
            "gold": str(ex.target),
            "predicted": pred.answer,
            "problem_type": ex.meta.get("problem_type", ""),
        }
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
