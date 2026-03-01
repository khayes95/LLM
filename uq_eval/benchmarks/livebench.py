from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


# Available LiveBench categories
LIVEBENCH_CATEGORIES = [
    "math",
    "reasoning",
    "coding",
    "language",
    "data_analysis",
    "instruction_following",
]


@dataclass(slots=True)
class LiveBenchBenchmark(BaseBenchmark):
    """LiveBench: Continuously updated benchmark.

    Monthly refreshed questions to prevent contamination.
    Categories: math, reasoning, coding, language, data_analysis, instruction_following.

    Dataset: https://huggingface.co/datasets/livebench/*
    Website: https://livebench.ai/

    GPT-5 scores ~79% overall, with reasoning at 98% and agentic coding ~50%.
    """

    name: str = "livebench"
    mode: str = "text"

    # Filter by category (math, reasoning, coding, language, data_analysis, instruction_following)
    # Use None for all categories
    category_filter: str | None = None

    # List of categories to include (alternative to category_filter for multiple)
    categories: list[str] = field(default_factory=lambda: LIVEBENCH_CATEGORIES.copy())

    def iter_examples(self, split: str) -> Iterable[Example]:
        # Determine which categories to load
        if self.category_filter:
            cats_to_load = [self.category_filter]
        else:
            cats_to_load = self.categories

        example_idx = 0
        for category in cats_to_load:
            if category not in LIVEBENCH_CATEGORIES:
                continue

            try:
                ds = load_dataset(f"livebench/{category}", split="test")
            except Exception:
                continue

            for row in ds:
                # Extract question from turns (list of conversation turns)
                turns = row.get("turns", [])
                if isinstance(turns, list) and len(turns) > 0:
                    question = turns[0]
                else:
                    question = str(turns) if turns else ""

                # Ground truth answer
                ground_truth = row.get("ground_truth", "")

                yield Example(
                    id=f"livebench_{category}_{example_idx}",
                    input=question,
                    target=ground_truth,
                    meta={
                        "split": split,
                        "category": category,
                        "task": row.get("task", ""),
                        "subtask": row.get("subtask", ""),
                        "question_id": row.get("question_id", ""),
                        "hardness": row.get("hardness", ""),
                        "expressions": row.get("expressions", ""),
                    },
                )
                example_idx += 1

    def build_request(self, ex: Example) -> ModelRequest:
        category = ex.meta.get("category", "")

        if category == "math":
            system = (
                "You are an expert mathematician. Solve the problem step by step.\n"
                'Return a JSON object with keys: "reasoning" (your solution steps), '
                '"answer" (your final answer), and "confidence" (0..1).'
            )
        elif category == "coding":
            system = (
                "You are an expert programmer. Solve the coding problem.\n"
                'Return a JSON object with keys: "reasoning" (your approach), '
                '"answer" (your code or solution), and "confidence" (0..1).'
            )
        elif category == "reasoning":
            system = (
                "You are an expert at logical reasoning. Think carefully step by step.\n"
                'Return a JSON object with keys: "reasoning" (your logical steps), '
                '"answer" (your conclusion), and "confidence" (0..1).'
            )
        else:
            system = (
                "You are a helpful assistant. Answer the question accurately.\n"
                'Return a JSON object with keys: "reasoning" (your thinking), '
                '"answer" (your answer), and "confidence" (0..1).'
            )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(ex.input)},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=4096)

    def parse_prediction(self, ex: Example, resp: ModelResponse) -> Prediction:
        raw_text = resp.text or ""
        obj = extract_first_json_obj(raw_text)

        answer = obj.get("answer", raw_text) if obj else raw_text
        confidence = None
        if obj and obj.get("confidence") is not None:
            try:
                confidence = clamp01(float(obj["confidence"]))
            except Exception:
                pass

        return Prediction(
            example_id=ex.id,
            answer=str(answer).strip(),
            confidence=confidence,
            raw_text=raw_text,
            extra={
                "parsed_json": obj is not None,
                "category": ex.meta.get("category", ""),
            },
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = str(ex.target).strip()
        got = str(pred.answer).strip()

        # Normalize for comparison
        gold_norm = re.sub(r"\s+", " ", gold.lower())
        got_norm = re.sub(r"\s+", " ", got.lower())

        # Check for exact match or containment
        correct = int(gold_norm == got_norm or gold_norm in got_norm)

        out = {
            "correct": correct,
            "gold": gold,
            "predicted": got,
            "category": ex.meta.get("category", ""),
            "task": ex.meta.get("task", ""),
        }
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
