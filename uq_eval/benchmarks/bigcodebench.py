from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


@dataclass(slots=True)
class BigCodeBenchBenchmark(BaseBenchmark):
    """BigCodeBench: Code generation benchmark.

    1,140 practical programming tasks with function calls.
    Dataset: https://huggingface.co/datasets/bigcode/bigcodebench
    """

    name: str = "bigcodebench"
    mode: str = "text"

    # Use "hard" subset (more difficult tasks)
    use_hard: bool = False

    def iter_examples(self, split: str) -> Iterable[Example]:
        # BigCodeBench uses version-based splits like v0.1.2
        ds = load_dataset("bigcode/bigcodebench", split="v0.1.2")

        for idx, row in enumerate(ds):
            # Get the instruction/prompt
            instruction = row.get("instruct_prompt", row.get("complete_prompt", ""))

            # Get canonical solution for reference
            solution = row.get("canonical_solution", "")

            yield Example(
                id=f"bigcodebench_{idx}",
                input=instruction,
                target=solution,  # For reference, actual eval uses test execution
                meta={
                    "split": split,
                    "task_id": row.get("task_id", ""),
                    "test": row.get("test", ""),
                    "libs": row.get("libs", []),
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are an expert Python programmer.\n"
            "Write clean, correct Python code to solve the given task.\n"
            "Return a JSON object with keys: \"code\" (your Python solution), and \"confidence\" (0..1).\n"
            "The code should be complete and runnable."
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

        # Try to extract code from markdown blocks if present
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
            extra={
                "parsed_json": obj is not None,
                "task_id": ex.meta.get("task_id", ""),
            },
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        # Full code evaluation requires execution
        # For now, return placeholder for offline evaluation
        out = {
            "correct": -1,  # Needs execution-based evaluation
            "needs_execution": True,
            "code_length": len(pred.answer),
            "task_id": ex.meta.get("task_id", ""),
        }
        if pred.confidence is not None:
            out["confidence"] = pred.confidence
        return out
