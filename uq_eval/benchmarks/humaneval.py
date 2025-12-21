from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


@dataclass(slots=True)
class HumanEvalBenchmark(BaseBenchmark):
    """HumanEval: Code generation benchmark.

    164 Python programming problems with test cases.
    Dataset: https://huggingface.co/datasets/openai/openai_humaneval
    """

    name: str = "humaneval"
    mode: str = "text"

    def iter_examples(self, split: str) -> Iterable[Example]:
        ds = load_dataset("openai/openai_humaneval", split="test")

        for idx, row in enumerate(ds):
            yield Example(
                id=f"humaneval_{idx}",
                input=row["prompt"],
                target=row["canonical_solution"],
                meta={
                    "split": split,
                    "task_id": row["task_id"],
                    "entry_point": row["entry_point"],
                    "test": row["test"],
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are an expert Python programmer.\n"
            "Complete the given function. Only output the code to complete the function.\n"
            'Return a JSON object with keys: "code" (your Python code) and "confidence" (0..1).'
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(ex.input)},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=1024)

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
            extra={
                "parsed_json": obj is not None,
                "task_id": ex.meta.get("task_id", ""),
            },
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        # Requires execution for accurate scoring
        out = {
            "correct": -1,  # Needs execution
            "needs_execution": True,
            "code_length": len(pred.answer),
            "task_id": ex.meta.get("task_id", ""),
        }
        if pred.confidence is not None:
            out["confidence"] = pred.confidence
        return out


@dataclass(slots=True)
class HumanEvalPlusBenchmark(BaseBenchmark):
    """HumanEval+: Extended HumanEval with more test cases.

    Same 164 problems but with 80x more test cases.
    Dataset: https://huggingface.co/datasets/evalplus/humanevalplus
    """

    name: str = "humanevalplus"
    mode: str = "text"

    def iter_examples(self, split: str) -> Iterable[Example]:
        ds = load_dataset("evalplus/humanevalplus", split="test")

        for idx, row in enumerate(ds):
            yield Example(
                id=f"humanevalplus_{idx}",
                input=row["prompt"],
                target=row["canonical_solution"],
                meta={
                    "split": split,
                    "task_id": row["task_id"],
                    "entry_point": row["entry_point"],
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are an expert Python programmer.\n"
            "Complete the given function. Only output the code to complete the function.\n"
            'Return a JSON object with keys: "code" (your Python code) and "confidence" (0..1).'
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(ex.input)},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=1024)

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
        out = {
            "correct": -1,
            "needs_execution": True,
            "code_length": len(pred.answer),
            "task_id": ex.meta.get("task_id", ""),
        }
        if pred.confidence is not None:
            out["confidence"] = pred.confidence
        return out
