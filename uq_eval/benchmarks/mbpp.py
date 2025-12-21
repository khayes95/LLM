from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


@dataclass(slots=True)
class MBPPBenchmark(BaseBenchmark):
    """MBPP: Mostly Basic Programming Problems.

    974 Python programming problems (500 test).
    Dataset: https://huggingface.co/datasets/google-research-datasets/mbpp
    """

    name: str = "mbpp"
    mode: str = "text"

    def iter_examples(self, split: str) -> Iterable[Example]:
        # MBPP has train, validation, test, prompt splits
        ds_split = "test" if split in ["test", "dev"] else split
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split=ds_split)

        for idx, row in enumerate(ds):
            # Format prompt with test cases
            prompt = row["prompt"]
            test_list = row.get("test_list", [])

            full_input = f"{prompt}\n\nTest cases:\n" + "\n".join(test_list[:3])

            yield Example(
                id=f"mbpp_{split}_{idx}",
                input=full_input,
                target=row["code"],
                meta={
                    "split": split,
                    "task_id": row.get("task_id", idx),
                    "test_list": test_list,
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are an expert Python programmer.\n"
            "Write a Python function to solve the given problem.\n"
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
            extra={"parsed_json": obj is not None},
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
