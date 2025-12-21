from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


@dataclass(slots=True)
class GSM8KBenchmark(BaseBenchmark):
    """GSM8K: Grade School Math 8K.

    8,500 grade school math word problems.
    Dataset: https://huggingface.co/datasets/openai/gsm8k
    """

    name: str = "gsm8k"
    mode: str = "text"

    def iter_examples(self, split: str) -> Iterable[Example]:
        # GSM8K splits are "train" and "test"
        ds_split = "test" if split in ["test", "dev"] else "train"
        ds = load_dataset("openai/gsm8k", "main", split=ds_split)

        for idx, row in enumerate(ds):
            question = row["question"]
            answer_text = row["answer"]

            # Extract the numeric answer (after ####)
            match = re.search(r"####\s*(-?[\d,]+)", answer_text)
            numeric_answer = match.group(1).replace(",", "") if match else ""

            yield Example(
                id=f"gsm8k_{split}_{idx}",
                input=question,
                target=numeric_answer,
                meta={
                    "split": split,
                    "full_solution": answer_text,
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are a helpful math tutor solving grade school math problems.\n"
            "Think step by step, show your work, then provide your final numeric answer.\n"
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
            # Remove commas and find last number
            answer = answer.replace(",", "")
            numbers = re.findall(r"-?\d+\.?\d*", str(answer))
            if numbers:
                answer = numbers[-1]

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
        }
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
