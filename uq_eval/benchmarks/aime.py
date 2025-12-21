from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


@dataclass(slots=True)
class AIMEBenchmark(BaseBenchmark):
    """AIME: American Invitational Mathematics Examination.

    Hard competition math problems (AIME 2024).
    Dataset: https://huggingface.co/datasets/AI-MO/aimo-validation-aime
    """

    name: str = "aime"
    mode: str = "text"

    def iter_examples(self, split: str) -> Iterable[Example]:
        try:
            ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
        except Exception:
            # Try alternative dataset
            ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")

        for idx, row in enumerate(ds):
            yield Example(
                id=f"aime_{idx}",
                input=row.get("problem", row.get("question", "")),
                target=str(row.get("answer", "")),
                meta={
                    "split": split,
                    "url": row.get("url", ""),
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are an expert mathematician solving AIME competition problems.\n"
            "Think step by step, show your work, then provide your final answer.\n"
            "AIME answers are integers from 000 to 999.\n"
            'Return a JSON object with keys: "reasoning" (your solution), "answer" (integer 0-999), and "confidence" (0..1).'
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
            except:
                pass

        # Extract numeric answer
        if isinstance(answer, str):
            numbers = re.findall(r"\d+", str(answer))
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
            gold = int(str(ex.target).strip())
            got = int(str(pred.answer).strip())
            correct = int(gold == got)
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
