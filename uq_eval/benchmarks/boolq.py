from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


@dataclass(slots=True)
class BoolQBenchmark(BaseBenchmark):
    """BoolQ: Boolean Questions.

    Yes/no questions from real Google queries.
    Dataset: https://huggingface.co/datasets/google/boolq
    """

    name: str = "boolq"
    mode: str = "text"

    def iter_examples(self, split: str) -> Iterable[Example]:
        # BoolQ only has train and validation splits
        ds_split = "validation" if split in ["test", "dev"] else "train"
        ds = load_dataset("google/boolq", split=ds_split)

        for idx, row in enumerate(ds):
            passage = row["passage"]
            question = row["question"]

            full_input = f"Passage: {passage}\n\nQuestion: {question}\n\nAnswer with Yes or No."

            answer = "Yes" if row["answer"] else "No"

            yield Example(
                id=f"boolq_{split}_{idx}",
                input=full_input,
                target=answer,
                meta={
                    "split": split,
                    "passage": passage,
                    "question": question,
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are a helpful assistant answering yes/no questions based on the passage.\n"
            'Return a JSON object with keys: "answer" (Yes or No) and "confidence" (0..1).'
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(ex.input)},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=128)

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

        # Normalize to Yes/No
        answer_lower = str(answer).lower().strip()
        if "yes" in answer_lower:
            answer = "Yes"
        elif "no" in answer_lower:
            answer = "No"

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

        correct = int(gold == got)

        out = {
            "correct": correct,
            "gold": str(ex.target),
            "predicted": pred.answer,
        }
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
