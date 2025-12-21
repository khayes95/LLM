from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    s = s.lower().strip()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = re.sub(r"[^\w\s]", "", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


@dataclass(slots=True)
class DROPBenchmark(BaseBenchmark):
    """DROP: Discrete Reasoning Over Paragraphs.

    Reading comprehension requiring discrete reasoning (counting, sorting, etc.)
    Dataset: https://huggingface.co/datasets/ucinlp/drop
    """

    name: str = "drop"
    mode: str = "text"

    def iter_examples(self, split: str) -> Iterable[Example]:
        # DROP only has train and validation splits
        ds_split = "validation" if split in ["test", "dev"] else "train"
        ds = load_dataset("ucinlp/drop", split=ds_split)

        for idx, row in enumerate(ds):
            passage = row["passage"]
            question = row["question"]

            # Get answers (DROP has multiple valid answers)
            answers_spans = row.get("answers_spans", {})
            spans = answers_spans.get("spans", [])
            answer = spans[0] if spans else ""

            full_input = f"Passage:\n{passage}\n\nQuestion: {question}"

            yield Example(
                id=f"drop_{split}_{idx}",
                input=full_input,
                target=answer,
                meta={
                    "split": split,
                    "passage": passage,
                    "question": question,
                    "all_answers": spans,
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are an expert at reading comprehension and discrete reasoning.\n"
            "Read the passage carefully and answer the question.\n"
            'Return a JSON object with keys: "reasoning" (your thinking), "answer" (your answer), and "confidence" (0..1).'
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(ex.input)},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=512)

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
        got_norm = normalize_answer(pred.answer)

        # Check against all valid answers
        all_answers = ex.meta.get("all_answers", [str(ex.target)])
        correct = 0
        for ans in all_answers:
            if normalize_answer(ans) == got_norm:
                correct = 1
                break
            # Also check containment
            if normalize_answer(ans) in got_norm or got_norm in normalize_answer(ans):
                correct = 1
                break

        out = {
            "correct": correct,
            "gold": str(ex.target),
            "predicted": pred.answer,
        }
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
