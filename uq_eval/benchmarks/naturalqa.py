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
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@dataclass(slots=True)
class NaturalQuestionsBenchmark(BaseBenchmark):
    """Natural Questions: Real Google search queries.

    Open-domain QA from real user questions.
    Dataset: https://huggingface.co/datasets/google-research-datasets/natural_questions
    """

    name: str = "naturalqa"
    mode: str = "text"

    # Use short answers only (vs long answers)
    short_only: bool = True

    def iter_examples(self, split: str) -> Iterable[Example]:
        # NQ is large, use validation for testing
        ds_split = "validation" if split in ["test", "dev"] else "train"
        ds = load_dataset("google-research-datasets/natural_questions", "default", split=ds_split)

        for idx, row in enumerate(ds):
            question = row["question"]["text"]

            # Get short answer if available
            annotations = row.get("annotations", {})
            short_answers = annotations.get("short_answers", [])

            if short_answers and short_answers[0]:
                answer_tokens = short_answers[0].get("text", [])
                answer = " ".join(answer_tokens) if isinstance(answer_tokens, list) else str(answer_tokens)
            else:
                # Skip if no short answer
                if self.short_only:
                    continue
                answer = ""

            yield Example(
                id=f"naturalqa_{split}_{idx}",
                input=question,
                target=answer,
                meta={
                    "split": split,
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are a helpful assistant answering questions.\n"
            "Provide a brief, factual answer.\n"
            'Return a JSON object with keys: "answer" (your answer) and "confidence" (0..1).'
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(ex.input)},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=256)

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
        gold = normalize_answer(str(ex.target))
        got = normalize_answer(pred.answer)

        correct = int(gold == got or gold in got or got in gold)

        out = {
            "correct": correct,
            "gold": str(ex.target),
            "predicted": pred.answer,
        }
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
