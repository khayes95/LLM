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
class TriviaQABenchmark(BaseBenchmark):
    """TriviaQA: Large scale trivia question answering.

    650K question-answer pairs with evidence documents.
    Dataset: https://huggingface.co/datasets/trivia_qa
    """

    name: str = "triviaqa"
    mode: str = "text"

    # Use "rc" (reading comprehension with evidence) or "unfiltered" (no context)
    subset: str = "rc"

    def iter_examples(self, split: str) -> Iterable[Example]:
        # TriviaQA has train, test, validation splits (no "dev")
        ds_split = "validation" if split == "dev" else split
        ds = load_dataset("trivia_qa", self.subset, split=ds_split)

        for idx, row in enumerate(ds):
            question = row["question"]

            # Get answer aliases
            answer_data = row.get("answer", {})
            answer = answer_data.get("value", "")
            aliases = answer_data.get("aliases", [])

            yield Example(
                id=f"triviaqa_{self.subset}_{split}_{idx}",
                input=question,
                target=answer,
                meta={
                    "split": split,
                    "subset": self.subset,
                    "aliases": aliases,
                    "normalized_aliases": answer_data.get("normalized_aliases", []),
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are a trivia expert. Answer the question concisely.\n"
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
        got_norm = normalize_answer(pred.answer)

        # Check against all aliases
        all_answers = [str(ex.target)] + ex.meta.get("aliases", [])
        correct = 0
        for ans in all_answers:
            ans_norm = normalize_answer(ans)
            if ans_norm == got_norm or ans_norm in got_norm:
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
