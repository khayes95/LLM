from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj, normalize_text


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison (lowercase, strip punctuation/articles)."""
    s = s.lower().strip()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = re.sub(r'[^\w\s]', '', s)
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def answers_match(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold answer.

    Uses fuzzy matching: gold answer should be contained in prediction,
    or normalized versions should match.
    """
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)

    # Exact match after normalization
    if pred_norm == gold_norm:
        return True

    # Gold contained in prediction (for longer responses)
    if gold_norm in pred_norm:
        return True

    return False


@dataclass(slots=True)
class SimpleQABenchmark(BaseBenchmark):
    """SimpleQA: Short-form factuality benchmark.

    Tests factual knowledge with questions that have short, verifiable answers.
    Dataset: basicv8vc/SimpleQA (4326 examples)

    Note: OpenAI's official eval uses an LLM grader. This implementation uses
    fuzzy string matching for simplicity. For production use, consider adding
    LLM-based grading.
    """

    name: str = "simpleqa"
    mode: str = "text"

    def iter_examples(self, split: str) -> Iterable[Example]:
        # SimpleQA only has test split
        ds = load_dataset("basicv8vc/SimpleQA", split="test")

        for idx, row in enumerate(ds):
            # metadata is stored as a JSON string, need to parse it
            meta_str = row.get("metadata", "{}")
            try:
                meta_dict = json.loads(meta_str) if isinstance(meta_str, str) else meta_str
            except (json.JSONDecodeError, TypeError):
                meta_dict = {}

            yield Example(
                id=f"simpleqa_{idx}",
                input=row["problem"],
                target=row["answer"],
                meta={
                    "split": split,
                    "topic": meta_dict.get("topic", "") if isinstance(meta_dict, dict) else "",
                    "answer_type": meta_dict.get("answer_type", "") if isinstance(meta_dict, dict) else "",
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are a helpful assistant answering factual questions.\n"
            "Answer the question directly and concisely.\n"
            'Return ONLY a JSON object with keys: "answer" (your answer) and "confidence" (0..1).\n'
            "No extra text or explanation."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(ex.input)},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=256)

    def parse_prediction(self, ex: Example, resp: ModelResponse) -> Prediction:
        raw_text = resp.text or ""
        obj = extract_first_json_obj(raw_text)

        answer = None
        confidence = None

        if obj:
            answer = obj.get("answer", None)
            conf = obj.get("confidence", None)
            try:
                if conf is not None:
                    confidence = clamp01(float(conf))
            except Exception:
                confidence = None

        if answer is None:
            # Fallback: use raw text as answer
            answer = raw_text.strip()

        return Prediction(
            example_id=ex.id,
            answer=str(answer).strip(),
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = str(ex.target).strip()
        got = str(pred.answer).strip()

        correct = int(answers_match(got, gold))
        out = {"correct": correct, "gold": gold, "predicted": got}
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
