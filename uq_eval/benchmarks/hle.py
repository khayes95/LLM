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


def check_mcq_answer(predicted: str, gold: str) -> bool:
    """Check if MCQ answer matches (e.g., 'A', '(A)', 'A)')."""
    pred_clean = predicted.strip().upper()
    gold_clean = gold.strip().upper()

    # Direct match
    if pred_clean == gold_clean:
        return True

    # Extract letter from various formats
    pred_letter = re.sub(r"[^A-Z]", "", pred_clean)
    gold_letter = re.sub(r"[^A-Z]", "", gold_clean)

    if pred_letter and gold_letter and pred_letter[0] == gold_letter[0]:
        return True

    return False


def check_short_answer(predicted: str, gold: str) -> bool:
    """Fuzzy check for short answer questions."""
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)

    # Exact match after normalization
    if pred_norm == gold_norm:
        return True

    # Gold contained in prediction
    if gold_norm and gold_norm in pred_norm:
        return True

    # All words in gold appear in prediction
    gold_words = set(gold_norm.split())
    pred_words = set(pred_norm.split())
    if gold_words and gold_words.issubset(pred_words):
        return True

    return False


@dataclass(slots=True)
class HLEBenchmark(BaseBenchmark):
    """Humanity's Last Exam (HLE) benchmark.

    A multi-modal benchmark with 2,500 expert-level questions across dozens of
    subjects (mathematics, humanities, natural sciences).

    Dataset: https://huggingface.co/datasets/cais/hle
    Paper: https://arxiv.org/abs/2501.14249

    Note: This implementation filters to text-only questions by default.
    For full multimodal evaluation, set text_only=False (requires vision model).
    """

    name: str = "hle"
    mode: str = "text"

    # Filter to text-only questions (no images required)
    text_only: bool = True

    # Filter by answer type: "mcq", "short_answer", or None for all
    answer_type_filter: str | None = None

    # Filter by subject category (e.g., "Mathematics", "Physics")
    category_filter: str | None = None

    def iter_examples(self, split: str) -> Iterable[Example]:
        # HLE only has test split
        ds = load_dataset("cais/hle", split="test")

        for idx, row in enumerate(ds):
            # Filter for text-only if requested
            if self.text_only and row.get("image") is not None:
                continue

            # Filter by answer type if specified
            answer_type = row.get("answer_type", "")
            if self.answer_type_filter:
                if self.answer_type_filter == "mcq" and answer_type != "mcq":
                    continue
                if self.answer_type_filter == "short_answer" and answer_type != "short_answer":
                    continue

            # Filter by category if specified
            if self.category_filter:
                category = row.get("category", "")
                if self.category_filter.lower() not in category.lower():
                    continue

            yield Example(
                id=f"hle_{idx}",
                input=row["question"],
                target=row["answer"],
                meta={
                    "split": split,
                    "answer_type": answer_type,
                    "category": row.get("category", ""),
                    "raw_subject": row.get("raw_subject", ""),
                    "has_image": row.get("image") is not None,
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        answer_type = ex.meta.get("answer_type", "")

        if answer_type == "mcq":
            system = (
                "You are an expert answering multiple-choice questions.\n"
                "Think step by step, then provide your final answer.\n"
                'Return ONLY a JSON object with keys: "reasoning" (your step-by-step thinking), '
                '"answer" (the letter of your answer, e.g., "A", "B", "C", or "D"), '
                'and "confidence" (0..1 how confident you are).'
            )
        else:
            system = (
                "You are an expert answering academic questions.\n"
                "Think step by step, then provide your final answer.\n"
                'Return ONLY a JSON object with keys: "reasoning" (your step-by-step thinking), '
                '"answer" (your concise answer), '
                'and "confidence" (0..1 how confident you are).'
            )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(ex.input)},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=2048)

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
            # Fallback: try to extract answer from raw text
            # Look for "answer is X" or "Answer: X" patterns
            match = re.search(r"(?:answer\s+is|answer:)\s*([A-Z]|\S+)", raw_text, re.IGNORECASE)
            if match:
                answer = match.group(1)
            else:
                answer = raw_text.strip()[:200]  # Truncate if very long

        return Prediction(
            example_id=ex.id,
            answer=str(answer).strip() if answer else "",
            confidence=confidence,
            raw_text=raw_text,
            extra={
                "parsed_json": obj is not None,
                "answer_type": ex.meta.get("answer_type", ""),
            },
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = str(ex.target).strip()
        got = str(pred.answer).strip()
        answer_type = ex.meta.get("answer_type", "")

        # Score based on answer type
        if answer_type == "mcq":
            correct = int(check_mcq_answer(got, gold))
        else:
            # Short answer - use fuzzy matching
            # Note: Official HLE uses LLM judge for short answers
            correct = int(check_short_answer(got, gold))

        out = {
            "correct": correct,
            "gold": gold,
            "predicted": got,
            "answer_type": answer_type,
            "category": ex.meta.get("category", ""),
        }
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
