from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


@dataclass(slots=True)
class MMLUProBenchmark(BaseBenchmark):
    """MMLU-Pro: Harder MMLU with 10-choice questions.

    A more challenging version of MMLU with expanded answer choices.
    Dataset: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
    """

    name: str = "mmlu_pro"
    mode: str = "text"

    # Filter by category (e.g., "math", "physics", "biology")
    category_filter: str | None = None

    def iter_examples(self, split: str) -> Iterable[Example]:
        # MMLU-Pro has test and validation splits
        ds_split = "test" if split == "test" else "validation"
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split=ds_split)

        for idx, row in enumerate(ds):
            category = row.get("category", "")
            if self.category_filter and self.category_filter.lower() not in category.lower():
                continue

            # Build the question with options
            question = row["question"]
            options = row.get("options", [])

            # Format options as A, B, C, etc.
            formatted_options = []
            for i, opt in enumerate(options):
                letter = chr(ord('A') + i)
                formatted_options.append(f"{letter}. {opt}")

            full_question = question + "\n\n" + "\n".join(formatted_options)

            # Get the correct answer letter
            answer_index = row.get("answer_index", row.get("answer", 0))
            if isinstance(answer_index, int):
                correct_letter = chr(ord('A') + answer_index)
            else:
                correct_letter = str(answer_index)

            yield Example(
                id=f"mmlu_pro_{split}_{idx}",
                input=full_question,
                target=correct_letter,
                meta={
                    "split": split,
                    "category": category,
                    "options": options,
                    "question_only": question,
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are an expert answering multiple-choice questions.\n"
            "Think step by step, then provide your final answer.\n"
            'Return ONLY a JSON object with keys: "reasoning" (your thinking), '
            '"answer" (the letter A-J of your answer), and "confidence" (0..1).'
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(ex.input)},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=1024)

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
            except:
                pass

        if answer is None:
            # Try to extract letter from raw text
            match = re.search(r"\b([A-J])\b", raw_text)
            if match:
                answer = match.group(1)
            else:
                answer = raw_text.strip()[:10]

        return Prediction(
            example_id=ex.id,
            answer=str(answer).strip().upper() if answer else "",
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = str(ex.target).strip().upper()
        got = str(pred.answer).strip().upper()

        # Extract just the letter
        got_letter = re.sub(r"[^A-J]", "", got)
        if got_letter:
            got = got_letter[0]

        correct = int(gold == got)

        out = {
            "correct": correct,
            "gold": gold,
            "predicted": pred.answer,
            "category": ex.meta.get("category", ""),
        }
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
