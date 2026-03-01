from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


def normalize_math_answer(s: str) -> str:
    """Normalize math answers for comparison."""
    s = s.strip()
    # Remove LaTeX boxed
    s = re.sub(r"\\boxed\{([^}]+)\}", r"\1", s)
    # Remove dollar signs
    s = re.sub(r"\$([^$]+)\$", r"\1", s)
    # Remove LaTeX commands
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    # Remove braces
    s = re.sub(r"[{}]", "", s)
    s = s.lower().strip()
    return s


def extract_boxed(s: str) -> str | None:
    """Extract content from \\boxed{...}."""
    match = re.search(r"\\boxed\{([^}]+)\}", s)
    return match.group(1) if match else None


@dataclass(slots=True)
class MATHBenchmark(BaseBenchmark):
    """MATH benchmark: Competition mathematics problems.

    12,500 problems from math competitions (AMC, AIME, etc.)
    Difficulty levels 1-5.
    Dataset: https://huggingface.co/datasets/hendrycks/competition_math
    """

    name: str = "math"
    mode: str = "text"

    # Filter by difficulty level (1-5, None = all)
    difficulty_filter: int | None = None

    # Filter by subject (e.g., "algebra", "geometry")
    subject_filter: str | None = None

    def iter_examples(self, split: str) -> Iterable[Example]:
        # MATH dataset - load all subjects from EleutherAI's version
        subjects = ["algebra", "counting_and_probability", "geometry",
                    "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]

        for subject in subjects:
            if self.subject_filter and self.subject_filter.lower() not in subject.lower():
                continue
            try:
                ds = load_dataset("EleutherAI/hendrycks_math", subject, split=split)
            except Exception:
                continue

            for idx, row in enumerate(ds):
                level = row.get("level", "")
                level_num = None
                if level:
                    import re
                    match = re.search(r"Level (\d)", level)
                    if match:
                        level_num = int(match.group(1))

                if self.difficulty_filter and level_num != self.difficulty_filter:
                    continue

                solution = row.get("solution", "")
                answer = self._extract_boxed(solution) or ""

                yield Example(
                    id=f"math_{subject}_{split}_{idx}",
                    input=row["problem"],
                    target=answer,
                    meta={
                        "split": split,
                        "level": level,
                        "level_num": level_num,
                        "type": subject,
                        "solution": solution,
                    },
                )

    def _extract_boxed(self, s: str) -> str | None:
        import re
        match = re.search(r"\\boxed\{([^}]+)\}", s)
        return match.group(1) if match else None

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are an expert mathematician solving competition math problems.\n"
            "Think step by step, show your work, then provide your final answer.\n"
            'Return a JSON object with keys: "reasoning" (your solution), "answer" (final answer), and "confidence" (0..1).'
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(ex.input)},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=2048)

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
        gold = normalize_math_answer(str(ex.target))
        got = normalize_math_answer(pred.answer)

        correct = 0
        if gold == got:
            correct = 1
        else:
            # Try numeric comparison
            try:
                if abs(float(gold) - float(got)) < 1e-6:
                    correct = 1
            except:
                pass

        out = {
            "correct": correct,
            "gold": str(ex.target),
            "predicted": pred.answer,
            "level": ex.meta.get("level", ""),
            "type": ex.meta.get("type", ""),
        }
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
