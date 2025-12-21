from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_choice_letter, extract_first_json_obj


def preprocess(text: str) -> str:
    """Clean up answer text (from lm-eval-harness)."""
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


@dataclass(slots=True)
class GPQABenchmark(BaseBenchmark):
    """GPQA: A Graduate-Level Google-Proof Q&A Benchmark.

    Multiple-choice questions in biology, physics, and chemistry.
    Requires HuggingFace authentication for gated dataset access.

    Subsets: gpqa_main (448), gpqa_diamond (198), gpqa_extended (546)
    """

    name: str = "gpqa"
    mode: str = "text"
    subset: str = "gpqa_diamond"  # gpqa_main, gpqa_diamond, gpqa_extended
    seed: int = 42  # for reproducible answer shuffling

    _rng: random.Random = field(default_factory=lambda: random.Random(42), repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def iter_examples(self, split: str) -> Iterable[Example]:
        # GPQA only has train split on HuggingFace
        ds = load_dataset("Idavidrein/gpqa", self.subset, split="train")

        for idx, row in enumerate(ds):
            # Shuffle choices (same logic as lm-eval-harness)
            choices = [
                preprocess(row["Incorrect Answer 1"]),
                preprocess(row["Incorrect Answer 2"]),
                preprocess(row["Incorrect Answer 3"]),
                preprocess(row["Correct Answer"]),
            ]

            # Use a seeded RNG for reproducibility
            rng = random.Random(self.seed + idx)
            rng.shuffle(choices)

            correct_answer = preprocess(row["Correct Answer"])
            correct_idx = choices.index(correct_answer)
            correct_letter = chr(65 + correct_idx)  # A, B, C, D

            question = row["Question"]

            yield Example(
                id=f"gpqa_{self.subset}_{idx}",
                input={
                    "question": question,
                    "choices": choices,
                },
                target=correct_letter,
                meta={
                    "split": split,
                    "subset": self.subset,
                    "correct_answer_text": correct_answer,
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        question = ex.input["question"]
        choices = ex.input["choices"]

        # Format choices
        choices_text = "\n".join([
            f"(A) {choices[0]}",
            f"(B) {choices[1]}",
            f"(C) {choices[2]}",
            f"(D) {choices[3]}",
        ])

        prompt = f"What is the correct answer to this question:\n{question}\n\nChoices:\n{choices_text}"

        system = (
            "You are an expert taking a graduate-level exam in science (biology, physics, or chemistry).\n"
            "Answer the multiple-choice question by selecting the correct option.\n"
            'Return ONLY a JSON object with keys: "answer" (one of A/B/C/D) and "confidence" (0..1).\n'
            "No extra text or explanation."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=128)

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
            # Fallback: extract last A/B/C/D in the text
            answer = extract_choice_letter(raw_text) or raw_text.strip()

        # Normalize answer to single letter
        answer = str(answer).strip().upper()
        if answer.startswith("(") and answer.endswith(")"):
            answer = answer[1:-1]
        if len(answer) > 1:
            answer = answer[0]

        return Prediction(
            example_id=ex.id,
            answer=answer,
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = str(ex.target).strip().upper()
        got = str(pred.answer).strip().upper()

        correct = int(gold == got)
        out = {"correct": correct}
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
