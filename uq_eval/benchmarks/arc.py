from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


@dataclass(slots=True)
class ARCBenchmark(BaseBenchmark):
    """ARC: AI2 Reasoning Challenge.

    Science questions from 3rd-9th grade.
    Challenge set has 2,590 questions (harder subset).
    Dataset: https://huggingface.co/datasets/allenai/ai2_arc
    """

    name: str = "arc"
    mode: str = "text"

    # Use challenge set (harder) or easy set
    use_challenge: bool = True

    def iter_examples(self, split: str) -> Iterable[Example]:
        subset = "ARC-Challenge" if self.use_challenge else "ARC-Easy"
        # ARC has train, test, validation splits (no "dev")
        ds_split = "validation" if split == "dev" else split
        ds = load_dataset("allenai/ai2_arc", subset, split=ds_split)

        for idx, row in enumerate(ds):
            question = row["question"]
            choices = row["choices"]

            # Format choices
            labels = choices["label"]
            texts = choices["text"]
            formatted = []
            for label, text in zip(labels, texts):
                formatted.append(f"{label}. {text}")

            full_question = question + "\n\n" + "\n".join(formatted)

            yield Example(
                id=f"arc_{subset}_{split}_{idx}",
                input=full_question,
                target=row["answerKey"],
                meta={
                    "split": split,
                    "subset": subset,
                    "question_only": question,
                    "choices": choices,
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are an expert answering science multiple-choice questions.\n"
            "Think through the question carefully, then provide your answer.\n"
            'Return ONLY a JSON object with keys: "reasoning" (your thinking), '
            '"answer" (the letter of your answer), and "confidence" (0..1).'
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(ex.input)},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=512)

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
            # Try to extract letter
            match = re.search(r"\b([A-E])\b", raw_text)
            if match:
                answer = match.group(1)
            else:
                answer = raw_text.strip()[:5]

        return Prediction(
            example_id=ex.id,
            answer=str(answer).strip().upper() if answer else "",
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = str(ex.target).strip().upper()
        got = re.sub(r"[^A-E]", "", str(pred.answer).upper())
        if got:
            got = got[0]

        correct = int(gold == got)

        out = {
            "correct": correct,
            "gold": gold,
            "predicted": pred.answer,
            "subset": ex.meta.get("subset", ""),
        }
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
