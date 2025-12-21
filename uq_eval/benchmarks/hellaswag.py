from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


@dataclass(slots=True)
class HellaSwagBenchmark(BaseBenchmark):
    """HellaSwag: Commonsense reasoning benchmark.

    ~70K sentence completion problems testing commonsense.
    Dataset: https://huggingface.co/datasets/Rowan/hellaswag
    """

    name: str = "hellaswag"
    mode: str = "text"

    def iter_examples(self, split: str) -> Iterable[Example]:
        # Test split has no labels, use validation
        ds_split = "validation" if split in ["test", "dev"] else split
        ds = load_dataset("Rowan/hellaswag", split=ds_split)

        for idx, row in enumerate(ds):
            # Build context and endings
            ctx = row["ctx"]
            endings = row["endings"]

            # Format as MCQ
            formatted = [f"{chr(ord('A') + i)}. {end}" for i, end in enumerate(endings)]
            full_input = f"{ctx}\n\nWhich ending is most likely?\n" + "\n".join(formatted)

            # Label is the index (0-3)
            label = row["label"]
            if isinstance(label, str):
                if label == "":
                    continue  # Skip examples with missing labels
                label = int(label)
            answer_letter = chr(ord('A') + label)

            yield Example(
                id=f"hellaswag_{split}_{idx}",
                input=full_input,
                target=answer_letter,
                meta={
                    "split": split,
                    "ctx": ctx,
                    "endings": endings,
                    "activity_label": row.get("activity_label", ""),
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are an expert at commonsense reasoning.\n"
            "Choose the most likely ending for the given context.\n"
            'Return ONLY a JSON object with keys: "reasoning" (your thinking), '
            '"answer" (the letter A, B, C, or D), and "confidence" (0..1).'
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
            match = re.search(r"\b([A-D])\b", raw_text)
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
        got = re.sub(r"[^A-D]", "", str(pred.answer).upper())
        if got:
            got = got[0]

        correct = int(gold == got)

        out = {
            "correct": correct,
            "gold": gold,
            "predicted": pred.answer,
        }
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
