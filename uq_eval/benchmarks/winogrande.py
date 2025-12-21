from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


@dataclass(slots=True)
class WinograndeBenchmark(BaseBenchmark):
    """WinoGrande: Large-scale Winograd schema challenge.

    44K problems testing commonsense reasoning via pronoun resolution.
    Dataset: https://huggingface.co/datasets/allenai/winogrande
    """

    name: str = "winogrande"
    mode: str = "text"

    # Size: "xs", "s", "m", "l", "xl", or "debiased"
    size: str = "xl"

    def iter_examples(self, split: str) -> Iterable[Example]:
        # Winogrande has train, test, validation splits (no "dev")
        ds_split = "validation" if split == "dev" else split
        ds = load_dataset("allenai/winogrande", f"winogrande_{self.size}", split=ds_split)

        for idx, row in enumerate(ds):
            sentence = row["sentence"]
            option1 = row["option1"]
            option2 = row["option2"]

            full_input = (
                f"Sentence: {sentence}\n\n"
                f"Which option should replace the blank?\n"
                f"1. {option1}\n"
                f"2. {option2}"
            )

            yield Example(
                id=f"winogrande_{self.size}_{split}_{idx}",
                input=full_input,
                target=row["answer"],  # "1" or "2"
                meta={
                    "split": split,
                    "size": self.size,
                    "sentence": sentence,
                    "option1": option1,
                    "option2": option2,
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are an expert at commonsense reasoning and pronoun resolution.\n"
            "Choose the option that best fills in the blank.\n"
            'Return ONLY a JSON object with keys: "reasoning" (your thinking), '
            '"answer" (either "1" or "2"), and "confidence" (0..1).'
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
            # Try to extract 1 or 2
            if "1" in raw_text and "2" not in raw_text:
                answer = "1"
            elif "2" in raw_text and "1" not in raw_text:
                answer = "2"
            else:
                match = re.search(r"\b([12])\b", raw_text)
                if match:
                    answer = match.group(1)
                else:
                    answer = raw_text.strip()[:5]

        return Prediction(
            example_id=ex.id,
            answer=str(answer).strip() if answer else "",
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = str(ex.target).strip()
        got = str(pred.answer).strip()

        # Extract just the number
        got_num = re.search(r"[12]", got)
        if got_num:
            got = got_num.group()

        correct = int(gold == got)

        out = {
            "correct": correct,
            "gold": gold,
            "predicted": pred.answer,
        }
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
