from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset
from PIL import Image

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import build_vision_content, clamp01, extract_choice_letter, extract_first_json_obj


@dataclass(slots=True)
class RealWorldQABenchmark(BaseBenchmark):
    """RealWorldQA: Real-World Visual Question Answering.

    Questions about real-world images requiring common sense reasoning.
    GPT-4V accuracy: ~60-70%

    Dataset: https://huggingface.co/datasets/xai-org/RealworldQA
    """

    name: str = "realworldqa"
    mode: str = "text+image"
    max_image_size: int = 1024

    def iter_examples(self, split: str) -> Iterable[Example]:
        ds = load_dataset("xai-org/RealworldQA", split="test")

        for idx, row in enumerate(ds):
            img = row.get("image")
            if img is None or not isinstance(img, Image.Image):
                continue

            yield Example(
                id=str(idx),
                input={
                    "question": row.get("question", ""),
                    "images": [img],
                },
                target=row.get("answer", ""),
                meta={"split": split},
            )

    def build_request(self, ex: Example) -> ModelRequest:
        inp = ex.input
        question = inp["question"]
        images = inp["images"]

        system = (
            "You are a visual question answering assistant specializing in real-world scenarios.\n"
            "Analyze the image and answer the question using common sense reasoning.\n"
            'Return a JSON object with keys: "reasoning" (your analysis), "answer" (letter A/B/C/D), '
            'and "confidence" (0..1).'
        )

        content = build_vision_content(question, images, max_size=self.max_image_size)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=512)

    def parse_prediction(self, ex: Example, resp: ModelResponse) -> Prediction:
        raw_text = resp.text or ""
        obj = extract_first_json_obj(raw_text)

        answer = None
        confidence = None

        if obj:
            answer = obj.get("answer", "")
            conf = obj.get("confidence")
            if conf is not None:
                try:
                    confidence = clamp01(float(conf))
                except:
                    pass

        if not answer:
            answer = extract_choice_letter(raw_text) or raw_text.strip()[:50]

        return Prediction(
            example_id=ex.id,
            answer=str(answer).strip().upper(),
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = str(ex.target).strip().upper()
        got = str(pred.answer).strip().upper()

        gold_letter = re.sub(r"[^A-Z]", "", gold)[:1]
        got_letter = re.sub(r"[^A-Z]", "", got)[:1]

        correct = int(gold_letter == got_letter) if gold_letter else 0

        out = {"correct": correct, "gold": gold, "predicted": pred.answer}
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
