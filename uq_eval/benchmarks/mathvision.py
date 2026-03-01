from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset
from PIL import Image

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import build_vision_content, clamp01, extract_first_json_obj, extract_last_number


@dataclass(slots=True)
class MathVisionBenchmark(BaseBenchmark):
    """MathVision: Mathematical Problem Solving with Visual Context.

    Math problems requiring diagram interpretation.
    GPT-4V accuracy: ~25-35%

    Dataset: https://huggingface.co/datasets/MathLLMs/MathVision
    """

    name: str = "mathvision"
    mode: str = "text+image"
    max_image_size: int = 1024

    def iter_examples(self, split: str) -> Iterable[Example]:
        # MathVision only has 'default' config, split is 'test' or 'testmini'
        ds_split = "testmini" if split in ("dev", "validation") else "test"

        ds = load_dataset("MathLLMs/MathVision", split=ds_split)

        for idx, row in enumerate(ds):
            # 'decoded_image' contains actual PIL Image, 'image' is just filename
            img = row.get("decoded_image")
            if img is None or not isinstance(img, Image.Image):
                continue

            yield Example(
                id=str(row.get("id", idx)),
                input={
                    "question": row.get("question", ""),
                    "images": [img],
                },
                target=row.get("answer", ""),
                meta={
                    "split": split,
                    "subject": row.get("subject", ""),
                    "level": row.get("level", ""),
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        inp = ex.input
        question = inp["question"]
        images = inp["images"]

        system = (
            "You are an expert math problem solver.\n"
            "Analyze the diagram, apply relevant mathematical concepts, and solve step by step.\n"
            'Return a JSON object with keys: "reasoning" (your work), "answer" (final answer), '
            'and "confidence" (0..1).'
        )

        content = build_vision_content(question, images, max_size=self.max_image_size)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=1024)

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
            answer = extract_last_number(raw_text) or raw_text.strip()[:50]

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

        # Try numerical comparison
        try:
            gold_num = float(re.sub(r"[^\d.\-]", "", gold))
            got_num = float(re.sub(r"[^\d.\-]", "", got))
            correct = int(abs(gold_num - got_num) < 0.01)
        except:
            correct = int(gold.lower() == got.lower())

        out = {"correct": correct, "gold": gold, "predicted": pred.answer}
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
