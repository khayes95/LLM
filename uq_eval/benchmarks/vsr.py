from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Iterable

import requests
from datasets import load_dataset
from PIL import Image

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import build_vision_content, clamp01, extract_first_json_obj


@dataclass(slots=True)
class VSRBenchmark(BaseBenchmark):
    """VSR: Visual Spatial Reasoning.

    Tests spatial relationship understanding in images.
    True/False questions about object positions.
    GPT-4V accuracy: ~60-70%

    Dataset: https://huggingface.co/datasets/cambridgeltl/vsr_random
    """

    name: str = "vsr"
    mode: str = "text+image"
    max_image_size: int = 1024

    def iter_examples(self, split: str) -> Iterable[Example]:
        # VSR has 'train', 'validation', 'test' splits (not 'dev')
        ds_split = "validation" if split in ("dev", "validation") else "test"

        ds = load_dataset("cambridgeltl/vsr_random", split=ds_split)

        for idx, row in enumerate(ds):
            # VSR stores image URLs in 'image_link', not PIL Images
            image_url = row.get("image_link", "")
            if not image_url:
                continue

            # Download image from COCO URL
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content)).convert("RGB")
            except Exception:
                continue

            yield Example(
                id=str(idx),
                input={
                    "question": row.get("caption", ""),  # The statement to verify
                    "images": [img],
                },
                target=row.get("label", 0),  # 1 = True, 0 = False
                meta={
                    "split": split,
                    "relation": row.get("relation", ""),
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        inp = ex.input
        caption = inp["question"]
        images = inp["images"]

        prompt_text = f'Is the following statement TRUE or FALSE about the image?\n\nStatement: "{caption}"'

        system = (
            "You are a visual reasoning assistant specializing in spatial relationships.\n"
            "Carefully examine the image and determine if the statement accurately describes it.\n"
            'Return a JSON object with keys: "reasoning" (your analysis), "answer" (True or False), '
            'and "confidence" (0..1).'
        )

        content = build_vision_content(prompt_text, images, max_size=self.max_image_size)

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
            text_lower = raw_text.lower()
            if "true" in text_lower:
                answer = "True"
            elif "false" in text_lower:
                answer = "False"
            else:
                answer = raw_text.strip()[:50]

        return Prediction(
            example_id=ex.id,
            answer=str(answer).strip(),
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        # Target is 1 (True) or 0 (False)
        gold_label = int(ex.target) if isinstance(ex.target, (int, float)) else 0
        gold = "True" if gold_label == 1 else "False"

        got = str(pred.answer).strip().lower()
        got_bool = 1 if "true" in got else (0 if "false" in got else -1)

        correct = int(gold_label == got_bool) if got_bool >= 0 else 0

        out = {"correct": correct, "gold": gold, "predicted": pred.answer}
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
