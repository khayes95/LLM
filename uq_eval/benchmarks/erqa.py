from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset
from PIL import Image

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import build_vision_content, clamp01, extract_choice_letter, extract_first_json_obj


@dataclass(slots=True)
class ERQABenchmark(BaseBenchmark):
    """ERQA: Embodied Reasoning Question Answering.

    Multimodal spatial reasoning benchmark with 400 examples.
    Tests reasoning about robot actions, trajectories, and spatial relationships.
    GPT-5 accuracy: 42-66%

    Dataset: https://huggingface.co/datasets/FlagEval/ERQA
    """

    name: str = "erqa"
    mode: str = "text+image"

    # Filter by question type or None for all
    question_type_filter: str | None = None

    # Max image size
    max_image_size: int = 1024

    def iter_examples(self, split: str) -> Iterable[Example]:
        ds = load_dataset("FlagEval/ERQA", split="test")

        for row in ds:
            # Filter by question type
            q_type = row.get("question_type", "")
            if self.question_type_filter and q_type != self.question_type_filter:
                continue

            # Parse images list
            images_raw = row.get("images", [])
            if isinstance(images_raw, str):
                try:
                    images_raw = ast.literal_eval(images_raw)
                except:
                    images_raw = []

            images = [img for img in images_raw if isinstance(img, Image.Image)]

            if not images:
                continue

            yield Example(
                id=row["question_id"],
                input={
                    "question": row["question"],
                    "images": images,
                },
                target=row.get("answer", ""),
                meta={
                    "split": split,
                    "question_type": q_type,
                    "visual_indices": row.get("visual_indices", ""),
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        inp = ex.input
        question = inp["question"]
        images = inp["images"]

        system = (
            "You are an expert at spatial reasoning and understanding robot actions.\n"
            "Analyze the image(s) carefully and answer the multiple choice question.\n"
            'Return a JSON object with keys: "reasoning" (your spatial analysis), '
            '"answer" (just the letter A/B/C/D), and "confidence" (0..1).'
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

        # Extract just the letter
        gold_letter = re.sub(r"[^A-Z]", "", gold)[:1]
        got_letter = re.sub(r"[^A-Z]", "", got)[:1]

        correct = int(gold_letter == got_letter) if gold_letter else 0

        out = {
            "correct": correct,
            "gold": gold,
            "predicted": pred.answer,
            "question_type": ex.meta.get("question_type", ""),
        }
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
