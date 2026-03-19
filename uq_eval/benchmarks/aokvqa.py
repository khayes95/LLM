from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset
from PIL import Image

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import build_vision_content, clamp01, extract_choice_letter, extract_first_json_obj, normalize_text


@dataclass(slots=True)
class AOKVQABenchmark(BaseBenchmark):
    """A-OKVQA: Augmented OK-VQA with rationales.

    Visual QA requiring outside knowledge and common sense reasoning.
    GPT-4V accuracy: ~60-70%

    Dataset: https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA
    """

    name: str = "aokvqa"
    mode: str = "text+image"
    max_image_size: int = 1024

    def iter_examples(self, split: str) -> Iterable[Example]:
        ds_split = "validation" if split in ("dev", "validation") else "test"

        ds = load_dataset("HuggingFaceM4/A-OKVQA", split=ds_split)

        for row in ds:
            img = row.get("image")
            if img is None or not isinstance(img, Image.Image):
                continue

            # A-OKVQA has multiple-choice options
            choices = row.get("choices", [])

            yield Example(
                id=str(row.get("question_id", "")),
                input={
                    "question": row.get("question", ""),
                    "choices": choices,
                    "images": [img],
                },
                target=row.get("correct_choice_idx", 0),  # Index of correct choice
                meta={
                    "split": split,
                    "direct_answers": row.get("direct_answers", []),
                    "difficult_direct_answer": row.get("difficult_direct_answer", False),
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        inp = ex.input
        question = inp["question"]
        choices = inp.get("choices", [])
        images = inp["images"]

        # Format with choices
        if choices:
            letters = "ABCDEFGH"
            prompt_text = question + "\n\nOptions:\n"
            for i, opt in enumerate(choices):
                if i < len(letters):
                    prompt_text += f"{letters[i]}. {opt}\n"
        else:
            prompt_text = question

        system = (
            "You are a visual question answering assistant with broad world knowledge.\n"
            "Use the image and your knowledge to answer the question.\n"
            'Return a JSON object with keys: "reasoning" (your analysis), "answer" (letter A/B/C/D), '
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
            answer = extract_choice_letter(raw_text) or raw_text.strip()[:50]

        return Prediction(
            example_id=ex.id,
            answer=str(answer).strip().upper(),
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        # Target is the index of correct choice
        gold_idx = int(ex.target) if isinstance(ex.target, (int, float)) else 0
        gold_letter = chr(ord('A') + gold_idx)

        got = str(pred.answer).strip().upper()
        standalone = re.findall(r'\b([A-J])\b', got)
        if standalone:
            got_letter = standalone[-1]
        elif len(got) <= 5:
            got_letter = re.sub(r"[^A-Z]", "", got)[:1]
        else:
            got_letter = ""

        correct = int(gold_letter == got_letter) if got_letter else 0

        out = {"correct": correct, "gold": gold_letter, "predicted": pred.answer}
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
