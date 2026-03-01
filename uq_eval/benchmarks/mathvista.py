from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset
from PIL import Image

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import build_vision_content, clamp01, extract_choice_letter, extract_first_json_obj, extract_last_number


@dataclass(slots=True)
class MathVistaBenchmark(BaseBenchmark):
    """MathVista: Visual Math Reasoning.

    Comprehensive math reasoning across multiple visual contexts.
    6,141 problems requiring numerical computation with visual inputs.
    GPT-4V accuracy: ~50%

    Dataset: https://huggingface.co/datasets/AI4Math/MathVista
    """

    name: str = "mathvista"
    mode: str = "text+image"
    max_image_size: int = 1024

    def iter_examples(self, split: str) -> Iterable[Example]:
        # MathVista: testmini (1000) has answers, test (5141) does NOT have answers
        # Always use testmini for evaluation since test split has empty answer fields
        ds_split = "testmini"

        ds = load_dataset("AI4Math/MathVista", split=ds_split)

        for row in ds:
            img = row.get("decoded_image") or row.get("image")
            if img is None or not isinstance(img, Image.Image):
                continue

            # Choices for MCQ
            choices = row.get("choices", [])
            if choices and isinstance(choices, str):
                try:
                    import ast
                    choices = ast.literal_eval(choices)
                except:
                    choices = []

            yield Example(
                id=str(row.get("pid", row.get("id", ""))),
                input={
                    "question": row.get("question", ""),
                    "choices": choices,
                    "images": [img],
                    "query": row.get("query", ""),
                },
                target=row.get("answer", ""),
                meta={
                    "split": split,
                    "question_type": row.get("question_type", ""),
                    "answer_type": row.get("answer_type", ""),
                    "precision": row.get("precision", None),
                    "skill": row.get("skills", []),
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        inp = ex.input
        question = inp.get("query") or inp["question"]
        choices = inp.get("choices", [])
        images = inp["images"]

        # Format question with options if MCQ
        if choices:
            letters = "ABCDEFGH"
            prompt_text = question + "\n\nOptions:\n"
            for i, opt in enumerate(choices):
                if i < len(letters):
                    prompt_text += f"{letters[i]}. {opt}\n"
        else:
            prompt_text = question

        system = (
            "You are an expert math problem solver with strong visual understanding.\n"
            "Analyze the image carefully, perform any required calculations step by step, then provide your answer.\n"
            'Return a JSON object with keys: "reasoning" (your work), "answer" (final answer - letter for MCQ, number for free response), '
            'and "confidence" (0..1).'
        )

        content = build_vision_content(prompt_text, images, max_size=self.max_image_size)

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

        # Fallback extraction
        if not answer:
            choices = ex.input.get("choices", [])
            if choices:
                answer = extract_choice_letter(raw_text) or raw_text.strip()[:50]
            else:
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
        choices = ex.input.get("choices", [])

        if choices:
            # MCQ: compare letters
            gold_letter = gold.upper()[:1] if gold else ""
            got_letter = got.upper()[:1] if got else ""
            correct = int(gold_letter == got_letter) if gold_letter else 0
        else:
            # Numerical: try to compare numbers
            try:
                precision = ex.meta.get("precision", 2)
                gold_num = float(re.sub(r"[^\d.\-]", "", gold))
                got_num = float(re.sub(r"[^\d.\-]", "", got))
                correct = int(abs(gold_num - got_num) < 10 ** (-precision + 1))
            except:
                # Fallback to string comparison
                correct = int(gold.lower() == got.lower())

        out = {"correct": correct, "gold": gold, "predicted": pred.answer}
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
