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
class MathVerseBenchmark(BaseBenchmark):
    """MathVerse: Visual Math Reasoning with Diverse Diagram Types.

    Diverse mathematical diagrams with step-by-step solutions.
    GPT-4V accuracy: ~30-40%

    Dataset: https://huggingface.co/datasets/AI4Math/MathVerse
    """

    name: str = "mathverse"
    mode: str = "text+image"
    max_image_size: int = 1024

    # Subset: "testmini" (788) or "test"
    subset: str = "testmini"

    def iter_examples(self, split: str) -> Iterable[Example]:
        # MathVerse requires config name: testmini or testmini_text_only
        # The split within each config is just "testmini"
        config = "testmini"

        ds = load_dataset("AI4Math/MathVerse", config, split="testmini")

        for idx, row in enumerate(ds):
            img = row.get("image") or row.get("decoded_image")
            if img is None or not isinstance(img, Image.Image):
                continue

            yield Example(
                id=str(row.get("sample_index", idx)),
                input={
                    "question": row.get("question", ""),
                    "query_cot": row.get("query_cot", ""),
                    "images": [img],
                },
                target=row.get("answer", ""),
                meta={
                    "split": split,
                    "problem_version": row.get("problem_version", ""),
                    "subfield": row.get("subfield", ""),
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        inp = ex.input
        # Use chain-of-thought query if available
        question = inp.get("query_cot") or inp["question"]
        images = inp["images"]

        system = (
            "You are an expert math problem solver with strong visual diagram understanding.\n"
            "Analyze the mathematical diagram carefully, work through the problem step by step, then provide your answer.\n"
            'Return a JSON object with keys: "reasoning" (your step-by-step work), "answer" (final numerical or letter answer), '
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
            # Fallback to string comparison
            correct = int(gold.lower() == got.lower())

        out = {"correct": correct, "gold": gold, "predicted": pred.answer}
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
