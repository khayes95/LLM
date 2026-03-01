from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset
from PIL import Image

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import build_vision_content, clamp01, extract_first_json_obj


@dataclass(slots=True)
class HallusionBenchBenchmark(BaseBenchmark):
    """HallusionBench: Detecting Hallucinations in VLMs.

    Benchmark for measuring hallucination in vision-language models.
    Questions designed to test visual grounding and factual accuracy.
    GPT-4V accuracy: ~30-50%

    Dataset: https://huggingface.co/datasets/opencompass/HallusionBench
    """

    name: str = "hallusionbench"
    mode: str = "text+image"
    max_image_size: int = 1024

    def iter_examples(self, split: str) -> Iterable[Example]:
        # HallusionBench - lmms-lab has 'image' and 'non_image' splits
        # Use 'image' split for visual questions
        ds = load_dataset("lmms-lab/HallusionBench", split="image")

        for idx, row in enumerate(ds):
            img = row.get("image")
            if img is None or not isinstance(img, Image.Image):
                continue

            yield Example(
                id=str(row.get("id", idx)),
                input={
                    "question": row.get("question", ""),
                    "images": [img],
                },
                target=row.get("gt_answer", row.get("answer", "")),
                meta={
                    "split": split,
                    "category": row.get("category", ""),
                    "subcategory": row.get("subcategory", ""),
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        inp = ex.input
        question = inp["question"]
        images = inp["images"]

        system = (
            "You are a visual question answering assistant. Answer the question based ONLY on what you can see in the image.\n"
            "Be precise and avoid making assumptions or hallucinating details not present in the image.\n"
            'Return a JSON object with keys: "reasoning" (your observation), "answer" (Yes/No or short answer), '
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
            # Extract Yes/No pattern
            text_lower = raw_text.lower()
            if "yes" in text_lower[:50]:
                answer = "Yes"
            elif "no" in text_lower[:50]:
                answer = "No"
            else:
                answer = raw_text.strip()[:100]

        return Prediction(
            example_id=ex.id,
            answer=str(answer).strip(),
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = str(ex.target).strip().lower()
        got = str(pred.answer).strip().lower()

        # Normalize Yes/No answers - handle both text and numeric labels
        # Gold can be "yes", "no", "1" (yes), "0" (no)
        if gold in ("1", "yes"):
            gold_norm = "yes"
        elif gold in ("0", "no"):
            gold_norm = "no"
        else:
            gold_norm = gold

        got_norm = "yes" if "yes" in got else ("no" if "no" in got else got)

        correct = int(gold_norm == got_norm)

        out = {"correct": correct, "gold": ex.target, "predicted": pred.answer}
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
