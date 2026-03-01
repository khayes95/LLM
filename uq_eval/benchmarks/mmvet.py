from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset
from PIL import Image

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import build_vision_content, clamp01, extract_first_json_obj, normalize_text


@dataclass(slots=True)
class MMVetBenchmark(BaseBenchmark):
    """MM-Vet: Multimodal Veterinary benchmark.

    Diverse visual-language tasks including recognition, OCR, knowledge, math.
    GPT-4V accuracy: ~50-60%

    Dataset: https://huggingface.co/datasets/lmms-lab/MMVet
    """

    name: str = "mmvet"
    mode: str = "text+image"
    max_image_size: int = 1024

    def iter_examples(self, split: str) -> Iterable[Example]:
        ds = load_dataset("lmms-lab/MMVet", split="test")

        for idx, row in enumerate(ds):
            img = row.get("image")
            if img is None or not isinstance(img, Image.Image):
                continue

            yield Example(
                id=str(row.get("question_id", idx)),
                input={
                    "question": row.get("question", ""),
                    "images": [img],
                },
                target=row.get("answer", ""),
                meta={
                    "split": split,
                    "capability": row.get("capability", ""),
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        inp = ex.input
        question = inp["question"]
        images = inp["images"]

        system = (
            "You are a multimodal assistant capable of recognition, OCR, knowledge retrieval, and reasoning.\n"
            "Analyze the image carefully and provide a concise, accurate answer.\n"
            'Return a JSON object with keys: "reasoning" (your analysis), "answer" (concise answer), '
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
            answer = raw_text.strip()[:200]

        return Prediction(
            example_id=ex.id,
            answer=str(answer).strip(),
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        # MM-Vet uses GPT-4 grading, return -1 for offline grading
        out = {
            "correct": -1,  # Requires LLM grading
            "gold": ex.target,
            "predicted": pred.answer,
        }
        if pred.confidence is not None:
            out["brier"] = None  # Cannot compute without correct
        return out
