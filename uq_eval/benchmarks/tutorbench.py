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
class TutorBenchBenchmark(BaseBenchmark):
    """TutorBench: Tutoring evaluation benchmark.

    Evaluates LLM tutoring capabilities with 1,490 examples.
    Dataset: https://huggingface.co/datasets/ScaleAI/TutorBench

    Supports both text-only and multimodal (image) modes.
    """

    name: str = "tutorbench"
    mode: str = "text"  # Changes to "text+image" when include_images=True

    # Include images in requests (enables multimodal mode)
    include_images: bool = False

    # Filter by subject (e.g., "Chemistry", "Math")
    subject_filter: str | None = None

    # Max image size for encoding
    max_image_size: int = 1024

    def __post_init__(self):
        if self.include_images:
            object.__setattr__(self, 'mode', 'text+image')

    def iter_examples(self, split: str) -> Iterable[Example]:
        # TutorBench only has train split
        ds = load_dataset("ScaleAI/TutorBench", split="train")

        for idx, row in enumerate(ds):
            has_image = row.get("Image") is not None and isinstance(row.get("Image"), Image.Image)

            # Filter logic based on mode
            if self.include_images:
                # In image mode, only include examples with images
                if not has_image:
                    continue
            else:
                # In text-only mode, skip examples that require images
                batch = row.get("BATCH", "")
                if "MULTIMODAL" in batch:
                    continue

            # Filter by subject
            subject = row.get("SUBJECT", "")
            if self.subject_filter and self.subject_filter.lower() not in subject.lower():
                continue

            # Get the image if in multimodal mode
            image = row.get("Image") if self.include_images and has_image else None

            # Build prompt from PROMPT and optionally FOLLOW_UP_PROMPT
            prompt = row.get("PROMPT", "")
            follow_up = row.get("FOLLOW_UP_PROMPT", "")

            yield Example(
                id=f"tutorbench_{idx}",
                input={
                    "prompt": prompt,
                    "follow_up": follow_up,
                    "image": image,
                } if self.include_images else prompt,
                target=row.get("UC1_INITIAL_EXPLANATION", ""),
                meta={
                    "split": split,
                    "subject": subject,
                    "batch": row.get("BATCH", ""),
                    "bloom_taxonomy": row.get("bloom_taxonomy", ""),
                    "has_image": has_image,
                    "rubrics": row.get("RUBRICS", ""),
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are a helpful tutor. Answer the student's question clearly and educationally.\n"
            "Explain your reasoning step by step.\n"
            'Return a JSON object with keys: "explanation" (your educational response) and "confidence" (0..1).'
        )

        # Handle both text-only and multimodal modes
        if self.include_images and isinstance(ex.input, dict):
            prompt = ex.input["prompt"]
            image = ex.input.get("image")
            if image is not None:
                content = build_vision_content(prompt, [image], max_size=self.max_image_size)
            else:
                content = prompt
        else:
            content = str(ex.input)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=2048)

    def parse_prediction(self, ex: Example, resp: ModelResponse) -> Prediction:
        raw_text = resp.text or ""
        obj = extract_first_json_obj(raw_text)

        # Try "explanation" first (new format), then "answer" (old format)
        answer = raw_text
        if obj:
            answer = obj.get("explanation", obj.get("answer", raw_text))

        confidence = None
        if obj and obj.get("confidence") is not None:
            try:
                confidence = clamp01(float(obj["confidence"]))
            except:
                pass

        return Prediction(
            example_id=ex.id,
            answer=str(answer).strip(),
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        # TutorBench typically uses LLM-as-judge for scoring
        # For now, return placeholder for offline grading
        out = {
            "correct": -1,  # Needs LLM judge
            "needs_grading": True,
            "response_length": len(pred.answer),
        }
        if pred.confidence is not None:
            out["confidence"] = pred.confidence
        return out
