from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Iterable

from datasets import get_dataset_config_names, load_dataset
from PIL import Image

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import build_vision_content, clamp01, extract_choice_letter, extract_first_json_obj


MMMU_SUBJECTS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory",
    "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science",
    "Design", "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics", "Energy_and_Power",
    "Finance", "Geography", "History", "Literature", "Manage", "Marketing", "Materials", "Math",
    "Mechanical_Engineering", "Music", "Pharmacy", "Physics", "Psychology", "Public_Health", "Sociology"
]


def parse_options(options_str: str) -> list[str]:
    """Parse options from string representation."""
    if not options_str:
        return []
    try:
        return ast.literal_eval(options_str)
    except:
        return []


def format_mcq_question(question: str, options: list[str]) -> str:
    """Format question with lettered options."""
    letters = "ABCDEFGH"
    formatted = question + "\n\nOptions:\n"
    for i, opt in enumerate(options):
        if i < len(letters):
            formatted += f"{letters[i]}. {opt}\n"
    return formatted.strip()


@dataclass(slots=True)
class MMMUBenchmark(BaseBenchmark):
    """MMMU: Massive Multi-discipline Multimodal Understanding.

    College-level visual problem-solving across 30 subjects.
    5,190 problems requiring domain-specific knowledge + visual understanding.
    GPT-5 accuracy: 62-78%

    Dataset: https://huggingface.co/datasets/MMMU/MMMU
    """

    name: str = "mmmu"
    mode: str = "text+image"

    # Filter by subject (None = all)
    subject_filter: str | None = None

    # Filter by difficulty: "Easy", "Medium", "Hard", or None
    difficulty_filter: str | None = None

    # Max image size for encoding
    max_image_size: int = 1024

    # Subjects to load (all by default)
    subjects: list[str] = field(default_factory=lambda: MMMU_SUBJECTS.copy())

    def iter_examples(self, split: str) -> Iterable[Example]:
        # MMMU has dev, validation, test splits
        # test split has no answers, use validation for eval
        ds_split = "validation" if split == "test" else split

        subjects_to_load = self.subjects
        if self.subject_filter:
            subjects_to_load = [s for s in self.subjects if self.subject_filter.lower() in s.lower()]

        for subject in subjects_to_load:
            try:
                ds = load_dataset("MMMU/MMMU", subject, split=ds_split)
            except Exception as e:
                print(f"Warning: Could not load MMMU/{subject}: {e}")
                continue

            for row in ds:
                # Filter by difficulty
                difficulty = row.get("topic_difficulty", "")
                if self.difficulty_filter and difficulty != self.difficulty_filter:
                    continue

                # Collect images
                images = []
                for i in range(1, 8):
                    img = row.get(f"image_{i}")
                    if img is not None and isinstance(img, Image.Image):
                        images.append(img)

                # Skip if no images (shouldn't happen for MMMU)
                if not images:
                    continue

                # Parse options
                options = parse_options(row.get("options", "[]"))

                yield Example(
                    id=row["id"],
                    input={
                        "question": row["question"],
                        "options": options,
                        "images": images,
                    },
                    target=row.get("answer", ""),
                    meta={
                        "split": split,
                        "subject": subject,
                        "subfield": row.get("subfield", ""),
                        "difficulty": difficulty,
                        "question_type": row.get("question_type", ""),
                        "img_type": row.get("img_type", ""),
                        "num_images": len(images),
                    },
                )

    def build_request(self, ex: Example) -> ModelRequest:
        inp = ex.input
        question = inp["question"]
        options = inp["options"]
        images = inp["images"]

        # Replace <image N> placeholders with generic reference
        question = re.sub(r"<image\s*\d+>", "[See image]", question)

        # Format question with options
        if options:
            prompt_text = format_mcq_question(question, options)
        else:
            prompt_text = question

        system = (
            "You are an expert solving college-level problems that require visual understanding.\n"
            "Analyze the image(s) carefully, think step by step, then provide your answer.\n"
            'Return a JSON object with keys: "reasoning" (your analysis), "answer" (just the letter A/B/C/D), '
            'and "confidence" (0..1).'
        )

        # Build vision content
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

        # Fallback: extract letter from raw text
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
            "subject": ex.meta.get("subject", ""),
            "difficulty": ex.meta.get("difficulty", ""),
        }
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
