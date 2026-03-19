from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset
from PIL import Image

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import build_vision_content, clamp01, extract_first_json_obj


def decode_data_url_to_pil(data_url: str) -> Image.Image | None:
    """Decode a data:image/... URL to a PIL Image."""
    if not data_url or not data_url.startswith("data:"):
        return None
    try:
        # Parse: data:image/jpeg;base64,/9j/4AAQ...
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        return Image.open(io.BytesIO(img_bytes))
    except Exception:
        return None


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    s = s.lower().strip()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = re.sub(r"[^\w\s]", "", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def check_mcq_answer(predicted: str, gold: str) -> bool:
    """Check if MCQ answer matches (e.g., 'A', '(A)', 'A)')."""
    pred_clean = predicted.strip().upper()
    gold_clean = gold.strip().upper()

    # Direct match
    if pred_clean == gold_clean:
        return True

    # Extract standalone choice letter (last occurrence, to handle "The answer is B")
    gold_letters = re.findall(r'\b([A-J])\b', gold_clean)
    pred_letters = re.findall(r'\b([A-J])\b', pred_clean)

    if gold_letters and pred_letters and gold_letters[-1] == pred_letters[-1]:
        return True

    # Fallback for short responses like "(B)" where \b won't match inside parens
    if not pred_letters and len(pred_clean) <= 5:
        pred_letter = re.sub(r"[^A-Z]", "", pred_clean)[:1]
        gold_letter = re.sub(r"[^A-Z]", "", gold_clean)[:1]
        if pred_letter and gold_letter and pred_letter == gold_letter:
            return True

    return False


def check_short_answer(predicted: str, gold: str) -> bool:
    """Fuzzy check for short answer questions."""
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)

    # Exact match after normalization
    if pred_norm == gold_norm:
        return True

    # Gold contained in prediction
    if gold_norm and gold_norm in pred_norm:
        return True

    return False


def question_references_image(question: str) -> bool:
    """Check if question text references an image/figure that would be required."""
    q_lower = question.lower()
    image_indicators = [
        'image', 'figure', 'diagram', 'picture', 'photo', 'graph',
        'shown above', 'shown below', 'see the', 'in the figure',
        'look at', 'observe the', 'the following image', 'attached',
    ]
    return any(ind in q_lower for ind in image_indicators)


@dataclass(slots=True)
class HLEBenchmark(BaseBenchmark):
    """Humanity's Last Exam (HLE) benchmark.

    A multi-modal benchmark with 2,500 expert-level questions across dozens of
    subjects (mathematics, humanities, natural sciences).

    Dataset: https://huggingface.co/datasets/cais/hle
    Paper: https://arxiv.org/abs/2501.14249

    Note: This implementation filters to text-answerable questions by default
    (questions that don't explicitly reference images/figures in the text).
    ~2000 of 2500 questions are answerable from text alone.
    """

    name: str = "hle"
    mode: str = "text"  # Changes to "text+image" when include_images=True

    # Filter to questions that don't reference images in text (ignored if include_images=True)
    text_only: bool = True

    # Include images in requests (enables multimodal mode)
    include_images: bool = False

    # Filter by answer type: "mcq", "exact_match", or None for all
    # Note: HLE uses "multipleChoice" and "exactMatch" internally
    answer_type_filter: str | None = None

    # Filter by subject category (e.g., "Mathematics", "Humanities")
    category_filter: str | None = None

    # Max image size for encoding
    max_image_size: int = 1024

    def __post_init__(self):
        if self.include_images:
            object.__setattr__(self, 'mode', 'text+image')

    def iter_examples(self, split: str) -> Iterable[Example]:
        # HLE only has test split
        ds = load_dataset("cais/hle", split="test")

        for idx, row in enumerate(ds):
            has_image = row.get("image") is not None
            references_image = question_references_image(row.get("question", ""))

            # Filter logic depends on mode
            if self.include_images:
                # In image mode, only include questions that have images
                if not has_image:
                    continue
            else:
                # In text-only mode, filter out questions that reference images
                if self.text_only and references_image:
                    continue

            # Filter by answer type if specified
            answer_type = row.get("answer_type", "")
            if self.answer_type_filter:
                if self.answer_type_filter == "mcq" and answer_type != "mcq":
                    continue
                if self.answer_type_filter == "short_answer" and answer_type != "short_answer":
                    continue

            # Filter by category if specified
            if self.category_filter:
                category = row.get("category", "")
                if self.category_filter.lower() not in category.lower():
                    continue

            # Decode image if present and in image mode
            image = None
            if self.include_images and has_image:
                image = decode_data_url_to_pil(row["image"])

            yield Example(
                id=f"hle_{idx}",
                input={
                    "question": row["question"],
                    "image": image,
                } if self.include_images else row["question"],
                target=row["answer"],
                meta={
                    "split": split,
                    "answer_type": answer_type,
                    "category": row.get("category", ""),
                    "raw_subject": row.get("raw_subject", ""),
                    "has_image": has_image,
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        answer_type = ex.meta.get("answer_type", "")

        if answer_type == "mcq":
            system = (
                "You are an expert answering multiple-choice questions.\n"
                "Think step by step, then provide your final answer.\n"
                'Return ONLY a JSON object with keys: "reasoning" (your step-by-step thinking), '
                '"answer" (the letter of your answer, e.g., "A", "B", "C", or "D"), '
                'and "confidence" (0..1 how confident you are).'
            )
        else:
            system = (
                "You are an expert answering academic questions.\n"
                "Think step by step, then provide your final answer.\n"
                'Return ONLY a JSON object with keys: "reasoning" (your step-by-step thinking), '
                '"answer" (your concise answer), '
                'and "confidence" (0..1 how confident you are).'
            )

        # Handle both text-only and multimodal modes
        if self.include_images and isinstance(ex.input, dict):
            question = ex.input["question"]
            image = ex.input.get("image")
            if image is not None:
                content = build_vision_content(question, [image], max_size=self.max_image_size)
            else:
                content = question
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

        answer = None
        confidence = None

        if obj:
            answer = obj.get("answer", None)
            conf = obj.get("confidence", None)
            try:
                if conf is not None:
                    confidence = clamp01(float(conf))
            except Exception:
                confidence = None

        if answer is None:
            # Fallback: try to extract answer from raw text
            # Look for "answer is X" or "Answer: X" patterns
            match = re.search(r"(?:answer\s+is|answer:)\s*([A-Z]|\S+)", raw_text, re.IGNORECASE)
            if match:
                answer = match.group(1)
            else:
                answer = raw_text.strip()[:200]  # Truncate if very long

        return Prediction(
            example_id=ex.id,
            answer=str(answer).strip() if answer else "",
            confidence=confidence,
            raw_text=raw_text,
            extra={
                "parsed_json": obj is not None,
                "answer_type": ex.meta.get("answer_type", ""),
            },
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = str(ex.target).strip()
        got = str(pred.answer).strip()
        answer_type = ex.meta.get("answer_type", "")

        # Score based on answer type
        if answer_type == "mcq":
            correct = int(check_mcq_answer(got, gold))
        else:
            # Short answer - use fuzzy matching
            # Note: Official HLE uses LLM judge for short answers
            correct = int(check_short_answer(got, gold))

        out = {
            "correct": correct,
            "gold": gold,
            "predicted": got,
            "answer_type": answer_type,
            "category": ex.meta.get("category", ""),
        }
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
