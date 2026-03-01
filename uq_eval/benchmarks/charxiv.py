from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset
from PIL import Image

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import build_vision_content, clamp01, extract_first_json_obj, normalize_text


@dataclass(slots=True)
class CharXivBenchmark(BaseBenchmark):
    """CharXiv: Scientific Figure Reasoning benchmark.

    2,323 figures from arXiv papers with reasoning questions.
    GPT-5 accuracy: 57-81%

    Dataset: https://huggingface.co/datasets/princeton-nlp/CharXiv
    """

    name: str = "charxiv"
    mode: str = "text+image"

    # Question type: "reasoning" (default) or "descriptive"
    question_type: str = "reasoning"

    # Filter by category (cs, physics, math, etc.) or None for all
    category_filter: str | None = None

    # Max image size
    max_image_size: int = 1024

    def iter_examples(self, split: str) -> Iterable[Example]:
        # CharXiv has 'validation' and 'test' splits (not 'dev')
        ds_split = "validation" if split in ("dev", "validation") else "test"
        ds = load_dataset("princeton-nlp/CharXiv", split=ds_split)

        for idx, row in enumerate(ds):
            # Filter by category
            category = row.get("category", "")
            if self.category_filter and category != self.category_filter:
                continue

            img = row.get("image")
            if not isinstance(img, Image.Image):
                continue

            if self.question_type == "reasoning":
                question = row.get("reasoning_q", "")
                answer = row.get("reasoning_a", "")
                answer_type = row.get("reasoning_a_type", "")
            else:
                # Use descriptive questions (q1-q4)
                # Pick the first non-empty one
                question = None
                answer = None
                for i in range(1, 5):
                    q = row.get(f"descriptive_q{i}", "")
                    a = row.get(f"descriptive_a{i}", "")
                    if q and a:
                        question = q
                        answer = a
                        break
                answer_type = "descriptive"

            if not question or not answer:
                continue

            yield Example(
                id=f"charxiv_{idx}",
                input={
                    "question": question,
                    "image": img,
                },
                target=answer,
                meta={
                    "split": split,
                    "category": category,
                    "answer_type": answer_type,
                    "question_type": self.question_type,
                    "year": row.get("year", ""),
                    "num_subplots": row.get("num_subplots", ""),
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        inp = ex.input
        question = inp["question"]
        img = inp["image"]

        system = (
            "You are an expert at analyzing scientific figures and charts.\n"
            "Study the figure carefully and answer the question.\n"
            'Return a JSON object with keys: "reasoning" (your analysis of the figure), '
            '"answer" (your concise answer), and "confidence" (0..1).'
        )

        content = build_vision_content(question, [img], max_size=self.max_image_size)

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
            answer = raw_text.strip()[:200]

        return Prediction(
            example_id=ex.id,
            answer=str(answer).strip(),
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = normalize_text(str(ex.target))
        got = normalize_text(str(pred.answer))

        # Fuzzy match: check containment in both directions
        correct = 0
        if gold == got:
            correct = 1
        elif gold and got:
            if gold in got or got in gold:
                correct = 1

        out = {
            "correct": correct,
            "gold": str(ex.target),
            "predicted": pred.answer,
            "category": ex.meta.get("category", ""),
            "question_type": ex.meta.get("question_type", ""),
        }
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
