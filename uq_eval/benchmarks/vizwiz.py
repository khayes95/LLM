from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset
from PIL import Image

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import build_vision_content, clamp01, extract_first_json_obj, normalize_text


@dataclass(slots=True)
class VizWizBenchmark(BaseBenchmark):
    """VizWiz: Visual QA from blind users.

    Real questions from blind/low-vision users about images they took.
    Tests practical visual assistance capabilities.
    GPT-4V accuracy: ~50-60%

    Dataset: https://huggingface.co/datasets/lmms-lab/VizWiz-VQA
    """

    name: str = "vizwiz"
    mode: str = "text+image"
    max_image_size: int = 1024

    def iter_examples(self, split: str) -> Iterable[Example]:
        # VizWiz: val has answers, test does NOT have answers (empty answers list)
        # Always use val for evaluation since test split has no ground truth
        ds_split = "val"

        ds = load_dataset("lmms-lab/VizWiz-VQA", split=ds_split)

        for idx, row in enumerate(ds):
            img = row.get("image")
            if img is None or not isinstance(img, Image.Image):
                continue

            # VizWiz has multiple answer annotations
            answers = row.get("answers", [])
            # Use most common answer as target
            if answers:
                answer_texts = [a.get("answer", "") for a in answers if isinstance(a, dict)]
                if answer_texts:
                    from collections import Counter
                    target = Counter(answer_texts).most_common(1)[0][0]
                else:
                    target = answers[0] if answers else ""
            else:
                target = row.get("answer", "")

            yield Example(
                id=str(idx),
                input={
                    "question": row.get("question", ""),
                    "images": [img],
                },
                target=target,
                meta={
                    "split": split,
                    "answers": answers,
                    "answerable": row.get("answerable", None),
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        inp = ex.input
        question = inp["question"]
        images = inp["images"]

        system = (
            "You are a helpful visual assistant answering questions from users who may be blind or have low vision.\n"
            "Carefully examine the image and provide a helpful, accurate answer.\n"
            'If the question cannot be answered from the image, say "unanswerable".\n'
            'Return a JSON object with keys: "reasoning" (your analysis), "answer" (short answer), '
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
            answer = raw_text.strip()[:100]

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

        # VizWiz uses soft accuracy: check if prediction matches any of the annotator answers
        answers = ex.meta.get("answers", [])
        if answers:
            answer_texts = [normalize_text(a.get("answer", "") if isinstance(a, dict) else str(a)) for a in answers]
            # Count matches
            matches = sum(1 for a in answer_texts if got == a or got in a or a in got)
            correct = min(1.0, matches / 3.0)  # VizWiz soft accuracy formula
        else:
            correct = int(gold == got or gold in got or got in gold)

        out = {"correct": float(correct), "gold": ex.target, "predicted": pred.answer}
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
