from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


# Oolong has multiple task types
OOLONG_TASKS = [
    "analogies",
    "associations",
    "collocations",
    "semantic_similarity",
    "text_classification",
]


@dataclass(slots=True)
class OolongBenchmark(BaseBenchmark):
    """Oolong: Long context understanding benchmark.

    Tests understanding of long documents with various tasks.
    Dataset: https://huggingface.co/datasets/yuchenlin/oolong

    Tasks:
    - analogies: Word analogy completion
    - associations: Word association tasks
    - collocations: Common word combinations
    - semantic_similarity: Sentence similarity
    - text_classification: Document classification
    """

    name: str = "oolong"
    mode: str = "longtext"

    # Which task to run (None = all tasks)
    task: str | None = None

    def iter_examples(self, split: str) -> Iterable[Example]:
        tasks_to_run = [self.task] if self.task else OOLONG_TASKS

        for task_name in tasks_to_run:
            try:
                ds = load_dataset("princeton-nlp/OolongBench", task_name, split="test", trust_remote_code=True)
            except Exception:
                try:
                    ds = load_dataset("yuchenlin/oolong", task_name, split="test", trust_remote_code=True)
                except Exception:
                    continue

            for idx, row in enumerate(ds):
                # Get input - could be "text", "sentence", "word", etc.
                input_text = row.get("text", row.get("sentence", row.get("input", "")))

                # Get target
                target = row.get("label", row.get("answer", row.get("target", "")))

                yield Example(
                    id=f"oolong_{task_name}_{idx}",
                    input=input_text,
                    target=str(target),
                    meta={
                        "split": split,
                        "task": task_name,
                    },
                )

    def build_request(self, ex: Example) -> ModelRequest:
        task = ex.meta.get("task", "")

        if task == "analogies":
            instruction = "Complete the word analogy. A is to B as C is to ?"
        elif task == "associations":
            instruction = "What word is most associated with the given words?"
        elif task == "collocations":
            instruction = "What word commonly appears with the given word?"
        elif task == "semantic_similarity":
            instruction = "Rate the semantic similarity of these sentences (0-5)."
        elif task == "text_classification":
            instruction = "Classify the given text into the appropriate category."
        else:
            instruction = "Answer the following question."

        system = (
            f"{instruction}\n"
            'Return a JSON object with keys: "answer" (your response) and "confidence" (0..1).'
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(ex.input)},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=512)

    def parse_prediction(self, ex: Example, resp: ModelResponse) -> Prediction:
        raw_text = resp.text or ""
        obj = extract_first_json_obj(raw_text)

        answer = obj.get("answer", raw_text) if obj else raw_text
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
            extra={"parsed_json": obj is not None, "task": ex.meta.get("task", "")},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = str(ex.target).strip().lower()
        got = str(pred.answer).strip().lower()

        # Task-specific scoring
        task = ex.meta.get("task", "")

        if task == "semantic_similarity":
            # Numeric comparison for similarity scores
            try:
                gold_num = float(gold)
                got_num = float(re.search(r"[\d.]+", got).group())
                correct = int(abs(gold_num - got_num) < 0.5)
            except:
                correct = int(gold == got)
        else:
            # String matching for other tasks
            correct = int(gold == got or gold in got)

        out = {
            "correct": correct,
            "gold": str(ex.target),
            "predicted": pred.answer,
            "task": task,
        }
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
