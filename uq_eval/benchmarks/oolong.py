from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


@dataclass(slots=True)
class OolongBenchmark(BaseBenchmark):
    """Oolong: Long context understanding benchmark.

    Tests understanding of long documents (8K-128K tokens) with various tasks.
    Dataset: https://huggingface.co/datasets/oolongbench/oolong-synth

    5,200 examples in the synthetic version.
    GPT-5 scores ~47-70% depending on context length.
    """

    name: str = "oolong"
    mode: str = "longtext"

    # Which variant: "synth" (synthetic, 5200 examples) or "real" (requires config)
    variant: str = "synth"

    # Filter by task group (e.g., "counting", "classification")
    task_filter: str | None = None

    def iter_examples(self, split: str) -> Iterable[Example]:
        # Load the appropriate dataset
        if self.variant == "synth":
            ds = load_dataset("oolongbench/oolong-synth", split="test")
        else:
            # Real variant requires a config (dnd, toy_dnd)
            ds = load_dataset("oolongbench/oolong-real", "dnd", split="test")

        for idx, row in enumerate(ds):
            task_group = row.get("task_group", "")
            task = row.get("task", "")

            if self.task_filter and self.task_filter.lower() not in task_group.lower():
                continue

            # Build input: context + question
            context = row.get("context_window_text", "")
            question = row.get("question", "")

            # Combine context and question
            if context and question:
                input_text = f"{context}\n\n{question}"
            else:
                input_text = question or context

            # Get answer - may be a list
            answer = row.get("answer", "")
            if isinstance(answer, list):
                answer = answer[0] if answer else ""

            yield Example(
                id=f"oolong_{self.variant}_{idx}",
                input=input_text,
                target=str(answer),
                meta={
                    "split": split,
                    "variant": self.variant,
                    "task_group": task_group,
                    "task": task,
                    "context_len": row.get("context_len", 0),
                    "answer_type": row.get("answer_type", ""),
                    "dataset": row.get("dataset", ""),
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        task_group = ex.meta.get("task_group", "")

        # Build task-specific instruction
        if "count" in task_group.lower():
            instruction = "Count the requested items in the given context and provide the number."
        elif "classif" in task_group.lower():
            instruction = "Classify or identify the label as requested based on the context."
        elif "extract" in task_group.lower():
            instruction = "Extract the requested information from the context."
        else:
            instruction = "Answer the question based on the given context."

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
            except Exception:
                pass

        return Prediction(
            example_id=ex.id,
            answer=str(answer).strip(),
            confidence=confidence,
            raw_text=raw_text,
            extra={
                "parsed_json": obj is not None,
                "task_group": ex.meta.get("task_group", ""),
            },
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = str(ex.target).strip().lower()
        got = str(pred.answer).strip().lower()

        answer_type = ex.meta.get("answer_type", "")

        # Scoring depends on answer type
        if answer_type == "number" or "count" in ex.meta.get("task_group", "").lower():
            # Numeric comparison
            try:
                gold_num = float(re.search(r"[\d.]+", gold).group())
                got_num = float(re.search(r"[\d.]+", got).group())
                correct = int(abs(gold_num - got_num) < 0.01)
            except Exception:
                correct = int(gold == got)
        else:
            # String matching
            correct = int(gold == got or gold in got or got in gold)

        out = {
            "correct": correct,
            "gold": str(ex.target),
            "predicted": pred.answer,
            "task_group": ex.meta.get("task_group", ""),
            "task": ex.meta.get("task", ""),
            "context_len": ex.meta.get("context_len", 0),
        }
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
