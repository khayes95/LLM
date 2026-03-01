from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj, normalize_text


@dataclass(slots=True)
class BABILongBenchmark(BaseBenchmark):
    """BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack.

    Needle-in-haystack reasoning benchmark with 20 tasks.
    Facts are hidden in irrelevant background text.
    Tests context lengths from 0 to 10M tokens.

    From: https://github.com/booydar/babilong
    HuggingFace: RMT-team/babilong
    """

    name: str = "babilong"
    mode: str = "longtext"
    task: Literal["qa1", "qa2", "qa3", "qa4", "qa5"] = "qa1"
    context_length: Literal["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"] = "4k"
    max_examples: int | None = 100  # Default limit

    def iter_examples(self, split: str) -> Iterable[Example]:
        # BABILong is available on HuggingFace
        # Configs are separate: task configs (qa1-qa10) and length configs (0k-10M)
        # We load by task and use context_length as the split
        try:
            ds = load_dataset(
                "RMT-team/babilong",
                self.task,
                split=self.context_length,
                trust_remote_code=True,
            )
        except Exception as e:
            # Fallback: try just the context length as config
            try:
                ds = load_dataset(
                    "RMT-team/babilong",
                    self.context_length,
                    split="train",
                    trust_remote_code=True,
                )
            except Exception:
                raise RuntimeError(f"Failed to load BABILong dataset: {e}")

        count = 0
        for idx, row in enumerate(ds):
            # Extract fields
            question = row.get("question", row.get("input", ""))
            context = row.get("context", row.get("passage", row.get("input", "")))
            target = row.get("target", row.get("answer", row.get("output", "")))

            # Handle case where input contains both context and question
            if not question and "input" in row:
                parts = str(row["input"]).split("\n")
                if len(parts) > 1:
                    question = parts[-1]
                    context = "\n".join(parts[:-1])

            if not question or not target:
                continue

            yield Example(
                id=f"babilong_{self.task}_{self.context_length}_{idx}",
                input={
                    "question": question,
                    "context": context,
                },
                target=str(target),
                meta={
                    "task": self.task,
                    "context_length": self.context_length,
                    "actual_context_chars": len(context),
                },
            )

            count += 1
            if self.max_examples and count >= self.max_examples:
                break

    def build_request(self, ex: Example) -> ModelRequest:
        question = ex.input["question"]
        context = ex.input["context"]

        prompt = (
            f"Read the following text carefully. Important information is scattered "
            f"throughout the text. Answer the question at the end.\n\n"
            f"Text:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer with just the answer, no explanation."
        )

        system = (
            "You are a careful reader. Answer questions based on information in the text.\n"
            "The relevant facts may be hidden among irrelevant background text.\n"
            'Return a JSON object with keys: "answer" (short answer) and "confidence" (0..1).\n'
            "Be concise - just give the answer."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        # Estimate max tokens based on context length
        max_input_tokens = {
            "0k": 512,
            "1k": 2048,
            "2k": 4096,
            "4k": 8192,
            "8k": 16384,
            "16k": 32768,
            "32k": 65536,
            "64k": 131072,
            "128k": 262144,
        }

        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=128)

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
            # Clean up the answer
            answer = raw_text.strip()
            # Remove common prefixes
            for prefix in ["Answer:", "The answer is", "A:"]:
                if answer.lower().startswith(prefix.lower()):
                    answer = answer[len(prefix):].strip()

        return Prediction(
            example_id=ex.id,
            answer=answer,
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = normalize_text(str(ex.target))
        got = normalize_text(str(pred.answer or ""))

        # BABILong uses short answers, so use exact/containment matching
        em = int(gold == got)
        contains = int(gold in got or got in gold) if gold else 0

        # For short answers, be stricter
        correct = int(em == 1 or (contains == 1 and len(gold) > 2))

        out = {
            "correct": correct,
            "em": em,
            "contains": contains,
        }

        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2

        return out
