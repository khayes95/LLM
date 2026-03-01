from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Literal

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj, normalize_text


@dataclass(slots=True)
class HELMETBenchmark(BaseBenchmark):
    """HELMET: Holistically Evaluating Long-context Models.

    7 diverse task categories: Recall, RAG, Rerank, Cite, LongQA, Summ, ICL.
    Tests models at context lengths from 8K to 128K tokens.

    From Princeton-NLP: https://github.com/princeton-nlp/HELMET
    """

    name: str = "helmet"
    mode: str = "longtext"
    task: Literal["recall", "rag", "longqa", "summ"] = "longqa"
    max_context_length: int = 8192  # Maximum context length in tokens (approximate)
    max_examples: int | None = None

    def iter_examples(self, split: str) -> Iterable[Example]:
        # HELMET data is stored on HuggingFace
        try:
            ds = load_dataset("princeton-nlp/HELMET", split="train", streaming=True)
        except Exception:
            # Fallback: load specific task files if available
            ds = load_dataset("princeton-nlp/HELMET", split="train")

        count = 0
        for idx, row in enumerate(ds):
            # Parse the jsonl field which contains the actual data
            if "jsonl" in row and isinstance(row["jsonl"], str):
                try:
                    data = json.loads(row["jsonl"])
                except json.JSONDecodeError:
                    continue
            else:
                data = row

            # Extract fields based on task type
            question = data.get("question", data.get("input", ""))
            context = data.get("context", data.get("passage", data.get("document", "")))
            answer = data.get("answer", data.get("output", data.get("target", "")))

            # Skip if missing required fields
            if not question or not answer:
                continue

            # Truncate context if too long (rough character estimate)
            # ~4 chars per token is a rough approximation
            max_chars = self.max_context_length * 4
            if len(context) > max_chars:
                context = context[:max_chars] + "..."

            yield Example(
                id=f"helmet_{self.task}_{idx}",
                input={
                    "question": question,
                    "context": context,
                },
                target=answer,
                meta={
                    "task": self.task,
                    "context_length": len(context),
                    "key": row.get("__key__", ""),
                },
            )

            count += 1
            if self.max_examples and count >= self.max_examples:
                break

    def build_request(self, ex: Example) -> ModelRequest:
        question = ex.input["question"]
        context = ex.input["context"]

        if self.task == "recall":
            prompt = (
                f"Read the following text carefully and answer the question.\n\n"
                f"Text:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Answer the question based only on the information in the text."
            )
        elif self.task == "rag":
            prompt = (
                f"Use the following context to answer the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Provide a concise answer based on the context."
            )
        elif self.task == "longqa":
            prompt = (
                f"Read the following document and answer the question.\n\n"
                f"Document:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Provide a detailed answer based on the document."
            )
        elif self.task == "summ":
            prompt = (
                f"Summarize the following text.\n\n"
                f"Text:\n{context}\n\n"
                f"Provide a concise summary capturing the main points."
            )
        else:
            prompt = f"Context:\n{context}\n\nQuestion: {question}"

        system = (
            "You are a helpful assistant that answers questions based on provided context.\n"
            'Return a JSON object with keys: "answer" (your response) and "confidence" (0..1).\n'
            "Be accurate and only use information from the given context."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=512)

    def parse_prediction(self, ex: Example, resp: ModelResponse) -> Prediction:
        raw_text = resp.text or ""
        obj = extract_first_json_obj(raw_text)

        answer = None
        confidence = None

        if obj:
            answer = obj.get("answer", obj.get("response", None))
            conf = obj.get("confidence", None)
            try:
                if conf is not None:
                    confidence = clamp01(float(conf))
            except Exception:
                confidence = None

        if answer is None:
            answer = raw_text.strip()

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

        # Use F1 scoring for long-form answers
        gold_tokens = set(gold.split())
        got_tokens = set(got.split())

        if not gold_tokens or not got_tokens:
            f1 = 0.0
        else:
            overlap = gold_tokens & got_tokens
            precision = len(overlap) / len(got_tokens) if got_tokens else 0
            recall = len(overlap) / len(gold_tokens) if gold_tokens else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Exact match
        em = int(gold == got)

        # Containment check for shorter gold answers
        contains = int(gold in got or got in gold) if gold else 0

        # Score as correct if F1 > 0.5 or contains match
        correct = int(f1 > 0.5 or contains == 1)

        out = {
            "correct": correct,
            "f1": f1,
            "em": em,
            "contains": contains,
        }

        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2

        return out
