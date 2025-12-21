from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@dataclass(slots=True)
class LongBenchBenchmark(BaseBenchmark):
    """LongBench: Long context understanding benchmark.

    Multi-task benchmark for long context (avg 6-15k tokens).
    Tasks: single/multi-doc QA, summarization, few-shot, code, synthetic.
    Dataset: https://huggingface.co/datasets/THUDM/LongBench
    """

    name: str = "longbench"
    mode: str = "longtext"

    # Filter by task (e.g., "narrativeqa", "qasper", "multifieldqa_en")
    task_filter: str | None = None

    def iter_examples(self, split: str) -> Iterable[Example]:
        # LongBench has multiple subsets
        subsets = [
            "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa",
            "2wikimqa", "musique", "gov_report", "qmsum", "multi_news",
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en",
            "lcc", "repobench-p"
        ]

        if self.task_filter:
            subsets = [s for s in subsets if self.task_filter.lower() in s.lower()]

        for subset in subsets:
            try:
                ds = load_dataset("THUDM/LongBench", subset, split="test", trust_remote_code=True)
            except Exception:
                continue

            for idx, row in enumerate(ds):
                context = row.get("context", "")
                question = row.get("input", "")
                answers = row.get("answers", [])
                answer = answers[0] if answers else ""

                full_input = f"Context:\n{context}\n\nQuestion: {question}"

                yield Example(
                    id=f"longbench_{subset}_{idx}",
                    input=full_input,
                    target=answer,
                    meta={
                        "split": split,
                        "task": subset,
                        "all_answers": answers,
                        "length": row.get("length", len(context)),
                    },
                )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are a helpful assistant answering questions based on the given context.\n"
            "Read the context carefully and answer accurately.\n"
            'Return a JSON object with keys: "answer" (your answer) and "confidence" (0..1).'
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(ex.input)},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=1024)

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
        got_norm = normalize_answer(pred.answer)

        # Check against all valid answers
        all_answers = ex.meta.get("all_answers", [str(ex.target)])
        correct = 0
        for ans in all_answers:
            ans_norm = normalize_answer(ans)
            if ans_norm == got_norm or ans_norm in got_norm or got_norm in ans_norm:
                correct = 1
                break

        out = {
            "correct": correct,
            "gold": str(ex.target),
            "predicted": pred.answer,
            "task": ex.meta.get("task", ""),
        }
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out


@dataclass(slots=True)
class LongBenchV2Benchmark(BaseBenchmark):
    """LongBench v2: Extended long context benchmark.

    More challenging long context tasks with 256k context support.
    Dataset: https://huggingface.co/datasets/THUDM/LongBench-v2
    """

    name: str = "longbench_v2"
    mode: str = "longtext"

    def iter_examples(self, split: str) -> Iterable[Example]:
        try:
            ds = load_dataset("THUDM/LongBench-v2", split="test")
        except Exception:
            # Fallback
            ds = load_dataset("THUDM/LongBench-v2", split="train")

        for idx, row in enumerate(ds):
            context = row.get("context", "")
            question = row.get("question", row.get("input", ""))
            answer = row.get("answer", row.get("answers", [""])[0] if isinstance(row.get("answers"), list) else "")

            full_input = f"Context:\n{context}\n\nQuestion: {question}"

            yield Example(
                id=f"longbench_v2_{idx}",
                input=full_input,
                target=answer,
                meta={
                    "split": split,
                    "task": row.get("task", ""),
                    "length": row.get("length", len(context)),
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are a helpful assistant answering questions based on long context.\n"
            "Read carefully and answer accurately.\n"
            'Return a JSON object with keys: "answer" (your answer) and "confidence" (0..1).'
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(ex.input)},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=1024)

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
            extra={"parsed_json": obj is not None},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = normalize_answer(str(ex.target))
        got = normalize_answer(pred.answer)

        correct = int(gold == got or gold in got or got in gold)

        out = {
            "correct": correct,
            "gold": str(ex.target),
            "predicted": pred.answer,
            "task": ex.meta.get("task", ""),
        }
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
