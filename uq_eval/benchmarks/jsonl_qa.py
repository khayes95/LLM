from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..io import iter_jsonl
from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj, normalize_text


@dataclass(slots=True)
class JsonlQABenchmark(BaseBenchmark):
    """Generic QA benchmark from a JSONL file with fields: id, input, target.

    Example line:
      {"id":"1","input":"What is 2+2?","target":"4"}
    """

    name: str = "jsonl_qa"
    mode: str = "text"
    data_path: str | None = None

    def __post_init__(self) -> None:
        if not self.data_path:
            raise ValueError("jsonl_qa requires --bench_data <path_to_jsonl>")

        p = Path(self.data_path)
        if not p.exists():
            raise FileNotFoundError(f"jsonl_qa cannot find data_path: {p}")

    def iter_examples(self, split: str) -> Iterable[Example]:
        assert self.data_path is not None
        for row in iter_jsonl(Path(self.data_path)):
            ex_id = str(row["id"])
            yield Example(
                id=ex_id,
                input=row["input"],
                target=row["target"],
                meta={"split": split, "source": str(self.data_path)},
            )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are running in an evaluation harness.\n"
            "Answer the user's question.\n"
            'Return ONLY a JSON object with keys: "answer" (string) and "confidence" (0..1).\n'
            "No extra text."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(ex.input)},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=256)

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
            answer = raw_text.strip()

        return Prediction(
            example_id=ex.id,
            answer=str(answer).strip(),
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = normalize_text(ex.target)
        got = normalize_text(pred.answer)

        correct = int(gold == got)
        out = {"correct": correct}
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
