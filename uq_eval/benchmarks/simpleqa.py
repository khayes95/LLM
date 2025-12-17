from __future__ import annotations

import ast
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark


SYSTEM_JSON_ANSWER_CONFIDENCE = """You are running in an evaluation harness.
Answer the user's question.

Return ONLY a JSON object with keys:
- "answer": string
- "confidence": number between 0 and 1

No extra text, no markdown.
If you are not confident enough to answer, set "answer" to "NOT_ATTEMPTED" and set "confidence" to a low value.
"""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return _project_root() / path_str


def _safe_parse_metadata(s: str) -> Any:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    try:
        return json.loads(s)
    except Exception:
        return s


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_obj(text: str) -> Optional[dict]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _coerce_confidence(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if v != v:  # NaN
        return None
    return max(0.0, min(1.0, v))


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _tokenize(s: str) -> Tuple[str, ...]:
    s = (s or "").casefold()
    parts = re.split(r"[^a-z0-9#+]+", s)
    return tuple(t for t in parts if t)


def _is_abstain(ans: str) -> bool:
    a = (ans or "").strip().casefold()
    return (not a) or ("not_attempted" in a) or (a in {"not attempted", "cannot answer", "can't answer"})


def _token_containment_correct(pred: str, gold: str) -> bool:
    pred_toks = set(_tokenize(pred))
    gold_toks = set(_tokenize(gold))
    if not gold_toks:
        return False
    return gold_toks.issubset(pred_toks)


@dataclass(slots=True)
class SimpleQABenchmark(BaseBenchmark):
    """SimpleQA CSV loader.

    Default file:
      data/simpleqa/simple_qa_test_set.csv

    You can override via env:
      UQ_SIMPLEQA_PATH=/abs/or/rel/path.csv
      UQ_USE_SAMPLE=1 to use sample file if it exists.
    """

    name: str = "simpleqa"
    mode: str = "text"

    data_path: str = "data/simpleqa/simple_qa_test_set.csv"
    sample_path: str = "data/simpleqa/simple_qa_test_set.sample.csv"

    def _choose_path(self) -> Path:
        p_env = os.environ.get("UQ_SIMPLEQA_PATH")
        if p_env:
            return _resolve_path(p_env)

        use_sample = os.environ.get("UQ_USE_SAMPLE", "").strip().lower() in {"1", "true", "yes"}
        if use_sample:
            sp = _resolve_path(self.sample_path)
            if sp.exists():
                return sp

        return _resolve_path(self.data_path)

    def iter_examples(self, split: str) -> Iterable[Example]:
        path = self._choose_path()
        if not path.exists():
            raise FileNotFoundError(f"SimpleQA CSV not found: {path}")

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                q = (row.get("problem") or row.get("question") or row.get("input") or "").strip()
                a = (row.get("answer") or row.get("target") or "").strip()

                # Optional metadata field
                meta_raw = row.get("metadata") or row.get("meta") or ""
                meta = _safe_parse_metadata(meta_raw)

                # If the CSV provides an id, use it; else use index.
                ex_id = row.get("id") or row.get("qid") or row.get("question_id") or f"simpleqa_{i}"

                yield Example(
                    id=str(ex_id),
                    input=q,
                    target=a,
                    meta={"split": split, "metadata": meta, "source": str(path)},
                )

    def build_request(self, ex: Example) -> ModelRequest:
        messages = [
            {"role": "system", "content": SYSTEM_JSON_ANSWER_CONFIDENCE},
            {"role": "user", "content": str(ex.input)},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=256)

    # ✅ 关键修复：签名必须是 (ex, resp)
    def parse_prediction(self, ex: Example, resp: ModelResponse) -> Prediction:
        raw_text = resp.text or ""
        obj = _extract_json_obj(raw_text)

        if not obj:
            return Prediction(
                example_id=ex.id,
                answer=raw_text.strip(),
                confidence=None,
                raw_text=raw_text,
                extra={"parsed_json": False},
            )

        ans = str(obj.get("answer", "")).strip()
        conf = _coerce_confidence(obj.get("confidence", None))

        return Prediction(
            example_id=ex.id,
            answer=ans,
            confidence=conf,
            raw_text=raw_text,
            extra={"parsed_json": True},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = str(ex.target or "").strip()
        ans = str(pred.answer or "").strip()

        abstain = _is_abstain(ans)

        correct = 0
        if not abstain:
            # baseline: exact match OR token containment
            correct = int(_normalize(ans) == _normalize(gold) or _token_containment_correct(ans, gold))

        out = {"correct": correct, "abstain": int(abstain)}
        if pred.confidence is not None:
            out["brier"] = float((pred.confidence - float(correct)) ** 2)
        return out
