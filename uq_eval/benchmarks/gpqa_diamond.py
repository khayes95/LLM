from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark


def _norm_key(k: str) -> str:
    # normalize csv column names: lower + remove non-alnum
    return re.sub(r"[^a-z0-9]+", "", k.lower())


def _norm_text(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    # drop some trailing punctuation
    s = re.sub(r"[\"'`]+", "", s)
    return s


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _extract_first_json_obj(text: str) -> dict | None:
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _extract_choice_letter(text: str) -> str | None:
    if not text:
        return None
    # match A/B/C/D as standalone tokens
    m = re.findall(r"\b([A-D])\b", text.upper())
    return m[-1] if m else None


def _parse_permutation(x: Any) -> list[int] | None:
    """Try parse permutation field like: [2,0,1,3] or '2 0 1 3' or '2,0,1,3'."""
    if x is None:
        return None
    if isinstance(x, list) and len(x) == 4 and all(isinstance(i, int) for i in x):
        return x
    s = str(x).strip()
    if not s:
        return None
    try:
        v = json.loads(s)
        if isinstance(v, list) and len(v) == 4 and all(isinstance(i, int) for i in v):
            return v
    except Exception:
        pass
    # fallback parse numbers
    nums = re.findall(r"-?\d+", s)
    if len(nums) == 4:
        try:
            v = [int(n) for n in nums]
            if sorted(v) == [0, 1, 2, 3]:
                return v
        except Exception:
            return None
    return None


def _stable_seed(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _format_mcq(question: str, choices: list[str]) -> str:
    letters = ["A", "B", "C", "D"]
    lines = ["Question:", question.strip(), "", "Options:"]
    for i, c in enumerate(choices[:4]):
        lines.append(f"{letters[i]}) {c}")
    return "\n".join(lines)


@dataclass(slots=True)
class GPQADiamondBenchmark(BaseBenchmark):
    """
    Real GPQA Diamond CSV benchmark.

    Supports common GPQA CSV schema (as used by many implementations):
      - Question
      - Correct Answer
      - Incorrect Answer 1
      - Incorrect Answer 2
      - Incorrect Answer 3
    Optional:
      - Subject / Domain
      - permutation (a fixed shuffle order of [0,1,2,3])

    Model output format (enforced by prompt):
      {"answer":"A|B|C|D", "confidence": 0..1}
    """

    name: str = "gpqa_diamond"
    mode: str = "text"

    # Path can be overridden by env var UQ_GPQA_DATA_PATH.
    data_path: str | None = None

    # If no permutation column exists, we will generate a deterministic shuffle using this seed.
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.data_path:
            self.data_path = os.environ.get("UQ_GPQA_DATA_PATH") or "data/gpqa/gpqa_diamond.csv"
        p = Path(self.data_path)
        if not p.exists():
            raise FileNotFoundError(
                f"GPQADiamondBenchmark cannot find CSV: {p}\n"
                f"Set env UQ_GPQA_DATA_PATH or place file at data/gpqa/gpqa_diamond.csv"
            )

    def iter_examples(self, split: str) -> Iterable[Example]:
        assert self.data_path is not None
        p = Path(self.data_path)

        with p.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {p}")

            # Build normalized-key lookup
            fieldnames = list(reader.fieldnames)
            kmap = {_norm_key(k): k for k in fieldnames}

            def get_col(*candidates: str) -> str | None:
                for c in candidates:
                    kk = _norm_key(c)
                    if kk in kmap:
                        return kmap[kk]
                return None

            # Required columns (try a few common variants)
            q_col = get_col("Question", "question", "problem", "prompt")
            ca_col = get_col("Correct Answer", "correct_answer", "correctanswer", "answer_correct", "gold")
            ia1_col = get_col("Incorrect Answer 1", "incorrect_answer_1", "incorrectanswer1", "wrong_answer_1", "distractor_1")
            ia2_col = get_col("Incorrect Answer 2", "incorrect_answer_2", "incorrectanswer2", "wrong_answer_2", "distractor_2")
            ia3_col = get_col("Incorrect Answer 3", "incorrect_answer_3", "incorrectanswer3", "wrong_answer_3", "distractor_3")

            if not (q_col and ca_col and ia1_col and ia2_col and ia3_col):
                raise ValueError(
                    "Cannot infer required GPQA columns. "
                    "Expected something like: Question / Correct Answer / Incorrect Answer 1/2/3.\n"
                    f"Found columns: {fieldnames}"
                )

            subj_col = get_col("Subject", "subject", "Domain", "domain", "Topic", "topic")
            perm_col = get_col("permutation", "perm", "choice_permutation")

            for idx, row in enumerate(reader):
                q = str(row.get(q_col, "")).strip()
                correct = str(row.get(ca_col, "")).strip()
                i1 = str(row.get(ia1_col, "")).strip()
                i2 = str(row.get(ia2_col, "")).strip()
                i3 = str(row.get(ia3_col, "")).strip()

                if not q or not correct or not i1 or not i2 or not i3:
                    # skip malformed rows but keep deterministic id
                    continue

                # Base choices: index 0 is always correct (so permutation can be interpreted consistently)
                base_choices = [correct, i1, i2, i3]

                perm = None
                if perm_col and row.get(perm_col) not in (None, ""):
                    perm = _parse_permutation(row.get(perm_col))

                if perm is None:
                    rng = random.Random(self.seed + _stable_seed(f"{p.name}:{idx}:{q[:40]}"))
                    perm = [0, 1, 2, 3]
                    rng.shuffle(perm)

                # Apply permutation
                choices = [base_choices[i] for i in perm]
                correct_pos = perm.index(0)  # where base index 0 moved to
                gold_letter = ["A", "B", "C", "D"][correct_pos]

                mcq_text = _format_mcq(q, choices)

                meta = {"split": split, "source": str(p)}
                if subj_col and row.get(subj_col) not in (None, ""):
                    meta["subject"] = row.get(subj_col)
                meta["perm"] = perm
                meta["correct_text"] = correct
                meta["choices"] = choices

                ex_id = f"gpqa_diamond_{idx}"
                yield Example(id=ex_id, input=mcq_text, target=gold_letter, meta=meta)

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are running in an evaluation harness.\n"
            "This is a 4-option multiple-choice question.\n"
            'Return ONLY a JSON object with keys: "answer" (one letter A/B/C/D) and "confidence" (number 0..1).\n'
            "No extra text, no markdown."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(ex.input)},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=128)

    def parse_prediction(self, ex: Example, resp: ModelResponse) -> Prediction:
        raw_text = resp.text or ""
        obj = _extract_first_json_obj(raw_text)

        answer: Any = None
        confidence: float | None = None

        if obj:
            answer = obj.get("answer", None)
            conf = obj.get("confidence", None)
            try:
                if conf is not None:
                    confidence = _clamp01(float(conf))
            except Exception:
                confidence = None

        # Build mapping: choice text -> letter (for cases model outputs full text)
        choice_map: dict[str, str] = {}
        choices = []
        try:
            choices = ex.meta.get("choices", [])
        except Exception:
            choices = []
        for i, c in enumerate(choices[:4]):
            choice_map[_norm_text(c)] = ["A", "B", "C", "D"][i]

        def coerce_to_letter(a: Any) -> str | None:
            if a is None:
                return None
            s = str(a).strip()
            if not s:
                return None
            # number 1-4
            if s in ("1", "2", "3", "4"):
                return ["A", "B", "C", "D"][int(s) - 1]
            # exact letter
            if len(s) == 1 and s.upper() in ("A", "B", "C", "D"):
                return s.upper()
            # try match to choice text
            ns = _norm_text(s)
            if ns in choice_map:
                return choice_map[ns]
            # sometimes answer contains the option text
            for k, v in choice_map.items():
                if k and k in ns:
                    return v
            return None

        letter = coerce_to_letter(answer)
        if letter is None:
            letter = _extract_choice_letter(raw_text)

        final_answer = letter or (str(answer).strip() if answer is not None else raw_text.strip())

        return Prediction(
            example_id=ex.id,
            answer=str(final_answer).strip().upper(),
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = str(ex.target).strip().upper()
        got = str(pred.answer).strip().upper()

        correct = int(gold == got)
        out = {"correct": correct}
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
