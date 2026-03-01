from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

from huggingface_hub import hf_hub_download

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


def normalize_math_answer(s: str) -> str:
    """Normalize math answers for comparison."""
    s = s.strip()
    # Remove LaTeX formatting
    s = re.sub(r"\\boxed\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\$([^$]+)\$", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = re.sub(r"[{}]", "", s)
    s = s.lower().strip()
    return s


@dataclass(slots=True)
class OmniMathBenchmark(BaseBenchmark):
    """Omni-MATH: Math olympiad problems.

    4,428 olympiad-level math problems.
    GPT-5 accuracy: 72%
    Dataset: https://huggingface.co/datasets/KbsdJames/Omni-MATH
    """

    name: str = "omnimath"
    mode: str = "text"

    # Filter by difficulty (None = all)
    difficulty_filter: str | None = None

    # Filter by domain (None = all)
    domain_filter: str | None = None

    def iter_examples(self, split: str) -> Iterable[Example]:
        # Download JSONL directly (datasets library has metadata bug)
        jsonl_path = hf_hub_download(
            "KbsdJames/Omni-MATH",
            "test.jsonl",
            repo_type="dataset"
        )

        with open(jsonl_path, "r") as f:
            for idx, line in enumerate(f):
                row = json.loads(line.strip())

                difficulty = row.get("difficulty", "")
                domain = row.get("domain", "")

                if self.difficulty_filter and difficulty != self.difficulty_filter:
                    continue
                if self.domain_filter and domain != self.domain_filter:
                    continue

                yield Example(
                    id=f"omnimath_{idx}",
                    input=row["problem"],
                    target=row.get("answer", ""),
                    meta={
                        "split": split,
                        "difficulty": difficulty,
                        "domain": domain,
                        "source": row.get("source", ""),
                        "solution": row.get("solution", ""),
                    },
                )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are an expert mathematician solving olympiad problems.\n"
            "Think step by step, show your work, then provide your final answer.\n"
            'Return a JSON object with keys: "reasoning" (your solution steps), "answer" (final answer), and "confidence" (0..1).'
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(ex.input)},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=2048)

    def parse_prediction(self, ex: Example, resp: ModelResponse) -> Prediction:
        raw_text = resp.text or ""
        obj = extract_first_json_obj(raw_text)

        answer = None
        confidence = None

        if obj:
            answer = obj.get("answer", "")
            if obj.get("confidence") is not None:
                try:
                    confidence = clamp01(float(obj["confidence"]))
                except:
                    pass

        # Fallback: try to extract answer from common patterns
        if not answer:
            # Try \boxed{...} pattern (LaTeX)
            boxed_match = re.search(r"\\boxed\{([^}]+)\}", raw_text)
            if boxed_match:
                answer = boxed_match.group(1)
            else:
                # Try "answer is X" or "answer: X" patterns
                ans_match = re.search(r"(?:answer|result|solution)\s*(?:is|:)\s*[\"']?([^\n\"']+)", raw_text, re.IGNORECASE)
                if ans_match:
                    answer = ans_match.group(1).strip()
                else:
                    # Try "= X" at end of reasoning
                    eq_match = re.search(r"=\s*([^\n=]+?)\s*(?:\.|$)", raw_text)
                    if eq_match:
                        answer = eq_match.group(1).strip()
                    else:
                        # Last resort: use last line if it's short
                        lines = [l.strip() for l in raw_text.strip().split('\n') if l.strip()]
                        if lines and len(lines[-1]) < 100:
                            answer = lines[-1]
                        else:
                            answer = raw_text[:200]  # Truncate to avoid huge answers

        return Prediction(
            example_id=ex.id,
            answer=str(answer).strip(),
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = normalize_math_answer(str(ex.target))
        got = normalize_math_answer(pred.answer)

        # Check for exact match or numeric equivalence
        correct = 0
        if gold == got:
            correct = 1
        else:
            # Try numeric comparison
            try:
                if abs(float(gold) - float(got)) < 1e-6:
                    correct = 1
            except:
                pass

        out = {
            "correct": correct,
            "gold": str(ex.target),
            "predicted": pred.answer,
            "difficulty": ex.meta.get("difficulty", ""),
            "domain": ex.meta.get("domain", ""),
        }
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
