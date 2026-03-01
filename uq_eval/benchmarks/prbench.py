from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


# Available PRBench splits
PRBENCH_SPLITS = ["finance", "legal", "finance_hard", "legal_hard"]


@dataclass(slots=True)
class PRBenchBenchmark(BaseBenchmark):
    """PRBench: Professional Reasoning Benchmark (Legal & Finance).

    Multi-turn expert-level reasoning in legal and finance domains.
    Evaluates complex professional reasoning with rubric-based scoring.

    Dataset: https://huggingface.co/datasets/ScaleAI/PRBench
    Leaderboard: https://scale.com/leaderboard/prbench-legal

    Splits:
    - finance: 600 examples (~51% GPT-5)
    - legal: 500 examples (~50% GPT-5)
    - finance_hard: 300 examples
    - legal_hard: 250 examples

    Note: This benchmark uses rubric-based evaluation. The score() method
    returns correct=-1 indicating offline/LLM-judge evaluation is needed.
    """

    name: str = "prbench"
    mode: str = "text"

    # Which domain: "finance", "legal", "finance_hard", "legal_hard", or "all"
    domain: str = "all"

    # Which turn to evaluate (0-9, or -1 for last turn with prompt)
    # Default: evaluate the final prompt in the conversation
    turn: int = -1

    def iter_examples(self, split: str) -> Iterable[Example]:
        # Determine which splits to load
        if self.domain == "all":
            splits_to_load = PRBENCH_SPLITS
        elif self.domain in PRBENCH_SPLITS:
            splits_to_load = [self.domain]
        else:
            splits_to_load = []

        example_idx = 0
        for domain_split in splits_to_load:
            try:
                ds = load_dataset("ScaleAI/PRBench", split=domain_split)
            except Exception:
                continue

            for row in ds:
                num_turns = row.get("turns", 10)

                # Determine which turn to use
                if self.turn >= 0:
                    turn_idx = min(self.turn, num_turns - 1)
                else:
                    # Find the last turn with a prompt
                    turn_idx = num_turns - 1
                    for i in range(num_turns - 1, -1, -1):
                        if row.get(f"prompt_{i}"):
                            turn_idx = i
                            break

                # Get the prompt for this turn
                prompt = row.get(f"prompt_{turn_idx}", "")
                if not prompt:
                    continue

                # Build conversation context from previous turns
                context_parts = []
                for i in range(turn_idx):
                    prev_prompt = row.get(f"prompt_{i}", "")
                    prev_response = row.get(f"response_{i}", "")
                    if prev_prompt:
                        context_parts.append(f"User: {prev_prompt}")
                    if prev_response:
                        context_parts.append(f"Assistant: {prev_response}")

                context = "\n\n".join(context_parts) if context_parts else ""

                # Get reference response if available
                reference_response = row.get(f"response_{turn_idx}", "")

                # Get rubric for evaluation
                rubric = row.get("rubric", [])

                yield Example(
                    id=f"prbench_{domain_split}_{example_idx}",
                    input=prompt,
                    target=reference_response,  # Reference response (for rubric comparison)
                    meta={
                        "split": split,
                        "domain": domain_split,
                        "field": row.get("field", ""),
                        "topic": row.get("topic", ""),
                        "expert": row.get("expert", ""),
                        "task_id": row.get("task", ""),
                        "turn": turn_idx,
                        "num_turns": num_turns,
                        "context": context,
                        "rubric": rubric,
                        "scratchpad": row.get("scratchpad", ""),
                        "decision_type": row.get("decision_type", ""),
                    },
                )
                example_idx += 1

    def build_request(self, ex: Example) -> ModelRequest:
        domain = ex.meta.get("domain", "")
        field = ex.meta.get("field", "Finance")
        context = ex.meta.get("context", "")

        if "legal" in domain:
            system = (
                f"You are an expert legal professional with deep knowledge of {field}.\n"
                "Provide thorough, well-reasoned advice based on relevant legal principles.\n"
                "Be precise about jurisdictional considerations and cite relevant precedents when applicable.\n"
                'Return a JSON object with keys: "reasoning" (your legal analysis), '
                '"answer" (your advice/conclusion), and "confidence" (0..1).'
            )
        else:
            system = (
                f"You are an expert finance professional with deep knowledge of {field}.\n"
                "Provide thorough, well-reasoned advice based on financial principles and regulations.\n"
                "Consider risk management, compliance requirements, and practical implementation.\n"
                'Return a JSON object with keys: "reasoning" (your analysis), '
                '"answer" (your advice/recommendation), and "confidence" (0..1).'
            )

        messages = [{"role": "system", "content": system}]

        # Add conversation context if available
        if context:
            messages.append({
                "role": "user",
                "content": f"[Previous conversation for context]\n\n{context}\n\n[Current question]\n\n{ex.input}"
            })
        else:
            messages.append({"role": "user", "content": str(ex.input)})

        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=4096)

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
                "domain": ex.meta.get("domain", ""),
                "field": ex.meta.get("field", ""),
            },
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        # PRBench requires rubric-based evaluation (LLM judge)
        # Return -1 to indicate offline evaluation needed
        out = {
            "correct": -1,  # Requires rubric-based LLM judge evaluation
            "gold": str(ex.target)[:500] if ex.target else "",  # Reference response (truncated)
            "predicted": pred.answer[:500] if pred.answer else "",
            "domain": ex.meta.get("domain", ""),
            "field": ex.meta.get("field", ""),
            "topic": ex.meta.get("topic", ""),
            "turn": ex.meta.get("turn", 0),
            "rubric_available": bool(ex.meta.get("rubric")),
        }
        if pred.confidence is not None:
            # Can't compute brier without correct label
            out["confidence"] = float(pred.confidence)
        return out
