from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


# Default path to local MultiChallenge data
DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "multichallenge" / "benchmark_questions.jsonl"


@dataclass(slots=True)
class MultiChallengeBenchmark(BaseBenchmark):
    """MultiChallenge: Multi-turn conversation benchmark.

    Tests LLMs on complex multi-turn conversations across four axes:
    - INFERENCE_MEMORY: Remembering and reasoning about prior context
    - INSTRUCTION_COMPLIANCE: Following complex/nested instructions
    - CONTEXT_SHIFT: Handling topic changes mid-conversation
    - PERSONA_CONSISTENCY: Maintaining consistent persona/role

    Dataset: https://github.com/ekwinox117/multi-challenge
    Paper: https://scale.com/leaderboard/multichallenge

    273 examples with binary YES/NO pass criteria.
    GPT-5 scores ~58-64%.
    """

    name: str = "multichallenge"
    mode: str = "text"

    # Path to local JSONL file
    data_path: str | None = None

    # Filter by axis (INFERENCE_MEMORY, INSTRUCTION_COMPLIANCE, CONTEXT_SHIFT, PERSONA_CONSISTENCY)
    axis_filter: str | None = None

    def iter_examples(self, split: str) -> Iterable[Example]:
        # Use provided path or default
        path = Path(self.data_path) if self.data_path else DEFAULT_DATA_PATH

        if not path.exists():
            raise FileNotFoundError(
                f"MultiChallenge data not found at {path}. "
                "Download from https://github.com/ekwinox117/multi-challenge/blob/main/data/benchmark_questions.jsonl"
            )

        with open(path, "r") as f:
            for idx, line in enumerate(f):
                row = json.loads(line.strip())

                axis = row.get("AXIS", "")
                if self.axis_filter and self.axis_filter.upper() != axis.upper():
                    continue

                # Build the full conversation context
                conversation = row.get("CONVERSATION", [])
                target_question = row.get("TARGET_QUESTION", "")
                pass_criteria = row.get("PASS_CRITERIA", "")

                # Format conversation as context
                conv_parts = []
                for msg in conversation:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    conv_parts.append(f"{role.capitalize()}: {content}")

                context = "\n\n".join(conv_parts)

                yield Example(
                    id=f"multichallenge_{row.get('QUESTION_ID', idx)}",
                    input=context,
                    target=pass_criteria,  # YES or NO
                    meta={
                        "split": split,
                        "axis": axis,
                        "target_question": target_question,
                        "question_id": row.get("QUESTION_ID", ""),
                        "num_turns": len(conversation),
                    },
                )

    def build_request(self, ex: Example) -> ModelRequest:
        target_question = ex.meta.get("target_question", "")

        system = (
            "You are evaluating a multi-turn conversation for correctness and consistency.\n"
            "You will be given a conversation history and a verification question.\n"
            "Answer the verification question with YES or NO based on the conversation.\n"
            'Return a JSON object with keys: "reasoning" (your analysis), '
            '"answer" (YES or NO), and "confidence" (0..1).'
        )

        user_content = (
            f"## Conversation History\n\n{ex.input}\n\n"
            f"## Verification Question\n\n{target_question}\n\n"
            "Based on the conversation above, answer the verification question with YES or NO."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=1024)

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
                except Exception:
                    pass

        # Normalize answer to YES/NO
        if answer:
            answer_upper = str(answer).strip().upper()
            if "YES" in answer_upper:
                answer = "YES"
            elif "NO" in answer_upper:
                answer = "NO"
            else:
                answer = answer_upper
        else:
            # Fallback: try to extract from raw text
            raw_upper = raw_text.upper()
            if "YES" in raw_upper and "NO" not in raw_upper:
                answer = "YES"
            elif "NO" in raw_upper and "YES" not in raw_upper:
                answer = "NO"
            else:
                answer = raw_text.strip()[:50]

        return Prediction(
            example_id=ex.id,
            answer=answer,
            confidence=confidence,
            raw_text=raw_text,
            extra={
                "parsed_json": obj is not None,
                "axis": ex.meta.get("axis", ""),
            },
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = str(ex.target).strip().upper()
        got = str(pred.answer).strip().upper()

        # Simple YES/NO matching
        correct = int(gold == got)

        out = {
            "correct": correct,
            "gold": gold,
            "predicted": got,
            "axis": ex.meta.get("axis", ""),
            "target_question": ex.meta.get("target_question", ""),
        }
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
