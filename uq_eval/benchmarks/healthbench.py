from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import hf_hub_download

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


@dataclass(slots=True)
class HealthBenchBenchmark(BaseBenchmark):
    """HealthBench benchmark for medical conversation evaluation.

    HealthBench contains 5,000 realistic health conversations with physician-created
    rubrics for grading model responses. Each conversation includes:
    - Multi-turn prompts (user/assistant history)
    - Multiple rubric criteria with point values

    Subsets:
    - hard: 1,000 challenging examples
    - consensus: 3,671 examples with multiply-validated criteria

    Note: Full rubric evaluation requires an LLM grader. This implementation
    generates responses and captures rubric metadata for later grading.

    Dataset: https://huggingface.co/datasets/openai/healthbench
    """

    name: str = "healthbench"
    mode: str = "text"

    # Which subset to use: "hard" or "consensus"
    subset: str = "hard"

    # Cache for downloaded file path
    _data_path: str | None = None

    def _get_data_path(self) -> str:
        """Download and cache the dataset file."""
        if self._data_path is None:
            if self.subset == "hard":
                filename = "hard_2025-05-08-21-00-10.jsonl"
            elif self.subset == "consensus":
                filename = "consensus_2025-05-09-20-00-46.jsonl"
            else:
                raise ValueError(f"Unknown subset: {self.subset}. Use 'hard' or 'consensus'")

            self._data_path = hf_hub_download(
                repo_id="openai/healthbench",
                filename=filename,
                repo_type="dataset",
            )
        return self._data_path

    def iter_examples(self, split: str) -> Iterable[Example]:
        data_path = self._get_data_path()

        with open(data_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                row = json.loads(line)

                # Build conversation from prompt messages
                prompt_messages = row.get("prompt", [])

                # Get the last user message as the main input
                last_user_msg = ""
                for msg in reversed(prompt_messages):
                    if msg.get("role") == "user":
                        last_user_msg = msg.get("content", "")
                        break

                # Get rubrics for metadata
                rubrics = row.get("rubrics", [])
                total_points = sum(r.get("points", 0) for r in rubrics)

                yield Example(
                    id=f"healthbench_{self.subset}_{idx}",
                    input=prompt_messages,  # Full conversation history
                    target=None,  # No single correct answer - uses rubric grading
                    meta={
                        "split": split,
                        "subset": self.subset,
                        "prompt_id": row.get("prompt_id", ""),
                        "rubrics": rubrics,
                        "total_points": total_points,
                        "num_criteria": len(rubrics),
                        "example_tags": row.get("example_tags", []),
                        "last_user_message": last_user_msg,
                    },
                )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are a helpful medical assistant providing accurate health information.\n"
            "Respond to the user's health question with accurate, helpful, and compassionate advice.\n"
            "Be thorough but concise. If you're unsure, say so.\n"
            "At the end, provide a confidence score (0-1) for how confident you are in your response.\n"
            'Format: Include "Confidence: X.XX" on its own line at the end.'
        )

        # Build messages from conversation history
        messages = [{"role": "system", "content": system}]

        # Add the conversation history
        prompt_messages = ex.input
        if isinstance(prompt_messages, list):
            for msg in prompt_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                # Skip any existing system messages
                if role != "system":
                    messages.append({"role": role, "content": content})
        else:
            # Fallback if input is a string
            messages.append({"role": "user", "content": str(ex.input)})

        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=2048)

    def parse_prediction(self, ex: Example, resp: ModelResponse) -> Prediction:
        raw_text = resp.text or ""

        # Try to extract confidence from "Confidence: X.XX" pattern
        confidence = None
        import re
        conf_match = re.search(r"confidence:\s*([\d.]+)", raw_text, re.IGNORECASE)
        if conf_match:
            try:
                confidence = clamp01(float(conf_match.group(1)))
            except ValueError:
                pass

        # Also try JSON extraction as fallback
        if confidence is None:
            obj = extract_first_json_obj(raw_text)
            if obj:
                conf = obj.get("confidence", None)
                if conf is not None:
                    try:
                        confidence = clamp01(float(conf))
                    except Exception:
                        pass

        return Prediction(
            example_id=ex.id,
            answer=raw_text.strip(),  # Full response is the answer
            confidence=confidence,
            raw_text=raw_text,
            extra={
                "rubrics": ex.meta.get("rubrics", []),
                "total_points": ex.meta.get("total_points", 0),
            },
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        """Score the prediction.

        Note: Full rubric-based scoring requires an LLM grader (like GPT-4).
        This implementation returns metadata for offline grading.
        For now, we return a placeholder score based on response quality heuristics.
        """
        response = pred.answer or ""

        # Basic quality heuristics (not real scoring - just for testing)
        # Real scoring should use the rubric criteria with an LLM judge
        has_content = len(response) > 50
        is_confident = pred.confidence is not None

        # Placeholder: mark as "needs_grading"
        out = {
            "correct": -1,  # -1 indicates needs rubric grading
            "needs_grading": True,
            "response_length": len(response),
            "has_confidence": is_confident,
            "num_criteria": ex.meta.get("num_criteria", 0),
            "total_points": ex.meta.get("total_points", 0),
            "subset": ex.meta.get("subset", ""),
        }

        if pred.confidence is not None:
            # Can't compute real brier without knowing if correct
            out["confidence"] = pred.confidence

        return out
