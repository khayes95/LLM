from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


@dataclass(slots=True)
class SWEBenchBenchmark(BaseBenchmark):
    """SWE-Bench: Software Engineering benchmark.

    Real GitHub issues requiring code changes.
    Subsets: full (2,294), lite (300), verified (500)
    Dataset: https://huggingface.co/datasets/princeton-nlp/SWE-bench
    """

    name: str = "swebench"
    mode: str = "text"

    # Which subset: "full", "lite", or "verified"
    subset: str = "lite"

    def iter_examples(self, split: str) -> Iterable[Example]:
        # Load appropriate subset
        if self.subset == "lite":
            ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        elif self.subset == "verified":
            ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        else:
            ds = load_dataset("princeton-nlp/SWE-bench", split="test")

        for idx, row in enumerate(ds):
            # Problem statement combines issue description and hints
            problem = row.get("problem_statement", "")

            # The patch is the ground truth solution
            patch = row.get("patch", "")

            yield Example(
                id=f"swebench_{self.subset}_{idx}",
                input=problem,
                target=patch,
                meta={
                    "split": split,
                    "subset": self.subset,
                    "instance_id": row.get("instance_id", ""),
                    "repo": row.get("repo", ""),
                    "base_commit": row.get("base_commit", ""),
                    "hints_text": row.get("hints_text", ""),
                },
            )

    def build_request(self, ex: Example) -> ModelRequest:
        hints = ex.meta.get("hints_text", "")
        context = str(ex.input)
        if hints:
            context += f"\n\nHints:\n{hints}"

        system = (
            "You are an expert software engineer fixing bugs in open source projects.\n"
            "Analyze the issue and provide a patch to fix it.\n"
            'Return a JSON object with keys: "analysis" (your understanding), "patch" (unified diff format), and "confidence" (0..1).'
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": context},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=4096)

    def parse_prediction(self, ex: Example, resp: ModelResponse) -> Prediction:
        raw_text = resp.text or ""
        obj = extract_first_json_obj(raw_text)

        patch = obj.get("patch", raw_text) if obj else raw_text
        confidence = None
        if obj and obj.get("confidence") is not None:
            try:
                confidence = clamp01(float(obj["confidence"]))
            except:
                pass

        # Try to extract diff from markdown code blocks
        if "```diff" in patch:
            match = re.search(r"```diff\s*(.*?)\s*```", patch, re.DOTALL)
            if match:
                patch = match.group(1)
        elif "```" in patch:
            match = re.search(r"```\s*(.*?)\s*```", patch, re.DOTALL)
            if match:
                patch = match.group(1)

        return Prediction(
            example_id=ex.id,
            answer=str(patch).strip(),
            confidence=confidence,
            raw_text=raw_text,
            extra={
                "parsed_json": obj is not None,
                "instance_id": ex.meta.get("instance_id", ""),
            },
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        # SWE-Bench requires execution-based evaluation
        # (apply patch, run tests)
        out = {
            "correct": -1,  # Needs execution-based evaluation
            "needs_execution": True,
            "patch_length": len(pred.answer),
            "instance_id": ex.meta.get("instance_id", ""),
            "repo": ex.meta.get("repo", ""),
            "subset": ex.meta.get("subset", ""),
        }
        if pred.confidence is not None:
            out["confidence"] = pred.confidence
        return out
