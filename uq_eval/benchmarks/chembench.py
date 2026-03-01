from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_choice_letter, extract_first_json_obj, normalize_text


CHEMBENCH_CONFIGS = [
    "analytical_chemistry",
    "chemical_preference",
    "general_chemistry",
    "inorganic_chemistry",
    "materials_science",
    "organic_chemistry",
    "physical_chemistry",
    "technical_chemistry",
    "toxicity_and_safety",
]


@dataclass(slots=True)
class ChemBenchBenchmark(BaseBenchmark):
    """ChemBench: Evaluating chemistry and materials science capabilities of LLMs.

    2,700+ high-quality questions manually curated by chemistry experts.
    Covers diverse chemical disciplines and complexity levels.

    From: https://huggingface.co/datasets/jablonkagroup/ChemBench
    """

    name: str = "chembench"
    mode: str = "text"
    subset: str | None = None  # None means all subsets, or one of CHEMBENCH_CONFIGS
    max_examples: int | None = None

    def iter_examples(self, split: str) -> Iterable[Example]:
        # Load from HuggingFace - need to specify a config
        configs = [self.subset] if self.subset else CHEMBENCH_CONFIGS

        datasets = []
        for config in configs:
            try:
                ds = load_dataset("jablonkagroup/ChemBench", config, split="train")
                datasets.append((config, ds))
            except Exception:
                continue

        if not datasets:
            raise RuntimeError("Failed to load any ChemBench configs")

        count = 0
        global_idx = 0
        for config, ds in datasets:
            for row in ds:
                # ChemBench has questions in 'examples' field as list of dicts
                examples_list = row.get("examples", [])
                task_name = row.get("name", "")

                for ex_idx, example in enumerate(examples_list):
                    question = example.get("input", "")
                    target = example.get("target", "")

                    # Get target_scores for MCQ - may be string or dict
                    target_scores_raw = example.get("target_scores", {})
                    if isinstance(target_scores_raw, str):
                        try:
                            target_scores = json.loads(target_scores_raw)
                        except json.JSONDecodeError:
                            target_scores = {}
                    else:
                        target_scores = target_scores_raw or {}

                    # Skip empty examples
                    if not question or (not target and not target_scores):
                        continue

                    # Determine if MCQ based on target_scores
                    is_mcq = bool(target_scores)
                    choices = list(target_scores.keys()) if is_mcq else None

                    # For MCQ, find correct answer (score=1)
                    if is_mcq and not target:
                        for choice, score in target_scores.items():
                            if score == 1:
                                target = choice
                                break

                    yield Example(
                        id=f"chembench_{config}_{global_idx}_{ex_idx}",
                        input={
                            "question": question,
                            "choices": choices if is_mcq else None,
                            "is_mcq": is_mcq,
                        },
                        target=target,
                        meta={
                            "category": config,
                            "task_name": task_name,
                            "subfield": row.get("subfield", ""),
                        },
                    )

                    count += 1
                    if self.max_examples and count >= self.max_examples:
                        return

                global_idx += 1

    def build_request(self, ex: Example) -> ModelRequest:
        question = ex.input["question"]
        is_mcq = ex.input["is_mcq"]
        choices = ex.input["choices"]

        if is_mcq and choices:
            # Format MCQ choices
            letters = "ABCDEFGH"
            choices_text = "\n".join([
                f"({letters[i]}) {choice}"
                for i, choice in enumerate(choices[:8])
            ])
            prompt = f"{question}\n\nChoices:\n{choices_text}"

            system = (
                "You are a chemistry and materials science expert.\n"
                "Answer the multiple-choice question by selecting the correct option.\n"
                'Return a JSON object with keys: "answer" (letter A-H) and "confidence" (0..1).\n'
                "No extra text or explanation."
            )
        else:
            prompt = question

            system = (
                "You are a chemistry and materials science expert.\n"
                "Answer the question accurately and concisely.\n"
                'Return a JSON object with keys: "answer" (your response) and "confidence" (0..1).\n'
                "For numerical answers, include units if applicable."
            )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
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

        is_mcq = ex.input.get("is_mcq", False)

        if answer is None:
            if is_mcq:
                # Extract letter for MCQ
                answer = extract_choice_letter(raw_text) or raw_text.strip()
            else:
                answer = raw_text.strip()

        # Normalize MCQ answer to single letter
        if is_mcq and answer:
            answer = str(answer).strip().upper()
            if answer.startswith("(") and answer.endswith(")"):
                answer = answer[1:-1]
            if len(answer) > 1 and answer[0].isalpha():
                answer = answer[0]

        return Prediction(
            example_id=ex.id,
            answer=answer,
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None, "is_mcq": is_mcq},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = str(ex.target).strip()
        got = str(pred.answer or "").strip()

        is_mcq = ex.input.get("is_mcq", False)

        if is_mcq:
            # MCQ: map letter to choice text, then compare with target (which is the text)
            choices = ex.input.get("choices", [])
            got_norm = got.upper().strip()
            if len(got_norm) > 1:
                got_norm = got_norm[0]

            # Map letter to choice text (A=0, B=1, etc.)
            letter_idx = ord(got_norm) - ord('A') if got_norm.isalpha() else -1
            if 0 <= letter_idx < len(choices):
                got_text = choices[letter_idx]
            else:
                got_text = got  # Fallback to raw answer

            # Compare choice text with target text
            correct = int(normalize_text(got_text) == normalize_text(gold))
        else:
            # Open-ended: normalize and check containment
            gold_norm = normalize_text(gold)
            got_norm = normalize_text(got)

            # Exact match
            if gold_norm == got_norm:
                correct = 1
            # Containment for shorter answers
            elif len(gold_norm) < 50 and (gold_norm in got_norm or got_norm in gold_norm):
                correct = 1
            # Try numeric comparison
            else:
                try:
                    # Extract numbers
                    import re
                    gold_nums = re.findall(r'-?[\d.]+', gold)
                    got_nums = re.findall(r'-?[\d.]+', got)
                    if gold_nums and got_nums:
                        # Compare first number
                        g1 = float(gold_nums[0])
                        g2 = float(got_nums[0])
                        correct = int(abs(g1 - g2) / (abs(g1) + 1e-10) < 0.05)  # 5% tolerance
                    else:
                        correct = 0
                except Exception:
                    correct = 0

        out = {"correct": correct}

        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2

        return out
