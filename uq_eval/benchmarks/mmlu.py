from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


# MMLU subjects
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions"
]


@dataclass(slots=True)
class MMLUBenchmark(BaseBenchmark):
    """MMLU: Massive Multitask Language Understanding.

    57 subjects across STEM, humanities, social sciences.
    Dataset: https://huggingface.co/datasets/cais/mmlu
    """

    name: str = "mmlu"
    mode: str = "text"

    # Filter by subject
    subject_filter: str | None = None

    # Use specific subjects (list of subject names)
    subjects: list | None = None

    def iter_examples(self, split: str) -> Iterable[Example]:
        subjects_to_use = self.subjects or MMLU_SUBJECTS

        if self.subject_filter:
            subjects_to_use = [s for s in subjects_to_use if self.subject_filter.lower() in s.lower()]

        for subject in subjects_to_use:
            try:
                ds = load_dataset("cais/mmlu", subject, split=split)
            except Exception:
                continue

            for idx, row in enumerate(ds):
                question = row["question"]
                choices = row["choices"]

                # Format as MCQ
                formatted = []
                for i, choice in enumerate(choices):
                    letter = chr(ord('A') + i)
                    formatted.append(f"{letter}. {choice}")

                full_input = question + "\n\n" + "\n".join(formatted)

                # Answer is index 0-3
                answer_idx = row["answer"]
                answer_letter = chr(ord('A') + answer_idx)

                yield Example(
                    id=f"mmlu_{subject}_{split}_{idx}",
                    input=full_input,
                    target=answer_letter,
                    meta={
                        "split": split,
                        "subject": subject,
                        "choices": choices,
                    },
                )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are an expert answering multiple-choice questions.\n"
            "Think carefully, then provide your answer.\n"
            'Return ONLY a JSON object with keys: "reasoning" (brief explanation), '
            '"answer" (letter A, B, C, or D), and "confidence" (0..1).'
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": str(ex.input)},
        ]
        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=512)

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
            except:
                pass

        if answer is None:
            match = re.search(r"\b([A-D])\b", raw_text)
            if match:
                answer = match.group(1)
            else:
                answer = raw_text.strip()[:5]

        return Prediction(
            example_id=ex.id,
            answer=str(answer).strip().upper() if answer else "",
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = str(ex.target).strip().upper()
        got = re.sub(r"[^A-D]", "", str(pred.answer).upper())
        if got:
            got = got[0]

        correct = int(gold == got)

        out = {
            "correct": correct,
            "gold": gold,
            "predicted": pred.answer,
            "subject": ex.meta.get("subject", ""),
        }
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
