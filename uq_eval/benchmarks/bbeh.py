from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import clamp01, extract_first_json_obj


# BBEH evaluation logic (adapted from google-deepmind/bbeh/evaluate.py)
def strip_latex(response: str) -> str:
    if response.startswith("$") and response.endswith("$"):
        response = response[1:-1]
    if "boxed{" in response and response.endswith("}"):
        response = response[0:-1].split("boxed{")[1]
    if "text{" in response and response.endswith("}"):
        response = response[0:-1].split("text{")[1]
    if "texttt{" in response and response.endswith("}"):
        response = response[0:-1].split("texttt{")[1]
    return response


def extract_answer(sample: str) -> str:
    """Extracts the final answer from the sample."""
    answer_prefixes = [
        "The answer is:",
        "The final answer is ",
        "The final answer is: ",
        "The answer is ",
    ]
    answer = sample
    for answer_prefix in answer_prefixes:
        if answer_prefix in answer:
            answer = answer.split(answer_prefix)[-1].strip()
    if answer.endswith("."):
        answer = answer[:-1]
    return strip_latex(answer)


def fuzzy_match(prediction: str, reference: str) -> bool:
    """Fuzzy match function for BigBench Extra Hard."""
    if prediction == reference:
        return True

    # (a) vs a
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        return reference[1] == prediction

    # Numbers
    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass

    # Quote issues
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True

    # Bracket issues
    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True

    # Question mark issues
    if prediction.endswith("?") and prediction[:-1] == reference:
        return True

    return False


def preprocess_sample(sample: str) -> str:
    prediction = extract_answer(sample.strip()).lower()
    prediction = prediction.replace(", ", ",").replace("**", "")
    prediction = prediction.split("\n")[0]
    prediction = prediction[0:-1] if prediction.endswith(".") else prediction
    return prediction


def preprocess_reference(reference: str) -> str:
    reference = reference.strip().lower()
    reference = reference.replace(", ", ",")
    return reference


def evaluate_correctness(sample: str, reference: str) -> bool:
    prediction = preprocess_sample(sample)
    reference = preprocess_reference(reference)
    return fuzzy_match(prediction, reference)


# All 23 BBEH tasks
BBEH_TASKS = [
    "bbeh_boardgame_qa",
    "bbeh_boolean_expressions",
    "bbeh_buggy_tables",
    "bbeh_causal_understanding",
    "bbeh_disambiguation_qa",
    "bbeh_dyck_languages",
    "bbeh_geometric_shapes",
    "bbeh_hyperbaton",
    "bbeh_linguini",
    "bbeh_movie_recommendation",
    "bbeh_multistep_arithmetic",
    "bbeh_nycc",
    "bbeh_object_counting",
    "bbeh_object_properties",
    "bbeh_sarc_triples",
    "bbeh_shuffled_objects",
    "bbeh_spatial_reasoning",
    "bbeh_sportqa",
    "bbeh_temporal_sequence",
    "bbeh_time_arithmetic",
    "bbeh_web_of_lies",
    "bbeh_word_sorting",
    "bbeh_zebra_puzzles",
]


@dataclass(slots=True)
class BBEHBenchmark(BaseBenchmark):
    """BIG-Bench Extra Hard (BBEH) benchmark.

    BBEH replaces each task in BBH with a novel task that probes similar
    reasoning capability but exhibits significantly increased difficulty.

    - Full version: 4520 examples across 23 tasks
    - Mini version: 460 examples (20 per task)

    Data source: https://github.com/google-deepmind/bbeh
    """

    name: str = "bbeh"
    mode: str = "text"

    # Path to cloned BBEH repo
    data_path: str = "data/bbeh"

    # Which tasks to include (None = all 23)
    tasks: list[str] | None = None

    # Use mini version (460 examples) instead of full (4520)
    use_mini: bool = False

    def iter_examples(self, split: str) -> Iterable[Example]:
        # Determine base path
        if os.path.isabs(self.data_path):
            base_path = Path(self.data_path)
        else:
            # Relative to current working directory
            base_path = Path(self.data_path)

        if self.use_mini:
            tasks_dir = base_path / "bbeh" / "mini"
        else:
            tasks_dir = base_path / "bbeh" / "benchmark_tasks"

        if not tasks_dir.exists():
            raise FileNotFoundError(
                f"BBEH data not found at {tasks_dir}. "
                f"Clone the repo: git clone https://github.com/google-deepmind/bbeh.git {self.data_path}"
            )

        # Which tasks to load
        task_list = self.tasks if self.tasks else BBEH_TASKS

        for task_name in task_list:
            task_dir = tasks_dir / task_name
            task_file = task_dir / "task.json"

            if not task_file.exists():
                print(f"Warning: Task file not found: {task_file}")
                continue

            with open(task_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            examples = data.get("examples", [])
            for idx, ex in enumerate(examples):
                yield Example(
                    id=f"{task_name}_{idx}",
                    input=ex["input"],
                    target=ex["target"],
                    meta={
                        "task": task_name,
                        "split": split,
                    },
                )

    def build_request(self, ex: Example) -> ModelRequest:
        system = (
            "You are a helpful assistant solving reasoning problems.\n"
            "Think step by step, then provide your final answer.\n"
            'Return ONLY a JSON object with keys: "reasoning" (your step-by-step thinking), '
            '"answer" (your final answer), and "confidence" (0..1 how confident you are).\n'
            "For the answer, be concise - just the answer value, not a full sentence."
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
            answer = obj.get("answer", None)
            conf = obj.get("confidence", None)
            try:
                if conf is not None:
                    confidence = clamp01(float(conf))
            except Exception:
                confidence = None

        if answer is None:
            # Fallback: try to extract answer using BBEH's extraction logic
            answer = extract_answer(raw_text)

        return Prediction(
            example_id=ex.id,
            answer=str(answer).strip() if answer else "",
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None, "task": ex.meta.get("task", "")},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = str(ex.target).strip()
        got = str(pred.answer).strip()

        # Use BBEH's official evaluation logic
        correct = int(evaluate_correctness(got, gold))

        out = {
            "correct": correct,
            "gold": gold,
            "predicted": got,
            "task": ex.meta.get("task", ""),
        }
        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2
        return out
