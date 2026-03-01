from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from ..types import Example, ModelRequest, ModelResponse, Prediction
from .base import BaseBenchmark
from .common import extract_first_json_obj


# Color mapping for text representation
COLOR_NAMES = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey",
    6: "magenta",
    7: "orange",
    8: "cyan",
    9: "maroon",
}


def grid_to_text(grid: list[list[int]], use_colors: bool = False) -> str:
    """Convert a grid to text representation.

    Args:
        grid: 2D list of integers 0-9
        use_colors: If True, use color names instead of numbers

    Returns:
        Text representation of the grid
    """
    if use_colors:
        lines = []
        for row in grid:
            line = " ".join(COLOR_NAMES.get(c, str(c)) for c in row)
            lines.append(line)
        return "\n".join(lines)
    else:
        # Use compact number representation
        lines = []
        for row in grid:
            line = " ".join(str(c) for c in row)
            lines.append(line)
        return "\n".join(lines)


def parse_grid_from_text(text: str) -> list[list[int]] | None:
    """Parse a grid from text representation.

    Handles both number format and color name format.
    """
    # Reverse color mapping
    name_to_num = {v: k for k, v in COLOR_NAMES.items()}

    lines = text.strip().split("\n")
    grid = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Try to parse as numbers
        row = []
        tokens = line.replace(",", " ").replace("[", "").replace("]", "").split()

        for token in tokens:
            token = token.strip().lower()
            if token.isdigit():
                row.append(int(token))
            elif token in name_to_num:
                row.append(name_to_num[token])
            else:
                # Try to extract number from token
                nums = re.findall(r'\d+', token)
                if nums:
                    row.append(int(nums[0]))

        if row:
            grid.append(row)

    return grid if grid else None


def grids_equal(grid1: list[list[int]], grid2: list[list[int]]) -> bool:
    """Check if two grids are equal."""
    if grid1 is None or grid2 is None:
        return False
    if len(grid1) != len(grid2):
        return False
    for row1, row2 in zip(grid1, grid2):
        if len(row1) != len(row2):
            return False
        if row1 != row2:
            return False
    return True


@dataclass(slots=True)
class ARCAGIBenchmark(BaseBenchmark):
    """ARC-AGI: Abstraction and Reasoning Corpus.

    Abstract reasoning puzzles with colored grids. Represented as text.
    Models must learn the transformation pattern from examples and apply to test.

    Very challenging: GPT-5 achieves ~10% on ARC-AGI-1.
    """

    name: str = "arc_agi"
    mode: str = "text"
    version: int = 1  # 1 for ARC-AGI-1, 2 for ARC-AGI-2
    use_color_names: bool = False  # Use color names instead of numbers

    def iter_examples(self, split: str) -> Iterable[Example]:
        # Load from HuggingFace - use community datasets
        if self.version == 1:
            # Try multiple dataset options
            try:
                ds = load_dataset("lordspline/arc-agi", split="evaluation")
            except Exception:
                ds = load_dataset("dataartist/arc-agi", split="test")
        else:
            ds = load_dataset("barc0/ARC-AGI-2-public-train", split="train")

        for idx, row in enumerate(ds):
            # Each task has train examples and test examples
            train_examples = row.get("train", [])
            test_examples = row.get("test", [])
            task_id = row.get("task_id", f"task_{idx}")

            # For each test case in the task
            for test_idx, test_case in enumerate(test_examples):
                test_input = test_case.get("input", [])
                test_output = test_case.get("output", [])

                yield Example(
                    id=f"arc_agi_{self.version}_{task_id}_{test_idx}",
                    input={
                        "train_examples": train_examples,
                        "test_input": test_input,
                        "task_id": task_id,
                    },
                    target=test_output,
                    meta={
                        "version": self.version,
                        "num_train_examples": len(train_examples),
                        "input_shape": f"{len(test_input)}x{len(test_input[0]) if test_input else 0}",
                        "output_shape": f"{len(test_output)}x{len(test_output[0]) if test_output else 0}",
                    },
                )

    def build_request(self, ex: Example) -> ModelRequest:
        train_examples = ex.input["train_examples"]
        test_input = ex.input["test_input"]

        # Format training examples
        examples_text = []
        for i, train_ex in enumerate(train_examples):
            input_grid = train_ex.get("input", [])
            output_grid = train_ex.get("output", [])

            input_text = grid_to_text(input_grid, self.use_color_names)
            output_text = grid_to_text(output_grid, self.use_color_names)

            examples_text.append(
                f"Example {i+1}:\n"
                f"Input:\n{input_text}\n\n"
                f"Output:\n{output_text}"
            )

        examples_section = "\n\n".join(examples_text)
        test_input_text = grid_to_text(test_input, self.use_color_names)

        prompt = (
            f"Here are some examples of input-output grid transformations:\n\n"
            f"{examples_section}\n\n"
            f"Now apply the same transformation to this test input:\n\n"
            f"Test Input:\n{test_input_text}\n\n"
            f"What is the output grid?"
        )

        system = (
            "You are solving abstract reasoning puzzles with colored grids.\n"
            "Each cell contains a number 0-9 representing a color:\n"
            "0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=grey, 6=magenta, 7=orange, 8=cyan, 9=maroon\n\n"
            "Study the input-output examples to understand the transformation pattern.\n"
            "Apply the same pattern to the test input.\n\n"
            'Return ONLY a JSON object with keys: "output" (2D array of numbers) and "confidence" (0..1).\n'
            'Example: {"output": [[0,1,2],[3,4,5]], "confidence": 0.8}\n'
            "No extra text or explanation."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        return ModelRequest(messages=messages, temperature=0.0, max_output_tokens=2048)

    def parse_prediction(self, ex: Example, resp: ModelResponse) -> Prediction:
        raw_text = resp.text or ""
        obj = extract_first_json_obj(raw_text)

        answer = None
        confidence = None

        if obj:
            answer = obj.get("output", None)
            conf = obj.get("confidence", None)
            try:
                if conf is not None:
                    confidence = max(0.0, min(1.0, float(conf)))
            except Exception:
                confidence = None

        # Fallback: try to parse grid from text
        if answer is None:
            answer = parse_grid_from_text(raw_text)

        return Prediction(
            example_id=ex.id,
            answer=answer,
            confidence=confidence,
            raw_text=raw_text,
            extra={"parsed_json": obj is not None},
        )

    def score(self, ex: Example, pred: Prediction) -> dict:
        gold = ex.target
        got = pred.answer

        correct = int(grids_equal(gold, got))
        out = {"correct": correct}

        if pred.confidence is not None:
            out["brier"] = (float(pred.confidence) - correct) ** 2

        return out
