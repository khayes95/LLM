"""LLM-as-Judge for rubric-based and open-ended evaluation.

Provides judge functions that use an LLM to evaluate model responses
against rubrics, reference answers, or quality criteria.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .models.base import BaseModelClient
from .types import ModelRequest, ModelResponse


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation."""
    correct: int  # 1 = correct, 0 = incorrect, -1 = uncertain
    score: float  # 0.0 to 1.0 continuous score
    reasoning: str  # Judge's explanation
    raw_response: str  # Full judge response
    extra: dict = None  # Additional metadata

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


def _parse_judge_response(text: str) -> tuple[int, float, str]:
    """Parse judge response, extracting verdict, score, and reasoning."""
    # Try JSON parsing first
    try:
        # Find JSON object in response
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            obj = json.loads(match.group(0))
            verdict = obj.get("verdict", obj.get("correct", ""))
            score = obj.get("score", obj.get("confidence", 0.5))
            reasoning = obj.get("reasoning", obj.get("explanation", ""))

            # Normalize verdict to int
            if isinstance(verdict, bool):
                correct = 1 if verdict else 0
            elif isinstance(verdict, (int, float)):
                correct = 1 if verdict > 0.5 else 0
            elif isinstance(verdict, str):
                v = verdict.lower().strip()
                if v in ("correct", "yes", "true", "pass", "1"):
                    correct = 1
                elif v in ("incorrect", "no", "false", "fail", "0"):
                    correct = 0
                else:
                    correct = -1
            else:
                correct = -1

            # Normalize score
            try:
                score = float(score)
                score = max(0.0, min(1.0, score))
            except:
                score = 0.5

            return correct, score, str(reasoning)
    except:
        pass

    # Fallback: look for keywords
    text_lower = text.lower()
    if "correct" in text_lower and "incorrect" not in text_lower:
        return 1, 0.8, text
    elif "incorrect" in text_lower or "wrong" in text_lower:
        return 0, 0.2, text

    return -1, 0.5, text


def judge_correctness(
    client: BaseModelClient,
    question: str,
    reference_answer: str,
    model_answer: str,
    context: str = "",
) -> JudgeResult:
    """Judge if a model answer is correct given a reference answer.

    Simple correctness check for factual/short-answer questions.
    """
    system = """You are an expert judge evaluating answer correctness.
Compare the model's answer to the reference answer and determine if it is correct.
Consider semantic equivalence - answers don't need to be word-for-word identical.

Return a JSON object with:
- "verdict": "correct" or "incorrect"
- "score": 0.0 to 1.0 confidence in your judgment
- "reasoning": brief explanation of your decision"""

    prompt = f"""Question: {question}

Reference Answer: {reference_answer}

Model's Answer: {model_answer}
{f'Additional Context: {context}' if context else ''}

Is the model's answer correct?"""

    req = ModelRequest(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_output_tokens=512,
    )

    resp = client.generate(req)
    correct, score, reasoning = _parse_judge_response(resp.text)

    return JudgeResult(
        correct=correct,
        score=score,
        reasoning=reasoning,
        raw_response=resp.text,
    )


def judge_rubric(
    client: BaseModelClient,
    question: str,
    model_answer: str,
    rubric: str | list[dict],
    reference_answer: str = "",
) -> JudgeResult:
    """Judge a model answer against a rubric.

    For benchmarks like HealthBench, TutorBench, PRBench that use detailed rubrics.

    Args:
        client: LLM client to use for judging
        question: The original question/prompt
        model_answer: The model's response to evaluate
        rubric: Either a string describing criteria, or a list of rubric items
        reference_answer: Optional reference/gold answer
    """
    # Format rubric if it's a list
    if isinstance(rubric, list):
        rubric_text = "\n".join([
            f"- {item.get('criterion', item.get('description', str(item)))}"
            for item in rubric
        ])
    else:
        rubric_text = str(rubric)

    system = """You are an expert judge evaluating responses against a rubric.
Carefully assess how well the response meets each criterion in the rubric.

Return a JSON object with:
- "verdict": "correct" if the response substantially meets the rubric, "incorrect" otherwise
- "score": 0.0 to 1.0 representing how well the rubric criteria are met
- "reasoning": detailed explanation citing specific rubric criteria
- "criteria_met": list of which criteria were satisfied (optional)"""

    prompt = f"""Question/Prompt:
{question}

Model's Response:
{model_answer}
{f'''
Reference Answer:
{reference_answer}''' if reference_answer else ''}

Evaluation Rubric:
{rubric_text}

Evaluate the response against this rubric."""

    req = ModelRequest(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_output_tokens=1024,
    )

    resp = client.generate(req)
    correct, score, reasoning = _parse_judge_response(resp.text)

    return JudgeResult(
        correct=correct,
        score=score,
        reasoning=reasoning,
        raw_response=resp.text,
    )


def judge_quality(
    client: BaseModelClient,
    question: str,
    model_answer: str,
    criteria: str = "helpfulness, accuracy, and clarity",
) -> JudgeResult:
    """Judge response quality on general criteria.

    For open-ended evaluation without specific rubric.
    """
    system = f"""You are an expert judge evaluating response quality.
Assess the response based on: {criteria}

Return a JSON object with:
- "verdict": "correct" if the response is high quality, "incorrect" if low quality
- "score": 0.0 to 1.0 quality score
- "reasoning": explanation of your assessment"""

    prompt = f"""Question:
{question}

Response:
{model_answer}

Evaluate this response."""

    req = ModelRequest(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_output_tokens=512,
    )

    resp = client.generate(req)
    correct, score, reasoning = _parse_judge_response(resp.text)

    return JudgeResult(
        correct=correct,
        score=score,
        reasoning=reasoning,
        raw_response=resp.text,
    )


def judge_bullshit(
    client: BaseModelClient,
    question: str,
    nonsensical_element: str,
    model_answer: str,
) -> JudgeResult:
    """Judge whether a model's response correctly identifies a nonsensical premise.

    Uses the BullshitBench scoring rubric:
        2 = Clear Pushback: model clearly rejects the broken premise
        1 = Partial Challenge: model flags issues but still engages the bad premise
        0 = Accepted Nonsense: model treats the nonsense as valid

    Args:
        client: LLM client to use for judging
        question: The nonsensical question that was posed
        nonsensical_element: Explanation of why the question is nonsensical
        model_answer: The model's response to evaluate
    """
    system = """You are an expert judge for the BullshitBench benchmark. Your job is to evaluate whether an AI model detected that a question contains a fabricated or nonsensical premise.

Score the response using this rubric:
- Score 2 (Clear Pushback): The model clearly identifies that the premise is flawed, fabricated, or nonsensical. It refuses to engage with the false premise as if it were real.
- Score 1 (Partial Challenge): The model expresses some doubt or asks for clarification, but still partially engages with the nonsensical premise as if it could be valid.
- Score 0 (Accepted Nonsense): The model treats the nonsensical premise as completely valid and provides a substantive answer without questioning it.

Return a JSON object with:
- "score": 0, 1, or 2
- "reasoning": brief explanation of your judgment"""

    prompt = f"""Question posed to the model:
{question}

Why the question is nonsensical:
{nonsensical_element}

Model's response:
{model_answer}

Score the model's response (0, 1, or 2)."""

    req = ModelRequest(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_output_tokens=512,
    )

    resp = client.generate(req)
    raw = resp.text or ""

    # Parse the judge response
    pushback_score = -1
    reasoning = raw
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            obj = json.loads(match.group(0))
            pushback_score = int(obj.get("score", -1))
            reasoning = obj.get("reasoning", raw)
    except Exception:
        pass

    # Fallback: look for bare score
    if pushback_score not in (0, 1, 2):
        m = re.search(r'\b([012])\b', raw)
        if m:
            pushback_score = int(m.group(1))
        else:
            pushback_score = -1

    correct = 1 if pushback_score >= 1 else (0 if pushback_score == 0 else -1)
    normalized_score = pushback_score / 2.0 if pushback_score >= 0 else 0.5

    return JudgeResult(
        correct=correct,
        score=normalized_score,
        reasoning=reasoning,
        raw_response=raw,
        extra={"pushback_score": pushback_score},
    )


def batch_judge(
    client: BaseModelClient,
    items: list[dict],
    judge_fn: str = "correctness",
) -> list[JudgeResult]:
    """Batch judge multiple items.

    Args:
        client: LLM client
        items: List of dicts with keys matching judge function args
        judge_fn: One of "correctness", "rubric", "quality"

    Returns:
        List of JudgeResult objects
    """
    fn_map = {
        "correctness": judge_correctness,
        "rubric": judge_rubric,
        "quality": judge_quality,
    }

    fn = fn_map.get(judge_fn, judge_correctness)
    results = []

    for item in items:
        try:
            result = fn(client, **item)
            results.append(result)
        except Exception as e:
            results.append(JudgeResult(
                correct=-1,
                score=0.0,
                reasoning=f"Judge error: {e}",
                raw_response="",
            ))

    return results
