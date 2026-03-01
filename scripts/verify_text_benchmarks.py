#!/usr/bin/env python3
"""
Verify text-only benchmarks load correctly and have proper grading format.

Similar to verify_benchmarks.py but for text-only benchmarks.
Tests that:
1. Dataset loads from HuggingFace
2. Samples have questions and answers
3. Grading format is correct (MCQ, open-ended, binary, etc.)
4. Category distribution is balanced (where applicable)

Usage:
    python scripts/verify_text_benchmarks.py [--benchmark NAME] [--max-samples N]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class BenchmarkVerification:
    """Results of benchmark verification."""
    name: str
    status: str  # "success", "error", "warning"
    total_samples: int = 0
    samples_with_questions: int = 0
    samples_with_answers: int = 0
    grading_type: str = ""  # "mcq", "binary", "open_ended", "numeric", "rubric"
    answer_format: str = ""  # Description of expected answer format
    categories: dict = field(default_factory=dict)  # Category distribution
    class_balance: dict = field(default_factory=dict)  # For binary/MCQ
    warnings: list = field(default_factory=list)
    error_message: str = ""
    gpt5_accuracy: str = ""  # Expected GPT-5 accuracy range


def verify_gpqa(max_samples: int = 100) -> BenchmarkVerification:
    """Verify GPQA Diamond - Graduate-level science questions."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="GPQA Diamond",
        status="pending",
        grading_type="mcq",
        answer_format="A/B/C/D letter",
        gpt5_accuracy="77-90%"
    )

    try:
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", trust_remote_code=True)
        result.total_samples = len(ds)

        answers = defaultdict(int)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            # Check question - GPQA uses "Question" field
            if row.get("Question") or row.get("Pre-Revision Question"):
                result.samples_with_questions += 1

            # Check answer - GPQA uses "Correct Answer" field
            answer = row.get("Correct Answer", "") or row.get("Pre-Revision Correct Answer", "")
            if answer:
                result.samples_with_answers += 1
                # Track which option (A/B/C/D) based on position
                answers["has_answer"] += 1

        result.class_balance = dict(answers)
        result.status = "success" if result.samples_with_answers > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_simpleqa(max_samples: int = 100) -> BenchmarkVerification:
    """Verify SimpleQA - Factual short-answer questions."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="SimpleQA",
        status="pending",
        grading_type="open_ended",
        answer_format="Short factual answer",
        gpt5_accuracy="19-54%"
    )

    try:
        # Use basicv8vc mirror (openai/SimpleQA is not public)
        ds = load_dataset("basicv8vc/SimpleQA", split="test", trust_remote_code=True)
        result.total_samples = len(ds)

        topics = defaultdict(int)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            if row.get("problem"):
                result.samples_with_questions += 1

            if row.get("answer"):
                result.samples_with_answers += 1

            # Track topics
            topic = row.get("metadata", {}).get("topic", "unknown") if isinstance(row.get("metadata"), dict) else "unknown"
            topics[topic] += 1

        result.categories = dict(topics)
        result.status = "success" if result.samples_with_answers > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_bbeh(max_samples: int = 100) -> BenchmarkVerification:
    """Verify BBEH (BIG-Bench Extra Hard)."""
    from pathlib import Path
    import json

    result = BenchmarkVerification(
        name="BBEH",
        status="pending",
        grading_type="mixed",
        answer_format="Task-dependent (MCQ/open-ended)",
        gpt5_accuracy="~50%"
    )

    try:
        # BBEH uses subdirectories with task.json files
        bbeh_dir = Path("data/bbeh/bbeh/benchmark_tasks")
        if not bbeh_dir.exists():
            result.status = "error"
            result.error_message = f"BBEH directory not found: {bbeh_dir}"
            return result

        tasks = defaultdict(int)
        total = 0
        with_answers = 0

        task_dirs = list(bbeh_dir.glob("bbeh_*"))
        samples_per_task = max(1, max_samples // len(task_dirs)) if task_dirs else max_samples

        for task_dir in task_dirs:
            task_name = task_dir.name
            task_file = task_dir / "task.json"

            if not task_file.exists():
                continue

            with open(task_file) as f:
                data = json.load(f)

            examples = data.get("examples", [])
            tasks[task_name] = len(examples)
            total += len(examples)

            for ex in examples[:samples_per_task]:
                if ex.get("target"):
                    with_answers += 1

        result.total_samples = total
        result.samples_with_questions = total
        result.samples_with_answers = with_answers
        result.categories = dict(tasks)
        result.status = "success" if with_answers > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_hle(max_samples: int = 100) -> BenchmarkVerification:
    """Verify HLE (Humanity's Last Exam) - Text only subset."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="HLE (Text Only)",
        status="pending",
        grading_type="mixed",
        answer_format="MCQ or exact match",
        gpt5_accuracy="25-30%"
    )

    try:
        ds = load_dataset("cais/hle", split="test", trust_remote_code=True)
        result.total_samples = len(ds)

        answer_types = defaultdict(int)
        text_only_count = 0

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            # Check if text-only (no image reference)
            question = row.get("question", "")
            has_image_ref = any(kw in question.lower() for kw in ["image", "figure", "diagram", "picture", "photo"])

            if not has_image_ref:
                text_only_count += 1
                result.samples_with_questions += 1

                if row.get("answer"):
                    result.samples_with_answers += 1

                # Track answer type
                atype = row.get("answer_type", "unknown")
                answer_types[atype] += 1

        result.categories = dict(answer_types)
        result.warnings.append(f"Text-only: {text_only_count}/{min(max_samples, len(ds))} samples (filtered out image questions)")
        result.status = "success" if result.samples_with_answers > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_healthbench(max_samples: int = 100) -> BenchmarkVerification:
    """Verify HealthBench Hard - Medical questions with rubrics."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="HealthBench Hard",
        status="pending",
        grading_type="rubric",
        answer_format="Open-ended (needs LLM grader)",
        gpt5_accuracy="~60%"
    )

    try:
        # Load specific hard subset file (default split has schema mismatch)
        ds = load_dataset(
            "openai/healthbench",
            data_files="hard_2025-05-08-21-00-10.jsonl",
            split="train",
            trust_remote_code=True
        )
        result.total_samples = len(ds)

        has_rubric = 0

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            if row.get("prompt"):
                result.samples_with_questions += 1

            rubrics = row.get("rubrics", [])
            if rubrics:
                result.samples_with_answers += 1
                has_rubric += 1

        result.warnings.append(f"Uses rubric-based grading: {has_rubric} samples have rubrics")
        result.status = "success" if result.samples_with_answers > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_tutorbench(max_samples: int = 100) -> BenchmarkVerification:
    """Verify TutorBench - Educational tutoring scenarios."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="TutorBench",
        status="pending",
        grading_type="rubric",
        answer_format="Open-ended (needs LLM grader)",
        gpt5_accuracy="~55%"
    )

    try:
        # Correct dataset ID from uq_eval/benchmarks/tutorbench.py
        ds = load_dataset("ScaleAI/TutorBench", split="train", trust_remote_code=True)
        result.total_samples = len(ds)

        subjects = defaultdict(int)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            # TutorBench uses PROMPT field for questions
            if row.get("PROMPT") or row.get("FOLLOW_UP_PROMPT"):
                result.samples_with_questions += 1

            # TutorBench uses RUBRICS field for grading criteria
            if row.get("RUBRICS"):
                result.samples_with_answers += 1

            # Track by SUBJECT field
            subject = row.get("SUBJECT", "unknown")
            subjects[subject] += 1

        result.categories = dict(subjects)
        result.status = "success" if result.samples_with_answers > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_multichallenge(max_samples: int = 100) -> BenchmarkVerification:
    """Verify MultiChallenge - Multi-turn reasoning."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="MultiChallenge",
        status="pending",
        grading_type="mcq",
        answer_format="A/B/C/D letter",
        gpt5_accuracy="58-64%"
    )

    try:
        ds = load_dataset("HendrixLab/MultiChallenge", split="test", trust_remote_code=True)
        result.total_samples = len(ds)

        answers = defaultdict(int)
        categories = defaultdict(int)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            if row.get("question") or row.get("context"):
                result.samples_with_questions += 1

            answer = row.get("answer", "")
            if answer:
                result.samples_with_answers += 1
                answers[str(answer).upper()] += 1

            cat = row.get("category", "unknown")
            categories[cat] += 1

        result.class_balance = dict(answers)
        result.categories = dict(categories)
        result.status = "success" if result.samples_with_answers > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_prbench(max_samples: int = 100) -> BenchmarkVerification:
    """Verify PRBench - Professional reasoning (legal/finance)."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="PRBench",
        status="pending",
        grading_type="rubric",
        answer_format="Open-ended (needs LLM grader)",
        gpt5_accuracy="~50-51%"
    )

    try:
        # Correct dataset ID from uq_eval/benchmarks/prbench.py
        ds = load_dataset("ScaleAI/PRBench", split="finance", trust_remote_code=True)
        result.total_samples = len(ds)

        domains = defaultdict(int)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            if row.get("prompt") or row.get("question") or row.get("input"):
                result.samples_with_questions += 1

            if row.get("rubric") or row.get("answer") or row.get("rubrics"):
                result.samples_with_answers += 1

            domain = row.get("domain", "finance")
            domains[domain] += 1

        result.categories = dict(domains)
        result.status = "success" if result.samples_with_answers > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_math(max_samples: int = 100) -> BenchmarkVerification:
    """Verify MATH benchmark - Competition math problems."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="MATH",
        status="pending",
        grading_type="numeric",
        answer_format="Boxed numeric/algebraic answer",
        gpt5_accuracy="~96%"
    )

    try:
        # Use EleutherAI mirror (hendrycks/competition_math is deprecated)
        ds = load_dataset("EleutherAI/hendrycks_math", "algebra", split="test", trust_remote_code=True)
        result.total_samples = len(ds)

        levels = defaultdict(int)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            if row.get("problem"):
                result.samples_with_questions += 1

            if row.get("solution"):
                result.samples_with_answers += 1

            level = row.get("level", "unknown")
            levels[level] += 1

        result.categories = dict(levels)
        result.warnings.append("Using algebra subset (7 subjects available)")
        result.status = "success" if result.samples_with_answers > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_gsm8k(max_samples: int = 100) -> BenchmarkVerification:
    """Verify GSM8K - Grade school math."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="GSM8K",
        status="pending",
        grading_type="numeric",
        answer_format="Numeric answer (extracted from solution)",
        gpt5_accuracy="~95%"
    )

    try:
        ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
        result.total_samples = len(ds)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            if row.get("question"):
                result.samples_with_questions += 1

            if row.get("answer"):
                result.samples_with_answers += 1

        result.status = "success" if result.samples_with_answers > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_mmlu_pro(max_samples: int = 100) -> BenchmarkVerification:
    """Verify MMLU-Pro - Harder MMLU variant."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="MMLU-Pro",
        status="pending",
        grading_type="mcq",
        answer_format="A-J letter (10 options)",
        gpt5_accuracy="~87%"
    )

    try:
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", trust_remote_code=True)
        result.total_samples = len(ds)

        subjects = defaultdict(int)
        answers = defaultdict(int)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            if row.get("question"):
                result.samples_with_questions += 1

            answer = row.get("answer", "")
            if answer:
                result.samples_with_answers += 1
                answers[str(answer).upper()] += 1

            subject = row.get("category", "unknown")
            subjects[subject] += 1

        result.class_balance = dict(answers)
        result.categories = dict(subjects)
        result.status = "success" if result.samples_with_answers > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_arc(max_samples: int = 100) -> BenchmarkVerification:
    """Verify ARC-Challenge - Science reasoning."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="ARC-Challenge",
        status="pending",
        grading_type="mcq",
        answer_format="A/B/C/D letter",
        gpt5_accuracy="~95%"
    )

    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", trust_remote_code=True)
        result.total_samples = len(ds)

        answers = defaultdict(int)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            if row.get("question"):
                result.samples_with_questions += 1

            answer = row.get("answerKey", "")
            if answer:
                result.samples_with_answers += 1
                answers[answer] += 1

        result.class_balance = dict(answers)
        result.status = "success" if result.samples_with_answers > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_drop(max_samples: int = 100) -> BenchmarkVerification:
    """Verify DROP - Reading comprehension with discrete reasoning."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="DROP",
        status="pending",
        grading_type="open_ended",
        answer_format="Short answer (number, span, or date)",
        gpt5_accuracy="~90%"
    )

    try:
        ds = load_dataset("ucinlp/drop", split="validation", trust_remote_code=True)
        result.total_samples = len(ds)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            if row.get("question"):
                result.samples_with_questions += 1

            answers = row.get("answers_spans", {})
            if answers:
                result.samples_with_answers += 1

        result.status = "success" if result.samples_with_answers > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_triviaqa(max_samples: int = 100) -> BenchmarkVerification:
    """Verify TriviaQA - Trivia questions."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="TriviaQA",
        status="pending",
        grading_type="open_ended",
        answer_format="Short factual answer",
        gpt5_accuracy="~90%"
    )

    try:
        ds = load_dataset("mandarjoshi/trivia_qa", "rc", split="validation", trust_remote_code=True)
        result.total_samples = len(ds)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            if row.get("question"):
                result.samples_with_questions += 1

            answer = row.get("answer", {})
            if answer and answer.get("value"):
                result.samples_with_answers += 1

        result.status = "success" if result.samples_with_answers > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_hellaswag(max_samples: int = 100) -> BenchmarkVerification:
    """Verify HellaSwag - Commonsense NLI."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="HellaSwag",
        status="pending",
        grading_type="mcq",
        answer_format="0/1/2/3 index",
        gpt5_accuracy="~95%"
    )

    try:
        ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
        result.total_samples = len(ds)

        answers = defaultdict(int)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            if row.get("ctx"):
                result.samples_with_questions += 1

            label = row.get("label", "")
            if label != "":
                result.samples_with_answers += 1
                answers[str(label)] += 1

        result.class_balance = dict(answers)
        result.status = "success" if result.samples_with_answers > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_winogrande(max_samples: int = 100) -> BenchmarkVerification:
    """Verify WinoGrande - Commonsense reasoning."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="WinoGrande",
        status="pending",
        grading_type="binary",
        answer_format="1 or 2 (option index)",
        gpt5_accuracy="~95%"
    )

    try:
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
        result.total_samples = len(ds)

        answers = defaultdict(int)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            if row.get("sentence"):
                result.samples_with_questions += 1

            answer = row.get("answer", "")
            if answer:
                result.samples_with_answers += 1
                answers[answer] += 1

        result.class_balance = dict(answers)
        result.status = "success" if result.samples_with_answers > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_livebench(max_samples: int = 100) -> BenchmarkVerification:
    """Verify LiveBench - Continuously updated benchmark."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="LiveBench",
        status="pending",
        grading_type="mixed",
        answer_format="Category-dependent",
        gpt5_accuracy="~79%"
    )

    try:
        # LiveBench has per-category datasets
        ds = load_dataset("livebench/math", split="test", trust_remote_code=True)
        result.total_samples = len(ds)

        categories = defaultdict(int)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            if row.get("turns") or row.get("question"):
                result.samples_with_questions += 1

            if row.get("ground_truth") or row.get("answer"):
                result.samples_with_answers += 1

            cat = row.get("category", "unknown")
            categories[cat] += 1

        result.categories = dict(categories)
        result.status = "success" if result.samples_with_answers > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_omnimath(max_samples: int = 100) -> BenchmarkVerification:
    """Verify OmniMath - Competition math."""
    from huggingface_hub import hf_hub_download
    import json

    result = BenchmarkVerification(
        name="OmniMath",
        status="pending",
        grading_type="open_ended",
        answer_format="Math proof/solution",
        gpt5_accuracy="~72%"
    )

    try:
        # Direct jsonl download (load_dataset has 'List' feature type issue)
        path = hf_hub_download("KbsdJames/Omni-MATH", "test.jsonl", repo_type="dataset")

        difficulties = defaultdict(int)
        domains = defaultdict(int)
        total = 0

        with open(path) as f:
            for i, line in enumerate(f):
                total += 1
                if i >= max_samples:
                    continue

                row = json.loads(line)

                if row.get("problem"):
                    result.samples_with_questions += 1

                if row.get("solution") or row.get("answer"):
                    result.samples_with_answers += 1

                diff = row.get("difficulty", "unknown")
                difficulties[str(int(diff)) if isinstance(diff, (int, float)) else str(diff)] += 1

                # Domain is a list, take first element
                domain_list = row.get("domain", ["unknown"])
                domain = domain_list[0] if isinstance(domain_list, list) and domain_list else str(domain_list)
                # Simplify domain to top category
                domain_simple = domain.split(" -> ")[0] if " -> " in domain else domain
                domains[domain_simple] += 1

        result.total_samples = total
        result.categories = dict(difficulties)
        result.warnings.append(f"Top-level domains: {dict(domains)}")
        result.status = "success" if result.samples_with_answers > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_oolong(max_samples: int = 100) -> BenchmarkVerification:
    """Verify Oolong - Long context benchmark."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="Oolong",
        status="pending",
        grading_type="mixed",
        answer_format="Task-dependent",
        gpt5_accuracy="47-70%"
    )

    try:
        # Correct dataset ID from uq_eval/benchmarks/oolong.py
        ds = load_dataset("oolongbench/oolong-real", "dnd", split="test", trust_remote_code=True)
        result.total_samples = len(ds)

        context_lengths = defaultdict(int)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            if row.get("question") or row.get("prompt"):
                result.samples_with_questions += 1

            if row.get("answer") or row.get("ground_truth"):
                result.samples_with_answers += 1

            # Track context length
            context = row.get("context", "")
            length_bucket = len(context) // 10000 * 10  # 10k buckets
            context_lengths[f"{length_bucket}k-{length_bucket+10}k"] += 1

        result.categories = dict(context_lengths)
        result.status = "success" if result.samples_with_answers > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_arc_agi(max_samples: int = 100) -> BenchmarkVerification:
    """Verify ARC-AGI - Abstract reasoning corpus."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="ARC-AGI",
        status="pending",
        grading_type="grid",
        answer_format="2D grid output",
        gpt5_accuracy="~10%"
    )

    try:
        # Try multiple dataset sources
        try:
            ds = load_dataset("dataartist/arc-agi", split="test", trust_remote_code=True)
        except:
            ds = load_dataset("lordspline/arc-agi", split="evaluation", trust_remote_code=True)

        result.total_samples = len(ds)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            if row.get("input") or row.get("train") or row.get("task"):
                result.samples_with_questions += 1

            if row.get("output") or row.get("test"):
                result.samples_with_answers += 1

        result.warnings.append("Very hard benchmark - models typically <10% accuracy")
        result.status = "success" if result.samples_with_answers > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_chembench(max_samples: int = 100) -> BenchmarkVerification:
    """Verify ChemBench - Chemistry questions."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="ChemBench",
        status="pending",
        grading_type="mcq",
        answer_format="Letter answer",
        gpt5_accuracy="8-70%"
    )

    try:
        # ChemBench uses jablonkagroup repo with config
        ds = load_dataset("jablonkagroup/ChemBench", "general_chemistry", split="train", trust_remote_code=True)
        result.total_samples = len(ds)

        subfields = defaultdict(int)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            # ChemBench uses 'examples' field containing question/answer pairs
            examples = row.get("examples", [])
            if examples or row.get("description"):
                result.samples_with_questions += 1

            # Each example has input/target pairs
            if examples:
                result.samples_with_answers += 1

            # Track by subfield
            subfield = row.get("subfield", "general_chemistry")
            subfields[subfield] += 1

        result.categories = dict(subfields)
        result.warnings.append("Multiple configs available: general_chemistry, organic_chemistry, etc.")
        result.status = "success" if result.samples_with_answers > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_bigcodebench(max_samples: int = 100) -> BenchmarkVerification:
    """Verify BigCodeBench - Code generation."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="BigCodeBench",
        status="pending",
        grading_type="execution",
        answer_format="Python code (needs execution)",
        gpt5_accuracy="~56%"
    )

    try:
        ds = load_dataset("bigcode/bigcodebench", split="v0.1.2", trust_remote_code=True)
        result.total_samples = len(ds)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            if row.get("instruct_prompt") or row.get("complete_prompt"):
                result.samples_with_questions += 1

            if row.get("canonical_solution") or row.get("test"):
                result.samples_with_answers += 1

        result.warnings.append("Needs code execution for grading")
        result.status = "success" if result.samples_with_answers > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_livecodebench(max_samples: int = 100) -> BenchmarkVerification:
    """Verify LiveCodeBench - Continuously updated code benchmark."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="LiveCodeBench",
        status="pending",
        grading_type="execution",
        answer_format="Code (needs execution)",
        gpt5_accuracy="4-90%"
    )

    try:
        ds = load_dataset("livecodebench/code_generation_lite", split="test", trust_remote_code=True)
        result.total_samples = len(ds)

        difficulties = defaultdict(int)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            if row.get("question_content") or row.get("problem"):
                result.samples_with_questions += 1

            if row.get("public_test_cases") or row.get("test"):
                result.samples_with_answers += 1

            diff = row.get("difficulty", "unknown")
            difficulties[str(diff)] += 1

        result.categories = dict(difficulties)
        result.warnings.append("Needs code execution for grading")
        result.status = "success" if result.samples_with_answers > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_swebench(max_samples: int = 100) -> BenchmarkVerification:
    """Verify SWE-Bench Lite - Software engineering."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="SWE-Bench Lite",
        status="pending",
        grading_type="execution",
        answer_format="Git patch (needs execution)",
        gpt5_accuracy="52-75%"
    )

    try:
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test", trust_remote_code=True)
        result.total_samples = len(ds)

        repos = defaultdict(int)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            if row.get("problem_statement"):
                result.samples_with_questions += 1

            if row.get("patch") or row.get("test_patch"):
                result.samples_with_answers += 1

            repo = row.get("repo", "unknown")
            repos[repo] += 1

        result.categories = dict(repos)
        result.warnings.append("Needs code execution and git for grading")
        result.status = "success" if result.samples_with_answers > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_babilong(max_samples: int = 100) -> BenchmarkVerification:
    """Verify BABILong - Long context reasoning."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="BABILong",
        status="pending",
        grading_type="open_ended",
        answer_format="Short answer",
        gpt5_accuracy="varies by length"
    )

    try:
        ds = load_dataset("RMT-team/babilong", "qa1", split="0k", trust_remote_code=True)
        result.total_samples = len(ds)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            if row.get("input") or row.get("question"):
                result.samples_with_questions += 1

            if row.get("target") or row.get("answer"):
                result.samples_with_answers += 1

        result.warnings.append("Multiple context lengths available: 0k, 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k")
        result.status = "success" if result.samples_with_answers > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


# Organized by priority tier
TIER1_BENCHMARKS = {
    "bbeh": verify_bbeh,
    "simpleqa": verify_simpleqa,
    "tutorbench": verify_tutorbench,
    "multichallenge": verify_multichallenge,
    "healthbench": verify_healthbench,
    "bigcodebench": verify_bigcodebench,
    "prbench": verify_prbench,
}

TIER2_BENCHMARKS = {
    "hle": verify_hle,
    "arc_agi": verify_arc_agi,
    "chembench": verify_chembench,
}

TIER3_BENCHMARKS = {
    "gpqa": verify_gpqa,
    "omnimath": verify_omnimath,
    "livebench": verify_livebench,
}

TIER4_BENCHMARKS = {
    "livecodebench": verify_livecodebench,
    "swebench": verify_swebench,
}

TIER5_BENCHMARKS = {
    "oolong": verify_oolong,
    "babilong": verify_babilong,
}

TIER6_BENCHMARKS = {
    "math": verify_math,
    "mmlu_pro": verify_mmlu_pro,
    "gsm8k": verify_gsm8k,
    "arc": verify_arc,
    "drop": verify_drop,
    "triviaqa": verify_triviaqa,
    "hellaswag": verify_hellaswag,
    "winogrande": verify_winogrande,
}

ALL_BENCHMARKS = {
    **TIER1_BENCHMARKS,
    **TIER2_BENCHMARKS,
    **TIER3_BENCHMARKS,
    **TIER4_BENCHMARKS,
    **TIER5_BENCHMARKS,
    **TIER6_BENCHMARKS,
}


def run_all_verifications(max_samples: int = 50) -> dict:
    """Run verification on all benchmarks."""
    results = {}

    print("=" * 80)
    print("TEXT BENCHMARK VERIFICATION REPORT")
    print("=" * 80)
    print()

    tiers = [
        ("TIER 1: Ideal Accuracy (40-60%)", TIER1_BENCHMARKS),
        ("TIER 2: Lower Accuracy (<40%)", TIER2_BENCHMARKS),
        ("TIER 3: Higher Accuracy (>70%)", TIER3_BENCHMARKS),
        ("TIER 4: Coding Benchmarks", TIER4_BENCHMARKS),
        ("TIER 5: Long Context", TIER5_BENCHMARKS),
        ("TIER 6: Standard Benchmarks (High Accuracy)", TIER6_BENCHMARKS),
    ]

    for tier_name, benchmarks in tiers:
        print(f"\n{'='*60}")
        print(tier_name)
        print("=" * 60)

        for name, verify_fn in benchmarks.items():
            print(f"\nVerifying {name}...", end=" ", flush=True)
            try:
                result = verify_fn(max_samples)
                results[name] = asdict(result)

                status_symbol = "✓" if result.status == "success" else "✗"
                print(f"{status_symbol} {result.status}")

                if result.status == "success":
                    print(f"  Samples: {result.total_samples} total")
                    print(f"  Questions: {result.samples_with_questions}/{max_samples} checked")
                    print(f"  Answers: {result.samples_with_answers}/{max_samples} checked")
                    print(f"  Grading: {result.grading_type} ({result.answer_format})")
                    print(f"  GPT-5 Acc: {result.gpt5_accuracy}")

                    if result.class_balance:
                        print(f"  Class balance: {result.class_balance}")
                    if result.categories and len(result.categories) <= 10:
                        print(f"  Categories: {result.categories}")
                    elif result.categories:
                        print(f"  Categories: {len(result.categories)} types")

                    for warning in result.warnings:
                        print(f"  ⚠️  {warning}")
                else:
                    print(f"  Error: {result.error_message}")

            except Exception as e:
                print(f"✗ ERROR: {e}")
                results[name] = {"name": name, "status": "error", "error_message": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(description="Verify text benchmarks")
    parser.add_argument("--benchmark", "-b", help="Specific benchmark to verify")
    parser.add_argument("--max-samples", "-n", type=int, default=50, help="Max samples to check per benchmark")
    parser.add_argument("--output", "-o", help="Output JSON file")

    args = parser.parse_args()

    if args.benchmark:
        if args.benchmark not in ALL_BENCHMARKS:
            print(f"Unknown benchmark: {args.benchmark}")
            print(f"Available: {list(ALL_BENCHMARKS.keys())}")
            return

        result = ALL_BENCHMARKS[args.benchmark](args.max_samples)
        results = {args.benchmark: asdict(result)}
        print(json.dumps(results, indent=2))
    else:
        results = run_all_verifications(args.max_samples)

    # Save results
    output_path = args.output or "data/TEXT_BENCHMARK_STATUS.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    success = sum(1 for r in results.values() if r.get("status") == "success")
    total = len(results)
    print(f"Verified: {success}/{total} benchmarks")

    # Group by grading type
    grading_types = defaultdict(list)
    for name, r in results.items():
        if r.get("status") == "success":
            grading_types[r.get("grading_type", "unknown")].append(name)

    print("\nBy grading type:")
    for gtype, benchmarks in sorted(grading_types.items()):
        print(f"  {gtype}: {', '.join(benchmarks)}")

    # List failures
    failures = [name for name, r in results.items() if r.get("status") != "success"]
    if failures:
        print(f"\nFailed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
