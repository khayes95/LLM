#!/usr/bin/env python3
"""Comprehensive benchmark verification script.

Tests all VLM benchmarks to ensure:
1. Data loads correctly
2. Images are real (not gray placeholders)
3. Grading format is correct
4. Category distribution is good
"""

import sys
import json
import traceback
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Iterator
from collections import defaultdict

import numpy as np
from PIL import Image

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class BenchmarkVerification:
    """Results of verifying a benchmark."""
    name: str
    status: str  # "success", "error", "partial"
    error_message: Optional[str] = None

    # Data loading
    total_samples: int = 0
    samples_with_images: int = 0
    samples_with_valid_images: int = 0
    samples_with_answers: int = 0

    # Image quality
    gray_images: int = 0
    real_images: int = 0
    image_load_failures: int = 0
    avg_image_std: float = 0.0

    # Grading format
    grading_type: str = ""  # "mcq", "binary", "open_ended", "rubric"
    answer_format: str = ""  # Description of answer format
    sample_answers: list = field(default_factory=list)

    # Categories
    categories: dict = field(default_factory=dict)
    class_balance: dict = field(default_factory=dict)

    # Warnings
    warnings: list = field(default_factory=list)


def is_gray_image(img: Image.Image, threshold: float = 5.0) -> bool:
    """Check if image is a gray placeholder (low variance)."""
    if img is None:
        return True
    try:
        arr = np.array(img.convert("RGB"))
        return arr.std() < threshold
    except:
        return True


def verify_vsr(max_samples: int = 100) -> BenchmarkVerification:
    """Verify VSR benchmark - Visual Spatial Reasoning."""
    from datasets import load_dataset
    import requests
    import io

    result = BenchmarkVerification(
        name="VSR",
        status="pending",
        grading_type="binary",
        answer_format="true/false (label: 1=True, 0=False)"
    )

    try:
        ds = load_dataset("cambridgeltl/vsr_random", split="test")
        result.total_samples = len(ds)

        image_stds = []
        relations = defaultdict(int)
        labels = defaultdict(int)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            # Check answer
            label = row.get("label")
            if label is not None:
                result.samples_with_answers += 1
                labels[str(label)] += 1

            # Track relation categories
            rel = row.get("relation", "unknown")
            relations[rel] += 1

            # Check image - VSR stores as filename/URL
            img = row.get("image")
            if img is not None:
                result.samples_with_images += 1

                # If it's a string (filename), need to download from image_link
                if isinstance(img, str):
                    img_url = row.get("image_link", "")
                    if img_url:
                        try:
                            resp = requests.get(img_url, timeout=5)
                            resp.raise_for_status()
                            pil_img = Image.open(io.BytesIO(resp.content))

                            if is_gray_image(pil_img):
                                result.gray_images += 1
                            else:
                                result.real_images += 1
                                result.samples_with_valid_images += 1
                                arr = np.array(pil_img.convert("RGB"))
                                image_stds.append(arr.std())
                        except Exception as e:
                            result.image_load_failures += 1
                    else:
                        result.image_load_failures += 1
                        result.warnings.append(f"Sample {i}: image is string but no image_link")

                elif isinstance(img, Image.Image):
                    if is_gray_image(img):
                        result.gray_images += 1
                    else:
                        result.real_images += 1
                        result.samples_with_valid_images += 1
                        arr = np.array(img.convert("RGB"))
                        image_stds.append(arr.std())

        result.avg_image_std = float(np.mean(image_stds)) if image_stds else 0.0
        result.categories = dict(list(relations.items())[:10])  # Top 10 relations
        result.class_balance = dict(labels)
        result.sample_answers = ["true", "false"]

        # Determine status
        if result.image_load_failures > max_samples * 0.5:
            result.status = "error"
            result.error_message = f"Too many image failures: {result.image_load_failures}/{max_samples}"
        elif result.gray_images > result.real_images:
            result.status = "partial"
            result.warnings.append("More gray images than real - need to download from image_link")
        else:
            result.status = "success"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_hallusionbench(max_samples: int = 100) -> BenchmarkVerification:
    """Verify HallusionBench - Hallucination Detection."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="HallusionBench",
        status="pending",
        grading_type="binary",
        answer_format="yes/no (gt_answer: '0'=No, '1'=Yes)"
    )

    try:
        ds = load_dataset("lmms-lab/HallusionBench", split="image", trust_remote_code=True)
        result.total_samples = len(ds)

        image_stds = []
        categories = defaultdict(int)
        answers = defaultdict(int)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            # Check answer format
            gt = row.get("gt_answer", "")
            if gt:
                result.samples_with_answers += 1
                answers[str(gt)] += 1

            # Track categories
            cat = row.get("category", "unknown")
            subcat = row.get("subcategory", "")
            categories[f"{cat}/{subcat}"] += 1

            # Check image
            img = row.get("image")
            if img is not None:
                result.samples_with_images += 1

                if isinstance(img, Image.Image):
                    if is_gray_image(img):
                        result.gray_images += 1
                    else:
                        result.real_images += 1
                        result.samples_with_valid_images += 1
                        arr = np.array(img.convert("RGB"))
                        image_stds.append(arr.std())

        result.avg_image_std = float(np.mean(image_stds)) if image_stds else 0.0
        result.categories = dict(list(categories.items())[:10])
        result.class_balance = dict(answers)
        result.sample_answers = list(answers.keys())[:5]

        # Check grading format warning
        if "0" in answers or "1" in answers:
            result.warnings.append("Ground truth uses '0'/'1' format, not 'yes'/'no' - ensure grader handles this")

        result.status = "success" if result.samples_with_valid_images > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_mmmu(max_samples: int = 100) -> BenchmarkVerification:
    """Verify MMMU - College-level Multimodal."""
    from datasets import load_dataset
    import ast

    result = BenchmarkVerification(
        name="MMMU",
        status="pending",
        grading_type="mcq",
        answer_format="Letter (A/B/C/D) - options stored as string repr of list"
    )

    subjects = ["Art", "Biology", "Chemistry", "Computer_Science", "Math", "Physics"]

    try:
        image_stds = []
        subject_counts = defaultdict(int)
        answers = defaultdict(int)
        samples_per_subject = max_samples // len(subjects)

        for subject in subjects:
            try:
                ds = load_dataset("MMMU/MMMU", subject, split="validation", trust_remote_code=True)
            except Exception as e:
                result.warnings.append(f"Could not load {subject}: {e}")
                continue

            for i, row in enumerate(ds):
                if i >= samples_per_subject:
                    break

                result.total_samples += 1
                subject_counts[subject] += 1

                # Check answer
                answer = row.get("answer", "")
                if answer:
                    result.samples_with_answers += 1
                    answers[answer.upper()] += 1

                # Check options format
                options_str = row.get("options", "[]")
                if isinstance(options_str, str):
                    try:
                        options = ast.literal_eval(options_str)
                        if i == 0 and subject == "Art":
                            result.warnings.append(f"Options are stored as string: '{options_str[:50]}...'")
                    except:
                        pass

                # Check images
                for img_key in ["image_1", "image_2", "image_3"]:
                    img = row.get(img_key)
                    if img is not None and isinstance(img, Image.Image):
                        result.samples_with_images += 1
                        if is_gray_image(img):
                            result.gray_images += 1
                        else:
                            result.real_images += 1
                            result.samples_with_valid_images += 1
                            arr = np.array(img.convert("RGB"))
                            image_stds.append(arr.std())
                        break  # Only count first image per sample

        result.avg_image_std = float(np.mean(image_stds)) if image_stds else 0.0
        result.categories = dict(subject_counts)
        result.class_balance = dict(answers)
        result.sample_answers = list(answers.keys())[:5]

        result.status = "success" if result.samples_with_valid_images > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_charxiv(max_samples: int = 100) -> BenchmarkVerification:
    """Verify CharXiv - Scientific Figure Reasoning."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="CharXiv",
        status="pending",
        grading_type="open_ended",
        answer_format="Free-form text (reasoning_a field)"
    )

    try:
        ds = load_dataset("princeton-nlp/CharXiv", split="validation", trust_remote_code=True)
        result.total_samples = len(ds)

        image_stds = []
        answer_types = defaultdict(int)
        answer_lengths = []

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            # Check answer
            answer = row.get("reasoning_a", "")
            if answer:
                result.samples_with_answers += 1
                answer_lengths.append(len(answer))

                # Track answer types
                a_type = row.get("reasoning_a_type", "unknown")
                answer_types[a_type] += 1

            # Check image
            img = row.get("image")
            if img is not None:
                result.samples_with_images += 1

                if isinstance(img, Image.Image):
                    if is_gray_image(img):
                        result.gray_images += 1
                    else:
                        result.real_images += 1
                        result.samples_with_valid_images += 1
                        arr = np.array(img.convert("RGB"))
                        image_stds.append(arr.std())

        result.avg_image_std = float(np.mean(image_stds)) if image_stds else 0.0
        result.categories = dict(answer_types)
        result.sample_answers = [f"Avg length: {np.mean(answer_lengths):.0f} chars" if answer_lengths else "N/A"]

        result.status = "success" if result.samples_with_valid_images > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_mmstar(max_samples: int = 100) -> BenchmarkVerification:
    """Verify MMStar - Vision-Indispensable Benchmark."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="MMStar",
        status="pending",
        grading_type="mcq",
        answer_format="Letter (A/B/C/D)"
    )

    try:
        ds = load_dataset("Lin-Chen/MMStar", split="val", trust_remote_code=True)
        result.total_samples = len(ds)

        image_stds = []
        categories = defaultdict(int)
        answers = defaultdict(int)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            # Check answer
            answer = row.get("answer", "")
            if answer:
                result.samples_with_answers += 1
                answers[answer.upper()] += 1

            # Track categories
            cat = row.get("category", "unknown")
            l2_cat = row.get("l2_category", "")
            categories[f"{cat}/{l2_cat}"] += 1

            # Check image
            img = row.get("image")
            if img is not None:
                result.samples_with_images += 1

                if isinstance(img, Image.Image):
                    if is_gray_image(img):
                        result.gray_images += 1
                    else:
                        result.real_images += 1
                        result.samples_with_valid_images += 1
                        arr = np.array(img.convert("RGB"))
                        image_stds.append(arr.std())

        result.avg_image_std = float(np.mean(image_stds)) if image_stds else 0.0
        result.categories = dict(list(categories.items())[:10])
        result.class_balance = dict(answers)
        result.sample_answers = list(answers.keys())

        result.status = "success" if result.samples_with_valid_images > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_realworldqa(max_samples: int = 100) -> BenchmarkVerification:
    """Verify RealWorldQA - Real-world Spatial Understanding."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="RealWorldQA",
        status="pending",
        grading_type="mcq",
        answer_format="Letter (A/B/C) or number"
    )

    try:
        ds = load_dataset("lmms-lab/RealWorldQA", split="test", trust_remote_code=True)
        result.total_samples = len(ds)

        image_stds = []
        answers = defaultdict(int)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            # Check answer
            answer = row.get("answer", "")
            if answer:
                result.samples_with_answers += 1
                answers[str(answer)] += 1

            # Check image
            img = row.get("image")
            if img is not None:
                result.samples_with_images += 1

                if isinstance(img, Image.Image):
                    if is_gray_image(img):
                        result.gray_images += 1
                    else:
                        result.real_images += 1
                        result.samples_with_valid_images += 1
                        arr = np.array(img.convert("RGB"))
                        image_stds.append(arr.std())

        result.avg_image_std = float(np.mean(image_stds)) if image_stds else 0.0
        result.class_balance = dict(list(answers.items())[:10])
        result.sample_answers = list(answers.keys())[:5]

        result.status = "success" if result.samples_with_valid_images > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_mathvision(max_samples: int = 100) -> BenchmarkVerification:
    """Verify MathVision - Competition Math with Images."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="MathVision",
        status="pending",
        grading_type="mixed",
        answer_format="MCQ (A-E) or numeric"
    )

    try:
        ds = load_dataset("MathLLMs/MathVision", split="test", trust_remote_code=True)
        result.total_samples = len(ds)

        image_stds = []
        subjects = defaultdict(int)
        levels = defaultdict(int)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            # Check answer
            answer = row.get("answer", "")
            if answer:
                result.samples_with_answers += 1

            # Track categories
            subjects[row.get("subject", "unknown")] += 1
            levels[str(row.get("level", "?"))] += 1

            # Check image - uses 'decoded_image' or 'image'
            img = row.get("decoded_image") or row.get("image")
            if img is not None:
                result.samples_with_images += 1

                if isinstance(img, Image.Image):
                    if is_gray_image(img):
                        result.gray_images += 1
                    else:
                        result.real_images += 1
                        result.samples_with_valid_images += 1
                        arr = np.array(img.convert("RGB"))
                        image_stds.append(arr.std())

        result.avg_image_std = float(np.mean(image_stds)) if image_stds else 0.0
        result.categories = dict(list(subjects.items())[:10])
        result.class_balance = dict(levels)
        result.sample_answers = ["numeric", "A-E MCQ"]

        result.status = "success" if result.samples_with_valid_images > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_mathverse(max_samples: int = 100) -> BenchmarkVerification:
    """Verify MathVerse - Visual Math with 6 versions per problem."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="MathVerse",
        status="pending",
        grading_type="mixed",
        answer_format="MCQ or numeric"
    )

    try:
        # MathVerse needs config
        ds = load_dataset("AI4Math/MathVerse", "testmini", split="testmini", trust_remote_code=True)
        result.total_samples = len(ds)

        image_stds = []
        problem_versions = defaultdict(int)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            # Check answer
            answer = row.get("answer", "")
            if answer:
                result.samples_with_answers += 1

            # Track problem versions
            pv = row.get("problem_version", "unknown")
            problem_versions[pv] += 1

            # Check image
            img = row.get("image") or row.get("decoded_image")
            if img is not None:
                result.samples_with_images += 1

                if isinstance(img, Image.Image):
                    if is_gray_image(img):
                        result.gray_images += 1
                    else:
                        result.real_images += 1
                        result.samples_with_valid_images += 1
                        arr = np.array(img.convert("RGB"))
                        image_stds.append(arr.std())

        result.avg_image_std = float(np.mean(image_stds)) if image_stds else 0.0
        result.categories = dict(problem_versions)

        result.status = "success" if result.samples_with_valid_images > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_ai2d(max_samples: int = 100) -> BenchmarkVerification:
    """Verify AI2D - Science Diagrams."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="AI2D",
        status="pending",
        grading_type="mcq",
        answer_format="Index (0-3)"
    )

    try:
        ds = load_dataset("lmms-lab/ai2d", split="test", trust_remote_code=True)
        result.total_samples = len(ds)

        image_stds = []
        answers = defaultdict(int)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            # Check answer
            answer = row.get("answer", "")
            if answer is not None:
                result.samples_with_answers += 1
                answers[str(answer)] += 1

            # Check image
            img = row.get("image")
            if img is not None:
                result.samples_with_images += 1

                if isinstance(img, Image.Image):
                    if is_gray_image(img):
                        result.gray_images += 1
                    else:
                        result.real_images += 1
                        result.samples_with_valid_images += 1
                        arr = np.array(img.convert("RGB"))
                        image_stds.append(arr.std())

        result.avg_image_std = float(np.mean(image_stds)) if image_stds else 0.0
        result.class_balance = dict(answers)

        result.status = "success" if result.samples_with_valid_images > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_chartqa(max_samples: int = 100) -> BenchmarkVerification:
    """Verify ChartQA - Chart Question Answering."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="ChartQA",
        status="pending",
        grading_type="open_ended",
        answer_format="Free-form (often numeric)"
    )

    try:
        ds = load_dataset("HuggingFaceM4/ChartQA", split="test", trust_remote_code=True)
        result.total_samples = len(ds)

        image_stds = []
        answer_lengths = []

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            # Check answer
            answer = row.get("label") or row.get("answer", "")
            if answer:
                result.samples_with_answers += 1
                if isinstance(answer, list):
                    answer = answer[0] if answer else ""
                answer_lengths.append(len(str(answer)))

            # Check image
            img = row.get("image")
            if img is not None:
                result.samples_with_images += 1

                if isinstance(img, Image.Image):
                    if is_gray_image(img):
                        result.gray_images += 1
                    else:
                        result.real_images += 1
                        result.samples_with_valid_images += 1
                        arr = np.array(img.convert("RGB"))
                        image_stds.append(arr.std())

        result.avg_image_std = float(np.mean(image_stds)) if image_stds else 0.0
        result.sample_answers = [f"Avg answer length: {np.mean(answer_lengths):.0f} chars" if answer_lengths else "N/A"]

        result.status = "success" if result.samples_with_valid_images > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_scienceqa(max_samples: int = 100) -> BenchmarkVerification:
    """Verify ScienceQA - Multimodal Science QA."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="ScienceQA",
        status="pending",
        grading_type="mcq",
        answer_format="Index (0-4)"
    )

    try:
        ds = load_dataset("derek-thomas/ScienceQA", split="test", trust_remote_code=True)
        result.total_samples = len(ds)

        image_stds = []
        subjects = defaultdict(int)
        has_image_count = 0

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            # Check answer
            answer = row.get("answer", "")
            if answer is not None:
                result.samples_with_answers += 1

            # Track subject
            subj = row.get("subject", "unknown")
            subjects[subj] += 1

            # Check image
            img = row.get("image")
            if img is not None:
                has_image_count += 1
                result.samples_with_images += 1

                if isinstance(img, Image.Image):
                    if is_gray_image(img):
                        result.gray_images += 1
                    else:
                        result.real_images += 1
                        result.samples_with_valid_images += 1
                        arr = np.array(img.convert("RGB"))
                        image_stds.append(arr.std())

        result.avg_image_std = float(np.mean(image_stds)) if image_stds else 0.0
        result.categories = dict(subjects)
        result.warnings.append(f"Only {has_image_count}/{max_samples} samples have images - rest are text-only")

        result.status = "success" if result.samples_with_answers > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_nlvr2(max_samples: int = 100) -> BenchmarkVerification:
    """Verify NLVR2 - Natural Language Visual Reasoning."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="NLVR2",
        status="pending",
        grading_type="binary",
        answer_format="True/False"
    )

    try:
        # Use balanced_test_public split
        ds = load_dataset("lmms-lab/NLVR2", split="balanced_test_public", trust_remote_code=True)
        result.total_samples = len(ds)

        image_stds = []
        labels = defaultdict(int)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            # Check answer
            label = row.get("label", "")
            if label is not None:
                result.samples_with_answers += 1
                labels[str(label)] += 1

            # Check images (NLVR2 has 2 images per sample)
            for img_key in ["image1", "image2", "left_image", "right_image"]:
                img = row.get(img_key)
                if img is not None and isinstance(img, Image.Image):
                    result.samples_with_images += 1
                    if is_gray_image(img):
                        result.gray_images += 1
                    else:
                        result.real_images += 1
                        result.samples_with_valid_images += 1
                        arr = np.array(img.convert("RGB"))
                        image_stds.append(arr.std())
                    break

        result.avg_image_std = float(np.mean(image_stds)) if image_stds else 0.0
        result.class_balance = dict(labels)
        result.warnings.append("Multi-image benchmark - uses 2 images per sample")

        result.status = "success" if result.samples_with_valid_images > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_ocrbench(max_samples: int = 100) -> BenchmarkVerification:
    """Verify OCRBench - Text Recognition."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="OCRBench",
        status="pending",
        grading_type="open_ended",
        answer_format="Extracted text"
    )

    try:
        ds = load_dataset("echo840/OCRBench", split="test", trust_remote_code=True)
        result.total_samples = len(ds)

        image_stds = []

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            # Check answer
            answer = row.get("answer", "") or row.get("answers", "")
            if answer:
                result.samples_with_answers += 1

            # Check image
            img = row.get("image")
            if img is not None:
                result.samples_with_images += 1

                if isinstance(img, Image.Image):
                    if is_gray_image(img):
                        result.gray_images += 1
                    else:
                        result.real_images += 1
                        result.samples_with_valid_images += 1
                        arr = np.array(img.convert("RGB"))
                        image_stds.append(arr.std())

        result.avg_image_std = float(np.mean(image_stds)) if image_stds else 0.0

        result.status = "success" if result.samples_with_valid_images > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_aokvqa(max_samples: int = 100) -> BenchmarkVerification:
    """Verify A-OKVQA - Outside Knowledge VQA."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="A-OKVQA",
        status="pending",
        grading_type="mcq",
        answer_format="MCQ with rationales"
    )

    try:
        ds = load_dataset("HuggingFaceM4/A-OKVQA", split="test", trust_remote_code=True)
        result.total_samples = len(ds)

        image_stds = []

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            # Check answer
            answer = row.get("correct_choice_idx", "")
            if answer is not None:
                result.samples_with_answers += 1

            # Check image
            img = row.get("image")
            if img is not None:
                result.samples_with_images += 1

                if isinstance(img, Image.Image):
                    if is_gray_image(img):
                        result.gray_images += 1
                    else:
                        result.real_images += 1
                        result.samples_with_valid_images += 1
                        arr = np.array(img.convert("RGB"))
                        image_stds.append(arr.std())

        result.avg_image_std = float(np.mean(image_stds)) if image_stds else 0.0

        result.status = "success" if result.samples_with_valid_images > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_textvqa(max_samples: int = 100) -> BenchmarkVerification:
    """Verify TextVQA - Reading text in images."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="TextVQA",
        status="pending",
        grading_type="open_ended",
        answer_format="Free-form text"
    )

    try:
        ds = load_dataset("textvqa", split="validation", trust_remote_code=True)
        result.total_samples = len(ds)

        image_stds = []

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            # Check answer
            answers = row.get("answers", [])
            if answers:
                result.samples_with_answers += 1

            # Check image
            img = row.get("image")
            if img is not None:
                result.samples_with_images += 1

                if isinstance(img, Image.Image):
                    if is_gray_image(img):
                        result.gray_images += 1
                    else:
                        result.real_images += 1
                        result.samples_with_valid_images += 1
                        arr = np.array(img.convert("RGB"))
                        image_stds.append(arr.std())

        result.avg_image_std = float(np.mean(image_stds)) if image_stds else 0.0
        result.status = "success" if result.samples_with_valid_images > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_docvqa(max_samples: int = 100) -> BenchmarkVerification:
    """Verify DocVQA - Document Visual QA."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="DocVQA",
        status="pending",
        grading_type="open_ended",
        answer_format="Free-form text (document reading)"
    )

    try:
        # DocVQA needs config name
        ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="test", trust_remote_code=True)
        result.total_samples = len(ds)

        image_stds = []

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            # Check answer
            answers = row.get("answers", []) or row.get("answer", "")
            if answers:
                result.samples_with_answers += 1

            # Check image
            img = row.get("image")
            if img is not None:
                result.samples_with_images += 1

                if isinstance(img, Image.Image):
                    if is_gray_image(img):
                        result.gray_images += 1
                    else:
                        result.real_images += 1
                        result.samples_with_valid_images += 1
                        arr = np.array(img.convert("RGB"))
                        image_stds.append(arr.std())

        result.avg_image_std = float(np.mean(image_stds)) if image_stds else 0.0
        result.status = "success" if result.samples_with_valid_images > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_mathvista(max_samples: int = 100) -> BenchmarkVerification:
    """Verify MathVista - IQ tests, plots, figures."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="MathVista",
        status="pending",
        grading_type="mixed",
        answer_format="MCQ or numeric"
    )

    try:
        # MathVista uses default config
        ds = load_dataset("AI4Math/MathVista", split="testmini", trust_remote_code=True)
        result.total_samples = len(ds)

        image_stds = []
        question_types = defaultdict(int)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            # Check answer
            answer = row.get("answer", "")
            if answer:
                result.samples_with_answers += 1

            # Track question types
            qtype = row.get("question_type", "unknown")
            question_types[qtype] += 1

            # Check image - MathVista uses decoded_image field
            img = row.get("decoded_image")
            if img is None:
                img = row.get("image")

            if img is not None:
                result.samples_with_images += 1

                if isinstance(img, Image.Image):
                    if is_gray_image(img):
                        result.gray_images += 1
                    else:
                        result.real_images += 1
                        result.samples_with_valid_images += 1
                        arr = np.array(img.convert("RGB"))
                        image_stds.append(arr.std())

        result.avg_image_std = float(np.mean(image_stds)) if image_stds else 0.0
        result.categories = dict(question_types)
        result.status = "success" if result.samples_with_valid_images > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_infographicvqa(max_samples: int = 100) -> BenchmarkVerification:
    """Verify InfographicVQA - Infographic understanding."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="InfographicVQA",
        status="pending",
        grading_type="open_ended",
        answer_format="Free-form text"
    )

    try:
        # Use DocVQA's InfographicVQA config
        ds = load_dataset("lmms-lab/DocVQA", "InfographicVQA", split="test", trust_remote_code=True)
        result.total_samples = len(ds)

        image_stds = []

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            # Check answer
            answers = row.get("answers", [])
            if answers:
                result.samples_with_answers += 1

            # Check image
            img = row.get("image")
            if img is not None:
                result.samples_with_images += 1

                if isinstance(img, Image.Image):
                    if is_gray_image(img):
                        result.gray_images += 1
                    else:
                        result.real_images += 1
                        result.samples_with_valid_images += 1
                        arr = np.array(img.convert("RGB"))
                        image_stds.append(arr.std())

        result.avg_image_std = float(np.mean(image_stds)) if image_stds else 0.0
        result.status = "success" if result.samples_with_valid_images > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_gqa(max_samples: int = 100) -> BenchmarkVerification:
    """Verify GQA - Visual Reasoning."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="GQA",
        status="pending",
        grading_type="open_ended",
        answer_format="Short answer"
    )

    try:
        # GQA needs config name - use testdev_balanced_images
        ds = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev", trust_remote_code=True)
        result.total_samples = len(ds)

        image_stds = []

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            # Check answer
            answer = row.get("answer", "")
            if answer:
                result.samples_with_answers += 1

            # Check image
            img = row.get("image")
            if img is not None:
                result.samples_with_images += 1

                if isinstance(img, Image.Image):
                    if is_gray_image(img):
                        result.gray_images += 1
                    else:
                        result.real_images += 1
                        result.samples_with_valid_images += 1
                        arr = np.array(img.convert("RGB"))
                        image_stds.append(arr.std())

        result.avg_image_std = float(np.mean(image_stds)) if image_stds else 0.0
        result.status = "success" if result.samples_with_valid_images > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_pope(max_samples: int = 100) -> BenchmarkVerification:
    """Verify POPE - Object Hallucination Detection."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="POPE",
        status="pending",
        grading_type="binary",
        answer_format="yes/no"
    )

    try:
        ds = load_dataset("lmms-lab/POPE", split="test", trust_remote_code=True)
        result.total_samples = len(ds)

        image_stds = []
        labels = defaultdict(int)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            # Check answer
            answer = row.get("answer", "")
            if answer:
                result.samples_with_answers += 1
                labels[str(answer).lower()] += 1

            # Check image
            img = row.get("image")
            if img is not None:
                result.samples_with_images += 1

                if isinstance(img, Image.Image):
                    if is_gray_image(img):
                        result.gray_images += 1
                    else:
                        result.real_images += 1
                        result.samples_with_valid_images += 1
                        arr = np.array(img.convert("RGB"))
                        image_stds.append(arr.std())

        result.avg_image_std = float(np.mean(image_stds)) if image_stds else 0.0
        result.class_balance = dict(labels)
        result.status = "success" if result.samples_with_valid_images > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_vizwiz(max_samples: int = 100) -> BenchmarkVerification:
    """Verify VizWiz-VQA - Visual QA from blind users."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="VizWiz",
        status="pending",
        grading_type="open_ended",
        answer_format="Free-form (some unanswerable)"
    )

    try:
        ds = load_dataset("lmms-lab/VizWiz-VQA", split="test", trust_remote_code=True)
        result.total_samples = len(ds)

        image_stds = []
        answerable_count = 0

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            # Check answer
            answers = row.get("answers", [])
            if answers:
                result.samples_with_answers += 1
                # Check if answerable
                if "unanswerable" not in str(answers).lower():
                    answerable_count += 1

            # Check image
            img = row.get("image")
            if img is not None:
                result.samples_with_images += 1

                if isinstance(img, Image.Image):
                    if is_gray_image(img):
                        result.gray_images += 1
                    else:
                        result.real_images += 1
                        result.samples_with_valid_images += 1
                        arr = np.array(img.convert("RGB"))
                        image_stds.append(arr.std())

        result.avg_image_std = float(np.mean(image_stds)) if image_stds else 0.0
        result.warnings.append(f"Answerable: {answerable_count}/{max_samples} samples")
        result.status = "success" if result.samples_with_valid_images > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_mmvet(max_samples: int = 100) -> BenchmarkVerification:
    """Verify MM-Vet - Integrated VL capabilities."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="MM-Vet",
        status="pending",
        grading_type="open_ended",
        answer_format="Free-form (needs LLM grader)"
    )

    try:
        ds = load_dataset("lmms-lab/MMVet", split="test", trust_remote_code=True)
        result.total_samples = len(ds)

        image_stds = []
        capabilities = defaultdict(int)

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            # Check answer
            answer = row.get("answer", "")
            if answer:
                result.samples_with_answers += 1

            # Track capabilities
            caps = row.get("capability", [])
            if isinstance(caps, str):
                caps = [caps]
            for cap in caps:
                capabilities[cap] += 1

            # Check image
            img = row.get("image")
            if img is not None:
                result.samples_with_images += 1

                if isinstance(img, Image.Image):
                    if is_gray_image(img):
                        result.gray_images += 1
                    else:
                        result.real_images += 1
                        result.samples_with_valid_images += 1
                        arr = np.array(img.convert("RGB"))
                        image_stds.append(arr.std())

        result.avg_image_std = float(np.mean(image_stds)) if image_stds else 0.0
        result.categories = dict(capabilities)
        result.status = "success" if result.samples_with_valid_images > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


def verify_zerobench(max_samples: int = 100) -> BenchmarkVerification:
    """Verify ZeroBench - Designed to be unsolvable."""
    from datasets import load_dataset

    result = BenchmarkVerification(
        name="ZeroBench",
        status="pending",
        grading_type="open_ended",
        answer_format="Free-form (very hard)"
    )

    try:
        ds = load_dataset("jonathan-roberts1/zerobench", split="zerobench", trust_remote_code=True)
        result.total_samples = len(ds)

        image_stds = []
        multi_image_count = 0

        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            # Check answer
            answer = row.get("question_answer", "")
            if answer:
                result.samples_with_answers += 1

            # Check images - ZeroBench often has multiple images
            images = row.get("question_images_decoded", [])
            if images:
                if len(images) > 1:
                    multi_image_count += 1
                result.samples_with_images += 1

                # Check first image
                img = images[0] if isinstance(images, list) else images
                if isinstance(img, Image.Image):
                    if is_gray_image(img):
                        result.gray_images += 1
                    else:
                        result.real_images += 1
                        result.samples_with_valid_images += 1
                        arr = np.array(img.convert("RGB"))
                        image_stds.append(arr.std())

        result.avg_image_std = float(np.mean(image_stds)) if image_stds else 0.0
        result.warnings.append(f"Multi-image samples: {multi_image_count}/{max_samples}")
        result.warnings.append("Expected 0% accuracy - designed to be unsolvable")
        result.status = "success" if result.samples_with_valid_images > 0 else "error"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)

    return result


# List of all verification functions
# Organized by difficulty/usefulness for UQ training

# TIER 1: SOTA Hard Benchmarks (<50% GPT-5 accuracy) - HIGH PRIORITY
HARD_BENCHMARKS = {
    "mathvision": verify_mathvision,       # ~24% GPT-5 accuracy
    "mathverse": verify_mathverse,         # ~25% models worse without vision
    "hallusionbench": verify_hallusionbench,  # ~31% GPT-5 accuracy
    # NOTE: ZeroBench excluded - 0% accuracy means no positive class for UQ training
}

# TIER 2: Challenging Benchmarks (50-70% GPT-5 accuracy) - GOOD FOR UQ
MEDIUM_BENCHMARKS = {
    "vsr": verify_vsr,                     # ~70% spatial reasoning
    "mathvista": verify_mathvista,         # ~50% IQ tests, plots
    "mmmu": verify_mmmu,                   # ~62% college-level
    "mmstar": verify_mmstar,               # ~55% vision-indispensable
    "charxiv": verify_charxiv,             # ~60% scientific charts
    "realworldqa": verify_realworldqa,     # ~70% real-world spatial
    "aokvqa": verify_aokvqa,               # ~60% outside knowledge VQA
    "vizwiz": verify_vizwiz,               # ~60% blind users VQA
    "mmvet": verify_mmvet,                 # ~55% integrated VL capabilities
}

# TIER 3: Easier Benchmarks (>70% accuracy) - Less useful for UQ
EASY_BENCHMARKS = {
    "ai2d": verify_ai2d,                   # ~85% science diagrams
    "chartqa": verify_chartqa,             # ~75% chart QA
    "docvqa": verify_docvqa,               # ~75% document QA
    "nlvr2": verify_nlvr2,                 # ~80% binary (multi-image)
    "pope": verify_pope,                   # ~80% object hallucination
    "ocrbench": verify_ocrbench,           # ~70% text recognition
    "gqa": verify_gqa,                     # ~75% visual reasoning
    "infographicvqa": verify_infographicvqa,  # infographic understanding
}

# EXCLUDED from UQ training:
# - textvqa, scienceqa: too easy (>85% accuracy, solved)
# - zerobench: 0% accuracy - no positive class for binary UQ training

# Keep zerobench verifier available for reference but not in main list
EXCLUDED_BENCHMARKS = {
    "zerobench": verify_zerobench,         # 0% - useless for UQ (no correct samples)
}

# Combined verifiers for running
BENCHMARK_VERIFIERS = {
    **HARD_BENCHMARKS,
    **MEDIUM_BENCHMARKS,
    **EASY_BENCHMARKS,
}

# Priority order for UQ training
PRIORITY_BENCHMARKS = list(HARD_BENCHMARKS.keys()) + list(MEDIUM_BENCHMARKS.keys())


def run_all_verifications(max_samples: int = 50) -> dict:
    """Run verification on all benchmarks."""
    results = {}

    print("=" * 80)
    print("BENCHMARK VERIFICATION REPORT")
    print("=" * 80)
    print()

    for name, verifier in BENCHMARK_VERIFIERS.items():
        print(f"Verifying {name}...", end=" ", flush=True)
        try:
            result = verifier(max_samples=max_samples)
            results[name] = asdict(result)

            # Print summary
            status_icon = "✓" if result.status == "success" else ("⚠" if result.status == "partial" else "✗")
            print(f"{status_icon} {result.status.upper()}")

            if result.error_message:
                print(f"   ERROR: {result.error_message}")
            if result.warnings:
                for w in result.warnings[:2]:
                    print(f"   WARNING: {w}")

            # Print key stats
            print(f"   Samples: {result.total_samples}, Images: {result.samples_with_valid_images}/{result.samples_with_images}")
            print(f"   Real images: {result.real_images}, Gray: {result.gray_images}")
            print(f"   Grading: {result.grading_type} - {result.answer_format}")
            print()

        except Exception as e:
            print(f"✗ EXCEPTION: {e}")
            results[name] = {"status": "exception", "error": str(e)}
            traceback.print_exc()
            print()

    return results


def print_summary(results: dict):
    """Print summary of all benchmark verifications."""
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    success = [k for k, v in results.items() if v.get("status") == "success"]
    partial = [k for k, v in results.items() if v.get("status") == "partial"]
    error = [k for k, v in results.items() if v.get("status") in ("error", "exception")]

    print(f"SUCCESS ({len(success)}): {', '.join(success)}")
    print(f"PARTIAL ({len(partial)}): {', '.join(partial)}")
    print(f"ERROR ({len(error)}): {', '.join(error)}")
    print()

    # Category distribution
    print("GRADING TYPES:")
    grading_types = defaultdict(list)
    for name, v in results.items():
        if v.get("grading_type"):
            grading_types[v["grading_type"]].append(name)

    for gtype, benchmarks in grading_types.items():
        print(f"  {gtype}: {', '.join(benchmarks)}")
    print()

    # Issues to fix
    print("ISSUES TO FIX:")
    for name, v in results.items():
        gray = v.get("gray_images", 0)
        real = v.get("real_images", 0)
        if gray > 0 and gray >= real:
            print(f"  {name}: {gray} gray images vs {real} real - needs image loading fix")
        if v.get("warnings"):
            for w in v["warnings"][:1]:
                print(f"  {name}: {w}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify VLM benchmarks")
    parser.add_argument("--max-samples", type=int, default=50, help="Max samples per benchmark")
    parser.add_argument("--benchmark", type=str, help="Verify specific benchmark only")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()

    if args.benchmark:
        if args.benchmark not in BENCHMARK_VERIFIERS:
            print(f"Unknown benchmark: {args.benchmark}")
            print(f"Available: {list(BENCHMARK_VERIFIERS.keys())}")
            sys.exit(1)

        result = BENCHMARK_VERIFIERS[args.benchmark](max_samples=args.max_samples)
        results = {args.benchmark: asdict(result)}

        print(json.dumps(results, indent=2))
    else:
        results = run_all_verifications(max_samples=args.max_samples)
        print_summary(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
