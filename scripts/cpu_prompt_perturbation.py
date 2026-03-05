#!/usr/bin/env python3
"""Generate perturbed questions/answers for prompt perturbation consistency analysis.

This is a CPU-only preprocessing script. It generates multiple perturbed versions
of each scored sample's question/response, which can later be scored by the UQ
judge on GPU. Agreement/disagreement across perturbations is itself a signal of
uncertainty.

Perturbation strategies (all rule-based, no API calls):
  - shuffle_sentences: Randomly reorder sentences in the response
  - drop_sentence: Drop one sentence at a time (N perturbations per sample)
  - synonym_replace: Replace common words with synonyms
  - case_change: Randomize capitalization of the response
  - add_filler: Add filler phrases at the start of the response
  - truncate_response: Truncate response at 25%, 50%, 75%
  - swap_answer_format: For MCQ, change answer format

Usage:
    # Smoke test (50 samples, 2 variants per strategy)
    python scripts/cpu_prompt_perturbation.py --smoke_test

    # Full run (all samples, all variants)
    python scripts/cpu_prompt_perturbation.py

    # Custom paths
    python scripts/cpu_prompt_perturbation.py \
        --scored_dir data/use_cases/scored_test_only \
        --output data/use_cases/perturbations/all_perturbations.jsonl
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where raw predictions live — same layout as score_all_unified.py
TARGET_CONFIGS = {
    "gpt5mini": {
        "data_dir": str(PROJECT_ROOT / "runs/gpt5_mini_combined"),
        "mode": "combined",
    },
    "gpt52": {
        "prefix": "gpt52_high_",
        "mode": "prefixed",
    },
    "qwen35": {
        "prefix": "qwen35_397b_",
        "mode": "prefixed",
    },
}

VLM_BENCHMARKS = {
    "charxiv", "mmmu", "mmstar", "hallusionbench", "mathverse",
    "mathvision", "mathvista", "realworldqa", "vizwiz", "hle_multimodal", "mmvet",
}

EXCLUDED_BENCHMARKS = {
    "triviaqa", "babilong", "vsr", "aokvqa", "erqa",
    "tutorbench", "healthbench", "arc", "oolong",
}

# Filler phrases for the add_filler strategy
FILLER_PHRASES = [
    "Well, ",
    "Let me think... ",
    "So, ",
    "Okay, ",
    "Hmm, ",
    "Right, so ",
    "To answer this, ",
    "Looking at this carefully, ",
    "After considering the question, ",
    "In my understanding, ",
]

# MCQ answer format variants
MCQ_FORMAT_VARIANTS = [
    # (pattern, replacements) — we try to detect the format and produce variants
    # Handled dynamically in swap_answer_format()
]

# Synonym dictionary for ~100 common words
SYNONYM_DICT = {
    "correct": ["right", "accurate", "proper"],
    "wrong": ["incorrect", "inaccurate", "erroneous"],
    "answer": ["response", "reply", "solution"],
    "question": ["query", "inquiry", "problem"],
    "result": ["outcome", "finding", "conclusion"],
    "show": ["demonstrate", "indicate", "reveal"],
    "shows": ["demonstrates", "indicates", "reveals"],
    "give": ["provide", "supply", "offer"],
    "gives": ["provides", "supplies", "offers"],
    "use": ["utilize", "employ", "apply"],
    "uses": ["utilizes", "employs", "applies"],
    "used": ["utilized", "employed", "applied"],
    "find": ["determine", "identify", "locate"],
    "found": ["determined", "identified", "located"],
    "get": ["obtain", "acquire", "retrieve"],
    "make": ["create", "produce", "construct"],
    "think": ["believe", "consider", "suppose"],
    "know": ["understand", "recognize", "realize"],
    "see": ["observe", "notice", "perceive"],
    "good": ["excellent", "fine", "satisfactory"],
    "bad": ["poor", "inadequate", "unsatisfactory"],
    "big": ["large", "substantial", "significant"],
    "small": ["little", "minor", "tiny"],
    "important": ["significant", "crucial", "essential"],
    "different": ["distinct", "various", "diverse"],
    "similar": ["alike", "comparable", "analogous"],
    "same": ["identical", "equal", "equivalent"],
    "first": ["initial", "primary", "foremost"],
    "last": ["final", "ultimate", "concluding"],
    "new": ["novel", "fresh", "recent"],
    "old": ["previous", "former", "prior"],
    "true": ["valid", "genuine", "authentic"],
    "false": ["invalid", "untrue", "spurious"],
    "many": ["numerous", "several", "multiple"],
    "few": ["some", "a handful of", "limited"],
    "also": ["additionally", "moreover", "furthermore"],
    "however": ["nevertheless", "nonetheless", "yet"],
    "because": ["since", "as", "due to the fact that"],
    "therefore": ["thus", "hence", "consequently"],
    "about": ["approximately", "roughly", "around"],
    "very": ["extremely", "highly", "exceedingly"],
    "example": ["instance", "illustration", "case"],
    "part": ["component", "element", "portion"],
    "change": ["modify", "alter", "adjust"],
    "changes": ["modifications", "alterations", "adjustments"],
    "help": ["assist", "aid", "support"],
    "helps": ["assists", "aids", "supports"],
    "start": ["begin", "commence", "initiate"],
    "end": ["finish", "conclude", "terminate"],
    "include": ["contain", "encompass", "comprise"],
    "includes": ["contains", "encompasses", "comprises"],
    "need": ["require", "necessitate", "demand"],
    "needs": ["requires", "necessitates", "demands"],
    "likely": ["probably", "presumably", "plausibly"],
    "unlikely": ["improbably", "doubtfully", "implausibly"],
    "clear": ["obvious", "evident", "apparent"],
    "certain": ["sure", "definite", "confident"],
    "possible": ["feasible", "potential", "viable"],
    "simple": ["straightforward", "basic", "elementary"],
    "complex": ["complicated", "intricate", "elaborate"],
    "value": ["amount", "quantity", "magnitude"],
    "total": ["sum", "aggregate", "overall"],
    "type": ["kind", "category", "variety"],
    "data": ["information", "evidence", "facts"],
    "method": ["approach", "technique", "procedure"],
    "process": ["procedure", "mechanism", "operation"],
    "system": ["framework", "structure", "arrangement"],
    "based": ["founded", "grounded", "rooted"],
    "increase": ["rise", "growth", "gain"],
    "decrease": ["decline", "reduction", "drop"],
    "high": ["elevated", "raised", "substantial"],
    "low": ["reduced", "minimal", "limited"],
    "best": ["optimal", "finest", "greatest"],
    "worst": ["poorest", "least favorable", "most inferior"],
    "problem": ["issue", "challenge", "difficulty"],
    "solution": ["resolution", "remedy", "fix"],
    "number": ["count", "quantity", "figure"],
    "point": ["aspect", "factor", "element"],
    "reason": ["cause", "basis", "rationale"],
    "way": ["manner", "approach", "means"],
    "work": ["function", "operate", "perform"],
    "works": ["functions", "operates", "performs"],
    "state": ["condition", "status", "situation"],
    "step": ["stage", "phase", "action"],
    "steps": ["stages", "phases", "actions"],
    "thus": ["therefore", "hence", "consequently"],
    "note": ["observe", "remark", "mention"],
    "equal": ["equivalent", "identical", "matching"],
    "contains": ["includes", "holds", "encompasses"],
    "requires": ["needs", "demands", "necessitates"],
    "provides": ["offers", "supplies", "delivers"],
    "describe": ["explain", "detail", "outline"],
    "describes": ["explains", "details", "outlines"],
    "calculate": ["compute", "determine", "evaluate"],
    "compare": ["contrast", "evaluate", "assess"],
    "analyze": ["examine", "investigate", "study"],
    "suggest": ["propose", "recommend", "indicate"],
    "suggests": ["proposes", "recommends", "indicates"],
}


# ============================================================
# DATA LOADING — mirror score_all_unified.py logic
# ============================================================

def extract_question_text(input_data) -> str:
    """Extract question text from input field (same as score_all_unified.py)."""
    if isinstance(input_data, str):
        return input_data
    if isinstance(input_data, dict):
        for key in ["question", "query", "query_cot", "prompt", "text"]:
            if key in input_data and input_data[key]:
                val = input_data[key]
                if isinstance(val, str):
                    return val
        if "messages" in input_data:
            for msg in input_data["messages"]:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        texts = [p.get("text", "") for p in content
                                 if isinstance(p, dict) and "text" in p]
                        return " ".join(texts)
        clean = {k: v for k, v in input_data.items() if k != "images"}
        return json.dumps(clean)[:2000]
    return str(input_data)[:2000]


def load_full_predictions(target_model: str) -> dict:
    """Load full question/response from raw prediction files, keyed by sample ID.

    Returns dict: id -> {"question": str, "response": str, "benchmark": str}
    """
    config = TARGET_CONFIGS.get(target_model)
    if config is None:
        print(f"WARNING: Unknown target model '{target_model}', skipping full data load")
        return {}

    id_to_data = {}

    if config["mode"] == "combined":
        data_path = Path(config["data_dir"])
        if not data_path.exists():
            print(f"WARNING: {data_path} does not exist")
            return {}
        for bench_dir in sorted(data_path.iterdir()):
            if not bench_dir.is_dir():
                continue
            benchmark = bench_dir.name
            if benchmark in EXCLUDED_BENCHMARKS:
                continue
            pred_file = bench_dir / "predictions.jsonl"
            if not pred_file.exists():
                continue
            _load_pred_file(pred_file, benchmark, id_to_data)
    else:
        runs_path = PROJECT_ROOT / "runs"
        prefix = config["prefix"]
        for run_dir in sorted(runs_path.iterdir()):
            if not run_dir.name.startswith(prefix):
                continue
            benchmark = run_dir.name[len(prefix):]
            if benchmark in EXCLUDED_BENCHMARKS:
                continue
            if "backup" in run_dir.name or "combined" in run_dir.name:
                continue
            pred_file = run_dir / "predictions.jsonl"
            if not pred_file.exists():
                continue
            _load_pred_file(pred_file, benchmark, id_to_data)

    return id_to_data


def _load_pred_file(pred_file: Path, benchmark: str, id_to_data: dict):
    """Load predictions from a single file into id_to_data dict."""
    with open(pred_file) as f:
        for line in f:
            try:
                pred = json.loads(line)
            except json.JSONDecodeError:
                continue

            sample_id = str(pred.get("id", ""))
            if not sample_id:
                continue

            question = extract_question_text(pred.get("input", {}))
            response = pred.get("response_text", "")
            if not response:
                response = str(pred.get("prediction", {}).get("answer", ""))

            if question and response:
                id_to_data[sample_id] = {
                    "question": question[:2000],
                    "response": response[:2000],
                    "benchmark": benchmark,
                }


def load_scored_samples(scored_dir: str) -> list:
    """Load all scored test-only samples from JSONL files.

    Returns list of dicts with full question/response text loaded from
    original prediction files.
    """
    scored_path = Path(scored_dir)
    if not scored_path.exists():
        print(f"ERROR: Scored directory {scored_dir} does not exist")
        sys.exit(1)

    # First pass: collect all sample IDs grouped by target model
    samples_by_target = defaultdict(list)
    for jsonl_file in sorted(scored_path.glob("*_scored.jsonl")):
        target_model = jsonl_file.stem.replace("_scored", "")
        with open(jsonl_file) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                samples_by_target[target_model].append(d)

    # Load full predictions for each target model
    all_samples = []
    for target_model, scored_list in samples_by_target.items():
        print(f"Loading full predictions for target={target_model} "
              f"({len(scored_list)} scored samples)...")
        full_data = load_full_predictions(target_model)
        print(f"  Loaded {len(full_data)} raw predictions")

        matched = 0
        for sample in scored_list:
            sid = sample["id"]
            if sid in full_data:
                sample["question"] = full_data[sid]["question"]
                sample["response"] = full_data[sid]["response"]
                matched += 1
            else:
                # Fall back to preview (truncated at 200 chars)
                sample["question"] = sample.get("question_preview", "")
                sample["response"] = sample.get("response_preview", "")

            sample["target_model"] = target_model
            all_samples.append(sample)

        print(f"  Matched {matched}/{len(scored_list)} samples to full text")

    return all_samples


# ============================================================
# SENTENCE UTILITIES
# ============================================================

def split_sentences(text: str) -> list:
    """Split text into sentences. Handles common abbreviations."""
    # Simple sentence splitting — split on .!? followed by space or end
    # Avoid splitting on common abbreviations
    abbrevs = r"(?<!\b(?:Mr|Mrs|Ms|Dr|Prof|Jr|Sr|vs|etc|e\.g|i\.e|approx))"
    parts = re.split(
        abbrevs + r'(?<=[.!?])\s+',
        text.strip()
    )
    # Filter out empty strings
    return [s.strip() for s in parts if s.strip()]


def is_mcq_response(response: str) -> bool:
    """Detect if a response looks like it contains an MCQ answer."""
    patterns = [
        r'\([A-J]\)',            # (A), (B), ...
        r'\b[A-J]\b',           # standalone A, B, ...
        r'Option\s+[A-J]',     # Option A
        r'The answer is\s+[A-J]',  # The answer is A
        r'"answer"\s*:\s*"[A-J]"', # JSON "answer": "A"
    ]
    return any(re.search(p, response[:500]) for p in patterns)


def extract_mcq_letter(response: str) -> str:
    """Try to extract the MCQ answer letter from a response."""
    # Try JSON format first: "answer": "A"
    m = re.search(r'"answer"\s*:\s*"([A-J])"', response)
    if m:
        return m.group(1)
    # Try "The answer is X"
    m = re.search(r'(?:the answer is|answer:)\s*\(?([A-J])\)?', response, re.IGNORECASE)
    if m:
        return m.group(1)
    # Try standalone (X) at start
    m = re.search(r'^\s*\(?([A-J])\)', response)
    if m:
        return m.group(1)
    return ""


# ============================================================
# PERTURBATION STRATEGIES
# ============================================================

def perturb_shuffle_sentences(sample: dict, max_variants: int = 5,
                              rng: random.Random = None) -> list:
    """Randomly reorder sentences in the response."""
    rng = rng or random.Random(42)
    response = sample["response"]
    sentences = split_sentences(response)

    if len(sentences) <= 1:
        return []

    variants = []
    seen = set()
    seen.add(tuple(range(len(sentences))))  # skip original order

    for _ in range(max_variants * 3):  # try extra times to find unique perms
        if len(variants) >= max_variants:
            break
        order = list(range(len(sentences)))
        rng.shuffle(order)
        order_key = tuple(order)
        if order_key in seen:
            continue
        seen.add(order_key)
        perturbed = " ".join(sentences[i] for i in order)
        variants.append(perturbed)

    return variants


def perturb_drop_sentence(sample: dict, max_variants: int = None,
                          rng: random.Random = None) -> list:
    """Drop one sentence at a time from the response."""
    response = sample["response"]
    sentences = split_sentences(response)

    if len(sentences) <= 1:
        return []

    variants = []
    indices = list(range(len(sentences)))
    if max_variants is not None and len(indices) > max_variants:
        rng = rng or random.Random(42)
        indices = sorted(rng.sample(indices, max_variants))

    for drop_idx in indices:
        remaining = [s for i, s in enumerate(sentences) if i != drop_idx]
        variants.append(" ".join(remaining))

    return variants


def perturb_synonym_replace(sample: dict, max_variants: int = 3,
                            rng: random.Random = None) -> list:
    """Replace common words with synonyms in the response."""
    rng = rng or random.Random(42)
    response = sample["response"]

    # Find replaceable words
    words = response.split()
    replaceable_indices = []
    for i, word in enumerate(words):
        clean = word.strip(".,;:!?\"'()[]{}").lower()
        if clean in SYNONYM_DICT:
            replaceable_indices.append(i)

    if not replaceable_indices:
        return []

    variants = []
    for v in range(max_variants):
        new_words = list(words)
        # Replace a random subset of replaceable words
        n_replace = max(1, len(replaceable_indices) // 3)
        to_replace = rng.sample(
            replaceable_indices,
            min(n_replace, len(replaceable_indices))
        )
        for idx in to_replace:
            original = new_words[idx]
            clean = original.strip(".,;:!?\"'()[]{}").lower()
            if clean in SYNONYM_DICT:
                synonyms = SYNONYM_DICT[clean]
                replacement = rng.choice(synonyms)
                # Preserve punctuation and capitalization
                prefix = ""
                suffix = ""
                stripped = original
                while stripped and not stripped[0].isalnum():
                    prefix += stripped[0]
                    stripped = stripped[1:]
                while stripped and not stripped[-1].isalnum():
                    suffix = stripped[-1] + suffix
                    stripped = stripped[:-1]
                # Match case of original
                if stripped and stripped[0].isupper():
                    replacement = replacement[0].upper() + replacement[1:]
                if stripped and stripped.isupper():
                    replacement = replacement.upper()
                new_words[idx] = prefix + replacement + suffix
        variants.append(" ".join(new_words))

    return variants


def perturb_case_change(sample: dict, max_variants: int = 3,
                        rng: random.Random = None) -> list:
    """Randomize capitalization of the response."""
    rng = rng or random.Random(42)
    response = sample["response"]

    if len(response) < 5:
        return []

    variants = []
    for _ in range(max_variants):
        chars = list(response)
        for i in range(len(chars)):
            if chars[i].isalpha():
                if rng.random() < 0.15:  # Flip ~15% of chars
                    if chars[i].isupper():
                        chars[i] = chars[i].lower()
                    else:
                        chars[i] = chars[i].upper()
        perturbed = "".join(chars)
        if perturbed != response:
            variants.append(perturbed)

    return variants


def perturb_add_filler(sample: dict, max_variants: int = None,
                       rng: random.Random = None) -> list:
    """Add filler phrases at the start of the response."""
    rng = rng or random.Random(42)
    response = sample["response"]
    fillers = list(FILLER_PHRASES)

    if max_variants is not None and len(fillers) > max_variants:
        fillers = rng.sample(fillers, max_variants)

    return [filler + response for filler in fillers]


def perturb_truncate_response(sample: dict, max_variants: int = None,
                              rng: random.Random = None) -> list:
    """Truncate response at 25%, 50%, 75% of length."""
    response = sample["response"]
    if len(response) < 20:
        return []

    fractions = [0.25, 0.50, 0.75]
    if max_variants is not None:
        fractions = fractions[:max_variants]

    variants = []
    for frac in fractions:
        cutoff = max(10, int(len(response) * frac))
        truncated = response[:cutoff]
        # Try to cut at a word boundary
        last_space = truncated.rfind(" ")
        if last_space > cutoff * 0.8:
            truncated = truncated[:last_space]
        variants.append(truncated + "...")

    return variants


def perturb_swap_answer_format(sample: dict, max_variants: int = None,
                               rng: random.Random = None) -> list:
    """For MCQ responses, change the answer format."""
    response = sample["response"]
    letter = extract_mcq_letter(response)

    if not letter:
        return []

    # Generate format variants
    format_variants = [
        f"({letter})",
        letter,
        f"Option {letter}",
        f"The answer is {letter}",
        f"The answer is ({letter})",
        f"Answer: {letter}",
        f"({letter}) is correct",
    ]

    if max_variants is not None:
        format_variants = format_variants[:max_variants]

    variants = []
    for fmt in format_variants:
        # Replace the detected answer format in the response
        new_response = response
        # Try multiple replacement patterns
        patterns = [
            (r'"answer"\s*:\s*"[A-J]"', f'"answer": "{letter}"'),
            (r'(?:the answer is|answer:)\s*\(?[A-J]\)?',
             f'the answer is {fmt}'),
            (r'\([A-J]\)', fmt),
        ]
        replaced = False
        for pattern, replacement in patterns:
            new_resp, n = re.subn(pattern, replacement, new_response,
                                  count=1, flags=re.IGNORECASE)
            if n > 0:
                new_response = new_resp
                replaced = True
                break

        if not replaced:
            # Prepend the answer format
            new_response = fmt + ". " + response

        if new_response != response:
            variants.append(new_response)

    return variants


# Registry of all perturbation strategies
PERTURBATION_STRATEGIES = {
    "shuffle_sentences": perturb_shuffle_sentences,
    "drop_sentence": perturb_drop_sentence,
    "synonym_replace": perturb_synonym_replace,
    "case_change": perturb_case_change,
    "add_filler": perturb_add_filler,
    "truncate_response": perturb_truncate_response,
    "swap_answer_format": perturb_swap_answer_format,
}


# ============================================================
# WORKER FUNCTION (for multiprocessing)
# ============================================================

def generate_perturbations_for_sample(args: tuple) -> list:
    """Generate all perturbations for a single sample.

    Args:
        args: (sample_dict, max_variants_per_strategy, seed_offset)

    Returns:
        List of perturbation dicts ready for JSONL output.
    """
    sample, max_variants, seed_offset = args
    sid = sample["id"]
    question = sample.get("question", sample.get("question_preview", ""))
    response = sample.get("response", sample.get("response_preview", ""))

    if not question or not response:
        return []

    # Build a sample dict for perturbation functions
    perturb_input = {
        "question": question,
        "response": response,
    }

    results = []
    for strategy_name, strategy_fn in PERTURBATION_STRATEGIES.items():
        # Deterministic seed per sample + strategy
        seed = hash((sid, strategy_name, seed_offset)) % (2**31)
        rng = random.Random(seed)

        try:
            variants = strategy_fn(
                perturb_input,
                max_variants=max_variants,
                rng=rng,
            )
        except Exception as e:
            # Skip on error, don't crash the whole pipeline
            continue

        for vi, perturbed_response in enumerate(variants):
            results.append({
                "original_id": sid,
                "perturbation_type": strategy_name,
                "variant_idx": vi,
                "perturbation_id": f"{sid}__{strategy_name}__{vi}",
                "benchmark": sample.get("benchmark", ""),
                "target_model": sample.get("target_model", ""),
                "is_correct": sample.get("is_correct", -1),
                "question": question,
                "response": perturbed_response,
                "has_image": sample.get("has_image", False),
            })

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate perturbed Q/A pairs for prompt perturbation consistency"
    )
    parser.add_argument(
        "--scored_dir",
        default=str(PROJECT_ROOT / "data/use_cases/scored_test_only"),
        help="Directory with scored JSONL files (default: data/use_cases/scored_test_only/)",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data/use_cases/perturbations/all_perturbations.jsonl"),
        help="Output JSONL file",
    )
    parser.add_argument(
        "--output_summary",
        default=str(PROJECT_ROOT / "data/use_cases/perturbations/summary.json"),
        help="Output summary JSON file",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Smoke test: first 50 samples, 2 variants per strategy",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples to process (overrides smoke_test count)",
    )
    parser.add_argument(
        "--max_variants",
        type=int,
        default=None,
        help="Max variants per strategy per sample (overrides smoke_test count)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: os.cpu_count())",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Apply smoke test defaults
    if args.smoke_test:
        if args.max_samples is None:
            args.max_samples = 50
        if args.max_variants is None:
            args.max_variants = 2
        print("=== SMOKE TEST MODE ===")
        print(f"  Max samples: {args.max_samples}")
        print(f"  Max variants per strategy: {args.max_variants}")
        print()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    print("=" * 70)
    print("LOADING SCORED SAMPLES WITH FULL TEXT")
    print("=" * 70)
    t0 = time.time()
    all_samples = load_scored_samples(args.scored_dir)
    load_time = time.time() - t0
    print(f"\nLoaded {len(all_samples)} total scored samples in {load_time:.1f}s")

    # Subsample if needed
    if args.max_samples is not None and len(all_samples) > args.max_samples:
        rng = random.Random(args.seed)
        all_samples = rng.sample(all_samples, args.max_samples)
        print(f"Subsampled to {len(all_samples)} samples")

    # Prepare worker arguments
    worker_args = [
        (sample, args.max_variants, args.seed)
        for sample in all_samples
    ]

    n_workers = args.workers or os.cpu_count() or 1
    print(f"\n{'=' * 70}")
    print(f"GENERATING PERTURBATIONS")
    print(f"{'=' * 70}")
    print(f"Samples: {len(all_samples)}")
    print(f"Strategies: {len(PERTURBATION_STRATEGIES)} "
          f"({', '.join(PERTURBATION_STRATEGIES.keys())})")
    print(f"Workers: {n_workers}")
    if args.max_variants:
        print(f"Max variants per strategy: {args.max_variants}")
    print()

    # Generate perturbations in parallel
    t0 = time.time()
    all_perturbations = []
    strategy_counts = defaultdict(int)
    samples_with_perturbations = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(generate_perturbations_for_sample, wa): i
            for i, wa in enumerate(worker_args)
        }

        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            try:
                perturbations = future.result()
            except Exception as e:
                print(f"  Worker error: {e}")
                continue

            if perturbations:
                samples_with_perturbations += 1
                all_perturbations.extend(perturbations)
                for p in perturbations:
                    strategy_counts[p["perturbation_type"]] += 1

            if done_count % 500 == 0 or done_count == len(worker_args):
                elapsed = time.time() - t0
                rate = done_count / elapsed if elapsed > 0 else 0
                print(f"  [{done_count}/{len(worker_args)}] "
                      f"{rate:.0f} samples/s, "
                      f"{len(all_perturbations)} perturbations so far")

    gen_time = time.time() - t0

    # Write output JSONL
    print(f"\nWriting {len(all_perturbations)} perturbations to {args.output}...")
    with open(output_path, "w") as f:
        for p in all_perturbations:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Build and write summary
    summary = {
        "total_input_samples": len(all_samples),
        "samples_with_perturbations": samples_with_perturbations,
        "total_perturbations": len(all_perturbations),
        "strategies": len(PERTURBATION_STRATEGIES),
        "strategy_counts": dict(sorted(strategy_counts.items())),
        "avg_perturbations_per_sample": (
            round(len(all_perturbations) / len(all_samples), 2)
            if all_samples else 0
        ),
        "generation_time_seconds": round(gen_time, 1),
        "workers": n_workers,
        "smoke_test": args.smoke_test,
        "max_variants": args.max_variants,
        "output_file": str(output_path),
        "scored_dir": args.scored_dir,
        "seed": args.seed,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Input samples:              {len(all_samples)}")
    print(f"Samples with perturbations: {samples_with_perturbations}")
    print(f"Total perturbations:        {len(all_perturbations)}")
    print(f"Avg per sample:             {summary['avg_perturbations_per_sample']}")
    print(f"Generation time:            {gen_time:.1f}s")
    print()
    print("Per-strategy counts:")
    for strategy, count in sorted(strategy_counts.items()):
        avg = count / len(all_samples) if all_samples else 0
        print(f"  {strategy:25s}  {count:6d}  ({avg:.1f} per sample)")
    print()
    print(f"Output:  {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
