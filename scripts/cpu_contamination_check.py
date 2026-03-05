#!/usr/bin/env python3
"""
Data contamination and near-duplicate detection for UQ training/test data.

Checks:
1. Exact train/test overlap (question + response)
2. Near-duplicate detection via character 5-gram Jaccard similarity
3. Cross-benchmark contamination (questions shared across benchmarks)
4. Token-level n-gram overlap statistics between train and test

Fully parallelized with multiprocessing. Supports --smoke_test for quick runs.

Usage:
    # Smoke test
    python scripts/cpu_contamination_check.py --smoke_test

    # Full run (submit via SLURM debug partition)
    python scripts/cpu_contamination_check.py

    # Custom paths
    python scripts/cpu_contamination_check.py \
        --train_file data/finetune/train_v2.jsonl \
        --test_file data/finetune/test_v2.jsonl \
        --output data/use_cases/results_test_only/contamination_report.json
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from functools import partial
from itertools import combinations
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_num_workers() -> int:
    """Get worker count from SLURM env or fall back to os.cpu_count()."""
    return int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count() or 4))


def extract_question_text(sample: dict) -> str:
    """Extract the question/input text from a sample, handling both str and dict inputs."""
    inp = sample.get("input", "")
    if isinstance(inp, dict):
        # GPQA-style: input is a dict with a 'question' key
        parts = []
        if "question" in inp:
            parts.append(str(inp["question"]))
        # Include options if present
        for key in sorted(inp.keys()):
            if key != "question":
                parts.append(f"{key}: {inp[key]}")
        return " ".join(parts)
    return str(inp)


def extract_response_text(sample: dict) -> str:
    """Extract model response text from a sample."""
    resp = sample.get("model_response", "")
    if isinstance(resp, dict):
        return json.dumps(resp, sort_keys=True)
    return str(resp)


def extract_benchmark_name(sample: dict) -> str:
    """Extract benchmark name from the sample ID prefix."""
    sample_id = sample.get("id", "")
    # IDs look like: bbeh_disambiguation_qa_78, gpqa_gpqa_diamond_17, simpleqa_94
    # Extract everything before the last _<number>
    # But some have nested underscores, so use the known benchmark prefixes
    # Heuristic: strip trailing _<digits>
    parts = sample_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        name = parts[0]
    else:
        name = sample_id

    # Further normalize: some benchmarks have sub-names like gpqa_gpqa_diamond
    # Map common prefixes
    known_prefixes = [
        "bbeh", "gpqa", "simpleqa", "boolq", "hellaswag", "triviaqa",
        "mmlu_pro", "mmlu", "drop", "gsm8k", "math", "naturalqa",
        "arc", "winogrande", "humaneval", "mbpp", "hle",
        "healthbench", "prbench", "mmmu", "vsr", "charxiv",
        "hallusionbench", "mmvet", "mgsm", "multinrc",
    ]
    for prefix in sorted(known_prefixes, key=len, reverse=True):
        if name.startswith(prefix):
            return prefix
    # Fallback: use first component
    return name.split("_")[0] if "_" in name else name


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace, strip."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def char_ngrams(text: str, n: int = 5) -> Set[str]:
    """Compute character n-grams from text."""
    text = normalize_text(text)
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def token_ngrams(text: str, n: int = 5) -> Set[Tuple[str, ...]]:
    """Compute word-token n-grams from text."""
    tokens = normalize_text(text).split()
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard_similarity(set_a: Set, set_b: Set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Parallel workers
# ---------------------------------------------------------------------------

def compute_ngrams_for_sample(args: Tuple[int, str, int]) -> Tuple[int, Set[str]]:
    """Worker: compute char n-grams for a single sample's question text."""
    idx, text, n = args
    return idx, char_ngrams(text, n)


def compute_token_ngrams_for_sample(args: Tuple[int, str, int]) -> Tuple[int, Set[Tuple[str, ...]]]:
    """Worker: compute token n-grams for a single sample's question text."""
    idx, text, n = args
    return idx, token_ngrams(text, n)


def compute_pairwise_chunk(
    args: Tuple[List[Tuple[int, int]], List[Set[str]], List[Set[str]], float]
) -> List[Tuple[int, int, float]]:
    """Worker: compute Jaccard similarity for a chunk of index pairs.

    Returns pairs where similarity >= threshold.
    """
    pairs, ngrams_a, ngrams_b, threshold = args
    results = []
    for i, j in pairs:
        sim = jaccard_similarity(ngrams_a[i], ngrams_b[j])
        if sim >= threshold:
            results.append((i, j, sim))
    return results


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str, max_samples: Optional[int] = None) -> List[dict]:
    """Load a JSONL file, optionally limiting to max_samples."""
    samples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_scored_data(scored_dir: str, max_samples: Optional[int] = None) -> List[dict]:
    """Load scored JSONL files from a directory."""
    samples = []
    if not os.path.isdir(scored_dir):
        print(f"  Warning: scored_dir {scored_dir} does not exist, skipping.")
        return samples
    for fname in sorted(os.listdir(scored_dir)):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(scored_dir, fname)
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
                    if max_samples is not None and len(samples) >= max_samples:
                        return samples
    return samples


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def check_exact_duplicates(
    train_samples: List[dict],
    test_samples: List[dict],
) -> dict:
    """Check for exact duplicates within and across splits.

    Uses hash of normalized (question + response) for exact matching,
    and hash of normalized question for question-only matching.
    """
    print("\n[1/5] Checking exact duplicates...")

    def make_key(sample: dict) -> str:
        q = normalize_text(extract_question_text(sample))
        r = normalize_text(extract_response_text(sample))
        return hashlib.sha256((q + "|||" + r).encode()).hexdigest()

    def make_question_key(sample: dict) -> str:
        q = normalize_text(extract_question_text(sample))
        return hashlib.sha256(q.encode()).hexdigest()

    # Build hash sets
    train_keys = {}  # hash -> list of (index, sample_id)
    test_keys = {}
    train_qkeys = {}
    test_qkeys = {}

    for i, s in enumerate(train_samples):
        k = make_key(s)
        train_keys.setdefault(k, []).append((i, s.get("id", f"train_{i}")))
        qk = make_question_key(s)
        train_qkeys.setdefault(qk, []).append((i, s.get("id", f"train_{i}")))

    for i, s in enumerate(test_samples):
        k = make_key(s)
        test_keys.setdefault(k, []).append((i, s.get("id", f"test_{i}")))
        qk = make_question_key(s)
        test_qkeys.setdefault(qk, []).append((i, s.get("id", f"test_{i}")))

    # Internal train duplicates (exact question+response)
    train_internal = []
    for k, entries in train_keys.items():
        if len(entries) > 1:
            train_internal.append({
                "ids": [e[1] for e in entries],
                "count": len(entries),
                "type": "exact_question_response",
            })

    # Internal test duplicates
    test_internal = []
    for k, entries in test_keys.items():
        if len(entries) > 1:
            test_internal.append({
                "ids": [e[1] for e in entries],
                "count": len(entries),
                "type": "exact_question_response",
            })

    # Train-test overlap (exact question+response)
    train_test_qr = []
    overlap_keys = set(train_keys.keys()) & set(test_keys.keys())
    for k in overlap_keys:
        train_test_qr.append({
            "train_ids": [e[1] for e in train_keys[k]],
            "test_ids": [e[1] for e in test_keys[k]],
            "type": "exact_question_response",
        })

    # Train-test overlap (question only -- same question, possibly different response)
    train_test_q = []
    overlap_qkeys = set(train_qkeys.keys()) & set(test_qkeys.keys())
    for qk in overlap_qkeys:
        # Only report if not already caught by exact q+r match
        train_ids = [e[1] for e in train_qkeys[qk]]
        test_ids = [e[1] for e in test_qkeys[qk]]
        train_test_q.append({
            "train_ids": train_ids,
            "test_ids": test_ids,
            "type": "exact_question_only",
        })

    result = {
        "train_internal_exact_qr": len(train_internal),
        "test_internal_exact_qr": len(test_internal),
        "train_test_overlap_exact_qr": len(train_test_qr),
        "train_test_overlap_exact_question": len(train_test_q),
        "details_train_internal": train_internal[:20],
        "details_test_internal": test_internal[:20],
        "details_train_test_qr": train_test_qr[:20],
        "details_train_test_question": train_test_q[:20],
    }

    print(f"  Train internal (exact q+r): {len(train_internal)} duplicate groups")
    print(f"  Test internal (exact q+r):  {len(test_internal)} duplicate groups")
    print(f"  Train-test overlap (q+r):   {len(train_test_qr)}")
    print(f"  Train-test overlap (q only): {len(train_test_q)}")

    return result


def compute_near_duplicates(
    train_questions: List[str],
    test_questions: List[str],
    train_ids: List[str],
    test_ids: List[str],
    threshold: float = 0.8,
    ngram_n: int = 5,
    num_workers: int = 4,
    chunk_size: int = 50000,
) -> dict:
    """Find near-duplicate pairs using character n-gram Jaccard similarity.

    Computes pairwise similarity in chunks to avoid OOM.
    """
    print(f"\n[2/5] Computing near-duplicates (Jaccard >= {threshold}, char {ngram_n}-grams)...")
    all_questions = train_questions + test_questions
    all_ids = train_ids + test_ids
    n_train = len(train_questions)
    n_test = len(test_questions)
    n_total = n_train + n_test

    # Step 1: Compute n-grams in parallel
    print(f"  Computing char {ngram_n}-grams for {n_total} samples ({num_workers} workers)...")
    t0 = time.time()
    tasks = [(i, q, ngram_n) for i, q in enumerate(all_questions)]

    with Pool(num_workers) as pool:
        results = pool.map(compute_ngrams_for_sample, tasks, chunksize=max(1, len(tasks) // num_workers))

    ngrams_list = [None] * n_total
    for idx, ng in results:
        ngrams_list[idx] = ng
    print(f"  N-grams computed in {time.time() - t0:.1f}s")

    # Step 2: Pairwise comparison in chunks
    # We need: train-internal, test-internal, cross-split
    near_dupes = {"train_internal": [], "test_internal": [], "cross_split": []}

    def find_near_dupes_in_range(
        indices_a: List[int],
        indices_b: List[int],
        label: str,
        is_symmetric: bool = False,
    ):
        """Find near-duplicate pairs between two index sets."""
        # Generate all pairs
        if is_symmetric:
            pairs = list(combinations(range(len(indices_a)), 2))
            ngrams_a_sub = [ngrams_list[indices_a[i]] for i in range(len(indices_a))]
            ngrams_b_sub = ngrams_a_sub  # same
        else:
            pairs = [(i, j) for i in range(len(indices_a)) for j in range(len(indices_b))]
            ngrams_a_sub = [ngrams_list[indices_a[i]] for i in range(len(indices_a))]
            ngrams_b_sub = [ngrams_list[indices_b[j]] for j in range(len(indices_b))]

        total_pairs = len(pairs)
        if total_pairs == 0:
            return

        print(f"  {label}: {total_pairs:,} pairs to check...")
        t1 = time.time()

        # Chunk the pairs
        found = []
        for start in range(0, total_pairs, chunk_size):
            end = min(start + chunk_size, total_pairs)
            chunk_pairs = pairs[start:end]

            # Split chunk across workers
            per_worker = max(1, len(chunk_pairs) // num_workers)
            worker_chunks = []
            for w_start in range(0, len(chunk_pairs), per_worker):
                w_end = min(w_start + per_worker, len(chunk_pairs))
                worker_chunks.append(
                    (chunk_pairs[w_start:w_end], ngrams_a_sub, ngrams_b_sub, threshold)
                )

            with Pool(num_workers) as pool:
                chunk_results = pool.map(compute_pairwise_chunk, worker_chunks)

            for cr in chunk_results:
                for i_local, j_local, sim in cr:
                    if is_symmetric:
                        id_a = all_ids[indices_a[i_local]]
                        id_b = all_ids[indices_a[j_local]]
                    else:
                        id_a = all_ids[indices_a[i_local]]
                        id_b = all_ids[indices_b[j_local]]
                    found.append({
                        "id_a": id_a,
                        "id_b": id_b,
                        "jaccard": round(sim, 4),
                    })

        elapsed = time.time() - t1
        print(f"    Found {len(found)} near-duplicate pairs in {elapsed:.1f}s")
        near_dupes[label] = found

    # Train-internal
    train_indices = list(range(n_train))
    find_near_dupes_in_range(train_indices, train_indices, "train_internal", is_symmetric=True)

    # Test-internal
    test_indices = list(range(n_train, n_total))
    find_near_dupes_in_range(test_indices, test_indices, "test_internal", is_symmetric=True)

    # Cross-split (train vs test)
    find_near_dupes_in_range(train_indices, test_indices, "cross_split", is_symmetric=False)

    result = {
        "threshold": threshold,
        "ngram_n": ngram_n,
        "train_internal": len(near_dupes["train_internal"]),
        "test_internal": len(near_dupes["test_internal"]),
        "cross_split": len(near_dupes["cross_split"]),
        "examples_train_internal": near_dupes["train_internal"][:30],
        "examples_test_internal": near_dupes["test_internal"][:30],
        "examples_cross_split": near_dupes["cross_split"][:30],
    }

    return result


def check_cross_benchmark_contamination(
    train_samples: List[dict],
    test_samples: List[dict],
    num_workers: int = 4,
    ngram_n: int = 5,
    threshold: float = 0.8,
) -> dict:
    """Check if questions from one benchmark appear in another benchmark.

    Groups questions by benchmark, then checks for near-duplicates across benchmarks.
    """
    print("\n[3/5] Checking cross-benchmark contamination...")

    all_samples = train_samples + test_samples
    # Group by benchmark
    bench_groups = defaultdict(list)
    for i, s in enumerate(all_samples):
        bname = extract_benchmark_name(s)
        bench_groups[bname].append((i, s.get("id", f"sample_{i}"), extract_question_text(s)))

    print(f"  Found {len(bench_groups)} benchmarks: {sorted(bench_groups.keys())}")

    # Compute n-grams for all questions
    all_questions_text = [extract_question_text(s) for s in all_samples]
    tasks = [(i, q, ngram_n) for i, q in enumerate(all_questions_text)]

    with Pool(num_workers) as pool:
        ngram_results = pool.map(compute_ngrams_for_sample, tasks, chunksize=max(1, len(tasks) // num_workers))

    ngrams_map = {}
    for idx, ng in ngram_results:
        ngrams_map[idx] = ng

    # Check each pair of benchmarks
    bench_names = sorted(bench_groups.keys())
    cross_bench_dupes = []

    for b_i in range(len(bench_names)):
        for b_j in range(b_i + 1, len(bench_names)):
            b1, b2 = bench_names[b_i], bench_names[b_j]
            group1 = bench_groups[b1]
            group2 = bench_groups[b2]

            # Only check if both groups have samples
            found = 0
            examples = []
            for idx1, id1, _ in group1:
                for idx2, id2, _ in group2:
                    sim = jaccard_similarity(ngrams_map[idx1], ngrams_map[idx2])
                    if sim >= threshold:
                        found += 1
                        if len(examples) < 5:
                            examples.append({
                                "benchmark_a": b1,
                                "benchmark_b": b2,
                                "id_a": id1,
                                "id_b": id2,
                                "jaccard": round(sim, 4),
                            })

            if found > 0:
                cross_bench_dupes.append({
                    "benchmark_a": b1,
                    "benchmark_b": b2,
                    "count": found,
                    "examples": examples,
                })
                print(f"    {b1} <-> {b2}: {found} near-duplicate pairs")

    if not cross_bench_dupes:
        print("  No cross-benchmark near-duplicates found.")

    return {
        "threshold": threshold,
        "total_cross_benchmark_pairs": sum(d["count"] for d in cross_bench_dupes),
        "benchmark_pairs_with_overlap": len(cross_bench_dupes),
        "details": cross_bench_dupes,
    }


def compute_ngram_overlap_stats(
    train_questions: List[str],
    test_questions: List[str],
    num_workers: int = 4,
    ngram_n: int = 5,
) -> dict:
    """Compute token-level n-gram overlap statistics between train and test.

    For each test question, compute Jaccard similarity of its token n-grams
    against the union of all train token n-grams. Also compute per-test-sample
    max Jaccard against individual train samples.
    """
    print(f"\n[4/5] Computing token {ngram_n}-gram overlap statistics...")
    t0 = time.time()

    # Compute token n-grams for all train and test in parallel
    all_questions = train_questions + test_questions
    n_train = len(train_questions)
    tasks = [(i, q, ngram_n) for i, q in enumerate(all_questions)]

    with Pool(num_workers) as pool:
        results = pool.map(compute_token_ngrams_for_sample, tasks, chunksize=max(1, len(tasks) // num_workers))

    token_ngrams_list = [None] * len(all_questions)
    for idx, ng in results:
        token_ngrams_list[idx] = ng

    # Build union of all train token n-grams
    train_union = set()
    for i in range(n_train):
        train_union |= token_ngrams_list[i]

    print(f"  Train token {ngram_n}-gram vocabulary: {len(train_union):,} unique n-grams")

    # For each test sample, compute overlap with train union
    test_overlaps = []
    for i in range(n_train, len(all_questions)):
        test_ng = token_ngrams_list[i]
        if not test_ng:
            test_overlaps.append(0.0)
            continue
        overlap = len(test_ng & train_union) / len(test_ng)
        test_overlaps.append(overlap)

    # Also compute per-test max Jaccard against any single train sample
    # (more expensive but gives fine-grained contamination signal)
    print("  Computing per-test max Jaccard against individual train samples...")
    max_jaccards = []
    for i in range(n_train, len(all_questions)):
        test_ng = token_ngrams_list[i]
        best = 0.0
        for j in range(n_train):
            sim = jaccard_similarity(test_ng, token_ngrams_list[j])
            if sim > best:
                best = sim
        max_jaccards.append(best)

    test_overlaps_arr = np.array(test_overlaps)
    max_jaccards_arr = np.array(max_jaccards)

    elapsed = time.time() - t0
    print(f"  Token n-gram stats computed in {elapsed:.1f}s")

    result = {
        "ngram_n": ngram_n,
        "train_unique_ngrams": len(train_union),
        "test_token_ngram_overlap_with_train_union": {
            "mean": round(float(np.mean(test_overlaps_arr)), 4),
            "median": round(float(np.median(test_overlaps_arr)), 4),
            "std": round(float(np.std(test_overlaps_arr)), 4),
            "min": round(float(np.min(test_overlaps_arr)), 4) if len(test_overlaps_arr) > 0 else None,
            "max": round(float(np.max(test_overlaps_arr)), 4) if len(test_overlaps_arr) > 0 else None,
            "pct_above_50": round(float(np.mean(test_overlaps_arr > 0.5) * 100), 2),
            "pct_above_80": round(float(np.mean(test_overlaps_arr > 0.8) * 100), 2),
        },
        "test_max_jaccard_vs_any_train_sample": {
            "mean": round(float(np.mean(max_jaccards_arr)), 4),
            "median": round(float(np.median(max_jaccards_arr)), 4),
            "std": round(float(np.std(max_jaccards_arr)), 4),
            "min": round(float(np.min(max_jaccards_arr)), 4) if len(max_jaccards_arr) > 0 else None,
            "max": round(float(np.max(max_jaccards_arr)), 4) if len(max_jaccards_arr) > 0 else None,
            "pct_above_50": round(float(np.mean(max_jaccards_arr > 0.5) * 100), 2),
            "pct_above_80": round(float(np.mean(max_jaccards_arr > 0.8) * 100), 2),
        },
    }

    return result


def compute_benchmark_distribution(
    train_samples: List[dict],
    test_samples: List[dict],
) -> dict:
    """Compute benchmark distribution for train and test splits."""
    print("\n[5/5] Computing benchmark distribution...")

    train_dist = Counter()
    test_dist = Counter()

    for s in train_samples:
        train_dist[extract_benchmark_name(s)] += 1
    for s in test_samples:
        test_dist[extract_benchmark_name(s)] += 1

    # Check for benchmarks only in one split
    all_benchmarks = sorted(set(train_dist.keys()) | set(test_dist.keys()))
    train_only = sorted(set(train_dist.keys()) - set(test_dist.keys()))
    test_only = sorted(set(test_dist.keys()) - set(train_dist.keys()))

    # Compute distribution skew
    total_train = sum(train_dist.values())
    total_test = sum(test_dist.values())

    distribution_comparison = {}
    for b in all_benchmarks:
        train_pct = (train_dist.get(b, 0) / total_train * 100) if total_train > 0 else 0
        test_pct = (test_dist.get(b, 0) / total_test * 100) if total_test > 0 else 0
        distribution_comparison[b] = {
            "train_count": train_dist.get(b, 0),
            "test_count": test_dist.get(b, 0),
            "train_pct": round(train_pct, 2),
            "test_pct": round(test_pct, 2),
            "pct_diff": round(abs(train_pct - test_pct), 2),
        }

    result = {
        "train": dict(sorted(train_dist.items())),
        "test": dict(sorted(test_dist.items())),
        "total_train": total_train,
        "total_test": total_test,
        "num_benchmarks_total": len(all_benchmarks),
        "benchmarks_train_only": train_only,
        "benchmarks_test_only": test_only,
        "distribution_comparison": distribution_comparison,
    }

    print(f"  Total benchmarks: {len(all_benchmarks)}")
    print(f"  Train-only benchmarks: {train_only}")
    print(f"  Test-only benchmarks: {test_only}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_summary(report: dict) -> str:
    """Generate a human-readable summary of the contamination report."""
    lines = []
    lines.append("=" * 70)
    lines.append("DATA CONTAMINATION CHECK SUMMARY")
    lines.append("=" * 70)

    # Exact duplicates
    ed = report["exact_duplicates"]
    lines.append("")
    lines.append("EXACT DUPLICATES:")
    lines.append(f"  Train internal (question+response): {ed['train_internal_exact_qr']} duplicate groups")
    lines.append(f"  Test internal (question+response):  {ed['test_internal_exact_qr']} duplicate groups")
    lines.append(f"  Train-test overlap (question+response): {ed['train_test_overlap_exact_qr']}")
    lines.append(f"  Train-test overlap (question only):     {ed['train_test_overlap_exact_question']}")

    if ed["train_test_overlap_exact_qr"] > 0:
        lines.append("  ** WARNING: Exact train/test overlap detected! This is data leakage. **")
    if ed["train_test_overlap_exact_question"] > 0:
        lines.append(f"  Note: {ed['train_test_overlap_exact_question']} questions appear in both splits")
        lines.append("  (same question, possibly different model responses -- expected for multi-model data)")

    # Near duplicates
    nd = report["near_duplicates"]
    lines.append("")
    lines.append(f"NEAR-DUPLICATES (char {nd['ngram_n']}-gram Jaccard >= {nd['threshold']}):")
    lines.append(f"  Train internal: {nd['train_internal']} pairs")
    lines.append(f"  Test internal:  {nd['test_internal']} pairs")
    lines.append(f"  Cross-split:    {nd['cross_split']} pairs")

    if nd["cross_split"] > 0:
        lines.append("  ** WARNING: Near-duplicate questions found across train/test split. **")
        for ex in nd.get("examples_cross_split", [])[:5]:
            lines.append(f"    {ex['id_a']} <-> {ex['id_b']} (Jaccard={ex['jaccard']})")

    # Cross-benchmark
    cb = report["cross_benchmark_contamination"]
    lines.append("")
    lines.append(f"CROSS-BENCHMARK CONTAMINATION (Jaccard >= {cb['threshold']}):")
    lines.append(f"  Total cross-benchmark near-duplicate pairs: {cb['total_cross_benchmark_pairs']}")
    lines.append(f"  Benchmark pairs with overlap: {cb['benchmark_pairs_with_overlap']}")
    for detail in cb.get("details", [])[:10]:
        lines.append(f"    {detail['benchmark_a']} <-> {detail['benchmark_b']}: {detail['count']} pairs")

    # N-gram overlap
    ngo = report["ngram_overlap"]
    lines.append("")
    lines.append(f"TOKEN {ngo['ngram_n']}-GRAM OVERLAP (test vs train):")
    union_stats = ngo["test_token_ngram_overlap_with_train_union"]
    lines.append(f"  Overlap with train union: mean={union_stats['mean']:.4f}, "
                 f"median={union_stats['median']:.4f}, std={union_stats['std']:.4f}")
    lines.append(f"  % test samples >50% overlap: {union_stats['pct_above_50']:.1f}%")
    lines.append(f"  % test samples >80% overlap: {union_stats['pct_above_80']:.1f}%")

    max_stats = ngo["test_max_jaccard_vs_any_train_sample"]
    lines.append(f"  Max Jaccard vs any train sample: mean={max_stats['mean']:.4f}, "
                 f"median={max_stats['median']:.4f}, max={max_stats['max']:.4f}")
    lines.append(f"  % test with max Jaccard >0.5: {max_stats['pct_above_50']:.1f}%")
    lines.append(f"  % test with max Jaccard >0.8: {max_stats['pct_above_80']:.1f}%")

    if max_stats["pct_above_80"] > 5:
        lines.append("  ** WARNING: >5% of test samples have >0.8 Jaccard with a train sample. **")

    # Benchmark distribution
    bd = report["benchmark_distribution"]
    lines.append("")
    lines.append("BENCHMARK DISTRIBUTION:")
    lines.append(f"  Train: {bd['total_train']} samples across {len(bd['train'])} benchmarks")
    lines.append(f"  Test:  {bd['total_test']} samples across {len(bd['test'])} benchmarks")
    if bd["benchmarks_train_only"]:
        lines.append(f"  Benchmarks in train only: {bd['benchmarks_train_only']}")
    if bd["benchmarks_test_only"]:
        lines.append(f"  Benchmarks in test only: {bd['benchmarks_test_only']}")

    # Overall verdict
    lines.append("")
    lines.append("-" * 70)
    issues = []
    if ed["train_test_overlap_exact_qr"] > 0:
        issues.append(f"{ed['train_test_overlap_exact_qr']} exact train/test duplicates (leakage)")
    if nd["cross_split"] > 0:
        issues.append(f"{nd['cross_split']} near-duplicate cross-split pairs")
    if max_stats["pct_above_80"] > 5:
        issues.append(f"{max_stats['pct_above_80']:.1f}% test samples with >0.8 token Jaccard to train")

    if issues:
        lines.append("VERDICT: POTENTIAL CONTAMINATION DETECTED")
        for issue in issues:
            lines.append(f"  - {issue}")
    else:
        lines.append("VERDICT: NO SIGNIFICANT CONTAMINATION DETECTED")
        lines.append("  The train/test split appears clean.")

    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Check for data contamination and near-duplicates in UQ training/test data."
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/finetune/train_v2.jsonl",
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/finetune/test_v2.jsonl",
        help="Path to test JSONL file",
    )
    parser.add_argument(
        "--scored_dir",
        type=str,
        default="data/use_cases/scored_test_only/",
        help="Path to directory with scored JSONL files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/use_cases/results_test_only/contamination_report.json",
        help="Path to output JSON report",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Jaccard similarity threshold for near-duplicate detection",
    )
    parser.add_argument(
        "--ngram_n",
        type=int,
        default=5,
        help="Character n-gram size for near-duplicate detection",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Process first 100 samples only (quick validation)",
    )
    args = parser.parse_args()

    num_workers = get_num_workers()
    max_samples = 100 if args.smoke_test else None

    print(f"Contamination check configuration:")
    print(f"  Train file:  {args.train_file}")
    print(f"  Test file:   {args.test_file}")
    print(f"  Scored dir:  {args.scored_dir}")
    print(f"  Output:      {args.output}")
    print(f"  Threshold:   {args.threshold}")
    print(f"  N-gram size: {args.ngram_n}")
    print(f"  Workers:     {num_workers}")
    print(f"  Smoke test:  {args.smoke_test}")
    if args.smoke_test:
        print(f"  Max samples: {max_samples}")
    print()

    # Load data
    print("Loading data...")
    t_start = time.time()

    train_samples = load_jsonl(args.train_file, max_samples)
    test_samples = load_jsonl(args.test_file, max_samples)
    print(f"  Loaded {len(train_samples)} train samples, {len(test_samples)} test samples")

    # Extract questions and IDs
    train_questions = [extract_question_text(s) for s in train_samples]
    test_questions = [extract_question_text(s) for s in test_samples]
    train_ids = [s.get("id", f"train_{i}") for i, s in enumerate(train_samples)]
    test_ids = [s.get("id", f"test_{i}") for i, s in enumerate(test_samples)]

    # Run all checks
    report = {}

    # 1. Exact duplicates
    report["exact_duplicates"] = check_exact_duplicates(train_samples, test_samples)

    # 2. Near-duplicate detection
    report["near_duplicates"] = compute_near_duplicates(
        train_questions,
        test_questions,
        train_ids,
        test_ids,
        threshold=args.threshold,
        ngram_n=args.ngram_n,
        num_workers=num_workers,
    )

    # 3. Cross-benchmark contamination
    report["cross_benchmark_contamination"] = check_cross_benchmark_contamination(
        train_samples,
        test_samples,
        num_workers=num_workers,
        ngram_n=args.ngram_n,
        threshold=args.threshold,
    )

    # 4. N-gram overlap statistics
    report["ngram_overlap"] = compute_ngram_overlap_stats(
        train_questions,
        test_questions,
        num_workers=num_workers,
        ngram_n=args.ngram_n,
    )

    # 5. Benchmark distribution
    report["benchmark_distribution"] = compute_benchmark_distribution(
        train_samples, test_samples
    )

    # Generate summary
    summary_text = generate_summary(report)
    report["summary"] = summary_text

    # Save report
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\nReport saved to: {args.output}")
    print(f"Total time: {elapsed:.1f}s")
    print()
    print(summary_text)


if __name__ == "__main__":
    main()
