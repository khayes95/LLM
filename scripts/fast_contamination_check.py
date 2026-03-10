#!/usr/bin/env python3
"""Fast cross-split contamination check for v3 data.

Only checks cross-split (train→test) near-duplicates, skipping the expensive
train-internal O(n^2) computation. This is what matters for paper integrity.

Usage:
    python scripts/fast_contamination_check.py
"""
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path


def char_ngrams(text, n=5):
    """Extract character n-grams from text."""
    text = text.lower().strip()
    if len(text) < n:
        return set()
    return set(text[i:i+n] for i in range(len(text) - n + 1))


def jaccard(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def extract_question(sample):
    """Extract question text from sample."""
    for key in ["input", "question", "query"]:
        if key in sample and sample[key]:
            val = sample[key]
            if isinstance(val, str):
                return val
            if isinstance(val, dict):
                return json.dumps(val)[:2000]
    return ""


def compute_ngrams_worker(args):
    idx, text, n = args
    return idx, char_ngrams(text, n)


def main():
    train_file = "data/finetune/train_v3.jsonl"
    test_file = "data/finetune/test_v3.jsonl"
    output_file = "data/use_cases/results_test_only_v3/contamination_report_v3.json"
    threshold = 0.8
    ngram_n = 5
    num_workers = min(cpu_count() or 4, 32)  # Don't use too many workers

    print(f"=== Fast Cross-Split Contamination Check (v3) ===")
    print(f"Workers: {num_workers}, Threshold: {threshold}")

    # Load data
    t0 = time.time()
    train_samples = [json.loads(l) for l in open(train_file)]
    test_samples = [json.loads(l) for l in open(test_file)]
    print(f"Loaded {len(train_samples)} train, {len(test_samples)} test samples ({time.time()-t0:.1f}s)")

    train_questions = [extract_question(s) for s in train_samples]
    test_questions = [extract_question(s) for s in test_samples]
    train_ids = [s.get("id", f"train_{i}") for i, s in enumerate(train_samples)]
    test_ids = [s.get("id", f"test_{i}") for i, s in enumerate(test_samples)]

    # 1. Exact duplicate check
    print("\n[1/3] Checking exact duplicates...")
    train_q_set = set(train_questions)
    exact_overlap_q = sum(1 for q in test_questions if q in train_q_set)
    print(f"  Exact question overlap: {exact_overlap_q} / {len(test_questions)} ({exact_overlap_q/len(test_questions)*100:.1f}%)")

    # Exact ID overlap
    train_id_set = set(train_ids)
    exact_id_overlap = sum(1 for tid in test_ids if tid in train_id_set)
    print(f"  Exact ID overlap: {exact_id_overlap} / {len(test_ids)}")

    # 2. Compute n-grams
    print(f"\n[2/3] Computing char {ngram_n}-grams...")
    t1 = time.time()
    all_questions = train_questions + test_questions
    args = [(i, q, ngram_n) for i, q in enumerate(all_questions)]

    with Pool(num_workers) as pool:
        results = pool.map(compute_ngrams_worker, args)

    ngram_map = {}
    for idx, ng in results:
        ngram_map[idx] = ng
    print(f"  Computed in {time.time()-t1:.1f}s")

    train_ngrams = [ngram_map[i] for i in range(len(train_questions))]
    test_ngrams = [ngram_map[i + len(train_questions)] for i in range(len(test_questions))]

    # 3. Cross-split near-duplicate detection
    print(f"\n[3/3] Computing cross-split Jaccard similarities...")
    t2 = time.time()

    cross_pairs = []
    per_test_max_jaccard = []

    for ti, (tng, tq) in enumerate(zip(test_ngrams, test_questions)):
        if ti % 200 == 0:
            print(f"  Test sample {ti}/{len(test_ngrams)}...")

        max_j = 0.0
        max_train_idx = -1
        for tri, trng in enumerate(train_ngrams):
            j = jaccard(tng, trng)
            if j > max_j:
                max_j = j
                max_train_idx = tri
            if j >= threshold:
                cross_pairs.append({
                    "test_id": test_ids[ti],
                    "train_id": train_ids[tri],
                    "test_benchmark": test_samples[ti].get("benchmark", ""),
                    "train_benchmark": train_samples[tri].get("benchmark", ""),
                    "jaccard": round(j, 4),
                })

        per_test_max_jaccard.append({
            "test_id": test_ids[ti],
            "benchmark": test_samples[ti].get("benchmark", ""),
            "max_jaccard": round(max_j, 4),
            "most_similar_train_id": train_ids[max_train_idx] if max_train_idx >= 0 else "",
        })

    elapsed = time.time() - t2
    print(f"  Computed in {elapsed:.1f}s")

    # Analyze results
    n_contaminated = sum(1 for p in per_test_max_jaccard if p["max_jaccard"] >= threshold)
    pct_contaminated = n_contaminated / len(test_samples) * 100

    # Per-benchmark breakdown
    bench_contamination = defaultdict(lambda: {"total": 0, "contaminated": 0})
    for p in per_test_max_jaccard:
        b = p["benchmark"]
        bench_contamination[b]["total"] += 1
        if p["max_jaccard"] >= threshold:
            bench_contamination[b]["contaminated"] += 1

    report = {
        "n_train": len(train_samples),
        "n_test": len(test_samples),
        "threshold": threshold,
        "ngram_n": ngram_n,
        "exact_question_overlap": exact_overlap_q,
        "exact_id_overlap": exact_id_overlap,
        "n_cross_split_near_duplicates": len(cross_pairs),
        "n_contaminated_test_samples": n_contaminated,
        "pct_contaminated": round(pct_contaminated, 2),
        "cross_split_pairs": cross_pairs[:500],  # Cap at 500 for file size
        "per_test_max_jaccard": per_test_max_jaccard,
        "per_benchmark_contamination": {
            b: {
                "total": d["total"],
                "contaminated": d["contaminated"],
                "pct": round(d["contaminated"] / d["total"] * 100, 1) if d["total"] > 0 else 0,
            }
            for b, d in sorted(bench_contamination.items())
        },
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"CROSS-SPLIT CONTAMINATION SUMMARY (Jaccard >= {threshold})")
    print(f"{'='*60}")
    print(f"  Near-duplicate pairs: {len(cross_pairs)}")
    print(f"  Contaminated test samples: {n_contaminated} / {len(test_samples)} ({pct_contaminated:.1f}%)")
    print(f"  Exact question overlap: {exact_overlap_q}")
    print(f"  Exact ID overlap: {exact_id_overlap}")
    print(f"\n  Per-benchmark:")
    for b, d in sorted(bench_contamination.items()):
        if d["contaminated"] > 0:
            print(f"    {b}: {d['contaminated']}/{d['total']} ({d['contaminated']/d['total']*100:.1f}%)")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
