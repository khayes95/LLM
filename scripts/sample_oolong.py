"""Sample oolong dataset with stratified context lengths.

Usage:
    python scripts/sample_oolong.py --output data/oolong_samples.json
"""
import json
import random
import argparse
from collections import defaultdict
from uq_eval.registry import load_benchmark

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/oolong_sample_ids.json")
    parser.add_argument("--total", type=int, default=250)
    parser.add_argument("--max_context_len", type=int, default=131072, 
                        help="Max context length to include (default 128K)")
    args = parser.parse_args()
    
    bench = load_benchmark("oolong")
    examples = list(bench.iter_examples("test"))
    
    # Group by context_len
    by_length = defaultdict(list)
    for ex in examples:
        ctx_len = ex.meta.get('context_len', 0)
        if ctx_len <= args.max_context_len:
            by_length[ctx_len].append(ex)
    
    print(f"Context lengths <= {args.max_context_len}:")
    for ctx_len in sorted(by_length.keys()):
        print(f"  {ctx_len}: {len(by_length[ctx_len])} examples")
    
    # Calculate samples per length category
    n_categories = len(by_length)
    base_per_cat = args.total // n_categories
    remainder = args.total % n_categories
    
    print(f"\nSampling {base_per_cat} per category, {remainder} extra")
    
    # Sample with two seeds
    # First 100 samples: seed 42
    # Next 150 samples: seed 43
    
    all_sample_ids = []
    
    # Phase 1: seed 42 for first 100
    random.seed(42)
    phase1_target = 100
    phase1_per_cat = phase1_target // n_categories
    
    for ctx_len in sorted(by_length.keys()):
        pool = by_length[ctx_len]
        random.shuffle(pool)
        sampled = pool[:phase1_per_cat]
        all_sample_ids.extend([ex.id for ex in sampled])
        # Remove sampled from pool for phase 2
        by_length[ctx_len] = pool[phase1_per_cat:]
    
    print(f"Phase 1 (seed 42): {len(all_sample_ids)} samples")
    
    # Phase 2: seed 43 for next 150
    random.seed(43)
    phase2_target = args.total - len(all_sample_ids)
    phase2_per_cat = phase2_target // n_categories
    extra = phase2_target % n_categories
    
    for i, ctx_len in enumerate(sorted(by_length.keys())):
        pool = by_length[ctx_len]
        random.shuffle(pool)
        n_sample = phase2_per_cat + (1 if i < extra else 0)
        sampled = pool[:n_sample]
        all_sample_ids.extend([ex.id for ex in sampled])
    
    print(f"Phase 2 (seed 43): {len(all_sample_ids) - phase1_target} samples")
    print(f"Total: {len(all_sample_ids)} samples")
    
    # Save
    with open(args.output, 'w') as f:
        json.dump(all_sample_ids, f, indent=2)
    
    print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()
