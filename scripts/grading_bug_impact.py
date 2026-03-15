#!/usr/bin/env python3
"""Impact analysis of 4 grading bugs on training labels.

READ-ONLY analysis — does not modify any files.
Loads all prediction JSONL files used for training, re-grades with
both buggy and fixed logic, and reports label flips.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# ============================================================
# Config (mirrors train_best_uq.py)
# ============================================================
EXCLUDED = {
    "triviaqa", "babilong", "vsr", "aokvqa", "erqa", "tutorbench",
    "healthbench", "arc", "oolong",
}

DATA_SOURCES = {
    "gpt5mini": {"dir": "runs/gpt5_mini_combined", "type": "combined"},
    "gpt52": {"dir": "runs", "prefix": "gpt52_high_", "type": "prefixed"},
    "qwen35": {"dir": "runs", "prefix": "qwen35_397b_", "type": "prefixed"},
}


# ============================================================
# Grading functions — buggy vs fixed
# ============================================================

def normalize_answer_simpleqa(s):
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def normalize_answer_hle(s):
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def simpleqa_buggy(predicted, gold):
    """Bug 1: word-subset matching. gold_words.issubset(pred_words)"""
    pred_norm = normalize_answer_simpleqa(predicted)
    gold_norm = normalize_answer_simpleqa(gold)
    if pred_norm == gold_norm:
        return True
    if gold_norm in pred_norm:
        return True
    gold_words = set(gold_norm.split())
    pred_words = set(pred_norm.split())
    if gold_words and gold_words.issubset(pred_words):
        return True
    return False


def simpleqa_fixed(predicted, gold):
    """Fixed: remove word-subset matching (only exact match + containment)."""
    pred_norm = normalize_answer_simpleqa(predicted)
    gold_norm = normalize_answer_simpleqa(gold)
    if pred_norm == gold_norm:
        return True
    if gold_norm in pred_norm:
        return True
    # Removed: gold_words.issubset(pred_words)
    return False


def hle_short_answer_buggy(predicted, gold):
    """Bug 1 variant in HLE: same word-subset matching."""
    pred_norm = normalize_answer_hle(predicted)
    gold_norm = normalize_answer_hle(gold)
    if pred_norm == gold_norm:
        return True
    if gold_norm and gold_norm in pred_norm:
        return True
    gold_words = set(gold_norm.split())
    pred_words = set(pred_norm.split())
    if gold_words and gold_words.issubset(pred_words):
        return True
    return False


def hle_short_answer_fixed(predicted, gold):
    """Fixed: remove word-subset matching."""
    pred_norm = normalize_answer_hle(predicted)
    gold_norm = normalize_answer_hle(gold)
    if pred_norm == gold_norm:
        return True
    if gold_norm and gold_norm in pred_norm:
        return True
    return False


def hle_mcq_buggy(predicted, gold):
    """HLE MCQ uses check_mcq_answer — has its own bug (extracts first letter after removing non-letters)."""
    pred_clean = predicted.strip().upper()
    gold_clean = gold.strip().upper()
    if pred_clean == gold_clean:
        return True
    pred_letter = re.sub(r"[^A-Z]", "", pred_clean)
    gold_letter = re.sub(r"[^A-Z]", "", gold_clean)
    if pred_letter and gold_letter and pred_letter[0] == gold_letter[0]:
        return True
    return False


def hle_mcq_fixed(predicted, gold):
    """Fixed: extract LAST standalone letter, not first letter after stripping."""
    pred_clean = predicted.strip().upper()
    gold_clean = gold.strip().upper()
    if pred_clean == gold_clean:
        return True
    # Use standalone letter matching (like common.py should)
    gold_letters = re.findall(r'\b([A-Z])\b', gold_clean)
    pred_letters = re.findall(r'\b([A-Z])\b', pred_clean)
    if gold_letters and pred_letters:
        return gold_letters[-1] == pred_letters[-1]
    return False


def normalize_text(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())


def mmmu_score_buggy(got, gold):
    """Bug 2: extracts first letter after removing ALL non-letters.
    'The best answer is B' -> 'THEBESTANSWERISB' -> 'T'
    """
    gold_upper = gold.strip().upper()
    got_upper = got.strip().upper()
    gold_letter = re.sub(r"[^A-Z]", "", gold_upper)[:1]
    got_letter = re.sub(r"[^A-Z]", "", got_upper)[:1]
    return int(gold_letter == got_letter) if gold_letter else 0


def mmmu_score_fixed(got, gold):
    """Fixed: look for standalone letter or last letter in 'answer is X' pattern."""
    gold_upper = gold.strip().upper()
    got_upper = got.strip().upper()

    # Gold should be a single letter
    gold_letter = re.sub(r"[^A-Z]", "", gold_upper)[:1]
    if not gold_letter:
        return 0

    # Try standalone letter match (last one, to handle "The answer is B")
    standalone = re.findall(r'\b([A-Z])\b', got_upper)
    if standalone:
        return int(standalone[-1] == gold_letter)

    # Fallback: first letter after stripping (but only if got is very short, like "B" or "(B)")
    got_letter = re.sub(r"[^A-Z]", "", got_upper)[:1]
    if got_letter and len(got_upper) <= 5:
        return int(got_letter == gold_letter)

    return 0


def livebench_score_buggy(got, gold):
    """Bug 3: only checks gold_norm in got_norm (asymmetric)."""
    gold_norm = re.sub(r"\s+", " ", gold.lower())
    got_norm = re.sub(r"\s+", " ", got.lower())
    return int(gold_norm == got_norm or gold_norm in got_norm)


def livebench_score_fixed(got, gold):
    """Fixed: check both directions."""
    gold_norm = re.sub(r"\s+", " ", gold.lower())
    got_norm = re.sub(r"\s+", " ", got.lower())
    return int(gold_norm == got_norm or gold_norm in got_norm or got_norm in gold_norm)


def extract_choice_letter_buggy(text, choices="ABCD"):
    """Bug 4: regex [A-D] misses E-J."""
    if not text:
        return None
    m = re.findall(r"\b([A-D])\b", text.upper())
    if m:
        c = m[-1]
        return c if c in choices else None
    return None


def extract_choice_letter_fixed(text, choices="ABCDEFGHIJ"):
    """Fixed: regex [A-J] to cover more options."""
    if not text:
        return None
    m = re.findall(r"\b([A-J])\b", text.upper())
    if m:
        c = m[-1]
        return c if c in choices else None
    return None


# ============================================================
# Load all prediction files
# ============================================================

def load_predictions(base_dir):
    """Load all prediction files matching training script logic."""
    all_preds = []  # list of (benchmark, source_model, pred_dict)

    for model_name, config in DATA_SOURCES.items():
        if config["type"] == "combined":
            combined_path = Path(base_dir) / config["dir"]
            for bench_dir in sorted(combined_path.iterdir()):
                if not bench_dir.is_dir() or bench_dir.name in EXCLUDED:
                    continue
                pred_file = bench_dir / "predictions.jsonl"
                if not pred_file.exists():
                    continue
                for line in open(pred_file):
                    try:
                        pred = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    score = pred.get("score", {})
                    if isinstance(score, dict):
                        correct = score.get("correct", -1)
                    else:
                        correct = score
                    if correct not in (0, 1):
                        continue
                    all_preds.append((bench_dir.name, model_name, pred))
        else:
            runs_path = Path(base_dir) / config["dir"]
            prefix = config["prefix"]
            for run_dir in sorted(runs_path.iterdir()):
                if not run_dir.name.startswith(prefix):
                    continue
                benchmark = run_dir.name[len(prefix):]
                if benchmark in EXCLUDED:
                    continue
                pred_file = run_dir / "predictions.jsonl"
                if not pred_file.exists():
                    continue
                for line in open(pred_file):
                    try:
                        pred = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    score = pred.get("score", {})
                    if isinstance(score, dict):
                        correct = score.get("correct", -1)
                    else:
                        correct = score
                    if correct not in (0, 1):
                        continue
                    all_preds.append((benchmark, model_name, pred))

    return all_preds


def get_score_fields(pred):
    """Extract gold, predicted, and other score fields."""
    score = pred.get("score", {})
    if isinstance(score, dict):
        return score
    return {"correct": score}


def get_response_text(pred):
    """Get the model's response text."""
    return pred.get("response_text", "") or str(pred.get("prediction", {}).get("answer", ""))


def main():
    base_dir = Path("/scratch/khayes/LLM")
    all_preds = load_predictions(base_dir)
    print(f"Loaded {len(all_preds)} total training predictions\n")

    # Group by benchmark
    by_bench = defaultdict(list)
    for bench, model, pred in all_preds:
        by_bench[bench].append((model, pred))

    # ============================================================
    # Bug 1: SimpleQA & HLE word-subset matching
    # ============================================================
    print("=" * 70)
    print("BUG 1: SimpleQA & HLE word-subset matching")
    print("=" * 70)

    bug1_flips = []

    # SimpleQA
    for model, pred in by_bench.get("simpleqa", []):
        score_data = get_score_fields(pred)
        gold = score_data.get("gold", str(pred.get("target", "")))
        predicted = score_data.get("predicted", get_response_text(pred))
        if not gold or not predicted:
            continue

        buggy_result = simpleqa_buggy(predicted, gold)
        fixed_result = simpleqa_fixed(predicted, gold)

        if buggy_result != fixed_result:
            bug1_flips.append({
                "benchmark": "simpleqa",
                "model": model,
                "id": pred.get("id", ""),
                "gold": gold[:80],
                "predicted": predicted[:120],
                "buggy": buggy_result,
                "fixed": fixed_result,
                "direction": "correct->incorrect" if buggy_result else "incorrect->correct",
            })

    # HLE (short answer only — MCQ uses check_mcq_answer, handled separately)
    for model, pred in by_bench.get("hle", []) + by_bench.get("hle_multimodal", []):
        score_data = get_score_fields(pred)
        gold = score_data.get("gold", str(pred.get("target", "")))
        predicted = score_data.get("predicted", get_response_text(pred))
        answer_type = score_data.get("answer_type", pred.get("meta", {}).get("answer_type", ""))
        if not gold or not predicted:
            continue

        bench_name = "hle" if (model, pred) in by_bench.get("hle", []) else "hle_multimodal"

        # Only apply to non-MCQ (short answer) questions
        if answer_type == "mcq":
            continue

        buggy_result = hle_short_answer_buggy(predicted, gold)
        fixed_result = hle_short_answer_fixed(predicted, gold)

        if buggy_result != fixed_result:
            bug1_flips.append({
                "benchmark": bench_name,
                "model": model,
                "id": pred.get("id", ""),
                "gold": gold[:80],
                "predicted": predicted[:120],
                "buggy": buggy_result,
                "fixed": fixed_result,
                "direction": "correct->incorrect" if buggy_result else "incorrect->correct",
            })

    # Count
    bug1_correct_to_incorrect = sum(1 for f in bug1_flips if f["direction"] == "correct->incorrect")
    bug1_incorrect_to_correct = sum(1 for f in bug1_flips if f["direction"] == "incorrect->correct")
    total_simpleqa_hle = len(by_bench.get("simpleqa", [])) + len(by_bench.get("hle", [])) + len(by_bench.get("hle_multimodal", []))

    print(f"\nAffected benchmarks: simpleqa ({len(by_bench.get('simpleqa', []))}), "
          f"hle ({len(by_bench.get('hle', []))}), hle_multimodal ({len(by_bench.get('hle_multimodal', []))})")
    print(f"Total samples checked: {total_simpleqa_hle}")
    print(f"Total flips: {len(bug1_flips)}")
    print(f"  correct->incorrect: {bug1_correct_to_incorrect}")
    print(f"  incorrect->correct: {bug1_incorrect_to_correct}")
    print(f"  % of affected benchmarks: {len(bug1_flips)/max(1,total_simpleqa_hle)*100:.2f}%")
    print(f"  % of total training data: {len(bug1_flips)/len(all_preds)*100:.3f}%")

    if bug1_flips:
        print(f"\nExample flips (up to 5):")
        for f in bug1_flips[:5]:
            print(f"  [{f['benchmark']}/{f['model']}] {f['direction']}")
            print(f"    Gold: {f['gold']}")
            print(f"    Pred: {f['predicted']}")
            print()

    # ============================================================
    # Bug 2: MMMU letter extraction
    # ============================================================
    print("=" * 70)
    print("BUG 2: MMMU letter extraction (first letter after stripping)")
    print("=" * 70)

    bug2_flips = []

    for model, pred in by_bench.get("mmmu", []):
        score_data = get_score_fields(pred)
        gold = score_data.get("gold", str(pred.get("target", "")))
        predicted = score_data.get("predicted", "")
        if not predicted:
            # Try prediction field
            prediction_obj = pred.get("prediction", {})
            if isinstance(prediction_obj, dict):
                predicted = str(prediction_obj.get("answer", ""))
            else:
                predicted = str(prediction_obj)
        if not gold or not predicted:
            continue

        buggy_result = mmmu_score_buggy(predicted, gold)
        fixed_result = mmmu_score_fixed(predicted, gold)

        if buggy_result != fixed_result:
            bug2_flips.append({
                "benchmark": "mmmu",
                "model": model,
                "id": pred.get("id", ""),
                "gold": gold,
                "predicted": predicted[:120],
                "buggy": buggy_result,
                "fixed": fixed_result,
                "direction": "correct->incorrect" if buggy_result else "incorrect->correct",
            })

    total_mmmu = len(by_bench.get("mmmu", []))
    bug2_correct_to_incorrect = sum(1 for f in bug2_flips if f["direction"] == "correct->incorrect")
    bug2_incorrect_to_correct = sum(1 for f in bug2_flips if f["direction"] == "incorrect->correct")

    print(f"\nAffected benchmarks: mmmu ({total_mmmu})")
    print(f"Total flips: {len(bug2_flips)}")
    print(f"  correct->incorrect: {bug2_correct_to_incorrect}")
    print(f"  incorrect->correct: {bug2_incorrect_to_correct}")
    print(f"  % of mmmu samples: {len(bug2_flips)/max(1,total_mmmu)*100:.2f}%")
    print(f"  % of total training data: {len(bug2_flips)/len(all_preds)*100:.3f}%")

    if bug2_flips:
        print(f"\nExample flips (up to 5):")
        for f in bug2_flips[:5]:
            print(f"  [{f['model']}] {f['direction']}")
            print(f"    Gold: {f['gold']}")
            print(f"    Pred: {f['predicted']}")
            print()

    # ============================================================
    # Bug 3: LiveBench asymmetric matching
    # ============================================================
    print("=" * 70)
    print("BUG 3: LiveBench asymmetric matching (gold_norm in got_norm only)")
    print("=" * 70)

    bug3_flips = []

    for model, pred in by_bench.get("livebench", []):
        score_data = get_score_fields(pred)
        gold = score_data.get("gold", str(pred.get("target", "")))
        predicted = score_data.get("predicted", get_response_text(pred))
        if not gold or not predicted:
            continue

        buggy_result = livebench_score_buggy(predicted, gold)
        fixed_result = livebench_score_fixed(predicted, gold)

        if buggy_result != fixed_result:
            bug3_flips.append({
                "benchmark": "livebench",
                "model": model,
                "id": pred.get("id", ""),
                "gold": gold[:120],
                "predicted": predicted[:120],
                "buggy": buggy_result,
                "fixed": fixed_result,
                "direction": "correct->incorrect" if buggy_result else "incorrect->correct",
            })

    total_livebench = len(by_bench.get("livebench", []))
    bug3_correct_to_incorrect = sum(1 for f in bug3_flips if f["direction"] == "correct->incorrect")
    bug3_incorrect_to_correct = sum(1 for f in bug3_flips if f["direction"] == "incorrect->correct")

    print(f"\nAffected benchmarks: livebench ({total_livebench})")
    print(f"Total flips: {len(bug3_flips)}")
    print(f"  correct->incorrect: {bug3_correct_to_incorrect}")
    print(f"  incorrect->correct: {bug3_incorrect_to_correct}")
    print(f"  % of livebench samples: {len(bug3_flips)/max(1,total_livebench)*100:.2f}%")
    print(f"  % of total training data: {len(bug3_flips)/len(all_preds)*100:.3f}%")

    if bug3_flips:
        print(f"\nExample flips (up to 5):")
        for f in bug3_flips[:5]:
            print(f"  [{f['model']}] {f['direction']}")
            print(f"    Gold: {f['gold']}")
            print(f"    Pred: {f['predicted']}")
            print()

    # ============================================================
    # Bug 4: common.py extract_choice_letter [A-D] limit
    # ============================================================
    print("=" * 70)
    print("BUG 4: common.py extract_choice_letter [A-D] misses E-J")
    print("=" * 70)
    print("\nNote: This bug affects the PARSE step (extract_choice_letter), not the SCORE step.")
    print("It matters when the JSON parse fails and the fallback regex is used.")
    print("We check all MCQ benchmarks for predictions with answer letters E-J.")

    # Benchmarks that use extract_choice_letter as fallback:
    # chembench, gpqa, mmmu, mmstar, mathvista, realworldqa, mathverse
    # But erqa and aokvqa are EXCLUDED
    mcq_benchmarks_with_fallback = ["chembench", "gpqa", "mmmu", "mmstar", "mathvista", "realworldqa", "mathverse"]

    bug4_flips = []
    bug4_details = defaultdict(int)

    for bench_name in mcq_benchmarks_with_fallback:
        for model, pred in by_bench.get(bench_name, []):
            score_data = get_score_fields(pred)
            original_correct = score_data.get("correct", -1)
            if original_correct not in (0, 1):
                continue

            gold = score_data.get("gold", str(pred.get("target", "")))
            predicted = score_data.get("predicted", "")

            # Get the raw response text
            raw_text = pred.get("response_text", "")
            prediction_obj = pred.get("prediction", {})
            if isinstance(prediction_obj, dict):
                parsed_answer = prediction_obj.get("answer", "")
            else:
                parsed_answer = str(prediction_obj)

            # Check if the gold answer contains letters E-J
            gold_upper = gold.strip().upper() if gold else ""
            gold_letter = re.sub(r"[^A-Z]", "", gold_upper)[:1] if gold_upper else ""

            # Only relevant if gold or prediction has letters E-J
            if gold_letter not in "EFGHIJ":
                # Even if gold is A-D, the prediction might have been E-J and got wrongly parsed
                # But this would only matter if it should have matched — skip for now
                # Actually, let's check if the prediction contains E-J that got missed
                pass

            # The bug is in parse_prediction's fallback path.
            # If JSON parsing succeeded and answer was extracted, the bug doesn't apply.
            # Let's check if the prediction's parsed answer differs from what
            # extract_choice_letter would give
            if raw_text:
                buggy_extracted = extract_choice_letter_buggy(raw_text)
                fixed_extracted = extract_choice_letter_fixed(raw_text)

                if buggy_extracted != fixed_extracted:
                    # The fallback would have produced different answers
                    # But did JSON parsing succeed?
                    extra = pred.get("prediction", {})
                    if isinstance(extra, dict):
                        json_parsed = extra.get("extra", {}).get("parsed_json", None)
                        # If extra doesn't have this info, check if answer looks like
                        # it came from JSON
                        if json_parsed is None:
                            # Heuristic: if answer is a single letter, probably from JSON
                            ans = extra.get("answer", "")
                            if isinstance(ans, str) and len(ans.strip()) <= 3:
                                json_parsed = True  # likely came from JSON

                    # For predictions where JSON parse failed, the fallback was used
                    # and the bug would have affected the result
                    # We can't be 100% sure without replaying, but we can check
                    # if the stored predicted answer matches what buggy extraction gives
                    if predicted and buggy_extracted and predicted.strip().upper() == buggy_extracted:
                        # The buggy extraction was likely used
                        # What would fixed extraction give?
                        if fixed_extracted and fixed_extracted != buggy_extracted:
                            # Would the score change?
                            if bench_name == "mmmu":
                                buggy_score = mmmu_score_buggy(buggy_extracted, gold)
                                fixed_score = mmmu_score_fixed(fixed_extracted, gold)
                            else:
                                buggy_score = int(buggy_extracted == gold_letter)
                                fixed_score = int(fixed_extracted == gold_letter)

                            if buggy_score != fixed_score:
                                bug4_flips.append({
                                    "benchmark": bench_name,
                                    "model": model,
                                    "id": pred.get("id", ""),
                                    "gold": gold,
                                    "predicted": predicted,
                                    "buggy_extract": buggy_extracted,
                                    "fixed_extract": fixed_extracted,
                                    "buggy_score": buggy_score,
                                    "fixed_score": fixed_score,
                                    "direction": "correct->incorrect" if buggy_score else "incorrect->correct",
                                })
                                bug4_details[bench_name] += 1

            # Also check: gold answer is E-J but predicted was scored
            # by the score() function which uses [A-D] or similar
            if gold_letter in "EFGHIJ" and bench_name in ("chembench",):
                # ChemBench maps letter to choice text, so the scoring doesn't
                # directly use extract_choice_letter — it maps letter_idx to choices
                # The scoring should work for E-H since it uses ord() arithmetic
                pass

    # More direct approach for Bug 4: find predictions where gold is E-J
    # and check if the score might be wrong
    print(f"\nDirect check: predictions with gold answer E-J across MCQ benchmarks:")
    ej_counts = defaultdict(lambda: {"total": 0, "correct": 0, "incorrect": 0})
    for bench_name in mcq_benchmarks_with_fallback:
        for model, pred in by_bench.get(bench_name, []):
            score_data = get_score_fields(pred)
            gold = score_data.get("gold", str(pred.get("target", "")))
            predicted = score_data.get("predicted", "")
            correct = score_data.get("correct", -1)
            if correct not in (0, 1):
                continue
            gold_upper = gold.strip().upper() if gold else ""
            gold_letter = re.sub(r"[^A-Z]", "", gold_upper)[:1] if gold_upper else ""
            if gold_letter in "EFGHIJ":
                ej_counts[bench_name]["total"] += 1
                if correct == 1:
                    ej_counts[bench_name]["correct"] += 1
                else:
                    ej_counts[bench_name]["incorrect"] += 1

    for bench in sorted(ej_counts):
        c = ej_counts[bench]
        print(f"  {bench}: {c['total']} samples with gold E-J, "
              f"{c['correct']} scored correct, {c['incorrect']} scored incorrect")

    total_affected_benchmarks = sum(len(by_bench.get(b, [])) for b in mcq_benchmarks_with_fallback)
    print(f"\nFallback-path flips detected: {len(bug4_flips)}")
    for bench in sorted(bug4_details):
        print(f"  {bench}: {bug4_details[bench]}")

    bug4_correct_to_incorrect = sum(1 for f in bug4_flips if f["direction"] == "correct->incorrect")
    bug4_incorrect_to_correct = sum(1 for f in bug4_flips if f["direction"] == "incorrect->correct")
    print(f"  correct->incorrect: {bug4_correct_to_incorrect}")
    print(f"  incorrect->correct: {bug4_incorrect_to_correct}")
    print(f"  % of total training data: {len(bug4_flips)/len(all_preds)*100:.3f}%")

    if bug4_flips:
        print(f"\nExample flips (up to 5):")
        for f in bug4_flips[:5]:
            print(f"  [{f['benchmark']}/{f['model']}] {f['direction']}")
            print(f"    Gold: {f['gold']}, Buggy extract: {f['buggy_extract']}, Fixed extract: {f['fixed_extract']}")
            print()

    # ============================================================
    # SUMMARY
    # ============================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_flips = len(bug1_flips) + len(bug2_flips) + len(bug3_flips) + len(bug4_flips)
    print(f"\nTotal training samples: {len(all_preds)}")
    print(f"Total label flips across all 4 bugs: {total_flips}")
    print(f"Overall flip rate: {total_flips/len(all_preds)*100:.2f}%")
    print()
    print(f"Bug 1 (SimpleQA/HLE word-subset):   {len(bug1_flips):4d} flips ({len(bug1_flips)/len(all_preds)*100:.2f}%)")
    print(f"Bug 2 (MMMU letter extraction):      {len(bug2_flips):4d} flips ({len(bug2_flips)/len(all_preds)*100:.2f}%)")
    print(f"Bug 3 (LiveBench asymmetric):        {len(bug3_flips):4d} flips ({len(bug3_flips)/len(all_preds)*100:.2f}%)")
    print(f"Bug 4 (common.py A-D limit):         {len(bug4_flips):4d} flips ({len(bug4_flips)/len(all_preds)*100:.2f}%)")
    print()

    total_c2i = bug1_correct_to_incorrect + bug2_correct_to_incorrect + bug3_correct_to_incorrect + bug4_correct_to_incorrect
    total_i2c = bug1_incorrect_to_correct + bug2_incorrect_to_correct + bug3_incorrect_to_correct + bug4_incorrect_to_correct
    print(f"Direction of flips:")
    print(f"  correct->incorrect (false positives removed): {total_c2i}")
    print(f"  incorrect->correct (false negatives fixed):   {total_i2c}")

    # Save detailed results
    results = {
        "total_training_samples": len(all_preds),
        "total_flips": total_flips,
        "flip_rate_pct": round(total_flips / len(all_preds) * 100, 3),
        "bug1_flips": len(bug1_flips),
        "bug2_flips": len(bug2_flips),
        "bug3_flips": len(bug3_flips),
        "bug4_flips": len(bug4_flips),
        "correct_to_incorrect": total_c2i,
        "incorrect_to_correct": total_i2c,
        "bug1_examples": bug1_flips[:10],
        "bug2_examples": bug2_flips[:10],
        "bug3_examples": bug3_flips[:10],
        "bug4_examples": bug4_flips[:10],
    }

    out_path = Path("/scratch/khayes/LLM/data/grading_bug_impact.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {out_path}")


if __name__ == "__main__":
    main()
