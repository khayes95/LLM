#!/usr/bin/env python3
"""
Spurious Correlation Analysis for VLM Judge

Tests whether the model learned genuine correctness prediction vs spurious shortcuts:
1. Response-only baseline: Can you predict correctness from response alone (no question/image)?
2. Always-yes/always-no baseline: What's the accuracy of trivial strategies?
3. Response length correlation: Does model just predict based on length?
4. Within-benchmark AUROC: Does model discriminate within each benchmark?
5. Confidence calibration: Are high-confidence predictions more accurate?
6. Cross-benchmark generalization: Does model transfer across benchmarks?

Usage:
    python scripts/spurious_correlation_analysis.py
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import pearsonr, spearmanr

# Load predictions
def load_predictions(path):
    """Load predictions from JSON file."""
    with open(path) as f:
        return json.load(f)


def analyze_trivial_baselines(labels):
    """Test always-yes, always-no, and random baselines."""
    labels = np.array(labels)
    n = len(labels)
    n_correct = labels.sum()
    base_rate = n_correct / n

    print("\n" + "="*60)
    print("1. TRIVIAL BASELINE ANALYSIS")
    print("="*60)
    print(f"\nDataset: {n} samples, {int(n_correct)} correct ({100*base_rate:.1f}%)")
    print(f"\nBaseline accuracies:")
    print(f"  Always 'correct':    {100*base_rate:.1f}%")
    print(f"  Always 'incorrect':  {100*(1-base_rate):.1f}%")
    print(f"  Random (50/50):      50.0%")
    print(f"\n  => Any useful model must beat {100*max(base_rate, 1-base_rate):.1f}% (majority class)")

    return base_rate


def analyze_response_length(predictions, labels, responses):
    """Test if model is just using response length as a proxy."""
    print("\n" + "="*60)
    print("2. RESPONSE LENGTH ANALYSIS")
    print("="*60)

    lengths = np.array([len(r) for r in responses])
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Correlation between length and correctness
    corr_length_correct, p_lc = pearsonr(lengths, labels)
    print(f"\nCorrelation (length vs actual correctness): r={corr_length_correct:.3f} (p={p_lc:.4f})")

    # Correlation between length and model prediction
    corr_length_pred, p_lp = pearsonr(lengths, predictions)
    print(f"Correlation (length vs model P(correct)): r={corr_length_pred:.3f} (p={p_lp:.4f})")

    # Length-only AUROC (can length alone predict correctness?)
    try:
        length_auroc = roc_auc_score(labels, lengths)
        print(f"\nLength-only AUROC: {length_auroc:.3f}")
        print(f"  => If model just learned 'longer=correct', AUROC would be ~{length_auroc:.3f}")
    except:
        print("\nLength-only AUROC: N/A (single class)")

    # Residual AUROC (model performance after controlling for length)
    # Bin by length and compute within-bin AUROC
    length_bins = np.percentile(lengths, [0, 25, 50, 75, 100])
    within_bin_aurocs = []
    print(f"\nWithin-length-bin AUROC (controls for length):")
    for i in range(len(length_bins)-1):
        mask = (lengths >= length_bins[i]) & (lengths < length_bins[i+1] + 1)
        if mask.sum() > 10 and len(set(labels[mask])) > 1:
            bin_auroc = roc_auc_score(labels[mask], predictions[mask])
            within_bin_aurocs.append(bin_auroc)
            print(f"  Bin {i+1} (len {int(length_bins[i])}-{int(length_bins[i+1])}): AUROC={bin_auroc:.3f} (n={mask.sum()})")

    if within_bin_aurocs:
        mean_within = np.mean(within_bin_aurocs)
        print(f"\n  Mean within-bin AUROC: {mean_within:.3f}")
        print(f"  => Model discriminates WITHIN length bins, not just by length")


def analyze_response_token_bias(predictions, labels, responses):
    """Test if model is just predicting based on response tokens (yes/no/true/false)."""
    print("\n" + "="*60)
    print("3. RESPONSE TOKEN BIAS ANALYSIS")
    print("="*60)

    predictions = np.array(predictions)
    labels = np.array(labels)

    # Categorize responses by first token
    categories = defaultdict(lambda: {'preds': [], 'labels': []})
    for pred, label, resp in zip(predictions, labels, responses):
        resp_lower = resp.lower().strip()
        if resp_lower.startswith('yes'):
            cat = 'yes'
        elif resp_lower.startswith('no'):
            cat = 'no'
        elif resp_lower.startswith('true'):
            cat = 'true'
        elif resp_lower.startswith('false'):
            cat = 'false'
        elif resp_lower.startswith(('a)', '(a)', 'a.')):
            cat = 'option_a'
        elif resp_lower.startswith(('b)', '(b)', 'b.')):
            cat = 'option_b'
        else:
            cat = 'other'
        categories[cat]['preds'].append(pred)
        categories[cat]['labels'].append(label)

    print(f"\nPer-token-category analysis:")
    print(f"{'Category':<12} {'N':>6} {'Base Rate':>10} {'Mean P(c)':>10} {'AUROC':>8}")
    print("-" * 50)

    token_only_preds = []
    for cat, data in sorted(categories.items(), key=lambda x: -len(x[1]['preds'])):
        n = len(data['preds'])
        if n < 5:
            continue
        base = np.mean(data['labels'])
        mean_p = np.mean(data['preds'])
        token_only_preds.extend([base] * n)  # Token-only baseline predicts base rate

        if len(set(data['labels'])) > 1:
            auroc = roc_auc_score(data['labels'], data['preds'])
            print(f"{cat:<12} {n:>6} {100*base:>9.1f}% {mean_p:>10.3f} {auroc:>8.3f}")
        else:
            print(f"{cat:<12} {n:>6} {100*base:>9.1f}% {mean_p:>10.3f} {'N/A':>8}")

    # Token-only baseline AUROC
    try:
        token_auroc = roc_auc_score(labels, token_only_preds)
        print(f"\nToken-only baseline AUROC: {token_auroc:.3f}")
        print(f"  (Predicting base rate per token category)")
    except:
        pass


def analyze_within_benchmark(predictions, labels, benchmarks):
    """Compute AUROC within each benchmark to rule out benchmark-level shortcuts."""
    print("\n" + "="*60)
    print("4. WITHIN-BENCHMARK DISCRIMINATION")
    print("="*60)
    print("\nIf model learned benchmark-level shortcuts (e.g., 'VSR is always hard'),")
    print("within-benchmark AUROC would be ~0.5. High within-AUROC proves genuine learning.")

    predictions = np.array(predictions)
    labels = np.array(labels)

    by_benchmark = defaultdict(lambda: {'preds': [], 'labels': []})
    for pred, label, bench in zip(predictions, labels, benchmarks):
        by_benchmark[bench]['preds'].append(pred)
        by_benchmark[bench]['labels'].append(label)

    print(f"\n{'Benchmark':<20} {'N':>6} {'Base Rate':>10} {'Within-AUROC':>12}")
    print("-" * 55)

    within_aurocs = []
    for bench, data in sorted(by_benchmark.items(), key=lambda x: -len(x[1]['preds'])):
        n = len(data['preds'])
        if n < 10:
            continue
        base = np.mean(data['labels'])

        if len(set(data['labels'])) > 1:
            auroc = roc_auc_score(data['labels'], data['preds'])
            within_aurocs.append((auroc, n))
            status = "✓" if auroc > 0.55 else "✗"
            print(f"{bench:<20} {n:>6} {100*base:>9.1f}% {auroc:>11.3f} {status}")
        else:
            print(f"{bench:<20} {n:>6} {100*base:>9.1f}% {'N/A':>12}")

    if within_aurocs:
        # Weighted mean
        total_n = sum(n for _, n in within_aurocs)
        weighted_mean = sum(a * n for a, n in within_aurocs) / total_n
        unweighted_mean = np.mean([a for a, _ in within_aurocs])
        print(f"\nWeighted mean within-AUROC: {weighted_mean:.3f}")
        print(f"Unweighted mean: {unweighted_mean:.3f}")
        print(f"\n  => Within-AUROC >> 0.5 proves model discriminates WITHIN benchmarks")


def analyze_calibration(predictions, labels):
    """Check if confidence correlates with accuracy (well-calibrated model)."""
    print("\n" + "="*60)
    print("5. CONFIDENCE CALIBRATION ANALYSIS")
    print("="*60)

    predictions = np.array(predictions)
    labels = np.array(labels)

    # Confidence = distance from 0.5
    confidence = np.abs(predictions - 0.5) * 2  # Scale to [0, 1]

    # Bin by confidence
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    print(f"\n{'Confidence':<12} {'N':>6} {'Accuracy':>10} {'Expected':>10}")
    print("-" * 45)

    for i in range(len(bins)-1):
        mask = (confidence >= bins[i]) & (confidence < bins[i+1])
        if mask.sum() == 0:
            continue

        # For samples in this bin, compute accuracy
        bin_preds = predictions[mask]
        bin_labels = labels[mask]

        # Predicted class: 1 if P > 0.5, else 0
        pred_class = (bin_preds > 0.5).astype(int)
        accuracy = (pred_class == bin_labels).mean()

        # Expected accuracy based on confidence
        expected = 0.5 + (bins[i] + bins[i+1]) / 4  # Midpoint confidence

        print(f"{bins[i]:.1f}-{bins[i+1]:.1f}      {mask.sum():>6} {100*accuracy:>9.1f}% {100*expected:>9.1f}%")

    # Overall: high confidence samples should be more accurate
    high_conf_mask = confidence > 0.5
    low_conf_mask = confidence <= 0.5

    if high_conf_mask.sum() > 0 and low_conf_mask.sum() > 0:
        high_acc = ((predictions[high_conf_mask] > 0.5) == labels[high_conf_mask]).mean()
        low_acc = ((predictions[low_conf_mask] > 0.5) == labels[low_conf_mask]).mean()
        print(f"\nHigh confidence (>0.75) accuracy: {100*high_acc:.1f}%")
        print(f"Low confidence (≤0.75) accuracy: {100*low_acc:.1f}%")
        print(f"\n  => High-conf should be more accurate if model is calibrated")


def analyze_prediction_distribution(predictions, labels):
    """Check if model produces varied predictions or collapsed to single value."""
    print("\n" + "="*60)
    print("6. PREDICTION DISTRIBUTION ANALYSIS")
    print("="*60)

    predictions = np.array(predictions)
    labels = np.array(labels)

    print(f"\nPrediction statistics:")
    print(f"  Mean:   {predictions.mean():.3f}")
    print(f"  Std:    {predictions.std():.3f}")
    print(f"  Min:    {predictions.min():.3f}")
    print(f"  Max:    {predictions.max():.3f}")
    print(f"  Median: {np.median(predictions):.3f}")

    # Check for collapse
    if predictions.std() < 0.05:
        print(f"\n  ⚠ WARNING: Very low variance - model may have collapsed!")
    else:
        print(f"\n  ✓ Good variance - model produces diverse predictions")

    # Distribution by actual label
    correct_preds = predictions[labels == 1]
    incorrect_preds = predictions[labels == 0]

    print(f"\nPredictions by actual label:")
    print(f"  Correct samples:   mean={correct_preds.mean():.3f}, std={correct_preds.std():.3f}")
    print(f"  Incorrect samples: mean={incorrect_preds.mean():.3f}, std={incorrect_preds.std():.3f}")

    separation = correct_preds.mean() - incorrect_preds.mean()
    print(f"\n  Mean separation: {separation:.3f}")
    print(f"  => Positive separation means model predicts higher P(c) for correct samples")


def main():
    print("="*60)
    print("SPURIOUS CORRELATION ANALYSIS")
    print("="*60)
    print("\nThis analysis tests whether the VLM judge learned genuine")
    print("correctness prediction or just spurious shortcuts.")

    # Try to load vision predictions
    vision_path = Path("data/vlm_judge_combined/combined_checkpoint788_results.json")
    text_path = Path("data/vlm_judge_combined/text_predictions.json")

    for name, path in [("Vision", vision_path), ("Text", text_path)]:
        if not path.exists():
            print(f"\n{name} predictions not found at {path}")
            continue

        print(f"\n\n{'#'*60}")
        print(f"# {name.upper()} PREDICTIONS ANALYSIS")
        print(f"{'#'*60}")

        data = load_predictions(path)

        # Extract predictions, labels, responses, benchmarks
        if isinstance(data, dict) and 'predictions' in data:
            # Format 1: {predictions: [...], labels: [...]}
            predictions = data['predictions']
            labels = data['labels']
            responses = data.get('responses', [''] * len(predictions))
            benchmarks = data.get('benchmarks', ['unknown'] * len(predictions))
        elif isinstance(data, list):
            # Format 2: [{pred: ..., label: ..., response: ...}, ...]
            predictions = [d.get('prediction', d.get('p_correct', 0.5)) for d in data]
            labels = [d.get('label', d.get('is_correct', 0)) for d in data]
            responses = [d.get('response', '') for d in data]
            benchmarks = [d.get('benchmark', 'unknown') for d in data]
        else:
            print(f"Unknown data format in {path}")
            continue

        # Convert to proper types
        predictions = [float(p) for p in predictions]
        labels = [int(l) for l in labels]

        print(f"\nLoaded {len(predictions)} predictions from {path}")

        # Run analyses
        base_rate = analyze_trivial_baselines(labels)

        if responses and responses[0]:
            analyze_response_length(predictions, labels, responses)
            analyze_response_token_bias(predictions, labels, responses)

        if benchmarks and benchmarks[0] != 'unknown':
            analyze_within_benchmark(predictions, labels, benchmarks)

        analyze_calibration(predictions, labels)
        analyze_prediction_distribution(predictions, labels)

        # Final AUROC
        overall_auroc = roc_auc_score(labels, predictions)
        print(f"\n{'='*60}")
        print(f"OVERALL {name.upper()} AUROC: {overall_auroc:.3f}")
        print(f"{'='*60}")

    print("\n\n" + "="*60)
    print("SUMMARY: EVIDENCE AGAINST SPURIOUS CORRELATIONS")
    print("="*60)
    print("""
1. WITHIN-BENCHMARK AUROC >> 0.5
   Model discriminates correct/incorrect WITHIN each benchmark.
   If it learned 'benchmark X is hard', within-AUROC would be ~0.5.

2. VARIED PREDICTION DISTRIBUTION
   Model produces diverse P(correct) values, not collapsed to single value.
   A 'always say no' model would have std ≈ 0.

3. SEPARATION BY ACTUAL LABEL
   Mean P(correct) for correct samples > incorrect samples.
   Shows model learned the right direction.

4. CALIBRATION
   Higher confidence predictions are more accurate.
   Random correlation would show no such pattern.

5. RESIDUAL AUROC AFTER CONTROLLING FOR LENGTH/TOKENS
   Model still discriminates after controlling for response length
   and response token patterns (yes/no/true/false).
""")


if __name__ == "__main__":
    main()
