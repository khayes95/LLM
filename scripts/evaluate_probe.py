#!/usr/bin/env python3
"""Publication-ready probe evaluation with proper baselines and statistics.

Produces:
1. Mean ± std AUROC across 3 seeds
2. Baselines: random, entropy-only, hidden-state-only
3. Per-benchmark breakdown
4. Saved test set IDs for VLM comparison

Usage:
    python scripts/evaluate_probe.py
"""
import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.feature_extraction import load_all_samples, UQSample


# ============================================================
# DATASETS
# ============================================================

class ProbeDataset(Dataset):
    """Dataset for probe training."""

    def __init__(self, samples: list[UQSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "hidden_state": s.hidden_state,
            "top_prob": torch.tensor(s.top_prob, dtype=torch.float32),
            "entropy": torch.tensor(s.entropy, dtype=torch.float32),
            "is_correct": torch.tensor(s.is_correct, dtype=torch.float32),
            "benchmark": s.benchmark if hasattr(s, 'benchmark') else "unknown",
            "question_id": s.question_id if hasattr(s, 'question_id') else str(idx),
        }


# ============================================================
# MODELS
# ============================================================

class FullProbe(nn.Module):
    """Hidden state + top_prob + entropy -> P(correct)."""

    def __init__(self, hidden_dim: int = 8192, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 2, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, hidden_state, top_prob, entropy):
        x = torch.cat([
            hidden_state,
            top_prob.unsqueeze(-1),
            entropy.unsqueeze(-1)
        ], dim=-1)
        return torch.sigmoid(self.net(x)).squeeze(-1)


class HiddenOnlyProbe(nn.Module):
    """Hidden state only -> P(correct). No entropy/prob features."""

    def __init__(self, hidden_dim: int = 8192, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, hidden_state, top_prob, entropy):
        # Ignore top_prob and entropy
        return torch.sigmoid(self.net(hidden_state)).squeeze(-1)


class EntropyOnlyProbe(nn.Module):
    """Entropy only -> P(correct). No hidden states, no top_prob."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, hidden_state, top_prob, entropy):
        # Ignore hidden_state and top_prob
        return torch.sigmoid(self.net(entropy.unsqueeze(-1))).squeeze(-1)


class TopProbOnlyProbe(nn.Module):
    """Top probability only -> P(correct). No hidden states, no entropy."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, hidden_state, top_prob, entropy):
        # Ignore hidden_state and entropy
        return torch.sigmoid(self.net(top_prob.unsqueeze(-1))).squeeze(-1)


# ============================================================
# CALIBRATION METRICS
# ============================================================

def compute_ece(preds: list, labels: list, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    preds = np.array(preds)
    labels = np.array(labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (preds >= bin_boundaries[i]) & (preds < bin_boundaries[i + 1])
        if in_bin.sum() == 0:
            continue

        bin_conf = preds[in_bin].mean()
        bin_acc = labels[in_bin].mean()
        bin_size = in_bin.sum() / len(preds)

        ece += bin_size * abs(bin_acc - bin_conf)

    return ece


def compute_brier(preds: list, labels: list) -> float:
    """Compute Brier score (lower is better)."""
    preds = np.array(preds)
    labels = np.array(labels)
    return float(np.mean((preds - labels) ** 2))


def compute_selective_accuracy(preds: list, labels: list, threshold: float) -> dict:
    """Compute accuracy when only predicting on confident samples."""
    preds = np.array(preds)
    labels = np.array(labels)

    # Samples where model is confident (pred > threshold for correct, pred < 1-threshold for incorrect)
    confident_correct = preds >= threshold
    confident_incorrect = preds <= (1 - threshold)
    confident = confident_correct | confident_incorrect

    if confident.sum() == 0:
        return {"coverage": 0.0, "accuracy": 0.0}

    # For confident samples, predict 1 if pred > 0.5, else 0
    predictions = (preds[confident] > 0.5).astype(float)
    actual = labels[confident]

    return {
        "coverage": float(confident.sum() / len(preds)),
        "accuracy": float((predictions == actual).mean()),
    }


# ============================================================
# TRAINING & EVALUATION
# ============================================================

def train_probe(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-4,
) -> nn.Module:
    """Train a probe model."""
    model = model.to(device).float()  # Ensure float32
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            hidden = batch["hidden_state"].to(device).float()  # Convert bf16 -> float32
            prob = batch["top_prob"].to(device).float()
            entropy = batch["entropy"].to(device).float()
            labels = batch["is_correct"].to(device).float()

            preds = model(hidden, prob, entropy)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def evaluate_probe(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate probe and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    per_benchmark = defaultdict(lambda: {"preds": [], "labels": []})

    with torch.no_grad():
        for batch in loader:
            hidden = batch["hidden_state"].to(device).float()
            prob = batch["top_prob"].to(device).float()
            entropy = batch["entropy"].to(device).float()
            labels = batch["is_correct"]
            benchmarks = batch["benchmark"]

            preds = model(hidden, prob, entropy).cpu()

            for i in range(len(preds)):
                p = preds[i].item()
                l = labels[i].item()
                b = benchmarks[i]

                all_preds.append(p)
                all_labels.append(l)
                per_benchmark[b]["preds"].append(p)
                per_benchmark[b]["labels"].append(l)

    # Overall metrics
    results = {
        "auroc": roc_auc_score(all_labels, all_preds),
        "auprc": average_precision_score(all_labels, all_preds),
        "n_samples": len(all_labels),
        "n_correct": sum(all_labels),
        "base_rate": sum(all_labels) / len(all_labels),
    }

    # Per-benchmark metrics
    results["per_benchmark"] = {}
    for bench, data in per_benchmark.items():
        if len(set(data["labels"])) < 2:
            # Skip if only one class
            continue
        results["per_benchmark"][bench] = {
            "auroc": roc_auc_score(data["labels"], data["preds"]),
            "auprc": average_precision_score(data["labels"], data["preds"]),
            "n_samples": len(data["labels"]),
            "n_correct": sum(data["labels"]),
        }

    return results


def random_baseline(labels: list) -> float:
    """Random baseline AUROC (should be ~0.5)."""
    preds = np.random.rand(len(labels))
    return roc_auc_score(labels, preds)


# ============================================================
# MAIN
# ============================================================

def load_all_benchmarks(base_dir: Path) -> list[UQSample]:
    """Load samples from all benchmark directories with benchmark labels."""
    all_samples = []
    for bench_dir in base_dir.iterdir():
        if not bench_dir.is_dir() or bench_dir.name == "smoke_test":
            continue

        benchmark = bench_dir.name
        samples = load_all_samples(bench_dir)

        # Add benchmark label to each sample
        for s in samples:
            s.benchmark = benchmark

        print(f"  {benchmark}: {len(samples)} samples")
        all_samples.extend(samples)

    return all_samples


def main():
    print("=" * 70)
    print("PUBLICATION-READY PROBE EVALUATION")
    print("=" * 70)

    # Config
    seeds = [42, 123, 456]
    test_ratio = 0.2
    hidden_dim = 8192
    epochs = 20
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load all samples
    print("\nLoading samples...")
    base_dir = Path("data/features")
    all_samples = load_all_benchmarks(base_dir)
    print(f"\nTotal: {len(all_samples)} samples")

    n_correct = sum(1 for s in all_samples if s.is_correct)
    print(f"Correct: {n_correct} ({100*n_correct/len(all_samples):.1f}%)")

    # Results storage
    results = {
        "full_probe": [],
        "hidden_only": [],
        "entropy_only": [],
        "top_prob_only": [],
        "random": [],
        "per_benchmark": defaultdict(list),
        # Calibration metrics (only for full probe, seed 0)
        "calibration": {},
    }

    test_ids_all = []

    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"SEED {seed}")
        print("=" * 50)

        # Split with seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        train_samples, test_samples = train_test_split(
            all_samples, test_size=test_ratio, random_state=seed
        )
        print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

        # Save test IDs (use first seed for canonical split)
        if seed == seeds[0]:
            test_ids_all = [
                {"question_id": s.question_id, "benchmark": s.benchmark}
                for s in test_samples
            ]

        train_loader = DataLoader(
            ProbeDataset(train_samples),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            ProbeDataset(test_samples),
            batch_size=batch_size,
            shuffle=False,
        )

        # 1. Full probe (hidden + entropy + prob)
        print("\n[1] Full Probe (hidden + entropy + prob)...")
        full_model = FullProbe(hidden_dim=hidden_dim)
        full_model = train_probe(full_model, train_loader, device, epochs=epochs)
        full_results = evaluate_probe(full_model, test_loader, device)
        results["full_probe"].append(full_results["auroc"])
        print(f"    AUROC: {full_results['auroc']:.4f}")

        # Store per-benchmark for first seed
        if seed == seeds[0]:
            for bench, metrics in full_results["per_benchmark"].items():
                results["per_benchmark"][bench].append(metrics)

        # 2. Hidden-only probe
        print("\n[2] Hidden-Only Probe...")
        hidden_model = HiddenOnlyProbe(hidden_dim=hidden_dim)
        hidden_model = train_probe(hidden_model, train_loader, device, epochs=epochs)
        hidden_results = evaluate_probe(hidden_model, test_loader, device)
        results["hidden_only"].append(hidden_results["auroc"])
        print(f"    AUROC: {hidden_results['auroc']:.4f}")

        # 3. Entropy-only probe
        print("\n[3] Entropy-Only Probe...")
        entropy_model = EntropyOnlyProbe()
        entropy_model = train_probe(entropy_model, train_loader, device, epochs=epochs, lr=1e-3)
        entropy_results = evaluate_probe(entropy_model, test_loader, device)
        results["entropy_only"].append(entropy_results["auroc"])
        print(f"    AUROC: {entropy_results['auroc']:.4f}")

        # 4. Top-prob-only probe
        print("\n[4] Top-Prob-Only Probe...")
        prob_model = TopProbOnlyProbe()
        prob_model = train_probe(prob_model, train_loader, device, epochs=epochs, lr=1e-3)
        prob_results = evaluate_probe(prob_model, test_loader, device)
        results["top_prob_only"].append(prob_results["auroc"])
        print(f"    AUROC: {prob_results['auroc']:.4f}")

        # 5. Random baseline
        test_labels = [s.is_correct for s in test_samples]
        rand_auroc = random_baseline(test_labels)
        results["random"].append(rand_auroc)
        print(f"\n[5] Random Baseline: {rand_auroc:.4f}")

        # Compute calibration metrics for full probe on first seed
        if seed == seeds[0]:
            print("\n[6] Computing calibration metrics...")
            # Get predictions from full model
            full_model.eval()
            all_preds = []
            all_labels_cal = []
            with torch.no_grad():
                for batch in test_loader:
                    hidden = batch["hidden_state"].to(device).float()
                    prob = batch["top_prob"].to(device).float()
                    entropy = batch["entropy"].to(device).float()
                    labels = batch["is_correct"]
                    preds = full_model(hidden, prob, entropy).cpu()
                    all_preds.extend(preds.tolist())
                    all_labels_cal.extend(labels.tolist())

            ece = compute_ece(all_preds, all_labels_cal)
            brier = compute_brier(all_preds, all_labels_cal)

            # Selective accuracy at different thresholds
            selective = {}
            for thresh in [0.6, 0.7, 0.8, 0.9]:
                selective[str(thresh)] = compute_selective_accuracy(all_preds, all_labels_cal, thresh)

            results["calibration"] = {
                "ece": ece,
                "brier": brier,
                "selective_accuracy": selective,
            }
            print(f"    ECE: {ece:.4f}")
            print(f"    Brier: {brier:.4f}")
            print(f"    Selective @0.8: cov={selective['0.8']['coverage']:.2%}, acc={selective['0.8']['accuracy']:.2%}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY (mean ± std across 3 seeds)")
    print("=" * 70)

    def summarize(values):
        return f"{np.mean(values):.4f} ± {np.std(values):.4f}"

    print(f"\n{'Method':<25} {'AUROC':>15}")
    print("-" * 42)
    print(f"{'Full Probe':<25} {summarize(results['full_probe']):>15}")
    print(f"{'Hidden-Only':<25} {summarize(results['hidden_only']):>15}")
    print(f"{'Entropy-Only':<25} {summarize(results['entropy_only']):>15}")
    print(f"{'Top-Prob-Only':<25} {summarize(results['top_prob_only']):>15}")
    print(f"{'Random':<25} {summarize(results['random']):>15}")

    # Calibration metrics
    if results["calibration"]:
        print("\n" + "-" * 42)
        print("CALIBRATION METRICS (Full Probe, Seed 42)")
        print("-" * 42)
        cal = results["calibration"]
        print(f"ECE (Expected Calibration Error): {cal['ece']:.4f}")
        print(f"Brier Score: {cal['brier']:.4f}")
        print("\nSelective Prediction (abstain on uncertain):")
        print(f"{'Threshold':<12} {'Coverage':>12} {'Accuracy':>12}")
        for thresh, metrics in cal["selective_accuracy"].items():
            print(f"{thresh:<12} {metrics['coverage']:>12.1%} {metrics['accuracy']:>12.1%}")

    # Per-benchmark breakdown
    print("\n" + "-" * 70)
    print("PER-BENCHMARK BREAKDOWN (Full Probe, Seed 42)")
    print("-" * 70)
    print(f"{'Benchmark':<20} {'AUROC':>10} {'AUPRC':>10} {'N':>8} {'Base Rate':>12}")
    print("-" * 70)

    for bench, metrics_list in sorted(results["per_benchmark"].items()):
        if metrics_list:
            m = metrics_list[0]
            base_rate = m["n_correct"] / m["n_samples"]
            print(f"{bench:<20} {m['auroc']:>10.4f} {m['auprc']:>10.4f} {m['n_samples']:>8} {base_rate:>12.1%}")

    # Save results
    output_dir = Path("data/probe_results")
    output_dir.mkdir(exist_ok=True)

    # Save test IDs
    with open(output_dir / "test_ids.json", "w") as f:
        json.dump(test_ids_all, f, indent=2)
    print(f"\nSaved {len(test_ids_all)} test IDs to {output_dir / 'test_ids.json'}")

    # Save full results
    summary = {
        "seeds": seeds,
        "test_ratio": test_ratio,
        "n_total_samples": len(all_samples),
        "n_test_samples": len(test_ids_all),
        "results": {
            "full_probe": {
                "auroc_mean": float(np.mean(results["full_probe"])),
                "auroc_std": float(np.std(results["full_probe"])),
                "auroc_values": results["full_probe"],
            },
            "hidden_only": {
                "auroc_mean": float(np.mean(results["hidden_only"])),
                "auroc_std": float(np.std(results["hidden_only"])),
                "auroc_values": results["hidden_only"],
            },
            "entropy_only": {
                "auroc_mean": float(np.mean(results["entropy_only"])),
                "auroc_std": float(np.std(results["entropy_only"])),
                "auroc_values": results["entropy_only"],
            },
            "top_prob_only": {
                "auroc_mean": float(np.mean(results["top_prob_only"])),
                "auroc_std": float(np.std(results["top_prob_only"])),
                "auroc_values": results["top_prob_only"],
            },
            "random": {
                "auroc_mean": float(np.mean(results["random"])),
                "auroc_std": float(np.std(results["random"])),
                "auroc_values": results["random"],
            },
        },
        "per_benchmark": {
            bench: metrics_list[0] if metrics_list else {}
            for bench, metrics_list in results["per_benchmark"].items()
        },
        "calibration": results["calibration"],
    }

    with open(output_dir / "probe_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved full results to {output_dir / 'probe_results.json'}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    full_mean = np.mean(results["full_probe"])
    entropy_mean = np.mean(results["entropy_only"])
    hidden_mean = np.mean(results["hidden_only"])

    print(f"\n• Full Probe AUROC: {full_mean:.3f}")
    if full_mean > 0.75:
        print("  → Strong signal for UQ. Hidden states are predictive of correctness.")
    elif full_mean > 0.65:
        print("  → Moderate signal. Some predictive power but room for improvement.")
    else:
        print("  → Weak signal. May need different features or more data.")

    print(f"\n• Hidden-only vs Entropy-only:")
    print(f"  Hidden: {hidden_mean:.3f}, Entropy: {entropy_mean:.3f}")
    if hidden_mean > entropy_mean + 0.05:
        print("  → Hidden states contribute significantly beyond simple uncertainty.")
    elif entropy_mean > hidden_mean + 0.05:
        print("  → Entropy is the main signal; hidden states add little value.")
    else:
        print("  → Both contribute roughly equally.")


if __name__ == "__main__":
    main()
