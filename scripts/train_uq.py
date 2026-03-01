#!/usr/bin/env python3
"""Train UQ classifier on all collected benchmark features.

Usage:
    python scripts/train_uq.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_extraction import load_all_samples, prepare_uq_dataset
from src.uq_classifier import UQClassifier, UQDataset, evaluate_model, train_classifier

import torch
from torch.utils.data import DataLoader
import torch.nn as nn


def load_all_benchmarks(base_dir: Path) -> list:
    """Load samples from all benchmark directories."""
    all_samples = []

    for bench_dir in base_dir.iterdir():
        if not bench_dir.is_dir():
            continue
        if bench_dir.name in ("smoke_test",):
            continue

        samples = load_all_samples(bench_dir)
        print(f"  {bench_dir.name}: {len(samples)} samples")
        all_samples.extend(samples)

    return all_samples


def main():
    print("=" * 60)
    print("UQ CLASSIFIER TRAINING")
    print("=" * 60)

    # Load all samples
    base_dir = Path("data/features")
    print(f"\nLoading samples from {base_dir}...")

    all_samples = load_all_benchmarks(base_dir)
    print(f"\nTotal: {len(all_samples)} samples")

    # Count correct/incorrect
    n_correct = sum(1 for s in all_samples if s.is_correct)
    n_incorrect = len(all_samples) - n_correct
    print(f"Correct: {n_correct} ({100*n_correct/len(all_samples):.1f}%)")
    print(f"Incorrect: {n_incorrect} ({100*n_incorrect/len(all_samples):.1f}%)")

    # Split data
    train_samples, test_samples = prepare_uq_dataset(all_samples, train_ratio=0.8, seed=42)
    print(f"\nTrain: {len(train_samples)}, Test: {len(test_samples)}")

    # Create dataloaders
    train_loader = DataLoader(UQDataset(train_samples), batch_size=64, shuffle=True)
    test_loader = DataLoader(UQDataset(test_samples), batch_size=64, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = UQClassifier(hidden_dim=8192).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    # Training loop
    epochs = 20
    best_auroc = 0
    best_epoch = 0
    output_path = "data/uq_classifier.pt"

    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            hidden = batch["hidden_state"].to(device)
            prob = batch["top_prob"].to(device)
            entropy = batch["entropy"].to(device)
            labels = batch["is_correct"].to(device)

            preds = model(hidden, prob, entropy)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Evaluate
        results = evaluate_model(model, test_loader, device)

        print(f"Epoch {epoch+1:2d}: loss={train_loss:.4f}, AUROC={results['auroc']:.4f}, AUPRC={results['auprc']:.4f}, acc={results['accuracy']:.4f}")

        # Save best model
        if results["auroc"] > best_auroc:
            best_auroc = results["auroc"]
            best_epoch = epoch + 1
            torch.save({
                "model_state": model.state_dict(),
                "hidden_dim": 8192,
                "auroc": best_auroc,
                "epoch": best_epoch,
            }, output_path)

    # Final evaluation
    checkpoint = torch.load(output_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    final_results = evaluate_model(model, test_loader, device)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best epoch: {best_epoch}")
    print(f"Best AUROC: {final_results['auroc']:.4f}")
    print(f"Best AUPRC: {final_results['auprc']:.4f}")
    print(f"Accuracy: {final_results['accuracy']:.4f}")
    print(f"Base rate: {final_results['base_rate']:.4f}")
    print(f"Model saved to: {output_path}")

    # Interpret results
    print("\n" + "-" * 40)
    if final_results["auroc"] > 0.75:
        print("✓ AUROC > 0.75: Strong signal! UQ method works well.")
    elif final_results["auroc"] > 0.60:
        print("⚠ AUROC 0.60-0.75: Weak signal. May need more features or data.")
    else:
        print("✗ AUROC ~0.50: No signal. Fundamental rethink needed.")


if __name__ == "__main__":
    main()
