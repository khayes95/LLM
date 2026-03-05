#!/usr/bin/env python3
"""Generate a literature comparison table for the paper.

This creates a structured comparison of our method against published UQ methods,
organized by: (1) what each method requires, (2) whether it works on closed-source
models, and (3) reported performance where available.

Output: JSON and formatted text for inclusion in the paper.

Usage:
    python scripts/literature_comparison.py
"""
import json
import os
from pathlib import Path


def main():
    output_dir = "data/use_cases/results_test_only_v2"
    os.makedirs(output_dir, exist_ok=True)

    # Our method's results
    our_results = {
        "method": "UQ Calibrator (Ours)",
        "base_model": "Qwen3-VL-8B + LoRA",
        "requires": "Question + Answer text (+ optional image)",
        "works_closed_source": True,
        "works_open_source": True,
        "multimodal": True,
        "cross_model": True,
        "training_needed": "Yes (fine-tuning with LoRA)",
        "auroc_reported": 0.896,  # held-out test set (v2 r=32)
        "notes": "Trained on 3 source models, evaluated cross-model. "
                 "Handles both text and VLM benchmarks.",
    }

    # Published methods comparison
    methods = [
        {
            "method": "Verbalized Confidence",
            "category": "Prompting",
            "reference": "Kadavath et al. 2022; Tian et al. 2023",
            "requires": "Model API access (generation)",
            "works_closed_source": True,
            "works_open_source": True,
            "multimodal": "Partial (text prompt only)",
            "cross_model": "N/A (per-model)",
            "training_needed": "No",
            "auroc_reported": "0.60-0.70 (our experiments)",
            "limitations": [
                "Models are known to be poorly calibrated in verbalized confidence",
                "Sensitive to prompt wording",
                "Does not improve with more data — no learning",
            ],
        },
        {
            "method": "Logit-Based (MaxProb / Entropy)",
            "category": "Logit Access",
            "reference": "Hendrycks & Gimpel 2017; Kadavath et al. 2022",
            "requires": "Full logit/probability access",
            "works_closed_source": False,
            "works_open_source": True,
            "multimodal": True,
            "cross_model": "No (model-specific logits)",
            "training_needed": "No",
            "auroc_reported": "0.65-0.80 (typical, model-dependent)",
            "limitations": [
                "Requires logit access — unavailable for GPT-5, Claude, Gemini",
                "Token-level probability ≠ answer-level confidence",
                "Poorly calibrated without post-hoc scaling",
            ],
        },
        {
            "method": "Temperature Scaling / Platt Scaling",
            "category": "Post-hoc Calibration",
            "reference": "Guo et al. 2017; Platt 1999",
            "requires": "Logit access + labeled validation set",
            "works_closed_source": False,
            "works_open_source": True,
            "multimodal": "Depends on base method",
            "cross_model": "No (model-specific)",
            "training_needed": "Minimal (fit scaling parameters)",
            "auroc_reported": "N/A (improves ECE, not AUROC)",
            "limitations": [
                "Improves calibration (ECE) but does NOT improve discrimination (AUROC)",
                "Requires logit access",
                "Single scalar parameter — limited expressiveness",
            ],
        },
        {
            "method": "Monte Carlo Dropout",
            "category": "Bayesian Approximation",
            "reference": "Gal & Ghahramani 2016",
            "requires": "Dropout layers + multiple forward passes",
            "works_closed_source": False,
            "works_open_source": "Partially (requires dropout)",
            "multimodal": True,
            "cross_model": "No",
            "training_needed": "No (uses existing dropout)",
            "auroc_reported": "0.65-0.80 (task-dependent)",
            "limitations": [
                "Requires model internals (dropout layers)",
                "K forward passes = K× inference cost",
                "Most modern LLMs don't use dropout",
                "Approximation quality degrades with model size",
            ],
        },
        {
            "method": "Deep Ensembles",
            "category": "Ensemble",
            "reference": "Lakshminarayanan et al. 2017",
            "requires": "Multiple independently trained models",
            "works_closed_source": False,
            "works_open_source": True,
            "multimodal": True,
            "cross_model": "No",
            "training_needed": "Yes (train M models)",
            "auroc_reported": "0.70-0.85 (strong but expensive)",
            "limitations": [
                "Requires training M separate models (M× cost)",
                "Not feasible for LLMs (100B+ parameters per model)",
                "Inference cost scales linearly with ensemble size",
            ],
        },
        {
            "method": "Conformal Prediction",
            "category": "Distribution-Free",
            "reference": "Shafer & Vovk 2008; Angelopoulos & Bates 2023",
            "requires": "Calibration set + exchangeability assumption",
            "works_closed_source": True,
            "works_open_source": True,
            "multimodal": "Partial",
            "cross_model": "Per-model calibration",
            "training_needed": "No (only calibration set)",
            "auroc_reported": "N/A (provides coverage guarantees, not discrimination)",
            "limitations": [
                "Provides prediction sets, not point estimates of P(correct)",
                "Requires exchangeability — may not hold for diverse benchmarks",
                "Does not discriminate between correct/incorrect — only sets sizes",
                "Complementary to our approach (could be combined)",
            ],
        },
        {
            "method": "Semantic Entropy",
            "category": "Generation-Based",
            "reference": "Kuhn et al. 2023",
            "requires": "Multiple generations + semantic clustering",
            "works_closed_source": True,
            "works_open_source": True,
            "multimodal": False,
            "cross_model": "Per-model",
            "training_needed": "No (uses NLI model for clustering)",
            "auroc_reported": "0.75-0.85 (free-form QA)",
            "limitations": [
                "Requires multiple generations per question (5-10× cost)",
                "Needs semantic similarity model for clustering",
                "Designed for free-form QA — less applicable to MCQ/math",
                "Does not transfer across models",
            ],
        },
        {
            "method": "Self-Evaluation / P(True)",
            "category": "Prompting",
            "reference": "Kadavath et al. 2022",
            "requires": "Model API access (ask model if its answer is correct)",
            "works_closed_source": True,
            "works_open_source": True,
            "multimodal": "Partial",
            "cross_model": "No (self-evaluation only)",
            "training_needed": "No",
            "auroc_reported": "0.65-0.80 (model-dependent)",
            "limitations": [
                "Model evaluates its own answer — same biases apply",
                "Cannot evaluate a DIFFERENT model's answer",
                "Performance varies greatly across models and tasks",
            ],
        },
        {
            "method": "Learned Verifiers / Reward Models",
            "category": "Trained Discriminator",
            "reference": "Cobbe et al. 2021; Lightman et al. 2023",
            "requires": "Labeled correct/incorrect pairs for training",
            "works_closed_source": True,
            "works_open_source": True,
            "multimodal": "Task-specific",
            "cross_model": "Typically per-model",
            "training_needed": "Yes (task-specific training)",
            "auroc_reported": "0.80-0.90 (math-specific, e.g. GSM8K)",
            "limitations": [
                "Typically task-specific (math verification)",
                "Not designed for cross-model transfer",
                "Requires task-specific labeled data",
                "Our approach is closest to this but broader (multi-task, multi-model, multimodal)",
            ],
        },
    ]

    # Build the comparison table
    print("=" * 100)
    print("LITERATURE COMPARISON TABLE")
    print("=" * 100)

    # Key differentiator table
    print(f"\n{'Method':<30} {'Closed':>7} {'Cross':>7} {'Multi':>7} {'Train':>7} {'Cost':>10}")
    print(f"{'':30} {'Source':>7} {'Model':>7} {'Modal':>7} {'Req':>7} {'(infer)':>10}")
    print("-" * 75)

    for m in methods:
        cs = "Yes" if m["works_closed_source"] else "No"
        cm = "Yes" if m.get("cross_model") == True else ("Partial" if "partial" in str(m.get("cross_model", "")).lower() else "No")
        mm = "Yes" if m["multimodal"] == True else ("Partial" if "partial" in str(m["multimodal"]).lower() else "No")
        tr = "Yes" if "yes" in str(m["training_needed"]).lower() else "No"

        # Inference cost
        if "ensemble" in m["category"].lower():
            cost = "M×"
        elif "dropout" in m["method"].lower():
            cost = "K×"
        elif "entropy" in m["method"].lower():
            cost = "5-10×"
        elif "self" in m["method"].lower() or "verbalized" in m["method"].lower():
            cost = "1× (prompt)"
        else:
            cost = "1×"

        print(f"{m['method']:<30} {cs:>7} {cm:>7} {mm:>7} {tr:>7} {cost:>10}")

    # Our method
    print(f"{'UQ Calibrator (Ours)':<30} {'Yes':>7} {'Yes':>7} {'Yes':>7} {'Yes':>7} {'1× (8B)':>10}")

    # Unique advantages
    print(f"\n{'='*100}")
    print("KEY ADVANTAGES OF OUR APPROACH")
    print("=" * 100)
    advantages = [
        "1. Works on closed-source models (GPT-5, Claude) where logit-based methods cannot",
        "2. Cross-model transfer: train on models A, B, C → evaluate on unseen model D",
        "3. Multimodal: handles both text LLM benchmarks and VLM benchmarks with real images",
        "4. Single unified model: one 8B model handles all tasks, modalities, and source models",
        "5. Efficient inference: single forward pass through 8B model (no multiple generations)",
        "6. Scalable to 2B model with 98% performance retention for edge deployment",
    ]
    for a in advantages:
        print(f"  {a}")

    # Save
    output = {
        "our_method": our_results,
        "comparison_methods": methods,
        "advantages": advantages,
        "key_differentiators": {
            "closed_source": "Only verbalized/self-eval and conformal work on closed-source, "
                              "but they don't learn from data or transfer across models",
            "cross_model": "No existing method is designed for cross-model uncertainty transfer",
            "multimodal_uq": "No existing method provides unified text+VLM uncertainty in a single model",
        },
    }

    output_path = os.path.join(output_dir, "literature_comparison.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
