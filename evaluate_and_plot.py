#!/usr/bin/env python3
"""
Evaluate UQ methods and generate publication figures.

Compares:
1. Fine-tuned UQ model (LoRA + Prompt method from Kapoor et al.)
2. Verbalized confidence (ask model for 0-100 score)
3. Zero-shot classifier (P(True) vs P(False))
4. Perplexity-based confidence

Outputs:
- results/metrics_summary.json
- results/results_table.tex
- figures/fig_main_results.pdf
- figures/fig_calibration.pdf
- figures/fig_distribution.pdf
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import torch
from tqdm import tqdm

# Paths
DEFAULT_MODEL_PATH = "uq_models/llama-8b-uq-lora"
DEFAULT_TEST_PATH = "data/finetune/val.jsonl"  # Use val for v1 model
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def load_test_data(path: str) -> list[dict]:
    """Load test examples from JSONL."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                # Extract benchmark from id (format: 'benchmark_*')
                ex_id = row.get("id", "")
                bench = ex_id.split("_")[0] if "_" in ex_id else "unknown"
                examples.append({
                    "id": ex_id,
                    "benchmark": bench,
                    "question": row.get("input", ""),
                    "answer": row.get("model_response", row.get("prediction", {}).get("answer", "")),
                    "correct": row.get("correct", 0) == 1,
                })
    return examples


# Benchmark categories for grouping
BENCHMARK_CATEGORIES = {
    "Math": ["gsm8k", "math", "mgsm", "aime", "omnimath"],
    "Science": ["gpqa", "arc"],
    "Reasoning": ["bbeh", "hellaswag", "winogrande", "drop"],
    "QA": ["triviaqa", "simpleqa", "boolq"],
    "Code": ["bigcodebench", "livecodebench", "swebench"],
    "Other": ["mmlu", "healthbench", "ether0", "hle", "multichallenge", "multinrc", "oolong", "tutorbench"],
}


def get_benchmark_category(bench: str) -> str:
    """Get category for a benchmark."""
    for category, benchmarks in BENCHMARK_CATEGORIES.items():
        if bench.lower() in benchmarks:
            return category
    return "Other"


def compute_ece(confidences: list[float], correct: list[bool], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    bin_correct = [0] * n_bins
    bin_conf = [0.0] * n_bins
    bin_count = [0] * n_bins

    for conf, corr in zip(confidences, correct):
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bin_correct[bin_idx] += int(corr)
        bin_conf[bin_idx] += conf
        bin_count[bin_idx] += 1

    ece = 0.0
    for i in range(n_bins):
        if bin_count[i] > 0:
            avg_conf = bin_conf[i] / bin_count[i]
            avg_acc = bin_correct[i] / bin_count[i]
            ece += bin_count[i] * abs(avg_conf - avg_acc)

    return ece / len(confidences)


def compute_brier(confidences: list[float], correct: list[bool]) -> float:
    """Compute Brier score."""
    return sum((c - int(corr)) ** 2 for c, corr in zip(confidences, correct)) / len(correct)


def compute_metrics(confidences: list[float], correct: list[bool]) -> dict:
    """Compute all calibration metrics."""
    predictions = [c > 0.5 for c in confidences]
    accuracy = sum(p == c for p, c in zip(predictions, correct)) / len(correct)

    try:
        auroc = roc_auc_score(correct, confidences)
    except ValueError:
        auroc = 0.5

    return {
        "accuracy": accuracy,
        "auroc": auroc,
        "ece": compute_ece(confidences, correct),
        "brier": compute_brier(confidences, correct),
        "mean_confidence": np.mean(confidences),
    }


# =============================================================================
# UQ Methods
# =============================================================================

def evaluate_finetuned(model, tokenizer, examples: list[dict], temperature: float = 1.0) -> list[float]:
    """Evaluate using fine-tuned UQ model (LoRA + Prompt method)."""
    from uq_eval.uq_finetune import get_confidence_batch
    return get_confidence_batch(model, tokenizer, examples, temperature=temperature)


def evaluate_verbalized(model, tokenizer, examples: list[dict]) -> list[float]:
    """Evaluate using verbalized confidence (ask model for 0-100)."""
    confidences = []

    prompt_template = """Question: {question}

Answer: {answer}

On a scale of 0 to 100, how confident are you that the above answer is correct?
Respond with just a number from 0 to 100.

Confidence:"""

    for ex in tqdm(examples, desc="Verbalized"):
        prompt = prompt_template.format(question=ex["question"], answer=ex["answer"])

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Parse confidence
        try:
            # Extract first number from response
            import re
            numbers = re.findall(r'\d+', response)
            if numbers:
                conf = min(100, max(0, int(numbers[0]))) / 100.0
            else:
                conf = 0.5
        except:
            conf = 0.5

        confidences.append(conf)

    return confidences


def evaluate_zero_shot(model, tokenizer, examples: list[dict]) -> list[float]:
    """Evaluate using zero-shot True/False classifier."""
    confidences = []

    # Get token IDs for True/False
    true_tokens = tokenizer.encode("True", add_special_tokens=False)
    false_tokens = tokenizer.encode("False", add_special_tokens=False)
    true_id = true_tokens[0]
    false_id = false_tokens[0]

    prompt_template = """Question: {question}

Answer: {answer}

Is the above answer correct? True or False:"""

    for ex in tqdm(examples, desc="Zero-shot"):
        prompt = prompt_template.format(question=ex["question"], answer=ex["answer"])

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]

            logit_true = logits[true_id].item()
            logit_false = logits[false_id].item()

            probs = torch.softmax(torch.tensor([logit_false, logit_true]), dim=0)
            conf = probs[1].item()  # P(True)

        confidences.append(conf)

    return confidences


def evaluate_perplexity(model, tokenizer, examples: list[dict]) -> list[float]:
    """Evaluate using perplexity-based confidence."""
    confidences = []

    for ex in tqdm(examples, desc="Perplexity"):
        # Compute perplexity of the answer given the question
        prompt = f"Question: {ex['question']}\n\nAnswer:"
        full_text = f"{prompt} {ex['answer']}"

        prompt_inputs = tokenizer(prompt, return_tensors="pt")
        full_inputs = tokenizer(full_text, return_tensors="pt")

        prompt_len = prompt_inputs["input_ids"].shape[1]

        full_inputs = {k: v.to(model.device) for k, v in full_inputs.items()}

        with torch.no_grad():
            outputs = model(**full_inputs, labels=full_inputs["input_ids"])

            # Get per-token losses for answer portion only
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = full_inputs["input_ids"][..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Only consider answer tokens (after prompt)
            answer_losses = losses[prompt_len-1:]
            if len(answer_losses) > 0:
                avg_loss = answer_losses.mean().item()
            else:
                avg_loss = 0.0

            # Convert to confidence: higher perplexity = higher confidence
            # (Empirically, the model is more fluent when wrong - confidently hallucinating)
            # Use sigmoid transformation, but INVERTED
            conf = 1.0 / (1.0 + np.exp(-(avg_loss - 2.0)))  # Inverted: higher loss = higher conf

        confidences.append(conf)

    return confidences


# =============================================================================
# Plotting
# =============================================================================

def plot_main_results(results: dict, output_path: str):
    """Bar chart comparing ECE and AUROC across methods."""
    methods = list(results.keys())
    ece_values = [results[m]["ece"] for m in methods]
    auroc_values = [results[m]["auroc"] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # ECE (lower is better)
    colors = ['#2ecc71' if m == 'Finetuned' else '#3498db' for m in methods]
    ax1.bar(x, ece_values, width, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('ECE (↓)', fontsize=12)
    ax1.set_title('Expected Calibration Error', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    ax1.set_ylim(0, max(ece_values) * 1.2)

    # AUROC (higher is better)
    ax2.bar(x, auroc_values, width, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('AUROC (↑)', fontsize=12)
    ax2.set_title('Area Under ROC Curve', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=15, ha='right')
    ax2.set_ylim(0.4, 1.0)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_calibration(all_confidences: dict, all_correct: list[bool], output_path: str):
    """Reliability diagram."""
    n_bins = 10

    fig, ax = plt.subplots(figsize=(6, 5))

    colors = {'Finetuned': '#2ecc71', 'Verbalized': '#e74c3c', 'Zero-shot': '#3498db', 'Perplexity': '#9b59b6'}

    for method, confidences in all_confidences.items():
        bin_correct = [0] * n_bins
        bin_conf = [0.0] * n_bins
        bin_count = [0] * n_bins

        for conf, corr in zip(confidences, all_correct):
            bin_idx = min(int(conf * n_bins), n_bins - 1)
            bin_correct[bin_idx] += int(corr)
            bin_conf[bin_idx] += conf
            bin_count[bin_idx] += 1

        # Compute averages
        bin_centers = []
        bin_accs = []
        for i in range(n_bins):
            if bin_count[i] > 5:  # Only plot bins with enough samples
                bin_centers.append(bin_conf[i] / bin_count[i])
                bin_accs.append(bin_correct[i] / bin_count[i])

        if bin_centers:
            ax.plot(bin_centers, bin_accs, 'o-', label=method, color=colors.get(method, '#333'),
                   markersize=6, linewidth=2)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')

    ax.set_xlabel('Predicted Confidence', fontsize=12)
    ax.set_ylabel('Actual Accuracy', fontsize=12)
    ax.set_title('Reliability Diagram', fontsize=12)
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_distribution(all_confidences: dict, all_correct: list[bool], output_path: str):
    """Histogram of confidence scores for correct vs incorrect."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, (method, confidences) in enumerate(all_confidences.items()):
        ax = axes[idx]

        correct_conf = [c for c, corr in zip(confidences, all_correct) if corr]
        incorrect_conf = [c for c, corr in zip(confidences, all_correct) if not corr]

        bins = np.linspace(0, 1, 21)
        ax.hist(correct_conf, bins=bins, alpha=0.7, label='Correct', color='#2ecc71', edgecolor='black', linewidth=0.5)
        ax.hist(incorrect_conf, bins=bins, alpha=0.7, label='Incorrect', color='#e74c3c', edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Confidence', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(method, fontsize=11)
        ax.legend(loc='upper center')
        ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_per_benchmark(examples: list[dict], confidences: list[float], output_path: str):
    """Generate per-benchmark breakdown figure."""
    from collections import defaultdict

    # Group by category
    category_data = defaultdict(lambda: {"confidences": [], "correct": []})

    for ex, conf in zip(examples, confidences):
        bench = ex.get("benchmark", "unknown")
        category = get_benchmark_category(bench)
        category_data[category]["confidences"].append(conf)
        category_data[category]["correct"].append(ex["correct"])

    # Compute metrics per category
    categories = []
    ece_values = []
    auroc_values = []
    counts = []

    for category in ["Math", "Science", "Reasoning", "QA", "Code", "Other"]:
        if category in category_data and len(category_data[category]["confidences"]) >= 10:
            data = category_data[category]
            metrics = compute_metrics(data["confidences"], data["correct"])
            categories.append(category)
            ece_values.append(metrics["ece"])
            auroc_values.append(metrics["auroc"])
            counts.append(len(data["confidences"]))

    if not categories:
        print("Not enough data for per-benchmark breakdown")
        return

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    x = np.arange(len(categories))
    width = 0.6

    # ECE by category
    colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
    bars1 = ax1.bar(x, ece_values, width, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('ECE (↓)', fontsize=12)
    ax1.set_title('Calibration Error by Category', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(categories, counts)], fontsize=10)
    ax1.set_ylim(0, max(ece_values) * 1.3)

    # AUROC by category
    bars2 = ax2.bar(x, auroc_values, width, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('AUROC (↑)', fontsize=12)
    ax2.set_title('Discrimination by Category', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(categories, counts)], fontsize=10)
    ax2.set_ylim(0.4, 1.0)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    # Also print breakdown
    print("\nPer-category breakdown:")
    print(f"{'Category':<12} {'Count':>6} {'AUROC':>8} {'ECE':>8}")
    print("-" * 36)
    for cat, cnt, auroc, ece in zip(categories, counts, auroc_values, ece_values):
        print(f"{cat:<12} {cnt:>6} {auroc:>8.3f} {ece:>8.3f}")


def plot_per_benchmark_detailed(examples: list[dict], confidences: list[float], output_path: str):
    """Generate detailed per-benchmark breakdown (individual benchmarks)."""
    from collections import defaultdict

    # Group by benchmark
    bench_data = defaultdict(lambda: {"confidences": [], "correct": []})

    for ex, conf in zip(examples, confidences):
        bench = ex.get("benchmark", "unknown")
        bench_data[bench]["confidences"].append(conf)
        bench_data[bench]["correct"].append(ex["correct"])

    # Compute metrics per benchmark (min 5 examples)
    benchmarks = []
    ece_values = []
    auroc_values = []
    counts = []

    for bench, data in sorted(bench_data.items(), key=lambda x: -len(x[1]["confidences"])):
        if len(data["confidences"]) >= 5:
            metrics = compute_metrics(data["confidences"], data["correct"])
            benchmarks.append(bench)
            ece_values.append(metrics["ece"])
            auroc_values.append(metrics["auroc"])
            counts.append(len(data["confidences"]))

    if not benchmarks:
        print("Not enough data for detailed per-benchmark breakdown")
        return

    # Limit to top 15 benchmarks by count
    if len(benchmarks) > 15:
        benchmarks = benchmarks[:15]
        ece_values = ece_values[:15]
        auroc_values = auroc_values[:15]
        counts = counts[:15]

    # Plot horizontal bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(6, len(benchmarks) * 0.4)))

    y = np.arange(len(benchmarks))
    height = 0.7

    # Color by category
    colors = [plt.cm.Set2(list(BENCHMARK_CATEGORIES.keys()).index(get_benchmark_category(b)) % 8)
              for b in benchmarks]

    # ECE
    ax1.barh(y, ece_values, height, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('ECE (↓)', fontsize=12)
    ax1.set_title('Calibration Error by Benchmark', fontsize=12)
    ax1.set_yticks(y)
    ax1.set_yticklabels([f"{b} ({c})" for b, c in zip(benchmarks, counts)], fontsize=9)
    ax1.invert_yaxis()

    # AUROC
    ax2.barh(y, auroc_values, height, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('AUROC (↑)', fontsize=12)
    ax2.set_title('Discrimination by Benchmark', fontsize=12)
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{b} ({c})" for b, c in zip(benchmarks, counts)], fontsize=9)
    ax2.set_xlim(0.4, 1.0)
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_latex_table(results: dict, output_path: str):
    """Generate LaTeX table."""
    latex = r"""\begin{table}[t]
\centering
\caption{Comparison of uncertainty quantification methods on held-out test set.}
\label{tab:results}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{ECE $\downarrow$} & \textbf{AUROC $\uparrow$} & \textbf{Brier $\downarrow$} \\
\midrule
"""

    for method, metrics in results.items():
        latex += f"{method} & {metrics['ece']:.3f} & {metrics['auroc']:.3f} & {metrics['brier']:.3f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_path, 'w') as f:
        f.write(latex)
    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate UQ methods and generate figures")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help="Path to finetuned LoRA model")
    parser.add_argument("--test_path", default=DEFAULT_TEST_PATH, help="Path to test data")
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL, help="Base model name")
    parser.add_argument("--skip_baselines", action="store_true", help="Skip baseline methods (faster)")
    parser.add_argument("--max_examples", type=int, default=None, help="Limit examples for testing")
    args = parser.parse_args()

    # Create output dirs
    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    # Load test data
    print(f"Loading test data from {args.test_path}...")
    examples = load_test_data(args.test_path)
    if args.max_examples:
        examples = examples[:args.max_examples]
    print(f"Loaded {len(examples)} examples")

    correct = [ex["correct"] for ex in examples]
    print(f"Class balance: {sum(correct)}/{len(correct)} correct ({100*sum(correct)/len(correct):.1f}%)")

    # Load models
    print(f"\nLoading base model: {args.base_model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load finetuned model
    print(f"Loading finetuned model from {args.model_path}...")
    from peft import PeftModel
    finetuned_model = PeftModel.from_pretrained(base_model, args.model_path)
    ft_tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Load calibration temperature
    calib_path = Path(args.model_path) / "calibration.json"
    if calib_path.exists():
        with open(calib_path) as f:
            calib = json.load(f)
        temperature = calib["temperature"]
        print(f"Using calibration temperature: {temperature:.4f}")
    else:
        temperature = 1.0
        print("No calibration file found, using T=1.0")

    # Evaluate all methods
    results = {}
    all_confidences = {}

    # 1. Finetuned model
    print("\n" + "="*50)
    print("Evaluating: Finetuned (LoRA + Prompt)")
    print("="*50)
    conf_finetuned = evaluate_finetuned(finetuned_model, ft_tokenizer, examples, temperature)
    results["Finetuned"] = compute_metrics(conf_finetuned, correct)
    all_confidences["Finetuned"] = conf_finetuned
    print(f"  AUROC: {results['Finetuned']['auroc']:.3f}, ECE: {results['Finetuned']['ece']:.3f}")

    if not args.skip_baselines:
        # 2. Verbalized
        print("\n" + "="*50)
        print("Evaluating: Verbalized Confidence")
        print("="*50)
        conf_verbalized = evaluate_verbalized(base_model, tokenizer, examples)
        results["Verbalized"] = compute_metrics(conf_verbalized, correct)
        all_confidences["Verbalized"] = conf_verbalized
        print(f"  AUROC: {results['Verbalized']['auroc']:.3f}, ECE: {results['Verbalized']['ece']:.3f}")

        # 3. Zero-shot
        print("\n" + "="*50)
        print("Evaluating: Zero-shot Classifier")
        print("="*50)
        conf_zeroshot = evaluate_zero_shot(base_model, tokenizer, examples)
        results["Zero-shot"] = compute_metrics(conf_zeroshot, correct)
        all_confidences["Zero-shot"] = conf_zeroshot
        print(f"  AUROC: {results['Zero-shot']['auroc']:.3f}, ECE: {results['Zero-shot']['ece']:.3f}")

        # 4. Perplexity
        print("\n" + "="*50)
        print("Evaluating: Perplexity-based")
        print("="*50)
        conf_perplexity = evaluate_perplexity(base_model, tokenizer, examples)
        results["Perplexity"] = compute_metrics(conf_perplexity, correct)
        all_confidences["Perplexity"] = conf_perplexity
        print(f"  AUROC: {results['Perplexity']['auroc']:.3f}, ECE: {results['Perplexity']['ece']:.3f}")

    # Save results
    print("\n" + "="*50)
    print("Generating outputs...")
    print("="*50)

    with open("results/metrics_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: results/metrics_summary.json")

    # Generate figures
    plot_main_results(results, "figures/fig_main_results.pdf")
    plot_calibration(all_confidences, correct, "figures/fig_calibration.pdf")
    plot_distribution(all_confidences, correct, "figures/fig_distribution.pdf")

    # Per-benchmark breakdown (using finetuned model confidences)
    plot_per_benchmark(examples, conf_finetuned, "figures/fig_benchmarks.pdf")
    plot_per_benchmark_detailed(examples, conf_finetuned, "figures/fig_benchmarks_detailed.pdf")

    # Save all confidences for per-benchmark analysis
    confidences_data = {
        "examples": [{"id": ex["id"], "benchmark": ex["benchmark"], "correct": ex["correct"]} for ex in examples],
        "confidences": {method: confs for method, confs in all_confidences.items()},
    }
    with open("results/all_confidences.json", "w") as f:
        json.dump(confidences_data, f)
    print("Saved: results/all_confidences.json")

    # Per-benchmark breakdown for ALL methods
    print("\n" + "="*70)
    print("PER-BENCHMARK AUROC BY METHOD")
    print("="*70)

    from collections import defaultdict
    bench_data = defaultdict(lambda: {"indices": [], "correct": []})
    for i, ex in enumerate(examples):
        bench_data[ex["benchmark"]]["indices"].append(i)
        bench_data[ex["benchmark"]]["correct"].append(ex["correct"])

    # Header
    methods = list(all_confidences.keys())
    header = f"{'Benchmark':<15} {'N':>5}"
    for m in methods:
        header += f" {m[:8]:>8}"
    print(header)
    print("-" * len(header))

    for bench in sorted(bench_data.keys(), key=lambda x: -len(bench_data[x]["correct"])):
        data = bench_data[bench]
        n = len(data["correct"])
        if n >= 10:  # Need enough samples for meaningful AUROC
            # Check if we have both classes
            if sum(data["correct"]) == 0 or sum(data["correct"]) == n:
                continue  # Skip if all same class

            row = f"{bench:<15} {n:>5}"
            for method in methods:
                method_confs = [all_confidences[method][i] for i in data["indices"]]
                try:
                    auroc = roc_auc_score(data["correct"], method_confs)
                    row += f" {auroc:>8.3f}"
                except:
                    row += f" {'N/A':>8}"
            print(row)

    # Generate LaTeX table
    generate_latex_table(results, "results/results_table.tex")

    # Print summary
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"{'Method':<15} {'AUROC':>8} {'ECE':>8} {'Brier':>8}")
    print("-"*43)
    for method, metrics in results.items():
        print(f"{method:<15} {metrics['auroc']:>8.3f} {metrics['ece']:>8.3f} {metrics['brier']:>8.3f}")


if __name__ == "__main__":
    main()
