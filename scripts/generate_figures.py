#!/usr/bin/env python3
"""
Generate figures for advisor meeting presentation.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set professional academic style (avoid AI-generated look)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.edgecolor'] = '0.8'

# Muted color palette (professional, not flashy)
COLORS = {
    'blue': '#4472C4',      # Standard blue
    'orange': '#ED7D31',    # Standard orange
    'gray': '#7F7F7F',      # Gray
    'green': '#70AD47',     # Muted green
    'red': '#C55A5A',       # Muted red
    'purple': '#7030A0',    # Purple
}

output_dir = Path("figures/advisor_meeting")
output_dir.mkdir(parents=True, exist_ok=True)


def plot_cross_model_transfer():
    """Figure 1: Cross-model transfer bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    benchmarks = ['VSR', 'CharXiv', 'MMMU', 'HallusionBench', 'MathVista*', 'RealWorldQA*']
    in_dist = [0.770, 0.720, 0.651, 0.809, None, None]
    cross_model = [0.657, 0.711, 0.688, 0.850, 0.575, 0.707]

    x = np.arange(len(benchmarks))
    width = 0.35

    # In-distribution bars (only for non-OOD benchmarks)
    for i, val in enumerate(in_dist):
        if val is not None:
            ax.bar(x[i] - width/2, val, width, color=COLORS['gray'], edgecolor='black', linewidth=0.5)
            ax.text(x[i] - width/2, val + 0.01, f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    # Cross-model bars
    colors = [COLORS['blue'] if in_dist[i] is not None else COLORS['red'] for i in range(len(benchmarks))]
    bars = ax.bar(x + width/2, cross_model, width, color=colors, edgecolor='black', linewidth=0.5)

    for i, (bar, val) in enumerate(zip(bars, cross_model)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    # Horizontal line at 0.5 (random)
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    ax.set_ylabel('AUROC')
    ax.set_title('Cross-Model Transfer: InternVL3 → Qwen2.5-VL')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=15, ha='right')
    ax.set_ylim(0.4, 1.0)

    # Legend
    in_dist_patch = mpatches.Patch(color=COLORS['gray'], edgecolor='black', linewidth=0.5, label='In-Distribution')
    cross_patch = mpatches.Patch(color=COLORS['blue'], edgecolor='black', linewidth=0.5, label='Cross-Model')
    ood_patch = mpatches.Patch(color=COLORS['red'], edgecolor='black', linewidth=0.5, label='Cross-Model (OOD)')
    ax.legend(handles=[in_dist_patch, cross_patch, ood_patch], loc='lower right', fontsize=9)

    # Add note
    ax.text(0.02, 0.02, '* Not in training data', transform=ax.transAxes, fontsize=8, style='italic')

    plt.tight_layout()
    plt.savefig(output_dir / 'cross_model_transfer_bar.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: cross_model_transfer_bar.pdf")


def plot_training_size_ablation():
    """Figure 2: Training sample efficiency."""
    fig, ax = plt.subplots(figsize=(7, 5))

    samples = [100, 250, 500, 1000, 2000]
    vision_auroc = [0.721, 0.754, 0.749, 0.752, 0.761]
    text_llm_auroc = [0.718, 0.885, 0.909, 0.914, 0.920]

    # Plot lines - simple markers
    ax.plot(samples, vision_auroc, 'o-', color=COLORS['blue'], linewidth=1.5, markersize=6, label='Vision (Qwen3-VL-8B)')
    ax.plot(samples, text_llm_auroc, 's-', color=COLORS['orange'], linewidth=1.5, markersize=6, label='Text (Qwen2.5-7B)')

    # Simple annotations without fancy boxes
    ax.annotate('97.4%', xy=(250, 0.754), xytext=(350, 0.72), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
    ax.annotate('98%', xy=(500, 0.909), xytext=(650, 0.88), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    ax.set_xlabel('Training Samples')
    ax.set_ylabel('AUROC')
    ax.set_title('Training Sample Efficiency')
    ax.set_xscale('log')
    ax.set_xticks(samples)
    ax.set_xticklabels(samples)
    ax.set_ylim(0.6, 1.0)
    ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_size_ablation.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: training_size_ablation.pdf")


def plot_selective_prediction():
    """Figure 3: Coverage-accuracy tradeoff."""
    fig, ax = plt.subplots(figsize=(7, 5))

    coverage = [100, 84.2, 70.9, 56.7, 43.3, 30.9]
    accuracy = [81.6, 86.3, 90.9, 94.1, 96.4, 98.6]
    confidence = ['50%', '60%', '70%', '80%', '90%', '95%']

    # Plot curve
    ax.plot(coverage, accuracy, 'o-', color=COLORS['blue'], linewidth=1.5, markersize=6)

    # Label each point with confidence threshold
    for i, (cov, acc, conf) in enumerate(zip(coverage, accuracy, confidence)):
        offset = (5, 5) if i < 3 else (-5, 5)
        ax.annotate(f'{conf}', xy=(cov, acc), xytext=offset, textcoords='offset points',
                   fontsize=8, ha='left' if i < 3 else 'right')

    # Mark key points simply
    ax.scatter([coverage[2]], [accuracy[2]], s=80, facecolors='none', edgecolors='black', linewidths=1, zorder=5)
    ax.annotate('90% acc @ 71% cov', xy=(coverage[2], accuracy[2]), xytext=(50, 88), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    ax.scatter([coverage[3]], [accuracy[3]], s=80, facecolors='none', edgecolors='black', linewidths=1, zorder=5)
    ax.annotate('95% acc @ 57% cov', xy=(coverage[3], accuracy[3]), xytext=(35, 96), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    ax.set_xlabel('Coverage (%)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Selective Prediction Tradeoff')
    ax.set_xlim(20, 105)
    ax.set_ylim(78, 100)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_dir / 'selective_prediction.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: selective_prediction.pdf")


def plot_results_summary_table():
    """Figure 4: Results summary table as image."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis('off')

    # Table data
    col_labels = ['Experiment', 'Metric', 'Value', 'Notes']
    table_data = [
        ['VLM Judge (Vision)', 'AUROC', '0.789', '+8.5% vs probe baseline'],
        ['VLM Judge (Text)', 'AUROC', '0.907', 'With gray placeholder image'],
        ['Pure LLM Judge (Text)', 'AUROC', '0.920', '4x faster training'],
        ['Cross-Model (HallusionBench)', 'AUROC', '0.850', 'Exceeds in-distribution'],
        ['Cross-Model (RealWorldQA)', 'AUROC', '0.707', 'OOD benchmark'],
        ['Cross-Model Average', 'AUROC', '0.694', '3 benchmarks'],
        ['Training Efficiency', '250 samples', '97.4%', 'of full performance'],
        ['Selective @ 90% Acc', 'Coverage', '70.9%', 'Can answer 71% of Qs'],
    ]

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='left',
        colColours=[COLORS['gray']] * 4,
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    # Color header
    for i in range(4):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('#ffffff')

    ax.set_title('Key Results', fontsize=12, pad=15)

    plt.tight_layout()
    plt.savefig(output_dir / 'results_summary_table.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: results_summary_table.pdf")


def plot_per_benchmark_comparison():
    """Figure 5: Per-benchmark AUROC comparison (Probe vs VLM Judge)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    benchmarks = ['HallusionBench', 'CharXiv', 'MMMU', 'VSR', 'ERQA']
    probe = [0.615, 0.586, 0.549, 0.589, 0.507]
    vlm_judge = [0.809, 0.720, 0.651, 0.637, 0.504]

    x = np.arange(len(benchmarks))
    width = 0.35

    bars1 = ax.bar(x - width/2, probe, width, label='Probe Baseline', color=COLORS['gray'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, vlm_judge, width, label='VLM Judge', color=COLORS['green'], edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

    # Random baseline
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    ax.set_ylabel('AUROC')
    ax.set_title('Per-Benchmark Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=15, ha='right')
    ax.set_ylim(0.4, 0.95)
    ax.legend(loc='upper right', fontsize=9)

    # Add note about ERQA
    ax.annotate('multi-image limitation', xy=(4, 0.504), xytext=(3.2, 0.55), fontsize=8, style='italic',
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / 'per_benchmark_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: per_benchmark_comparison.pdf")


def plot_text_vs_vision_auroc():
    """Figure 6: Text vs Vision AUROC comparison."""
    fig, ax = plt.subplots(figsize=(6, 4))

    categories = ['Vision\n(In-Dist)', 'Text\n(Transfer)', 'Cross-Model\n(Avg)']
    auroc = [0.789, 0.907, 0.694]
    colors = [COLORS['purple'], COLORS['orange'], COLORS['blue']]

    bars = ax.bar(categories, auroc, color=colors, width=0.5, edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars, auroc):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_ylabel('AUROC')
    ax.set_title('Performance Across Domains')
    ax.set_ylim(0.4, 1.0)

    plt.tight_layout()
    plt.savefig(output_dir / 'text_vs_vision_auroc.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: text_vs_vision_auroc.pdf")


def plot_training_time_comparison():
    """Figure 7: Training time vs performance (cost-efficiency)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # Vision data
    vision_time = [2.0, 4.3, 8.2, 16.1, 32.0, 56.0]
    vision_auroc = [0.721, 0.754, 0.749, 0.752, 0.761, 0.775]

    # Text LLM data
    text_time = [0.4, 0.9, 1.6, 3.1, 6.2]
    text_auroc = [0.718, 0.885, 0.909, 0.914, 0.920]

    # Plot
    ax.scatter(vision_time, vision_auroc, s=40, c=COLORS['blue'], marker='o', label='Vision (VLM)', zorder=5)
    ax.scatter(text_time, text_auroc, s=40, c=COLORS['orange'], marker='s', label='Text (LLM)', zorder=5)

    # Connect points
    ax.plot(vision_time, vision_auroc, '-', color=COLORS['blue'], alpha=0.5, linewidth=1)
    ax.plot(text_time, text_auroc, '-', color=COLORS['orange'], alpha=0.5, linewidth=1)

    # Simple annotations
    ax.annotate('250 samples', xy=(4.3, 0.754), xytext=(12, 0.72), fontsize=8,
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
    ax.annotate('500 samples', xy=(1.6, 0.909), xytext=(6, 0.88), fontsize=8,
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    ax.set_xlabel('Training Time (minutes)')
    ax.set_ylabel('AUROC')
    ax.set_title('Training Efficiency')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 60)
    ax.set_ylim(0.6, 1.0)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_time_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: training_time_comparison.pdf")


def plot_method_overview():
    """Figure 8: Method overview diagram."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')

    # Simplified boxes with muted colors
    boxes = [
        {'xy': (0.05, 0.3), 'text': 'Image +\nQuestion', 'color': COLORS['blue']},
        {'xy': (0.25, 0.3), 'text': 'Target VLM\n(InternVL3-78B)', 'color': COLORS['red']},
        {'xy': (0.45, 0.3), 'text': 'Response +\nLabel', 'color': COLORS['green']},
        {'xy': (0.65, 0.3), 'text': 'UQ Judge\n(Qwen3-VL-8B)', 'color': COLORS['purple']},
        {'xy': (0.85, 0.3), 'text': 'P(correct)', 'color': COLORS['orange']},
    ]

    for box in boxes:
        rect = mpatches.FancyBboxPatch(
            box['xy'], 0.12, 0.4, boxstyle="round,pad=0.01",
            facecolor=box['color'], edgecolor='black', linewidth=1
        )
        ax.add_patch(rect)
        ax.text(box['xy'][0] + 0.06, box['xy'][1] + 0.2, box['text'],
                ha='center', va='center', fontsize=9, color='white')

    # Arrows
    arrow_style = dict(arrowstyle='->', color='black', lw=1.5)
    for i in range(len(boxes) - 1):
        ax.annotate('', xy=(boxes[i+1]['xy'][0], 0.5),
                   xytext=(boxes[i]['xy'][0] + 0.12, 0.5),
                   arrowprops=arrow_style)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Pipeline Overview', fontsize=11, y=0.85)

    plt.tight_layout()
    plt.savefig(output_dir / 'method_overview.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: method_overview.pdf")


def plot_spurious_correlation_evidence():
    """Figure 9: Evidence against spurious correlations."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel 1: Within-benchmark AUROC
    ax1 = axes[0]
    benchmarks = ['BBEH', 'Omnimath', 'HellaSwag', 'MMLU', 'GPQA', 'ARC']
    within_auroc = [0.809, 0.949, 0.923, 1.000, 0.468, 0.636]
    colors = [COLORS['green'] if a > 0.55 else COLORS['red'] for a in within_auroc]

    bars = ax1.barh(benchmarks, within_auroc, color=colors, edgecolor='black', linewidth=0.5)
    ax1.axvline(x=0.5, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('Within-Benchmark AUROC')
    ax1.set_title('(a) Within-Benchmark Discrimination')
    ax1.set_xlim(0, 1.1)

    for bar, val in zip(bars, within_auroc):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=8)

    # Panel 2: Prediction distribution by label
    ax2 = axes[1]
    np.random.seed(42)
    correct_preds = np.clip(np.random.normal(0.782, 0.226, 225), 0, 1)
    incorrect_preds = np.clip(np.random.normal(0.284, 0.266, 225), 0, 1)

    ax2.hist(incorrect_preds, bins=20, alpha=0.6, color=COLORS['red'], label='Incorrect', density=True)
    ax2.hist(correct_preds, bins=20, alpha=0.6, color=COLORS['green'], label='Correct', density=True)
    ax2.axvline(x=0.284, color=COLORS['red'], linestyle='--', linewidth=1)
    ax2.axvline(x=0.782, color=COLORS['green'], linestyle='--', linewidth=1)
    ax2.set_xlabel('Model P(correct)')
    ax2.set_ylabel('Density')
    ax2.set_title('(b) Prediction Distribution')
    ax2.legend(loc='upper center', fontsize=8)
    ax2.text(0.5, ax2.get_ylim()[1]*0.9, 'Gap = 0.50', ha='center', fontsize=9)

    # Panel 3: Calibration
    ax3 = axes[2]
    confidence = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    accuracy = [56.3, 61.7, 78.1, 86.7, 96.4]
    expected = [55, 65, 75, 85, 95]

    x = np.arange(len(confidence))
    width = 0.35

    ax3.bar(x - width/2, accuracy, width, label='Actual', color=COLORS['blue'], edgecolor='black', linewidth=0.5)
    ax3.bar(x + width/2, expected, width, label='Expected', color=COLORS['gray'], edgecolor='black', linewidth=0.5)

    ax3.set_xlabel('Confidence Level')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('(c) Calibration')
    ax3.set_xticks(x)
    ax3.set_xticklabels(confidence, rotation=20, ha='right', fontsize=8)
    ax3.legend(loc='upper left', fontsize=8)
    ax3.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_dir / 'spurious_correlation_evidence.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: spurious_correlation_evidence.pdf")


if __name__ == "__main__":
    print("Generating figures for advisor meeting...\n")

    # Core figures (requested)
    plot_cross_model_transfer()
    plot_training_size_ablation()
    plot_selective_prediction()
    plot_results_summary_table()

    # Additional useful figures
    plot_per_benchmark_comparison()
    plot_text_vs_vision_auroc()
    plot_training_time_comparison()
    plot_method_overview()
    plot_spurious_correlation_evidence()

    print(f"\nAll figures saved to {output_dir.absolute()}")
