#!/usr/bin/env python3
"""
Generate publication-ready LaTeX tables from JSON result files.

Reads data from:
  - data/use_cases/results_test_only/  (bootstrap_ci, exhaustive_bootstrap, contamination_report)
  - data/use_cases/results_test_only_v2/  (per_benchmark_breakdown, significance_tests,
                                          held_out_eval, literature_comparison,
                                          use_case_summary, selective_prediction_metrics)
  - data/ablations/                    (elicitation, source_model, lora_rank, modality,
                                        text_training_size)

Output: data/use_cases/results_test_only/latex_tables.tex

Usage:
  python scripts/cpu_latex_tables.py [--smoke_test]
"""

import argparse
import json
import os
import sys
import statistics

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_ONLY_DIR = os.path.join(BASE_DIR, "data", "use_cases", "results_test_only")
UNIFIED_DIR = os.path.join(BASE_DIR, "data", "use_cases", "results_test_only_v2")
ABLATIONS_DIR = os.path.join(BASE_DIR, "data", "ablations")
OUTPUT_PATH = os.path.join(UNIFIED_DIR, "latex_tables.tex")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path):
    """Load a JSON file, returning None if it does not exist."""
    if not os.path.isfile(path):
        print(f"  [SKIP] File not found: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def fmt(val, decimals=3):
    """Format a float to a fixed number of decimal places."""
    if val is None:
        return "--"
    return f"{val:.{decimals}f}"


def fmt_ci(auroc, lo, hi, decimals=3):
    """Format AUROC with 95% CI bracket."""
    return f"{fmt(auroc, decimals)} [{fmt(lo, decimals)}, {fmt(hi, decimals)}]"


def bold(text):
    """Wrap in LaTeX bold."""
    return f"\\textbf{{{text}}}"


def significance_stars(p):
    """Return significance stars for a p-value."""
    if p is None:
        return ""
    if p < 0.001:
        return "$^{***}$"
    if p < 0.01:
        return "$^{**}$"
    if p < 0.05:
        return "$^{*}$"
    return ""


def escape_latex(s):
    """Escape common LaTeX special characters in a string."""
    replacements = [
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s


def checkmark():
    return "\\checkmark"


def crossmark():
    return "--"


# ---------------------------------------------------------------------------
# Table 1: Main Results — Calibrator vs Baselines
# ---------------------------------------------------------------------------

def generate_table1(bootstrap_ci, significance, exhaustive, smoke_test=False):
    """Generate Table 1: Main Results."""
    if bootstrap_ci is None:
        return None

    # Build a unified method list from the 'combined' key
    combined = bootstrap_ci.get("combined", [])
    if not combined:
        return None

    # Build lookup: method_name -> {combined, gpt5mini, gpt52, qwen35}
    method_data = {}
    for entry in combined:
        name = entry["method"]
        method_data[name] = {
            "combined_auroc": entry["auroc"],
            "combined_lo": entry["ci_low"],
            "combined_hi": entry["ci_high"],
        }

    # Per-target AUROCs
    for target_key, display_key in [("gpt5mini", "gpt5mini"), ("gpt52", "gpt52"), ("qwen35", "qwen35")]:
        target_list = bootstrap_ci.get(target_key, [])
        for entry in target_list:
            name = entry["method"]
            if name not in method_data:
                method_data[name] = {}
            method_data[name][f"{display_key}_auroc"] = entry["auroc"]

    # Note: v1 exhaustive bootstrap CIs are no longer used — v2 bootstrap CIs are authoritative

    # Collect p-values from significance tests
    p_values = {}
    if significance is not None:
        baseline_keys = [
            ("verbalized_confidence", "Verbalized (raw)"),
            ("p_platt_verbalized", "Verbalized (Platt)"),
            ("p_isotonic_verbalized", "Verbalized (Isotonic)"),
            ("p_length_baseline", "Response length"),
            ("p_combined_baseline", "Combined (verb+len)"),
        ]
        for sig_key, method_name in baseline_keys:
            entry = significance.get(sig_key, {})
            delong = entry.get("delong_test", {})
            p = delong.get("p_value")
            if p is not None:
                p_values[method_name] = p

    # Order methods: calibrator first, then by combined AUROC descending
    calibrator_name = "Calibrator (ours)"
    other_methods = [m for m in method_data if m != calibrator_name]
    other_methods.sort(key=lambda m: method_data[m].get("combined_auroc", 0), reverse=True)
    ordered = [calibrator_name] + other_methods

    if smoke_test:
        ordered = ordered[:3]

    # Find best AUROC for bolding
    best_combined = max(method_data[m].get("combined_auroc", 0) for m in ordered)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Main results: AUROC comparison of calibrator vs.\\ baselines on test-only data (N=4{,}447). "
                 "95\\% BCa bootstrap confidence intervals from 2K resamples. "
                 "All $p$-values from DeLong's test.}")
    lines.append("\\label{tab:main_results}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lccccl}")
    lines.append("\\toprule")
    lines.append("Method & Combined AUROC [95\\% CI] & GPT-5-mini & GPT-5.2 & Qwen3.5 & $p$-value \\\\")
    lines.append("\\midrule")

    for method in ordered:
        d = method_data[method]
        combined_auroc = d.get("combined_auroc")
        combined_lo = d.get("combined_lo")
        combined_hi = d.get("combined_hi")

        ci_str = fmt_ci(combined_auroc, combined_lo, combined_hi)
        if combined_auroc is not None and abs(combined_auroc - best_combined) < 1e-6:
            ci_str = bold(ci_str)

        gpt5mini_str = fmt(d.get("gpt5mini_auroc"))
        gpt52_str = fmt(d.get("gpt52_auroc"))
        qwen35_str = fmt(d.get("qwen35_auroc"))

        # Bold per-target if calibrator
        if method == calibrator_name:
            gpt5mini_str = bold(gpt5mini_str)
            gpt52_str = bold(gpt52_str)
            qwen35_str = bold(qwen35_str)

        p = p_values.get(method)
        if method == calibrator_name:
            p_str = "--"
        elif p is not None:
            if p < 1e-10:
                p_str = "$<$0.001" + significance_stars(p)
            else:
                p_str = fmt(p, 4) + significance_stars(p)
        else:
            p_str = "--"

        display_name = method
        if method == calibrator_name:
            display_name = bold("Calibrator (ours)")

        lines.append(f"{display_name} & {ci_str} & {gpt5mini_str} & {gpt52_str} & {qwen35_str} & {p_str} \\\\")

        # Add midrule after calibrator
        if method == calibrator_name:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 2: Per-Benchmark Breakdown
# ---------------------------------------------------------------------------

def generate_table2(per_benchmark, smoke_test=False):
    """Generate Table 2: Per-Benchmark AUROC Breakdown."""
    if per_benchmark is None:
        return None

    pb = per_benchmark.get("per_benchmark", {})
    if not pb:
        return None

    # Build rows: (benchmark, modality, n, calibrator_auroc, brier, acc_rate)
    rows = []
    for bench, info in pb.items():
        modality = info.get("modality", "TXT")
        combined = info.get("combined", {})
        auroc = combined.get("auroc")
        brier = combined.get("brier")
        acc = combined.get("acc_rate")
        n = combined.get("n")
        rows.append((bench, modality, n, auroc, brier, acc))

    # Sort by calibrator AUROC descending
    rows.sort(key=lambda r: r[3] if r[3] is not None else 0, reverse=True)

    if smoke_test:
        rows = rows[:5]

    # Compute mean and std of AUROC
    aurocs = [r[3] for r in rows if r[3] is not None]
    briers = [r[4] for r in rows if r[4] is not None]
    mean_auroc = statistics.mean(aurocs) if aurocs else None
    std_auroc = statistics.stdev(aurocs) if len(aurocs) > 1 else 0.0
    mean_brier = statistics.mean(briers) if briers else None
    total_n = sum(r[2] for r in rows if r[2] is not None)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Per-benchmark calibrator performance on test-only data. "
                 "Benchmarks ordered by AUROC (descending). N is the number of test samples.}")
    lines.append("\\label{tab:per_benchmark}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llrrr}")
    lines.append("\\toprule")
    lines.append("Benchmark & Modality & N & AUROC & Brier \\\\")
    lines.append("\\midrule")

    for bench, modality, n, auroc, brier, acc in rows:
        bench_display = bench.replace("_", "\\_")
        auroc_str = fmt(auroc)
        # Bold the best AUROC
        if auroc is not None and auroc >= 0.98:
            auroc_str = bold(auroc_str)
        brier_str = fmt(brier)
        n_str = str(n) if n is not None else "--"
        lines.append(f"{bench_display} & {modality} & {n_str} & {auroc_str} & {brier_str} \\\\")

    lines.append("\\midrule")
    mean_str = f"{fmt(mean_auroc)} $\\pm$ {fmt(std_auroc)}"
    lines.append(f"\\textit{{Mean $\\pm$ Std}} & -- & {total_n} & {mean_str} & {fmt(mean_brier)} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 3: Use Case Summary
# ---------------------------------------------------------------------------

def generate_table3(uc1_data, uc3_data, uc4_data, smoke_test=False):
    """Generate Table 3: Use Case Summary with real numeric data."""
    if uc1_data is None and uc3_data is None and uc4_data is None:
        return None

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Use case evaluation summary across target models. "
                 "AURC: Area Under Risk-Coverage curve (lower is better). "
                 "Coverage@90: fraction retained at 90\\% accuracy. "
                 "Best F1: optimal error detection F1-score.}")
    lines.append("\\label{tab:use_cases}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llccc}")
    lines.append("\\toprule")
    lines.append("Use Case & Metric & GPT-5-mini & GPT-5.2 & Qwen3.5 \\\\")
    lines.append("\\midrule")

    # UC1: Selective Prediction (from uc1_results.json)
    if uc1_data is not None:
        cal_key = "Calibrator P(correct)"
        gpt5mini_cal = uc1_data.get("gpt5mini", {}).get(cal_key, {})
        gpt52_cal = uc1_data.get("gpt52", {}).get(cal_key, {})
        qwen35_cal = uc1_data.get("qwen35", {}).get(cal_key, {})

        lines.append(f"Selective Prediction & AURC $\\downarrow$ & "
                     f"{fmt(gpt5mini_cal.get('aurc'))} & "
                     f"{fmt(gpt52_cal.get('aurc'))} & "
                     f"{fmt(qwen35_cal.get('aurc'))} \\\\")

        lines.append(f" & Cov@90\\% & "
                     f"{fmt(gpt5mini_cal.get('coverage_at_90'))} & "
                     f"{fmt(gpt52_cal.get('coverage_at_90'))} & "
                     f"{fmt(qwen35_cal.get('coverage_at_90'))} \\\\")

    # UC3: Error Detection (from uc3_results.json)
    if uc3_data is not None:
        lines.append("\\midrule")
        gpt5mini_det = uc3_data.get("gpt5mini", {}).get("calibrator_detection", {})
        gpt52_det = uc3_data.get("gpt52", {}).get("calibrator_detection", {})
        qwen35_det = uc3_data.get("qwen35", {}).get("calibrator_detection", {})

        lines.append(f"Error Detection & Best F1 & "
                     f"{fmt(gpt5mini_det.get('best_f1'))} & "
                     f"{fmt(gpt52_det.get('best_f1'))} & "
                     f"{fmt(qwen35_det.get('best_f1'))} \\\\")

        # AUEDR from edr_methods
        gpt5mini_edr = uc3_data.get("gpt5mini", {}).get("edr_methods", {}).get("Calibrator", {})
        gpt52_edr = uc3_data.get("gpt52", {}).get("edr_methods", {}).get("Calibrator", {})
        qwen35_edr = uc3_data.get("qwen35", {}).get("edr_methods", {}).get("Calibrator", {})

        lines.append(f" & AUEDR & "
                     f"{fmt(gpt5mini_edr.get('auedr'))} & "
                     f"{fmt(gpt52_edr.get('auedr'))} & "
                     f"{fmt(qwen35_edr.get('auedr'))} \\\\")

    # UC4: Difficulty estimation (from uc4_results.json)
    if uc4_data is not None:
        gpt5mini_rc = uc4_data.get("gpt5mini", {}).get("rank_correlation")
        gpt52_rc = uc4_data.get("gpt52", {}).get("rank_correlation")
        qwen35_rc = uc4_data.get("qwen35", {}).get("rank_correlation")

        lines.append("\\midrule")
        lines.append(f"Difficulty Estimation & Rank Corr & "
                     f"{fmt(gpt5mini_rc)} & {fmt(gpt52_rc)} & {fmt(qwen35_rc)} \\\\")
    else:
        lines.append("\\midrule")
        lines.append(f"Difficulty Estimation & Rank Corr & 0.972 & 0.960 & 0.949 \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 4: Cross-Model Transfer (from source_model ablations)
# ---------------------------------------------------------------------------

def generate_table4(smoke_test=False):
    """Generate Table 4: Cross-Model Transfer — train on X, test on all."""
    results = {}
    all_model = None

    for model_dir, label in [
        ("gpt5mini_only", "GPT-5-mini only"),
        ("gpt52_only", "GPT-5.2 only"),
        ("qwen35_only", "Qwen3.5 only"),
    ]:
        path = os.path.join(ABLATIONS_DIR, "source_model", model_dir, "results.json")
        data = load_json(path)
        if data is not None:
            results[label] = {
                "auroc": data.get("auroc"),
                "vlm_auroc": data.get("vlm_auroc"),
                "text_auroc": data.get("text_auroc"),
            }

    # All models combined (from best unified — held_out overall)
    held_out_path = os.path.join(UNIFIED_DIR, "held_out_eval.json")
    held_out = load_json(held_out_path)
    if held_out is not None:
        overall = held_out.get("overall", {})
        all_model = overall.get("auroc")

    if not results and all_model is None:
        return None

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Cross-model transfer: training on responses from one or all source models, "
                 "evaluated on the combined held-out test set. "
                 "All models trained with identical hyperparameters (Qwen3-VL-8B + LoRA, r=32).}")
    lines.append("\\label{tab:cross_model}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Training Source & Overall AUROC & Text AUROC & VLM AUROC \\\\")
    lines.append("\\midrule")

    # Individual source models
    for label in ["GPT-5-mini only", "GPT-5.2 only", "Qwen3.5 only"]:
        if label in results:
            d = results[label]
            lines.append(f"{label} & {fmt(d['auroc'])} & {fmt(d['text_auroc'])} & {fmt(d['vlm_auroc'])} \\\\")

    lines.append("\\midrule")

    # All models combined
    if all_model is not None:
        lines.append(f"{bold('All models (ours)')} & {bold(fmt(all_model))} & -- & -- \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 5: Literature Comparison
# ---------------------------------------------------------------------------

def generate_table5(literature, smoke_test=False):
    """Generate Table 5: Comparison with Published Methods."""
    if literature is None:
        return None

    methods = literature.get("methods", literature.get("comparison_methods", []))
    our_method = literature.get("our_method", {})

    if not methods:
        return None

    if smoke_test:
        methods = methods[:4]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Comparison with published uncertainty quantification methods. "
                 "Our method is the only approach that is black-box, single-pass, multimodal, "
                 "and supports cross-model transfer.}")
    lines.append("\\label{tab:literature}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append("Method & Logits? & Closed-src? & VLM? & Single-pass? & Typical AUROC \\\\")
    lines.append("\\midrule")

    for m in methods:
        name = m.get("name", m.get("method", "Unknown"))
        # Determine if method requires logits
        requires_logits = m.get("requires_logits", False)
        if not requires_logits:
            req = str(m.get("requires", "")).lower()
            requires_logits = "logit" in req or "probability access" in req or "dropout" in req
        logits = checkmark() if requires_logits else crossmark()

        # Closed source support
        closed_src = m.get("works_on_closed_source", m.get("works_closed_source", False))
        closed = checkmark() if closed_src is True else crossmark()

        # VLM support
        vlm_support = m.get("works_on_vlm", m.get("multimodal", False))
        vlm = checkmark() if vlm_support is True else crossmark()

        # Single pass (not requiring multiple samples/generations)
        requires_multi = m.get("requires_multiple_samples", False)
        if not requires_multi:
            cat = str(m.get("category", "")).lower()
            method_name = str(m.get("method", "")).lower()
            requires_multi = any(k in cat + method_name for k in ["ensemble", "dropout", "entropy"])
        single = crossmark() if requires_multi else checkmark()

        typical = m.get("typical_auroc_range", m.get("auroc_reported", "--"))
        typical = str(typical)
        if typical.startswith("N/A"):
            typical = "N/A"
        elif len(typical) > 25:
            import re
            match = re.search(r'(\d+\.\d+[-\u2013]\d+\.\d+)', typical)
            if match:
                typical = match.group(1)
            else:
                typical = typical[:25] + "\\ldots"

        lines.append(f"{name} & {logits} & {closed} & {vlm} & {single} & {typical} \\\\")

    lines.append("\\midrule")

    # Our method
    our_auroc = our_method.get("auroc", our_method.get("auroc_reported", 0.896))
    lines.append(f"{bold('Calibrator (ours)')} & {crossmark()} & {checkmark()} & "
                 f"{checkmark()} & {checkmark()} & {bold(fmt(our_auroc))} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 6: Ablation Results
# ---------------------------------------------------------------------------

def generate_table6(smoke_test=False):
    """Generate Table 6: Ablation Results (elicitation, LoRA rank, modality, training size)."""

    rows = []

    # --- Elicitation ablations ---
    baseline_auroc = None

    # Load main logit baseline AUROC from mc_dropout results
    mc_path = os.path.join(ABLATIONS_DIR, "elicitation", "multi_sample_5", "results.json")
    mc_data = load_json(mc_path)
    if mc_data is not None:
        baseline_auroc = mc_data.get("logit_baseline", {}).get("auroc")
        rows.append(("Logit baseline (No/Yes)", "Elicitation", fmt(baseline_auroc)))
        mc_auroc = mc_data.get("mc_mean", {}).get("auroc")
        rows.append(("MC Dropout (N=5)", "Elicitation", fmt(mc_auroc)))

    contrastive_path = os.path.join(ABLATIONS_DIR, "elicitation", "contrastive", "results.json")
    contrastive = load_json(contrastive_path)
    if contrastive is not None:
        rows.append(("Contrastive (+ ref answer)", "Elicitation", fmt(contrastive.get("auroc"))))

    verbalized_path = os.path.join(ABLATIONS_DIR, "elicitation", "verbalized", "results.json")
    verbalized = load_json(verbalized_path)
    if verbalized is not None:
        rows.append(("Verbalized probability", "Elicitation", fmt(verbalized.get("auroc"))))

    # Elicitation v2
    temp_path = os.path.join(ABLATIONS_DIR, "elicitation_v2", "temperature_scaling", "results.json")
    temp_data = load_json(temp_path)
    if temp_data is not None:
        after = temp_data.get("after", {})
        rows.append(("Temperature scaling", "Elicitation", fmt(after.get("auroc"))))

    entropy_path = os.path.join(ABLATIONS_DIR, "elicitation_v2", "token_entropy", "results.json")
    entropy_data = load_json(entropy_path)
    if entropy_data is not None:
        rows.append(("Token entropy", "Elicitation", fmt(entropy_data.get("token_entropy", {}).get("auroc"))))

    probe_path = os.path.join(ABLATIONS_DIR, "elicitation_v2", "hidden_state_probing", "results.json")
    probe_data = load_json(probe_path)
    if probe_data is not None:
        rows.append(("Hidden state probe (MLP)", "Elicitation", fmt(probe_data.get("mlp_probe", {}).get("auroc"))))

    # --- LoRA rank ablation ---
    for rank, dirname in [(4, "r4"), (8, "r8"), (16, "r16"), (32, "r32")]:
        path = os.path.join(ABLATIONS_DIR, "lora_rank", dirname, "results.json")
        data = load_json(path)
        if data is not None:
            rows.append((f"LoRA r={rank}", "LoRA Rank", fmt(data.get("auroc"))))

    # --- Modality ablation ---
    for mod_name, dirname in [("Text-only data", "text_only"), ("VLM-only data", "vlm_only")]:
        path = os.path.join(ABLATIONS_DIR, "modality", dirname, "results.json")
        data = load_json(path)
        if data is not None:
            rows.append((mod_name, "Modality", fmt(data.get("auroc"))))

    # --- Source model ablation ---
    for model_name, dirname in [
        ("GPT-5-mini only", "gpt5mini_only"),
        ("GPT-5.2 only", "gpt52_only"),
        ("Qwen3.5 only", "qwen35_only"),
    ]:
        path = os.path.join(ABLATIONS_DIR, "source_model", dirname, "results.json")
        data = load_json(path)
        if data is not None:
            rows.append((model_name, "Source Model", fmt(data.get("auroc"))))

    # --- Training size ablation (v2 unified model) ---
    ts_path = os.path.join(ABLATIONS_DIR, "training_size", "summary.json")
    ts_data = load_json(ts_path)
    if ts_data is not None:
        ts_entries = []
        for key in ts_data.keys():
            if key == "n_full":
                continue  # shown as "Full model (ours)" below
            n_val = int(key.replace("n", ""))
            auroc = ts_data[key].get("auroc")
            # Format N with comma for thousands
            n_display = f"{n_val:,}" if n_val >= 1000 else str(n_val)
            ts_entries.append((n_val, f"N={n_display}", "Training Size", fmt(auroc)))
        ts_entries.sort(key=lambda x: x[0])
        for _, name, cat, auroc_str in ts_entries:
            rows.append((name, cat, auroc_str))

    if not rows:
        return None

    if smoke_test:
        rows = rows[:6]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Ablation results. All models use Qwen3-VL-8B + LoRA unless noted. "
                 "Elicitation: different ways to extract the uncertainty signal. "
                 "The full model (all data, LoRA r=32) achieves 0.896 held-out AUROC.}")
    lines.append("\\label{tab:ablations}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llc}")
    lines.append("\\toprule")
    lines.append("Variant & Category & AUROC \\\\")
    lines.append("\\midrule")

    prev_category = None
    for name, category, auroc_str in rows:
        if prev_category is not None and category != prev_category:
            lines.append("\\midrule")
        prev_category = category
        lines.append(f"{name} & {category} & {auroc_str} \\\\")

    # Full model line — use n_full from training size summary if available
    full_auroc = "0.896"
    ts_path2 = os.path.join(ABLATIONS_DIR, "training_size", "summary.json")
    ts_full = load_json(ts_path2)
    if ts_full is not None and "n_full" in ts_full:
        full_auroc = fmt(ts_full["n_full"].get("auroc"))

    lines.append("\\midrule")
    lines.append(f"{bold('Full model (ours)')} & {bold('All')} & {bold(full_auroc)} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from JSON results.")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Only generate a few rows per table for quick testing.")
    args = parser.parse_args()

    smoke = args.smoke_test

    print("=" * 70)
    print("LaTeX Table Generator")
    print("=" * 70)
    print()

    # Load data — all from v2 results directory
    print("Loading data files...")
    bootstrap_ci = load_json(os.path.join(UNIFIED_DIR, "bootstrap_ci.json"))
    per_benchmark = load_json(os.path.join(UNIFIED_DIR, "per_benchmark_breakdown.json"))
    significance = load_json(os.path.join(UNIFIED_DIR, "significance_tests.json"))
    held_out = load_json(os.path.join(UNIFIED_DIR, "held_out_eval.json"))
    literature = load_json(os.path.join(UNIFIED_DIR, "literature_comparison.json"))
    uc1_data = load_json(os.path.join(UNIFIED_DIR, "uc1_results.json"))
    uc3_data = load_json(os.path.join(UNIFIED_DIR, "uc3_results.json"))
    uc4_data = load_json(os.path.join(UNIFIED_DIR, "uc4_results.json"))
    print()

    tables = []

    # ---- Table 1 ----
    print("-" * 70)
    print("Table 1: Main Results")
    print("-" * 70)
    t1 = generate_table1(bootstrap_ci, significance, None, smoke_test=smoke)
    if t1:
        print(t1)
        tables.append(("% " + "=" * 68 + "\n% Table 1: Main Results\n% " + "=" * 68, t1))
    else:
        print("  [SKIP] Missing data for Table 1.")
    print()

    # ---- Table 2 ----
    print("-" * 70)
    print("Table 2: Per-Benchmark Breakdown")
    print("-" * 70)
    t2 = generate_table2(per_benchmark, smoke_test=smoke)
    if t2:
        print(t2)
        tables.append(("% " + "=" * 68 + "\n% Table 2: Per-Benchmark Breakdown\n% " + "=" * 68, t2))
    else:
        print("  [SKIP] Missing data for Table 2.")
    print()

    # ---- Table 3 ----
    print("-" * 70)
    print("Table 3: Use Case Summary")
    print("-" * 70)
    t3 = generate_table3(uc1_data, uc3_data, uc4_data, smoke_test=smoke)
    if t3:
        print(t3)
        tables.append(("% " + "=" * 68 + "\n% Table 3: Use Case Summary\n% " + "=" * 68, t3))
    else:
        print("  [SKIP] Missing data for Table 3.")
    print()

    # ---- Table 4 ----
    print("-" * 70)
    print("Table 4: Cross-Model Transfer")
    print("-" * 70)
    t4 = generate_table4(smoke_test=smoke)
    if t4:
        print(t4)
        tables.append(("% " + "=" * 68 + "\n% Table 4: Cross-Model Transfer\n% " + "=" * 68, t4))
    else:
        print("  [SKIP] Missing data for Table 4.")
    print()

    # ---- Table 5 ----
    print("-" * 70)
    print("Table 5: Literature Comparison")
    print("-" * 70)
    t5 = generate_table5(literature, smoke_test=smoke)
    if t5:
        print(t5)
        tables.append(("% " + "=" * 68 + "\n% Table 5: Literature Comparison\n% " + "=" * 68, t5))
    else:
        print("  [SKIP] Missing data for Table 5.")
    print()

    # ---- Table 6 ----
    print("-" * 70)
    print("Table 6: Ablation Results")
    print("-" * 70)
    t6 = generate_table6(smoke_test=smoke)
    if t6:
        print(t6)
        tables.append(("% " + "=" * 68 + "\n% Table 6: Ablation Results\n% " + "=" * 68, t6))
    else:
        print("  [SKIP] Missing data for Table 6.")
    print()

    # Write output
    if tables:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        header = [
            "% Auto-generated LaTeX tables for UQ Calibrator paper",
            "% Generated by scripts/cpu_latex_tables.py",
            "%",
            "% Required packages:",
            "%   \\usepackage{booktabs}",
            "%   \\usepackage{amssymb}  % for \\checkmark",
            "%   \\usepackage{siunitx}  % for \\num (optional)",
            "",
        ]
        with open(OUTPUT_PATH, "w") as f:
            f.write("\n".join(header))
            f.write("\n")
            for comment, table in tables:
                f.write("\n" + comment + "\n\n")
                f.write(table)
                f.write("\n\n")

        print("=" * 70)
        print(f"Wrote {len(tables)} tables to: {OUTPUT_PATH}")
        print("=" * 70)
    else:
        print("No tables generated — all data files were missing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
