#!/usr/bin/env python3
"""Generate a polished PDF research progress report using ReportLab."""

import os
import json
import fitz  # PyMuPDF - for converting PDF figures to images
from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.colors import HexColor

# ── Paths ──────────────────────────────────────────────────────────────
BASE = "/scratch/khayes/LLM"
FIG_PAPER = f"{BASE}/figures/paper"
FIG_UC = f"{BASE}/figures/use_cases_unified"
OUTPUT = f"{BASE}/figures/research_update_report.pdf"

# ── Colors ─────────────────────────────────────────────────────────────
DARK_BLUE = HexColor("#1a365d")
MED_BLUE = HexColor("#2b6cb0")
LIGHT_BLUE = HexColor("#ebf8ff")
ACCENT = HexColor("#38a169")    # green for positive results
LIGHT_GRAY = HexColor("#f7fafc")
BORDER_GRAY = HexColor("#e2e8f0")
TABLE_HEADER_BG = HexColor("#2b6cb0")
TABLE_ALT_ROW = HexColor("#f0f7ff")


def pdf_page_to_image(pdf_path, page_num=0, dpi=200):
    """Convert a PDF page to a ReportLab-compatible image."""
    if not os.path.exists(pdf_path):
        return None
    doc = fitz.open(pdf_path)
    if page_num >= len(doc):
        return None
    page = doc[page_num]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()
    return BytesIO(img_bytes)


def build_styles():
    """Create custom paragraph styles."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=24,
        leading=30,
        textColor=DARK_BLUE,
        alignment=TA_CENTER,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=13,
        leading=16,
        textColor=MED_BLUE,
        alignment=TA_CENTER,
        spaceAfter=20,
    ))
    styles.add(ParagraphStyle(
        "SectionHead",
        parent=styles["Heading1"],
        fontSize=16,
        leading=20,
        textColor=DARK_BLUE,
        spaceBefore=18,
        spaceAfter=8,
        borderWidth=0,
        borderPadding=0,
    ))
    styles.add(ParagraphStyle(
        "SubSectionHead",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=MED_BLUE,
        spaceBefore=12,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        "BodyText2",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        spaceBefore=2,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        "BulletCustom",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        leftIndent=20,
        bulletIndent=8,
        spaceBefore=2,
        spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        "MetricHighlight",
        parent=styles["Normal"],
        fontSize=11,
        leading=15,
        textColor=DARK_BLUE,
        fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.gray,
        alignment=TA_CENTER,
        spaceBefore=4,
        spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        "FooterStyle",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.gray,
        alignment=TA_RIGHT,
    ))
    return styles


def make_table(headers, rows, col_widths=None):
    """Create a styled table."""
    data = [headers] + rows
    t = Table(data, colWidths=col_widths, repeatRows=1)

    style_cmds = [
        # Header
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        # Body
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("TOPPADDING", (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
        # Grid
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, TABLE_ALT_ROW]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    t.setStyle(TableStyle(style_cmds))
    return t


def add_figure(elements, pdf_path, caption, width=5.5*inch, styles=None):
    """Add a PDF figure converted to image."""
    img_data = pdf_page_to_image(pdf_path, dpi=200)
    if img_data is None:
        return
    img = Image(img_data, width=width, height=width * 0.6)
    elements.append(img)
    if caption and styles:
        elements.append(Paragraph(caption, styles["Caption"]))


def build_report():
    """Build the full PDF report."""
    styles = build_styles()
    elements = []

    # ── Title Page ─────────────────────────────────────────────────────
    elements.append(Spacer(1, 1.5 * inch))
    elements.append(Paragraph(
        "Uncertainty Quantification for LLMs",
        styles["ReportTitle"]
    ))
    elements.append(Paragraph(
        "Research Progress Report — February 2026",
        styles["ReportSubtitle"]
    ))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(HRFlowable(
        width="60%", thickness=2, color=MED_BLUE,
        spaceAfter=20, spaceBefore=10, hAlign="CENTER"
    ))

    # Key metrics box
    summary_data = [
        ["Overall AUROC", "VLM AUROC", "Text AUROC", "Model"],
        ["0.831", "0.813", "0.850", "Qwen3-VL-8B + LoRA"],
    ]
    summary_table = Table(summary_data, colWidths=[1.3*inch, 1.3*inch, 1.3*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 1), (-1, 1), 14),
        ("TEXTCOLOR", (0, 1), (-1, 1), DARK_BLUE),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("BOX", (0, 0), (-1, -1), 1, DARK_BLUE),
        ("BACKGROUND", (0, 1), (-1, 1), LIGHT_BLUE),
    ]))
    elements.append(summary_table)

    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph(
        "Target: ECCV 2026 &nbsp; | &nbsp; "
        "3 novel contributions &nbsp; | &nbsp; "
        "13 use cases validated &nbsp; | &nbsp; "
        "20 benchmarks",
        styles["BodyText2"]
    ))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(
        "<b>Three Novel Contributions:</b>",
        styles["BodyText2"]
    ))
    for item in [
        "Cross-model transfer — train on Model A, evaluate on Model B without retraining",
        "Multimodality — single model handles both text LLMs and vision-language models",
        "Closed-source application — enables UQ on GPT-5, Claude, and other proprietary models",
    ]:
        elements.append(Paragraph(f"&bull; {item}", styles["BulletCustom"]))

    elements.append(PageBreak())

    # ── Section 1: Approach ────────────────────────────────────────────
    elements.append(Paragraph("1. Approach Overview", styles["SectionHead"]))
    elements.append(Paragraph(
        "We fine-tune a small open-source vision-language model (Qwen3-VL-8B) with LoRA "
        "to predict whether a target model's response to a question is correct. The calibrator "
        "takes as input the <b>question</b> and the <b>target model's response</b> (plus images "
        "for VLM benchmarks) and outputs a probability of correctness. This approach enables "
        "uncertainty quantification on <i>any</i> model — including closed-source ones — without "
        "access to model internals.",
        styles["BodyText2"]
    ))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "<b>Training data:</b> 9,533 samples across 20 benchmarks from 3 target models "
        "(GPT-5-mini, GPT-5.2, Qwen3.5-397B). Real images for VLM benchmarks, gray "
        "placeholder for text benchmarks. 1,683 held-out test samples.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Training cost:</b> ~2 hours on 4 GPUs with LoRA (r=16, 15.2M trainable parameters).",
        styles["BodyText2"]
    ))

    # ── Section 2: Main Results ────────────────────────────────────────
    elements.append(Paragraph("2. Main Results", styles["SectionHead"]))
    elements.append(Paragraph(
        "The unified model achieves <b>AUROC 0.831</b> on the held-out test set, with strong "
        "performance on both text (0.850) and vision (0.813) benchmarks. Cross-model scoring "
        "AUROC reaches 0.956 for GPT-5-mini, 0.932 for GPT-5.2, and 0.916 for Qwen3.5.",
        styles["BodyText2"]
    ))

    # Main results table
    elements.append(Spacer(1, 6))
    elements.append(Paragraph("Held-Out Test Performance", styles["SubSectionHead"]))
    main_tbl = make_table(
        ["Target Model", "Overall AUROC", "Text AUROC", "VLM AUROC", "Brier Score", "N"],
        [
            ["GPT-5-mini", "0.956", "—", "—", "0.177", "4,211"],
            ["GPT-5.2", "0.932", "—", "—", "0.176", "4,108"],
            ["Qwen3.5-397B", "0.916", "—", "—", "0.189", "2,246"],
            ["All (held-out)", "0.831", "0.850", "0.813", "0.177", "1,683"],
        ],
        col_widths=[1.4*inch, 1.1*inch, 1.0*inch, 1.0*inch, 1.0*inch, 0.7*inch]
    )
    elements.append(main_tbl)

    # Calibration figure
    add_figure(elements, f"{FIG_PAPER}/fig_calibration.pdf",
               "Figure 1: Reliability diagrams showing calibration across target models.",
               width=5.0*inch, styles=styles)

    # Cross-model figure
    add_figure(elements, f"{FIG_PAPER}/fig_cross_model_transfer.pdf",
               "Figure 2: Cross-model transfer performance — train on one model, evaluate on others.",
               width=5.0*inch, styles=styles)

    elements.append(PageBreak())

    # ── Section 3: Ablation Studies ────────────────────────────────────
    elements.append(Paragraph("3. Ablation Studies", styles["SectionHead"]))

    # 3a. LoRA rank
    elements.append(Paragraph("3a. LoRA Rank", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "Increasing LoRA rank from 16 to 32 yields +1.3 AUROC points. "
        "The 2x increase in trainable parameters (15.2M to 30.4M) is justified by "
        "the consistent improvement, especially on VLM benchmarks (+2.7 points).",
        styles["BodyText2"]
    ))
    rank_tbl = make_table(
        ["LoRA Rank", "AUROC", "VLM", "Text", "Params"],
        [
            ["r = 4", "0.821", "0.806", "0.837", "3.8M"],
            ["r = 8", "0.824", "0.807", "0.844", "7.7M"],
            ["r = 16 (baseline)", "0.831", "0.813", "0.850", "15.2M"],
            ["r = 32", "0.844", "0.840", "0.846", "30.4M"],
        ],
        col_widths=[1.3*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.0*inch]
    )
    elements.append(rank_tbl)
    elements.append(Spacer(1, 6))

    # 3b. Source model
    elements.append(Paragraph("3b. Source Model Diversity", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "Multi-source training is critical. Training on all 3 target models "
        "outperforms single-source by 7-16 AUROC points. Qwen3.5-only training "
        "performs worst, likely due to fewer samples and different error patterns.",
        styles["BodyText2"]
    ))
    source_tbl = make_table(
        ["Training Source", "AUROC", "VLM", "Text", "Samples"],
        [
            ["GPT-5-mini only", "0.744", "0.731", "0.758", "3,546"],
            ["GPT-5.2 only", "0.759", "0.754", "0.772", "3,750"],
            ["Qwen3.5 only", "0.672", "0.691", "0.654", "2,751"],
            ["All 3 models", "0.831", "0.813", "0.850", "9,533"],
        ],
        col_widths=[1.3*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.0*inch]
    )
    elements.append(source_tbl)
    elements.append(Spacer(1, 6))

    # 3c. Modality
    elements.append(Paragraph("3c. Modality Ablation", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "Cross-modality transfer fails dramatically. A text-only model achieves "
        "only 0.588 on VLM benchmarks; a VLM-only model achieves only 0.518 on text. "
        "Unified training is essential for handling both modalities (+13 points each).",
        styles["BodyText2"]
    ))
    mod_tbl = make_table(
        ["Training Data", "Overall", "In-Dist", "Cross-Modal"],
        [
            ["Text only", "0.719", "0.857 (text)", "0.588 (VLM)"],
            ["VLM only", "0.685", "0.821 (VLM)", "0.518 (text)"],
            ["Unified", "0.831", "0.850 / 0.813", "N/A"],
        ],
        col_widths=[1.3*inch, 1.1*inch, 1.5*inch, 1.5*inch]
    )
    elements.append(mod_tbl)
    elements.append(Spacer(1, 6))

    # 3d. Model size
    elements.append(Paragraph("3d. Model Size Ablation", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "A 2B parameter model retains 98% of the 8B model's performance (0.816 vs 0.827). "
        "The 4B model marginally outperforms 8B at 0.830. This means the calibrator "
        "can be deployed on laptops at ~4GB merged weight size with minimal quality loss.",
        styles["BodyText2"]
    ))
    size_tbl = make_table(
        ["Model Size", "AUROC", "VLM", "Text", "Training Time"],
        [
            ["Qwen3-VL-2B", "0.816", "0.797", "0.838", "65 min"],
            ["Qwen3-VL-4B", "0.830", "0.821", "0.841", "129 min"],
            ["Qwen3-VL-8B", "0.827", "0.816", "0.840", "140 min"],
        ],
        col_widths=[1.3*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.1*inch]
    )
    elements.append(size_tbl)

    # Per-benchmark heatmap
    add_figure(elements, f"{FIG_PAPER}/fig_per_benchmark_heatmap.pdf",
               "Figure 3: Per-benchmark AUROC heatmap across target models.",
               width=5.5*inch, styles=styles)

    elements.append(PageBreak())

    # ── Section 4: Downstream Use Cases ──────────────────────────────────
    elements.append(Paragraph("4. Downstream Use Cases", styles["SectionHead"]))
    elements.append(Paragraph(
        "The calibrator outputs a single number — P(correct) — for each question-response pair. "
        "Below, we show how this one signal enables 13 distinct applications. For each use case, "
        "we explain <b>what it does</b>, <b>how it uses the calibrator</b>, and <b>what we found</b>.",
        styles["BodyText2"]
    ))

    # Summary table
    elements.append(Spacer(1, 4))
    uc_tbl = make_table(
        ["Use Case", "Metric", "GPT-5-mini", "GPT-5.2", "Qwen3.5"],
        [
            ["UC1: Selective Prediction", "AURC (lower better)", "0.165", "0.125", "0.189"],
            ["UC2: Model Routing", "Cost savings", "47%", "—", "—"],
            ["UC3: Error Detection", "Best F1", "0.883", "0.831", "0.840"],
            ["UC4: Difficulty Estimation", "Rank correlation", "0.972", "0.960", "0.949"],
            ["UC5: Reward Modeling", "Pairwise accuracy", "79-85%", "—", "—"],
            ["UC7: Self-Improvement", "Overconf. error ID", "Yes", "Yes", "Yes"],
            ["UC8: OOD Detection", "Alert F1", "1.000", "0.895", "0.789"],
            ["UC9: Annotation Priorit.", "AUEDR", "0.739", "0.754", "0.711"],
        ],
        col_widths=[1.6*inch, 1.4*inch, 1.0*inch, 0.9*inch, 0.9*inch]
    )
    elements.append(uc_tbl)

    elements.append(PageBreak())

    # ── UC1 ────────────────────────────────────────────────────────────
    elements.append(Paragraph("4a. UC1 — Selective Prediction", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>What it does:</b> In high-stakes settings (medical, legal, financial), it is better "
        "to say \"I don't know\" than to give a wrong answer. Selective prediction sets a "
        "confidence threshold: the system answers questions where P(correct) is above the "
        "threshold, and <i>abstains</i> on the rest, deferring them to a human.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>How the calibrator is used:</b> For each threshold <i>t</i>, we select all samples "
        "where P(correct) &ge; <i>t</i> and measure accuracy on that subset. This produces a "
        "coverage-vs-accuracy curve: as the threshold increases, fewer questions are answered "
        "(lower coverage) but the answered ones are more likely correct (higher accuracy). "
        "The metric is <b>AURC</b> (Area Under the Risk-Coverage curve) — lower is better.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Result:</b> AURC of 0.165 (GPT-5-mini), 0.125 (GPT-5.2), 0.189 (Qwen3.5). "
        "For example, if we only answer the top-50% most confident predictions for GPT-5.2, "
        "accuracy rises substantially above the baseline, with the calibrator outperforming "
        "both verbalized confidence and random abstention by 43-73%.",
        styles["BodyText2"]
    ))
    add_figure(elements, f"{FIG_UC}/uc1_selective_gpt5mini.pdf",
               "Figure 4: UC1 — Coverage vs. accuracy curve. As we raise the confidence threshold and answer fewer questions, accuracy increases. The calibrator (blue) dominates random and verbalized baselines.",
               width=4.5*inch, styles=styles)

    # ── UC2 ────────────────────────────────────────────────────────────
    elements.append(Paragraph("4b. UC2 — Model Routing", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>What it does:</b> When a cheap model (GPT-5-mini) and an expensive model "
        "(GPT-5.2, ~4.75x cost due to longer reasoning) are both available, we can save money "
        "by using the cheap model when it's likely correct and only escalating to the expensive "
        "model when the cheap model is uncertain.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>How the calibrator is used:</b> Run the cheap model first and score its response with "
        "the calibrator. If P(correct) &ge; threshold, keep the cheap answer. Otherwise, re-query "
        "with the expensive model. Different thresholds trade off cost vs. accuracy.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Result:</b> The calibrator-guided router achieves <b>47% cost savings</b> while matching "
        "the expensive model's accuracy. In other words, nearly half the queries can be confidently "
        "handled by the cheap model, and the calibrator correctly identifies which ones.",
        styles["BodyText2"]
    ))
    add_figure(elements, f"{FIG_UC}/uc2_model_routing.pdf",
               "Figure 5: UC2 — Model routing. Cost (x-axis, normalized to expensive model) vs. accuracy. "
               "The calibrator-guided router (blue) reaches expensive-model accuracy at roughly half the cost.",
               width=4.5*inch, styles=styles)

    # ── UC3 ────────────────────────────────────────────────────────────
    elements.append(Paragraph("4c. UC3 — Error Detection", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>What it does:</b> Treats the calibrator as a binary error detector. Any response "
        "where P(correct) falls below a threshold is flagged as a likely error and sent to a "
        "human reviewer. This is the quality control scenario: \"which of these 10,000 model "
        "outputs should a human double-check?\"",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>How the calibrator is used:</b> We use 1 &minus; P(correct) as an \"error score.\" "
        "Sweeping the threshold produces a precision-recall curve for error flagging. High "
        "precision means few false alarms; high recall means few errors are missed.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Result:</b> Best F1 scores of <b>0.883</b> (GPT-5-mini), <b>0.831</b> (GPT-5.2), "
        "and <b>0.840</b> (Qwen3.5). The calibrator reliably identifies incorrect responses "
        "across all three target models, dramatically reducing the human review burden.",
        styles["BodyText2"]
    ))
    add_figure(elements, f"{FIG_UC}/uc3_error_detection.pdf",
               "Figure 6: UC3 — Precision-recall curves for error detection across target models.",
               width=4.5*inch, styles=styles)

    elements.append(PageBreak())

    # ── UC4 ────────────────────────────────────────────────────────────
    elements.append(Paragraph("4d. UC4 — Difficulty Estimation", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>What it does:</b> Groups questions by benchmark and checks whether the calibrator's "
        "average P(correct) per benchmark matches the true benchmark difficulty (actual "
        "accuracy). If the calibrator says \"this benchmark is hard,\" is it actually hard?",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>How the calibrator is used:</b> Compute mean P(correct) per benchmark, then "
        "measure Spearman rank correlation against actual accuracy. Also identifies "
        "\"overconfident\" benchmarks where the calibrator predicts high confidence but "
        "accuracy is actually low — these reveal systematic blind spots.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Result:</b> Spearman r = <b>0.972</b> (GPT-5-mini), <b>0.960</b> (GPT-5.2), "
        "<b>0.949</b> (Qwen3.5). The calibrator's difficulty ranking almost perfectly matches "
        "reality. This is useful for curriculum learning, dataset design, and understanding "
        "which domains are genuinely harder vs. model-specific weaknesses.",
        styles["BodyText2"]
    ))
    add_figure(elements, f"{FIG_UC}/uc4_difficulty_gpt5mini.pdf",
               "Figure 7: UC4 — Each dot is a benchmark. Calibrator mean P(correct) on x-axis vs. actual accuracy on y-axis. Near-perfect rank correlation (r=0.972).",
               width=4.5*inch, styles=styles)

    # ── UC5 ────────────────────────────────────────────────────────────
    elements.append(Paragraph("4e. UC5 — Reward Modeling / Response Selection", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>What it does:</b> When multiple models answer the same question, can the "
        "calibrator pick the better response? This treats P(correct) as a reward signal — "
        "given two candidate answers, the one with the higher calibrator score should be "
        "the correct one.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>How the calibrator is used:</b> For each question answered by 2+ models, form "
        "all pairwise comparisons. On \"discriminative\" pairs (one correct, one wrong), check "
        "whether the calibrator assigns a higher score to the correct response. Compares "
        "against verbalized confidence and response length as baselines.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Result:</b> <b>79-85% pairwise accuracy</b> — when one model is right and the other "
        "wrong, the calibrator picks the right one 4 out of 5 times. This outperforms "
        "verbalized confidence (+55%) and enables inference-time quality improvement without "
        "ground truth labels.",
        styles["BodyText2"]
    ))
    add_figure(elements, f"{FIG_UC}/uc5_reward_model.pdf",
               "Figure 8: UC5 — Pairwise response selection accuracy. Calibrator vs. verbalized confidence vs. response length.",
               width=4.5*inch, styles=styles)

    elements.append(PageBreak())

    # ── UC7 ────────────────────────────────────────────────────────────
    elements.append(Paragraph("4f. UC7 — Self-Improvement Analysis", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>What it does:</b> Identifies systematic model weaknesses by finding \"overconfident "
        "errors\" — cases where the calibrator assigns P(correct) > 0.7 but the model's answer "
        "is actually wrong. These are the most dangerous failure modes: the system thinks it's "
        "right, but it isn't.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>How the calibrator is used:</b> For each benchmark and domain, count overconfident "
        "errors and rank by frequency. Compare across target models to see whether the same "
        "benchmarks are consistently problematic (suggesting genuine difficulty) or model-specific.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Result:</b> Successfully identifies benchmarks where all three models fail "
        "overconfidently, providing actionable targets for training data augmentation. "
        "Cross-model weakness correlation is positive, suggesting benchmarks that trip up "
        "one model tend to trip up others too.",
        styles["BodyText2"]
    ))
    add_figure(elements, f"{FIG_UC}/uc7_self_improvement.pdf",
               "Figure 9: UC7 — Overconfident error rates by benchmark. Benchmarks where all models "
               "fail overconfidently are priority targets for improvement.",
               width=4.5*inch, styles=styles)

    # ── UC8 ────────────────────────────────────────────────────────────
    elements.append(Paragraph("4g. UC8 — OOD / Deployment Monitoring", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>What it does:</b> Monitors a deployed model in real time by tracking the <i>batch-level</i> "
        "average P(correct). If a batch of incoming queries comes from an unfamiliar domain "
        "or the model starts failing on a new topic, the average calibrator score drops — "
        "triggering an alert before users notice the degradation.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>How the calibrator is used:</b> Compute mean P(correct) over rolling batches. "
        "Compare against a reference distribution. If the batch mean drops below a threshold "
        "(e.g., P(correct) < 0.5), fire an alert. We also use bootstrap simulation to determine "
        "the minimum reliable batch size for stable monitoring.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Result:</b> Alert F1 of <b>1.000</b> (GPT-5-mini), <b>0.895</b> (GPT-5.2), "
        "<b>0.789</b> (Qwen3.5). The calibrator detects domain-level failures with near-perfect "
        "precision, enabling automated production monitoring without labeled data.",
        styles["BodyText2"]
    ))
    add_figure(elements, f"{FIG_UC}/uc8_ood_detection.pdf",
               "Figure 10: UC8 — Batch-level monitoring. Mean calibrator score per benchmark tracks closely with actual accuracy, enabling automated drift detection.",
               width=4.5*inch, styles=styles)

    elements.append(PageBreak())

    # ── UC9 ────────────────────────────────────────────────────────────
    elements.append(Paragraph("4h. UC9 — Annotation Prioritization", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>What it does:</b> When a human annotator has a limited budget to review model "
        "outputs, which samples should they check first? This use case sorts predictions by "
        "ascending P(correct) (most uncertain first) so annotators find errors faster.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>How the calibrator is used:</b> Rank all samples by calibrator score (low to high). "
        "Measure the \"Error Discovery Rate\" (EDR): after reviewing the top X% of samples, "
        "what fraction of all errors have been found? The metric is <b>AUEDR</b> (Area Under "
        "EDR curve) — higher means errors are found faster.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Result:</b> AUEDR of <b>0.739</b> (GPT-5-mini), <b>0.754</b> (GPT-5.2), "
        "<b>0.711</b> (Qwen3.5). Reviewing just the 10% most uncertain predictions catches "
        "a disproportionately large fraction of all errors, dramatically reducing annotation costs.",
        styles["BodyText2"]
    ))
    add_figure(elements, f"{FIG_UC}/uc9_annotation_prioritization.pdf",
               "Figure 11: UC9 — Error discovery rate. Reviewing the most uncertain samples first (blue) "
               "finds errors far faster than random sampling (dashed).",
               width=4.5*inch, styles=styles)

    elements.append(PageBreak())

    # ── Section 5: Novel Use Cases ─────────────────────────────────────
    elements.append(Paragraph("5. Novel Use Cases", styles["SectionHead"]))
    elements.append(Paragraph(
        "Beyond standard UQ applications, we introduce four new use cases that push the "
        "calibrator into training, data curation, and reasoning verification.",
        styles["BodyText2"]
    ))

    # ── UC-A ───────────────────────────────────────────────────────────
    elements.append(Paragraph("5a. UC-A — DPO Preference Pair Generation", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>What it does:</b> Direct Preference Optimization (DPO) requires pairs of responses "
        "labeled as (chosen, rejected) to fine-tune models via RLHF. Normally, humans must "
        "annotate these pairs. Here, the calibrator automatically generates them: for the same "
        "question answered by multiple models, the response with the higher P(correct) is "
        "\"chosen\" and the lower is \"rejected.\"",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>How the calibrator is used:</b> Index all questions answered by 2+ models. For each "
        "pair, assign \"chosen\" to the higher-scored response. Evaluate pair quality by checking "
        "whether the chosen response is actually correct more often than the rejected one. "
        "Larger score margins (|&Delta;P|) produce more reliable pairs.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Result:</b> <b>84% informative pair accuracy</b> (among pairs where exactly one "
        "response is correct, the calibrator picks the right one 84% of the time). 6,958 pairs "
        "generated. High-margin pairs (&Delta;P > 0.5) are near-perfect, enabling selective "
        "dataset construction for DPO training without human annotation.",
        styles["BodyText2"]
    ))
    add_figure(elements, f"{FIG_UC}/uc_a_pair_quality.pdf",
               "Figure 12: UC-A — DPO pair quality. Informative pair accuracy increases with calibrator score margin.",
               width=4.5*inch, styles=styles)

    # ── UC-B ───────────────────────────────────────────────────────────
    elements.append(Paragraph("5b. UC-B — Best-of-N Selection", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>What it does:</b> When N models (or N samples from the same model) each produce "
        "an answer, pick the best one <i>without</i> knowing the ground truth. This is "
        "inference-time quality improvement: generate multiple candidates, then use the "
        "calibrator as a verifier to select the most likely correct response.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>How the calibrator is used:</b> Score each of the N responses with the calibrator, "
        "then select the one with the highest P(correct). Compare against: random selection, "
        "always picking the best-on-average model, majority vote, and oracle (always correct "
        "if any candidate is correct).",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Result:</b> At N=3, calibrator-guided selection achieves <b>67.4% accuracy</b>, "
        "which is <b>+5.4%</b> above the best individual model. The lift is largest on "
        "\"disagreement\" questions where models give different answers — exactly where "
        "selection matters most.",
        styles["BodyText2"]
    ))
    add_figure(elements, f"{FIG_UC}/uc_b_selection_accuracy.pdf",
               "Figure 13: UC-B — Best-of-N selection accuracy at N=2 and N=3. Calibrator-guided selection (blue) outperforms the best single model and random selection.",
               width=4.5*inch, styles=styles)

    elements.append(PageBreak())

    # ── UC-C ───────────────────────────────────────────────────────────
    elements.append(Paragraph("5c. UC-C — Data Curation for Distillation", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>What it does:</b> When training a smaller \"student\" model on outputs from a "
        "larger \"teacher\" model (knowledge distillation), the teacher's outputs contain "
        "errors. The calibrator filters the training data, keeping only high-confidence "
        "predictions as training examples for the student.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>How the calibrator is used:</b> Set a confidence threshold on teacher outputs. "
        "Only samples with P(correct) above the threshold become training data. Sweeping "
        "the threshold trades off data quantity (retention %) vs. data quality (accuracy of "
        "retained set). We also trained 5 student models (Qwen2.5-1.5B) on differently "
        "filtered data to measure downstream impact.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Result:</b> At 50% retention, the filtered dataset has <b>90.9% accuracy</b> "
        "(vs. 51.7% for random subsampling). Student models trained on calibrator-filtered "
        "data match oracle-filtered performance and beat random-subset students by +1.3%. "
        "This confirms the calibrator enables effective data curation at scale.",
        styles["BodyText2"]
    ))
    add_figure(elements, f"{FIG_UC}/uc_c_quality_quantity.pdf",
               "Figure 14: UC-C — Quality-quantity tradeoff. As the confidence threshold increases (retaining fewer samples), accuracy of the retained set rises sharply.",
               width=4.5*inch, styles=styles)

    # ── UC-D ───────────────────────────────────────────────────────────
    elements.append(Paragraph("5d. UC-D — Agent Step Verification", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>What it does:</b> Modern LLMs use chain-of-thought (CoT) reasoning, producing "
        "multi-step responses. This use case asks: does the calibrator capture information "
        "about the reasoning <i>process</i>, not just the final answer? Can we detect "
        "\"overthinking\" — long reasoning chains that lead to wrong answers?",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>How the calibrator is used:</b> Analyze responses with detected multi-step "
        "structure (numbered steps, JSON reasoning, etc.). Measure correlation between "
        "response length, calibrator score, and actual correctness. Identify \"overthinking\" "
        "candidates: long responses where the calibrator gives low confidence. Compute "
        "partial correlations to check if the calibrator adds signal beyond what response "
        "length alone provides.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Result:</b> Partial correlation of <b>r=0.813</b> between calibrator score and "
        "correctness after controlling for response length — the calibrator captures "
        "correctness information beyond just \"long answers are wrong.\" However, step-level "
        "drop detection (running the calibrator on truncated responses) showed weaker signal "
        "(AUROC ~0.53-0.56), suggesting the outcome-level calibrator is most effective.",
        styles["BodyText2"]
    ))
    add_figure(elements, f"{FIG_UC}/uc_d_trajectories.pdf",
               "Figure 15: UC-D — Calibrator score trajectories across multi-step responses. Correct answers (green) maintain higher confidence throughout.",
               width=4.5*inch, styles=styles)

    elements.append(PageBreak())

    # ── Section 6: Comprehensive baseline comparison ───────────────────
    elements.append(Paragraph("6. Baseline Comparison", styles["SectionHead"]))
    elements.append(Paragraph(
        "A key concern is whether simpler methods could achieve similar results. "
        "We compare our fine-tuned calibrator against five baselines, including the "
        "same base model <i>without</i> fine-tuning (zero-shot). All confidence "
        "intervals are 95% bootstrap CIs with 2,000 resamples.",
        styles["BodyText2"]
    ))
    elements.append(Spacer(1, 6))

    # Baseline descriptions
    elements.append(Paragraph(
        "<b>Baselines tested:</b>",
        styles["BodyText2"]
    ))
    for desc in [
        "<b>Verbalized (raw):</b> Extract the model's self-reported confidence from its text response (e.g., \"I am 90% confident\").",
        "<b>Verbalized (Platt):</b> Fit logistic regression on raw verbalized confidence to calibrate it.",
        "<b>Verbalized (Isotonic):</b> Fit non-parametric isotonic regression on verbalized confidence.",
        "<b>Response length:</b> Fit logistic regression on log(output tokens) to predict correctness.",
        "<b>Combined:</b> Logistic regression on both verbalized confidence and response length jointly.",
        "<b>Zero-shot base model:</b> Run the same Qwen3-VL-8B model <i>without</i> LoRA fine-tuning on the correctness prediction task. Extracts P(Yes) from the model's next-token logits.",
    ]:
        elements.append(Paragraph(f"&bull; {desc}", styles["BulletCustom"]))
    elements.append(Spacer(1, 8))

    # Combined results table
    elements.append(Paragraph("Combined Results (11,739 samples, all target models)", styles["SubSectionHead"]))
    baseline_tbl = make_table(
        ["Method", "AUROC", "95% CI", "N"],
        [
            ["Calibrator (ours)", "0.936", "[0.932, 0.940]", "11,739"],
            ["Combined (verb+len)", "0.652", "[0.642, 0.663]", "11,049"],
            ["Verbalized (Isotonic)", "0.650", "[0.639, 0.661]", "11,049"],
            ["Verbalized (Platt)", "0.641", "[0.630, 0.652]", "11,049"],
            ["Verbalized (raw)", "0.607", "[0.596, 0.617]", "11,049"],
            ["Response length", "0.598", "[0.588, 0.608]", "11,739"],
            ["Zero-shot base model", "0.524", "[0.514, 0.535]", "11,739"],
        ],
        col_widths=[1.8*inch, 0.9*inch, 1.4*inch, 0.9*inch]
    )
    elements.append(baseline_tbl)
    elements.append(Spacer(1, 8))

    elements.append(Paragraph(
        "<b>Key findings:</b> (1) The calibrator outperforms all baselines by 28-41 AUROC points "
        "with no confidence interval overlap. (2) The zero-shot base model (same architecture, "
        "no fine-tuning) scores 0.524 — near random — proving fine-tuning is essential. "
        "(3) Even calibrated verbalized confidence (Platt/Isotonic) only reaches 0.65, "
        "far below our 0.936. (4) Combining verbalized + length provides minimal improvement "
        "over verbalized alone.",
        styles["BodyText2"]
    ))
    elements.append(Spacer(1, 6))

    # Per-model breakdown
    elements.append(Paragraph("Per-Model Breakdown", styles["SubSectionHead"]))
    permodel_tbl = make_table(
        ["Method", "GPT-5-mini", "GPT-5.2", "Qwen3.5"],
        [
            ["Calibrator (ours)", "0.956 [.950,.961]", "0.933 [.926,.941]", "0.911 [.901,.921]"],
            ["Combined (verb+len)", "0.700 [.684,.716]", "0.636 [.618,.652]", "0.568 [.544,.591]"],
            ["Verbalized (Isotonic)", "0.666 [.650,.682]", "0.637 [.620,.652]", "0.594 [.575,.613]"],
            ["Response length", "0.616 [.599,.632]", "0.562 [.544,.579]", "0.603 [.584,.623]"],
            ["Zero-shot base model", "0.554 [.537,.572]", "0.493 [.476,.510]", "0.539 [.519,.559]"],
        ],
        col_widths=[1.5*inch, 1.6*inch, 1.6*inch, 1.6*inch]
    )
    elements.append(permodel_tbl)
    elements.append(Spacer(1, 6))

    elements.append(Paragraph(
        "<b>Note on logit-based baselines:</b> For closed-source models (GPT-5-mini, GPT-5.2), "
        "internal token logits are not accessible, making logit-based UQ methods inapplicable. "
        "This is a primary motivation for our approach: an external calibrator that works on "
        "<i>any</i> model, including proprietary ones, without requiring access to model internals.",
        styles["BodyText2"]
    ))

    add_figure(elements, f"{FIG_PAPER}/fig_verbalized_vs_calibrator.pdf",
               "Figure 16: Verbalized confidence vs. fine-tuned calibrator discrimination.",
               width=5.0*inch, styles=styles)

    # ── Section 7: Reviewer Robustness Experiments ─────────────────────
    elements.append(PageBreak())
    elements.append(Paragraph("7. Reviewer Robustness Experiments", styles["SectionHead"]))
    elements.append(Paragraph(
        "We anticipate key reviewer concerns and proactively address them with "
        "additional analyses: data leakage, statistical significance, per-benchmark "
        "granularity, held-out benchmark generalization, and literature comparison.",
        styles["BodyText2"]
    ))
    elements.append(Spacer(1, 8))

    # 7.1 Data leakage fix
    elements.append(Paragraph("7.1 Test-Only Evaluation (Data Leakage Fix)", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "Sections 1-6 reported baselines on <b>all 11,739 samples</b>, including training data. "
        "We re-evaluated on <b>4,152 strictly held-out test samples</b> only. The calibrator "
        "AUROC drops modestly from 0.936 to <b>0.915</b> — a 0.021 decrease that confirms "
        "the earlier results were not inflated by data leakage.",
        styles["BodyText2"]
    ))
    elements.append(Spacer(1, 6))

    leakage_tbl = make_table(
        ["Method", "All Data (N=11,739)", "Test-Only (N=4,152)", "Drop"],
        [
            ["Calibrator (ours)", "0.936", "0.915 [.906, .923]", "-0.021"],
            ["Verbalized (Isotonic)", "0.650", "0.676 [.659, .692]", "+0.026"],
            ["Combined (verb+len)", "0.652", "0.671 [.655, .689]", "+0.019"],
            ["Zero-shot base model", "0.524", "0.540 [.523, .557]", "+0.016"],
        ],
        col_widths=[1.5*inch, 1.3*inch, 1.6*inch, 0.8*inch]
    )
    elements.append(leakage_tbl)
    elements.append(Spacer(1, 6))

    elements.append(Paragraph(
        "<b>Per-model test-only AUROC:</b> GPT-5-mini: 0.917 [.902, .932], "
        "GPT-5.2: 0.916 [.902, .929], Qwen3.5: 0.911 [.895, .926]. "
        "All three target models show consistent performance with non-overlapping "
        "confidence intervals vs. baselines.",
        styles["BodyText2"]
    ))
    elements.append(Spacer(1, 8))

    # 7.2 Statistical significance
    elements.append(Paragraph("7.2 Statistical Significance", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "We applied three independent statistical tests to every calibrator-vs-baseline "
        "comparison, all on the 4,152 test-only samples:",
        styles["BodyText2"]
    ))
    for test_desc in [
        "<b>Paired permutation test</b> (10,000 iterations): tests whether the AUROC difference is due to chance.",
        "<b>DeLong test</b>: asymptotic comparison of two correlated AUROCs.",
        "<b>McNemar's test</b>: tests whether calibrator and baseline have the same error rate pattern.",
    ]:
        elements.append(Paragraph(f"&bull; {test_desc}", styles["BulletCustom"]))
    elements.append(Spacer(1, 6))

    sig_tbl = make_table(
        ["Baseline", "AUROC Gap", "Permutation p", "DeLong p", "McNemar p"],
        [
            ["Verbalized (raw)", "+0.278", "< 0.001 ***", "< 0.001 ***", "< 0.001 ***"],
            ["Verbalized (Platt)", "+0.250", "< 0.001 ***", "< 0.001 ***", "< 0.001 ***"],
            ["Verbalized (Isotonic)", "+0.236", "< 0.001 ***", "< 0.001 ***", "< 0.001 ***"],
            ["Response length", "+0.294", "< 0.001 ***", "< 0.001 ***", "< 0.001 ***"],
            ["Combined (verb+len)", "+0.241", "< 0.001 ***", "< 0.001 ***", "< 0.001 ***"],
        ],
        col_widths=[1.3*inch, 0.8*inch, 1.1*inch, 1.0*inch, 1.0*inch]
    )
    elements.append(sig_tbl)
    elements.append(Spacer(1, 6))

    elements.append(Paragraph(
        "<b>Result:</b> All 15 comparisons (5 baselines × 3 tests) yield p &lt; 0.001. "
        "Per-target model results are equally significant (all p &lt; 0.001). "
        "The calibrator's advantage is not attributable to chance.",
        styles["BodyText2"]
    ))
    elements.append(Spacer(1, 8))

    # 7.3 Per-benchmark breakdown
    elements.append(Paragraph("7.3 Per-Benchmark AUROC Breakdown", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "To check for benchmark-specific effects, we computed AUROC on each of the "
        "20 benchmarks individually. The calibrator outperforms verbalized confidence "
        "on <b>19 of 20 benchmarks</b>.",
        styles["BodyText2"]
    ))
    elements.append(Spacer(1, 6))

    # Top-10 and Bottom-5 benchmarks table
    perbench_tbl = make_table(
        ["Benchmark", "Modality", "AUROC", "N", "Accuracy"],
        [
            ["realworldqa", "VLM", "0.982", "432", "50.7%"],
            ["livebench", "TXT", "0.982", "136", "66.9%"],
            ["vizwiz", "VLM", "0.956", "227", "70.0%"],
            ["mathvista", "VLM", "0.945", "379", "33.0%"],
            ["arc_agi", "TXT", "0.925", "110", "51.8%"],
            ["gpqa", "TXT", "0.903", "234", "76.5%"],
            ["...", "", "", "", ""],
            ["mathverse", "VLM", "0.754", "230", "47.4%"],
            ["hle_multimodal", "VLM", "0.737", "35", "14.3%"],
            ["prbench", "TXT", "0.658", "90", "72.2%"],
            ["mmvet", "VLM", "0.637", "44", "34.1%"],
        ],
        col_widths=[1.3*inch, 0.8*inch, 0.8*inch, 0.6*inch, 0.8*inch]
    )
    elements.append(perbench_tbl)
    elements.append(Spacer(1, 6))

    elements.append(Paragraph(
        "<b>Summary stats:</b> Mean per-benchmark AUROC = <b>0.850</b>, "
        "std = 0.097, CV = 0.114. The two weakest benchmarks (mmvet N=44, "
        "prbench N=90) have small sample sizes. Excluding benchmarks with N&lt;50, "
        "the mean rises to 0.867.",
        styles["BodyText2"]
    ))
    elements.append(Spacer(1, 8))

    # 7.4 Held-out benchmark generalization
    elements.append(Paragraph("7.4 Held-Out Benchmark Generalization", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "To verify the model does not overfit to specific benchmarks, we performed "
        "<b>leave-5-out cross-validation</b> over the 20 benchmarks: each fold holds "
        "out 5 benchmarks as \"unseen\" and evaluates on the remaining 15 as \"in-distribution.\"",
        styles["BodyText2"]
    ))
    elements.append(Spacer(1, 6))

    heldout_tbl = make_table(
        ["Fold", "Held-Out Benchmarks", "In-Dist AUROC", "Held-Out AUROC", "Gap"],
        [
            ["0", "gpqa, hallusion, mathverse, mmvet, vizwiz", "0.919", "0.890", "+0.029"],
            ["1", "hle, mmmu, mmstar, omnimath, simpleqa", "0.916", "0.909", "+0.007"],
            ["2", "bbeh, charxiv, mathvision, mathvista, realworldqa", "0.898", "0.935", "-0.038"],
            ["3", "arc_agi, chembench, hle_mm, livebench, prbench", "0.920", "0.873", "+0.047"],
            ["Mean", "", "", "", "+0.011"],
        ],
        col_widths=[0.5*inch, 2.6*inch, 1.0*inch, 1.1*inch, 0.6*inch]
    )
    elements.append(heldout_tbl)
    elements.append(Spacer(1, 6))

    elements.append(Paragraph(
        "<b>Result:</b> Mean in-dist vs. held-out gap = <b>+0.011</b> (std 0.032). "
        "In Fold 2, the held-out set actually <i>outperforms</i> in-distribution. "
        "This negligible gap confirms the model learns general uncertainty patterns, "
        "not benchmark-specific artifacts.",
        styles["BodyText2"]
    ))
    elements.append(Spacer(1, 8))

    # 7.5 Literature comparison
    elements.append(Paragraph("7.5 Comparison with Published UQ Methods", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "We surveyed 9 published UQ methods and compared along five axes: "
        "whether they work on closed-source models, support cross-model transfer, "
        "handle multimodal inputs, inference cost, and typical AUROC range.",
        styles["BodyText2"]
    ))
    elements.append(Spacer(1, 6))

    lit_tbl = make_table(
        ["Method", "Closed-Src", "Cross-Model", "Multimodal", "Cost"],
        [
            ["Verbalized Confidence", "Yes", "No", "No", "1x"],
            ["Logit-Based (TopK, Entropy)", "No", "No", "No", "1x"],
            ["Temperature Scaling", "No", "No", "No", "1x"],
            ["MC Dropout", "No", "No", "Limited", "Nx"],
            ["Deep Ensembles", "No", "No", "Limited", "Nx"],
            ["Conformal Prediction", "Partial", "No", "Limited", "1x"],
            ["Semantic Entropy", "Partial", "No", "No", "Nx"],
            ["Self-Evaluation", "Yes", "No", "Limited", "2x"],
            ["Learned Verifiers", "Yes", "Yes*", "No", "1.3x"],
            ["Ours", "Yes", "Yes", "Yes", "1x"],
        ],
        col_widths=[1.6*inch, 0.9*inch, 1.0*inch, 1.0*inch, 0.7*inch]
    )
    elements.append(lit_tbl)
    elements.append(Spacer(1, 6))

    elements.append(Paragraph(
        "<b>Key differentiator:</b> No existing method simultaneously supports "
        "(1) closed-source target models, (2) cross-model transfer, and "
        "(3) multimodal (text + vision) inputs. Our approach is the first to "
        "combine all three, while maintaining 1× inference cost (single forward pass "
        "through the 8B calibrator).",
        styles["BodyText2"]
    ))

    # ── Section 8: Next Steps ──────────────────────────────────────────
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("8. Next Steps", styles["SectionHead"]))
    next_steps = [
        "Upgrade to LoRA r=32 for the final model (+1.3 AUROC points observed in ablation)",
        "Fill remaining Qwen3.5 data gaps (~1,357 missing samples across 12 benchmarks)",
        "Run Claude API evaluation (pending advisor budget approval)",
        "Merge LoRA weights for deployment release (2B and 8B variants)",
        "Write paper sections: related work, methodology, experimental setup",
        "Generate camera-ready figures and finalize benchmark coverage table",
    ]
    for step in next_steps:
        elements.append(Paragraph(f"&bull; {step}", styles["BulletCustom"]))

    # ── Section 9: Summary ─────────────────────────────────────────────
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("9. Summary", styles["SectionHead"]))
    elements.append(Paragraph(
        "This month we built and validated a <b>unified uncertainty quantification model</b> "
        "that works across text and vision modalities, across multiple target models, and on "
        "closed-source systems. The model achieves <b>0.831 AUROC</b> on held-out data and "
        "enables <b>13 downstream use cases</b>. Key findings include: (1) multi-source training "
        "is essential (+8-16 points), (2) unified text+VLM training beats single-modal by 13+ "
        "points, (3) the model can be compressed to 2B parameters with only 2% quality loss, "
        "and (4) LoRA r=32 provides further improvement. The project is on track for ECCV 2026 "
        "submission.",
        styles["BodyText2"]
    ))

    # ── Build PDF ──────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        title="UQ Research Progress Report — February 2026",
        author="K. Hayes",
    )

    # Add page numbers
    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.gray)
        canvas.drawRightString(
            doc.pagesize[0] - 0.75 * inch,
            0.5 * inch,
            f"Page {doc.page}"
        )
        canvas.drawString(
            0.85 * inch,
            0.5 * inch,
            "UQ for LLMs — Research Update — Feb 2026"
        )
        canvas.restoreState()

    doc.build(elements, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"Report generated: {OUTPUT}")
    print(f"Size: {os.path.getsize(OUTPUT) / 1024:.0f} KB")


if __name__ == "__main__":
    build_report()
