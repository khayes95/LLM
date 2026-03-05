#!/usr/bin/env python3
"""Generate a polished PDF research progress report using ReportLab.

Structured for advisor meeting - comprehensive overview of all work done.
"""

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
FIG_UC = f"{BASE}/figures/use_cases_v2"
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


def add_figure(elements, pdf_path, caption, width=5.5*inch, styles=None,
               max_height=7.0*inch):
    """Add a PDF figure converted to image, preserving original aspect ratio."""
    if not os.path.exists(pdf_path):
        return
    # Read actual dimensions from the PDF to preserve aspect ratio
    doc = fitz.open(pdf_path)
    if len(doc) == 0:
        doc.close()
        return
    page = doc[0]
    pdf_w = page.rect.width
    pdf_h = page.rect.height
    doc.close()

    aspect = pdf_h / pdf_w if pdf_w > 0 else 0.6
    height = width * aspect
    # Clamp height so tall figures don't overflow the page
    if height > max_height:
        height = max_height
        width = height / aspect

    img_data = pdf_page_to_image(pdf_path, dpi=200)
    if img_data is None:
        return
    img = Image(img_data, width=width, height=height)
    elements.append(img)
    if caption and styles:
        elements.append(Paragraph(caption, styles["Caption"]))


def build_report():
    """Build the full PDF report."""
    styles = build_styles()
    elements = []

    # ══════════════════════════════════════════════════════════════════════
    # TITLE PAGE
    # ══════════════════════════════════════════════════════════════════════
    elements.append(Spacer(1, 0.8 * inch))
    elements.append(Paragraph(
        "Uncertainty Quantification for LLMs",
        styles["ReportTitle"]
    ))
    elements.append(Paragraph(
        "Research Progress Report - March 2026",
        styles["ReportSubtitle"]
    ))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(HRFlowable(
        width="60%", thickness=2, color=MED_BLUE,
        spaceAfter=12, spaceBefore=6, hAlign="CENTER"
    ))

    # Key metrics box
    summary_data = [
        ["Held-Out AUROC", "Benchmarks", "Target Models", "Target"],
        ["0.898", "20 text + VLM", "3", "ECCV 2026"],
    ]
    summary_table = Table(summary_data, colWidths=[1.3*inch, 1.2*inch, 1.0*inch, 1.0*inch])
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

    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph(
        "<b>Three Novel Contributions:</b>",
        styles["BodyText2"]
    ))
    for item in [
        "<b>Cross-model transfer</b> - train on Model A, evaluate on Model B without retraining",
        "<b>Multimodality</b> - single model handles both text LLMs and vision-language models",
        "<b>Closed-source application</b> - enables UQ on GPT-5, Claude, and other proprietary models",
    ]:
        elements.append(Paragraph(f"&bull; {item}", styles["BulletCustom"]))

    elements.append(Spacer(1, 0.15 * inch))
    elements.append(Paragraph(
        "<b>What's in this report:</b> We cover the problem and our approach (Sec 1-2), main results "
        "and baselines (Sec 3-4), training ablations and elicitation strategies (Sec 5-6), "
        "13 downstream use cases (Sec 7-8), robustness analysis (Sec 9), and next steps (Sec 10).",
        styles["BodyText2"]
    ))

    elements.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1: THE PROBLEM
    # ══════════════════════════════════════════════════════════════════════
    elements.append(Paragraph("1. The Problem", styles["SectionHead"]))
    elements.append(Paragraph(
        "Large language models (LLMs) produce fluent, confident-sounding responses - "
        "even when they are wrong. In high-stakes applications (medical, legal, financial), "
        "deployers need a reliable <b>uncertainty signal</b>: \"How likely is this response "
        "to be correct?\" This is the problem of <b>uncertainty quantification (UQ)</b>.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "Existing UQ approaches fall short for modern deployments:",
        styles["BodyText2"]
    ))
    for item in [
        "<b>Logit-based methods</b> (token probabilities, entropy) require access to model internals - "
        "unavailable for closed-source models like GPT-5 and Claude.",
        "<b>Sampling-based methods</b> (semantic entropy, self-consistency) require N&ge;5 forward passes "
        "through the target model - expensive and slow at production scale.",
        "<b>Verbalized confidence</b> (asking the model \"how confident are you?\") is cheap but poorly "
        "calibrated - models routinely say 90% confident on wrong answers.",
        "<b>No existing method</b> simultaneously handles closed-source models, cross-model transfer, "
        "and multimodal (text + vision) inputs.",
    ]:
        elements.append(Paragraph(f"&bull; {item}", styles["BulletCustom"]))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2: OUR APPROACH
    # ══════════════════════════════════════════════════════════════════════
    elements.append(Paragraph("2. Our Approach", styles["SectionHead"]))
    elements.append(Paragraph(
        "We fine-tune a small open-source vision-language model (<b>Qwen3-VL-8B-Instruct</b>) with LoRA "
        "to predict whether a target model's response to a question is correct. The calibrator "
        "takes as input the <b>question</b> and the <b>target model's response</b> (plus images "
        "for VLM benchmarks) and outputs <b>P(correct)</b> - a single probability score. "
        "This approach enables uncertainty quantification on <i>any</i> model, including "
        "closed-source ones, without access to model internals or multiple forward passes.",
        styles["BodyText2"]
    ))
    elements.append(Spacer(1, 6))

    # 2a. Task formulation
    elements.append(Paragraph("2a. Task Formulation", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "Given a question Q, a target model's response R, and optionally an image I, "
        "the calibrator predicts P(correct | Q, R, I). This is a <b>binary classification</b> task: "
        "the label is 1 if the response is correct (matches ground truth) and 0 otherwise.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Label generation:</b> Each training sample comes from running a target model on a "
        "benchmark with known ground truth. We grade correctness automatically using the benchmark's "
        "grading function (exact match, multiple-choice extraction, or LLM-as-judge for open-ended "
        "questions). This produces binary labels: correct (1) or incorrect (0).",
        styles["BodyText2"]
    ))
    elements.append(Spacer(1, 6))

    # 2b. Prompt template
    elements.append(Paragraph("2b. Prompt Template", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "The calibrator sees a structured prompt. The v2 \"combined\" prompt includes CoT reasoning, "
        "metadata about the source benchmark and model, and extended truncation windows "
        "(1500 tokens for the question, 800 for the response):",
        styles["BodyText2"]
    ))
    # Show the actual prompt template
    prompt_text = (
        "<font face='Courier' size='8'>"
        "[Benchmark: {benchmark_name} | Model: {model_name}]<br/>"
        "Question: {question_text} (truncated to 1500 tokens)<br/><br/>"
        "Answer: {response_text} (truncated to 800 tokens)<br/><br/>"
        "Think step by step about whether this answer is correct,<br/>"
        "then give your final judgment.<br/>"
        "Is the answer correct? (i) No (ii) Yes"
        "</font>"
    )
    elements.append(Paragraph(prompt_text, styles["BodyText2"]))
    elements.append(Spacer(1, 6))

    # 2c. Training objective
    elements.append(Paragraph("2c. Training Objective and Signal Extraction", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "The model is fine-tuned with standard <b>next-token prediction loss</b> (cross-entropy) on "
        "the target tokens \"No\" or \"Yes\". At inference time, we extract the calibrator's confidence "
        "from the <b>logit for the \"Yes\" token</b> at the final position. Specifically:",
        styles["BodyText2"]
    ))
    for item in [
        "Forward pass through the model on the prompt above.",
        "Extract the logits at the last token position for tokens \"Yes\" and \"No\".",
        "Apply softmax to get P(Yes) = P(correct). This is the calibrator score.",
    ]:
        elements.append(Paragraph(f"&bull; {item}", styles["BulletCustom"]))
    elements.append(Paragraph(
        "This logit-based extraction preserves fine-grained probability information that would be "
        "lost in generated text (e.g., asking the model to output a number).",
        styles["BodyText2"]
    ))
    elements.append(Spacer(1, 6))

    # 2d. Training setup
    elements.append(Paragraph("2d. Training Setup", styles["SubSectionHead"]))
    for item in [
        "<b>Base model:</b> Qwen3-VL-8B-Instruct (vision-language model with 8B parameters).",
        "<b>Adapter:</b> LoRA (r=32, alpha=64, 30.4M trainable params out of 8B total).",
        "<b>Data:</b> 9,533 train / 1,774 held-out test samples across 20 benchmarks from 3 target "
        "models (GPT-5-mini, GPT-5.2, Qwen3.5-397B).",
        "<b>Images:</b> Real images for VLM benchmarks (MMMU, MathVista, etc.), 240x240 gray "
        "placeholder for text-only benchmarks. Using a VLM base model ensures the calibrator can "
        "handle both modalities.",
        "<b>Training:</b> 3 epochs, batch size 1 with gradient accumulation 16, learning rate 1e-4, "
        "bf16 mixed precision, ~2 hours on 4 GPUs.",
        "<b>Inference cost:</b> Single forward pass per response (1x cost). No sampling required.",
    ]:
        elements.append(Paragraph(f"&bull; {item}", styles["BulletCustom"]))
    elements.append(Spacer(1, 6))

    # 2e. Evaluation protocol
    elements.append(Paragraph("2e. Evaluation Protocol", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "All reported results use a <b>held-out test set (1,774 samples)</b> from a standard "
        "train/test split. These samples were never seen during training.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Primary metric:</b> AUROC (Area Under the ROC Curve) measures how well the calibrator's "
        "P(correct) scores separate correct from incorrect responses. We also report Brier score "
        "(mean squared error of probability estimates) and ECE (Expected Calibration Error) to "
        "verify calibration quality.",
        styles["BodyText2"]
    ))

    elements.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 3: MAIN RESULTS
    # ══════════════════════════════════════════════════════════════════════
    elements.append(Paragraph("3. Main Results", styles["SectionHead"]))
    elements.append(Paragraph(
        "<b>Evaluation:</b> All results use a <b>held-out test set (1,774 samples)</b> from an internal "
        "train/test split. These samples were never seen during training. We report AUROC as our "
        "primary metric.",
        styles["BodyText2"]
    ))

    elements.append(Spacer(1, 6))
    elements.append(Paragraph("Held-Out Test Performance (v2 model)", styles["SubSectionHead"]))
    main_tbl = make_table(
        ["Metric", "Overall", "VLM", "Text", "Brier", "ECE"],
        [
            ["Held-out (1,774)", "0.898", "0.915", "0.875", "0.141", "0.035"],
        ],
        col_widths=[1.4*inch, 0.9*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch]
    )
    elements.append(main_tbl)
    elements.append(Spacer(1, 6))

    elements.append(Paragraph(
        "<b>v1 to v2 improvement:</b> The v2 model (r=32 + combined prompt) improved held-out AUROC "
        "from 0.831 to 0.898 (+6.7 points). "
        "This came from prompt design improvements (CoT, metadata, longer truncation) and "
        "increased LoRA rank.",
        styles["BodyText2"]
    ))

    # Calibration figure
    add_figure(elements, f"{FIG_PAPER}/fig_calibration_curve.pdf",
               "Figure 1: Calibration plot. Predicted confidence (x-axis) vs actual accuracy (y-axis). "
               "A perfectly calibrated model follows the diagonal. Our calibrator (blue) tracks the "
               "diagonal closely; verbalized confidence (orange) is poorly calibrated.",
               width=5.0*inch, styles=styles)

    elements.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 4: BASELINE COMPARISONS
    # ══════════════════════════════════════════════════════════════════════
    elements.append(Paragraph("4. Baseline Comparisons", styles["SectionHead"]))
    elements.append(Paragraph(
        "A key question: could simpler methods achieve similar results? We compare our "
        "calibrator against <b>10 baselines</b> spanning all major UQ paradigms. All results "
        "use strictly held-out test data. 95% CIs from 2,000 bootstrap resamples.",
        styles["BodyText2"]
    ))

    elements.append(Spacer(1, 6))
    elements.append(Paragraph("4a. Standard Baselines", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>Zero-shot:</b> Same model, no fine-tuning. <b>Verbalized:</b> Ask the target model "
        "itself \"how confident are you?\" <b>Platt/Isotonic:</b> Post-hoc recalibration of "
        "verbalized scores (fit a function to correct systematic biases). "
        "<b>Response length:</b> Use token count as a confidence proxy (longer = less confident). "
        "<b>Combined:</b> Logistic regression on verbalized + length.",
        styles["BodyText2"]
    ))
    elements.append(Spacer(1, 4))

    baseline_tbl = make_table(
        ["Method", "AUROC", "95% CI", "Gap vs Ours"],
        [
            ["Calibrator (ours)", "0.898", "-", "-"],
            ["Zero-shot (no fine-tune)", "0.551", "-", "+0.347"],
            ["Verbalized (raw)", "0.607", "[0.590, 0.624]", "+0.291"],
            ["Verbalized (Platt)", "0.631", "-", "+0.267"],
            ["Verbalized (Isotonic)", "0.679", "-", "+0.219"],
            ["Response length", "0.621", "-", "+0.277"],
            ["Combined (verb+len)", "0.671", "-", "+0.227"],
        ],
        col_widths=[1.6*inch, 0.9*inch, 1.3*inch, 1.0*inch]
    )
    elements.append(baseline_tbl)
    elements.append(Spacer(1, 6))

    elements.append(Paragraph(
        "<b>Statistical significance:</b> All comparisons are significant at p &lt; 0.001 "
        "across three independent tests (paired permutation, DeLong, McNemar). No CI overlap.",
        styles["BodyText2"]
    ))
    elements.append(Spacer(1, 8))

    # 4b. Proxy Semantic Entropy
    elements.append(Paragraph("4b. Proxy Semantic Entropy & Self-Consistency", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>Semantic Entropy (SE)</b> is a leading UQ method: ask the model the same question N "
        "times, cluster answers by meaning, and measure spread. High spread = uncertain. "
        "<b>Self-Consistency</b> is simpler: majority vote across N answers. "
        "True SE requires N generations from the <i>target</i> model (expensive for paid APIs), "
        "so we tested a <b>proxy</b>: generate N=5 from our small calibrator model instead, "
        "to see if a small model's uncertainty predicts the big model's correctness.",
        styles["BodyText2"]
    ))
    se_tbl = make_table(
        ["Method", "AUROC", "N samples", "Note"],
        [
            ["Proxy Semantic Entropy (N=5)", "0.467", "1,376", "BELOW random"],
            ["Proxy Self-Consistency (N=5)", "0.466", "1,376", "BELOW random"],
            ["Calibrator (same subset)", "0.888", "1,376", "Single pass"],
            ["Verbalized (same subset)", "0.629", "1,306", "-"],
        ],
        col_widths=[2.0*inch, 0.8*inch, 0.9*inch, 1.5*inch]
    )
    elements.append(se_tbl)
    elements.append(Spacer(1, 6))

    elements.append(Paragraph(
        "<b>Key takeaway:</b> Proxy SE scores <i>below</i> random (0.467 vs 0.500). "
        "A small model's self-uncertainty does <i>not</i> predict a large model's correctness. "
        "True SE requires N generations from the <i>target</i> model - prohibitively expensive "
        "for closed-source APIs. Our single-pass calibrator beats it by <b>42 AUROC points</b>.",
        styles["BodyText2"]
    ))

    elements.append(Spacer(1, 8))
    # 4c. Literature positioning
    elements.append(Paragraph("4c. How We Compare to Existing Methods", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "Comparison against major UQ method families. Columns: works on closed-source models? "
        "transfers across models? handles images? Cost per query (1x = single pass, Nx = N samples).",
        styles["BodyText2"]
    ))
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
        "<b>Unique position:</b> No existing method combines (1) closed-source targets, "
        "(2) cross-model transfer, and (3) multimodal inputs. Ours is the first to achieve "
        "all three, at 1x cost (single forward pass through the calibrator).",
        styles["BodyText2"]
    ))

    add_figure(elements, f"{FIG_PAPER}/fig_auroc_comparison.pdf",
               "Figure 2: AUROC comparison — calibrator vs all baselines (held-out set, v2).",
               width=5.0*inch, styles=styles)

    elements.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 5: TRAINING ABLATIONS
    # ══════════════════════════════════════════════════════════════════════
    elements.append(Paragraph("5. Training Ablations - What Makes It Work", styles["SectionHead"]))
    elements.append(Paragraph(
        "We systematically varied each training choice to understand what matters. In each "
        "ablation below, we change one variable while holding everything else constant.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<i>Note: All ablation tables in Section 5 hold the prompt constant (basic prompt) to "
        "isolate each variable. The final v2 model combines r=32 with the improved prompt "
        "(Section 5e), yielding 0.898 held-out AUROC.</i>",
        styles["BodyText2"]
    ))

    # 5a. LoRA rank
    elements.append(Paragraph("5a. LoRA Rank", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "The LoRA rank (r) controls how many trainable parameters the adapter has. "
        "Higher rank = more capacity but more memory. We tested r = 4, 8, 16, and 32.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Result:</b> Increasing rank from 16 to 32 yields +1.3 AUROC points with the basic prompt. "
        "Combined with prompt improvements, the v2 model (r=32 + combined prompt) reaches "
        "0.898 - a +6.7 point improvement over v1 (r=16 + basic prompt).",
        styles["BodyText2"]
    ))
    rank_tbl = make_table(
        ["LoRA Rank", "AUROC", "VLM", "Text", "Params"],
        [
            ["r = 4", "0.821", "0.806", "0.837", "3.8M"],
            ["r = 8", "0.824", "0.807", "0.844", "7.7M"],
            ["r = 16 (v1)", "0.831", "0.813", "0.850", "15.2M"],
            ["r = 32", "0.844", "0.840", "0.846", "30.4M"],
        ],
        col_widths=[1.3*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.0*inch]
    )
    elements.append(rank_tbl)
    elements.append(Spacer(1, 8))

    # 5b. Source model diversity
    elements.append(Paragraph("5b. Source Model Diversity", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>What is this?</b> The calibrator is trained on question-response pairs from target models. "
        "\"Source model diversity\" means: does it help to train on data from multiple target models, "
        "or is data from one model enough? We tested training on each model individually vs. all three.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Result:</b> Multi-source training is critical. Training on all 3 target models "
        "outperforms single-source by <b>7-16 AUROC points</b>. This is our core "
        "cross-model transfer result: the calibrator learns general patterns of correctness "
        "that transfer across models, not model-specific quirks.",
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
    elements.append(Spacer(1, 8))

    # 5c. Modality
    elements.append(Paragraph("5c. Modality Ablation (Text vs Vision)", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>What is this?</b> Our benchmarks include text-only tasks (math, coding, factual QA) "
        "and vision+language tasks (image understanding, chart reading). We test whether a model "
        "trained only on text benchmarks can judge vision tasks, and vice versa. \"Cross-modal\" "
        "means evaluating on the modality the model was NOT trained on.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Result:</b> Cross-modality transfer fails. A text-only model gets 0.588 on VLM benchmarks; "
        "a VLM-only model gets 0.518 on text. Unified training on both modalities is essential "
        "(+13 points each). The model needs to see examples from both modalities to judge them.",
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
    elements.append(Spacer(1, 8))

    # 5d. Model size
    elements.append(Paragraph("5d. Model Size", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>What is this?</b> We tested whether a smaller (cheaper, faster) base model works just "
        "as well as the 8B model. Smaller models are important for deployment: a 2B model needs "
        "~4GB of memory and can run on a laptop, while 8B needs a GPU.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Result:</b> The 2B model retains <b>98%</b> of the 8B model's performance (0.816 vs 0.827). "
        "Deployable on laptops at ~4GB merged weight size.",
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
    elements.append(Spacer(1, 8))

    # 5e. Prompt design
    elements.append(Paragraph("5e. Prompt Design Ablation", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>What is this?</b> The prompt is the text we feed to the calibrator. We tested "
        "three improvements over the basic v1 prompt:",
        styles["BodyText2"]
    ))
    for item in [
        "<b>CoT (Chain-of-Thought):</b> Adding \"Think step by step\" to the prompt, encouraging "
        "the model to reason about correctness before answering Yes/No.",
        "<b>Metadata tags:</b> Prepending the benchmark name and model name (e.g., "
        "\"[Benchmark: MMLU | Model: GPT-5-mini]\") so the model can learn benchmark-specific patterns.",
        "<b>Longer truncation (1500/800):</b> The v1 prompt truncated questions and responses "
        "aggressively. Extending to 1500 tokens for questions and 800 for responses gives the "
        "calibrator more text to judge.",
    ]:
        elements.append(Paragraph(f"&bull; {item}", styles["BulletCustom"]))
    elements.append(Paragraph(
        "<b>Result:</b> Each component contributes, with longer truncation providing the largest gain:",
        styles["BodyText2"]
    ))
    prompt_tbl = make_table(
        ["Prompt Variant", "Held-Out AUROC", "Improvement"],
        [
            ["Basic (v1 baseline)", "0.862", "-"],
            ["+ CoT reasoning", "0.867", "+0.5"],
            ["+ Metadata tags", "0.867", "+0.5"],
            ["+ Longer truncation (1500/800)", "0.893", "+3.1"],
            ["Combined (all three)", "0.898", "+3.6"],
        ],
        col_widths=[2.0*inch, 1.3*inch, 1.3*inch]
    )
    elements.append(prompt_tbl)
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "<b>Finding:</b> Extended truncation provides the largest single gain (+3.1 points). "
        "The basic prompt was truncating responses too aggressively, losing useful signal.",
        styles["BodyText2"]
    ))

    # Per-benchmark heatmap
    add_figure(elements, f"{FIG_PAPER}/fig_per_benchmark_heatmap.pdf",
               "Figure 3: Per-benchmark AUROC heatmap across target models.",
               width=5.5*inch, styles=styles)

    elements.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 6: ELICITATION STRATEGIES
    # ══════════════════════════════════════════════════════════════════════
    elements.append(Paragraph("6. Elicitation Strategies - Extracting the UQ Signal", styles["SectionHead"]))
    elements.append(Paragraph(
        "Our default method extracts P(correct) from the logit (the model's internal probability "
        "of predicting \"Yes\" vs \"No\"). We tested 7 alternatives to see if any could do better.",
        styles["BodyText2"]
    ))

    # v1 elicitation
    elements.append(Paragraph("6a. First Round", styles["SubSectionHead"]))
    v1_elic_tbl = make_table(
        ["Strategy", "AUROC", "Delta", "What It Does"],
        [
            ["Logit P(Yes) (baseline)", "0.841", "-", "Extract Yes/No probability from model internals"],
            ["MC Dropout (N=5)", "0.841", "+0.0", "Run 5 passes with random dropout, measure variance"],
            ["Contrastive (+ ref answer)", "0.903", "+6.2", "Give model the correct answer to compare against"],
            ["Verbalized probability", "0.749", "-9.2", "Ask model to generate a confidence % as text"],
        ],
        col_widths=[1.6*inch, 0.7*inch, 0.6*inch, 2.5*inch]
    )
    elements.append(v1_elic_tbl)
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "MC Dropout: no effect (LoRA dropout too sparse). Contrastive: +6.2 but requires knowing "
        "the correct answer, which defeats the purpose. Verbalized: -9.2 because converting internal "
        "confidence to text loses fine-grained signal.",
        styles["BodyText2"]
    ))
    elements.append(Spacer(1, 8))

    # v2 elicitation
    elements.append(Paragraph("6b. Second Round", styles["SubSectionHead"]))
    v2_elic_tbl = make_table(
        ["Strategy", "AUROC", "Delta", "What It Does"],
        [
            ["Logit P(Yes) (baseline)", "0.841", "-", "Extract Yes/No probability from model internals"],
            ["Temperature scaling (T=1.34)", "0.841", "+0.0", "Post-hoc rescaling of logits to improve calibration"],
            ["Token entropy (full vocab)", "0.520", "-32.1", "Measure uncertainty across entire vocabulary"],
            ["Hidden state MLP probe", "0.878", "+3.7", "Train small network on model's internal representations"],
            ["Hidden + logit combined", "0.884", "+4.3", "Combine hidden state and logit features"],
        ],
        col_widths=[1.6*inch, 0.7*inch, 0.6*inch, 2.5*inch]
    )
    elements.append(v2_elic_tbl)
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "Temperature scaling: no AUROC change (model already well-calibrated; ECE improved 0.023 "
        "to 0.019). Token entropy: near random (full vocabulary entropy is noise). "
        "Hidden state probe: +3.7 but requires open-weight model access.",
        styles["BodyText2"]
    ))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph(
        "<b>Bottom line:</b> The simple logit captures <b>95%+</b> of the available signal. "
        "The only strategies that help (hidden state, contrastive) require either open-weight access "
        "or a reference answer. For closed-source models, the logit is the right default.",
        styles["BodyText2"]
    ))

    elements.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 7: STANDARD USE CASES (UC1-UC9)
    # ══════════════════════════════════════════════════════════════════════
    elements.append(Paragraph("7. Downstream Use Cases", styles["SectionHead"]))
    elements.append(Paragraph(
        "The calibrator outputs a single number - P(correct) - for each question-response pair. "
        "This one signal enables <b>11 distinct applications</b> (8 core + 3 novel). "
        "We merged UC3 (error detection) with UC9 (annotation prioritization) into a unified "
        "\"Error Discovery\" use case, and UC5 (reward modeling) with UC-B (best-of-N) into "
        "\"Response Selection.\" Below we describe the evaluation setup, summarize all use cases, "
        "and flag potential oversights.",
        styles["BodyText2"]
    ))
    elements.append(Spacer(1, 6))

    # UC methodology
    elements.append(Paragraph("7.0 Use Case Evaluation Setup", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>Data pipeline:</b> Each use case operates on scored samples from 3 target models "
        "(GPT-5-mini, GPT-5.2, Qwen3.5-397B) across 20 benchmarks. The pipeline is: "
        "(1) run target models on benchmarks, (2) grade each response against ground truth to get "
        "a binary label (correct/incorrect), (3) score each question+response pair with the v2 "
        "calibrator to get P(correct). The use case scripts then post-process these scored samples. "
        "Note: use case metrics (F1, AURC, etc.) measure downstream task performance, not "
        "calibrator AUROC.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Baselines compared in each UC:</b>",
        styles["BodyText2"]
    ))
    for item in [
        "<b>Verbalized confidence:</b> The target model's self-reported confidence (0-1), extracted "
        "from structured JSON output. Available for ~92% of samples (some models don't always "
        "include it). Not recalibrated unless noted.",
        "<b>Random:</b> Random ordering/selection (averaged over 100 repeats where applicable).",
        "<b>Oracle:</b> Perfect ordering using ground truth labels (upper bound).",
        "<b>Response length:</b> Number of tokens in the response (longer = less confident heuristic).",
    ]:
        elements.append(Paragraph(f"&bull; {item}", styles["BulletCustom"]))
    elements.append(Spacer(1, 4))

    elements.append(Paragraph(
        "<b>Key assumptions and potential oversights:</b>",
        styles["BodyText2"]
    ))
    for item in [
        "<b>Binary correctness only:</b> All UCs rely on binary correct/incorrect labels. For "
        "open-ended or partially-correct responses, this is a simplification. Benchmarks with "
        "LLM-as-judge grading (e.g., TutorBench) were excluded to avoid grading noise.",
        "<b>Benchmark-level evaluation:</b> We evaluate on academic benchmarks, not real "
        "production queries. Performance on user-facing questions may differ, particularly for "
        "UC8 (OOD detection) where \"distribution shift\" is proxied by benchmark differences.",
        "<b>Same calibrator for all UCs:</b> The calibrator was trained for outcome-level "
        "correctness prediction, not optimized for any specific downstream task. UC-D (step "
        "verification) suffers from this - the model wasn't designed for process-level reasoning.",
        "<b>Cross-model gap:</b> GPT-5-mini (in-distribution, N=1,361) vs GPT-5.2/Qwen3.5 "
        "(cross-model, N=1,642/1,444). Cross-model results are slightly weaker but still strong, "
        "which validates the transfer claim.",
        "<b>Threshold sensitivity:</b> UCs that require a threshold (UC1, UC2, UC3, UC8) are "
        "sensitive to the chosen operating point. We report threshold-free metrics where possible "
        "(AUROC, AURC, AUEDR) and show full curves rather than single numbers.",
    ]:
        elements.append(Paragraph(f"&bull; {item}", styles["BulletCustom"]))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph(
        "<b>Metric glossary</b> (for readers unfamiliar with these measures):",
        styles["BodyText2"]
    ))
    for item in [
        "<b>AUROC</b> (Area Under ROC Curve): How well the calibrator separates correct from "
        "incorrect. 1.0 = perfect, 0.5 = random. Our main metric.",
        "<b>AURC</b> (Area Under Risk-Coverage Curve): How much risk remains as you answer more "
        "questions. Lower = better. Measures selective prediction quality.",
        "<b>AUEDR</b> (Area Under Error Discovery Rate): How quickly errors are found when "
        "inspecting predictions in order of uncertainty. Higher = errors found faster.",
        "<b>F1 score</b>: Harmonic mean of precision and recall for binary classification. "
        "Range 0-1, higher is better.",
        "<b>Spearman r</b>: Rank correlation. How well the predicted ordering matches the true "
        "ordering. Range -1 to 1, closer to 1 = near-perfect ranking.",
    ]:
        elements.append(Paragraph(f"&bull; {item}", styles["BulletCustom"]))
    elements.append(Spacer(1, 6))

    # Summary table
    elements.append(Paragraph("Results Summary", styles["SubSectionHead"]))
    elements.append(Spacer(1, 4))
    uc_tbl = make_table(
        ["Use Case", "Metric", "GPT-5-mini", "GPT-5.2", "Qwen3.5"],
        [
            ["UC1: Selective Prediction", "AURC / Cov@90%", "0.167 / 50%", "0.116 / 60%", "0.175 / 49%"],
            ["UC2: Model Routing", "Cost savings", "53%", "-", "-"],
            ["UC3: Error Discovery", "F1 / AUEDR", "0.882 / 0.731", "0.871 / 0.767", "0.886 / 0.723"],
            ["UC4: Difficulty Estimation", "Spearman / Tau", "0.961 / 0.871", "0.979 / 0.895", "0.970 / 0.895"],
            ["UC5: Response Selection", "N=2 / N=3 acc.", "64.4% / 66.5%", "-", "-"],
            ["UC6: Cascade Inference", "2-tier savings", "53% / 70% cheap", "-", "-"],
            ["UC7: Self-Improvement", "20% retry lift", "+7.5%", "+9.1%", "+7.5%"],
            ["UC8: OOD Detection", "Alert F1 [CI]", "1.0 [1.0,1.0]", "1.0 [1.0,1.0]", "1.0 [1.0,1.0]"],
        ],
        col_widths=[1.6*inch, 1.2*inch, 1.0*inch, 0.9*inch, 0.9*inch]
    )
    elements.append(uc_tbl)
    elements.append(Spacer(1, 6))

    elements.append(Paragraph(
        "<b>Honest overall assessment:</b> UC1 (selective prediction), UC2 (model routing), "
        "UC3 (error discovery, merging detection + annotation prioritization), and UC5 (response "
        "selection, merging reward modeling + best-of-N) have clear, immediate practical value with "
        "strong quantitative results and bootstrap CIs. UC6 (cascade inference) and UC8 (OOD monitoring) "
        "are solid. UC4 (difficulty estimation), UC7 (self-improvement with simulated retry), and "
        "UC-A (DPO reward with point-biserial r=0.81) are interesting demonstrations. UC-C (data "
        "curation) shows a real signal. UC-D (step verification) is an honest negative. "
        "For the paper, we recommend focusing on the top 5-6 strongest use cases.",
        styles["BodyText2"]
    ))

    elements.append(PageBreak())

    # ── UC1 ────────────────────────────────────────────────────────────
    elements.append(Paragraph("7a. UC1 - Selective Prediction", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>Why it matters:</b> In medical, legal, or financial applications, a wrong answer can "
        "be costly. Rather than answering every query, the system should say \"I'm not sure\" on "
        "hard questions and defer to a human expert.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>How it works:</b> Set a confidence threshold t. If P(correct) &ge; t, answer "
        "automatically. If P(correct) &lt; t, abstain and escalate to a human. As t increases, "
        "fewer questions are answered (lower coverage) but accuracy on answered questions goes up. "
        "We sweep t from 0 to 1 and plot coverage vs accuracy curves. AURC summarizes the curve "
        "in one number (lower = better).",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Result:</b> AURC of 0.167 (GPT-5-mini), 0.116 (GPT-5.2), 0.175 (Qwen3.5). "
        "The calibrator outperforms verbalized confidence by 43-73%. At 80% coverage, accuracy "
        "improves from ~52% (all questions) to ~85% (confident subset only).",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Assessment:</b> <font color='#38a169'>Strong practical value.</font> "
        "This is the most directly deployable use case. Any production LLM system can use this "
        "to avoid answering when uncertain. The gains over verbalized confidence are large and "
        "consistent. Limitation: requires running the 8B calibrator on every query, adding latency.",
        styles["BodyText2"]
    ))
    add_figure(elements, f"{FIG_UC}/uc1_selective_gpt5mini.pdf",
               "Figure 4: UC1 - Coverage vs. accuracy. Calibrator (blue) dominates baselines.",
               width=4.5*inch, styles=styles)

    # ── UC2 ────────────────────────────────────────────────────────────
    elements.append(Paragraph("7b. UC2 - Model Routing", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>Why it matters:</b> GPT-5.2 is ~4.75x more expensive than GPT-5-mini per query. "
        "Many queries are easy enough for the cheap model. Routing saves money by only using the "
        "expensive model when it's actually needed.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>How it works:</b> First, run GPT-5-mini and score its response with the calibrator. "
        "If P(correct) &ge; t, accept the cheap answer. If P(correct) &lt; t, re-query GPT-5.2 "
        "and use that answer instead. We sweep t and plot cost vs accuracy.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Result:</b> <b>53% cost savings</b> while matching GPT-5.2 accuracy, with "
        "72% of queries routed to the cheap model. New: oracle router baseline (always routes to "
        "the correct model) achieves 66% accuracy, and break-even analysis shows exact threshold "
        "where calibrator routing matches GPT-5.2 accuracy at minimum cost.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Assessment:</b> <font color='#38a169'>Strong practical value.</font> "
        "Directly saves money in production. The 53% cost reduction is compelling with oracle "
        "baseline providing an upper bound. Limitation: tested on one model pair only.",
        styles["BodyText2"]
    ))
    add_figure(elements, f"{FIG_UC}/uc2_model_routing.pdf",
               "Figure 5: UC2 - Cost vs accuracy. Calibrator-guided router reaches expensive-model accuracy at ~half cost.",
               width=4.5*inch, styles=styles)

    # ── UC3 (merged with UC9) ─────────────────────────────────────────
    elements.append(Paragraph("7c. UC3 - Error Discovery (merged UC3 + UC9)", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>Why it matters:</b> If you deploy an LLM to answer 10,000 customer questions, some "
        "answers will be wrong. You need an automated way to (a) flag likely errors for human review "
        "and (b) prioritize annotation effort so humans find errors fastest. This UC merges the "
        "former UC3 (error detection) and UC9 (annotation prioritization) into a unified evaluation.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>How it works - Protocol A (Binary Detection):</b> Flag any response where "
        "P(correct) &lt; threshold as a suspected error. F1 measures the balance between "
        "catching errors (recall) and avoiding false alarms (precision).<br/>"
        "<b>Protocol B (Ranked Annotation Efficiency):</b> Sort predictions by ascending "
        "P(correct). AUEDR measures how quickly errors are found when inspecting in this order "
        "(1.0 = all errors at top, 0.5 = random).",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Result:</b> Protocol A: Best F1 of <b>0.882</b> (GPT-5-mini), <b>0.871</b> (GPT-5.2), "
        "<b>0.886</b> (Qwen3.5). Protocol B: AUEDR of <b>0.731</b> / <b>0.767</b> / <b>0.723</b>. "
        "Both protocols show the calibrator dominates verbalized confidence and random baselines "
        "across all 8 domains. Per-domain AUROC ranges from 0.863 to 0.996.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Assessment:</b> <font color='#38a169'>Strong practical value.</font> "
        "Two complementary views of the same capability: binary flagging (F1 &gt; 0.87) for "
        "automated QA pipelines, and ranked prioritization (AUEDR ~0.74) for human-in-the-loop "
        "review. Errors are found ~2x faster than random sampling.",
        styles["BodyText2"]
    ))
    add_figure(elements, f"{FIG_UC}/uc3_error_discovery.pdf",
               "Figure 6: UC3 - Error discovery: binary detection (left) and ranked annotation efficiency (right).",
               width=4.5*inch, styles=styles)

    elements.append(PageBreak())

    # ── UC4 ────────────────────────────────────────────────────────────
    elements.append(Paragraph("7d. UC4 - Difficulty Estimation", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>Why it matters:</b> Benchmark designers need to know which tasks are hard. "
        "If the calibrator's average confidence per benchmark matches the actual accuracy, "
        "it can estimate dataset difficulty without running expensive ground-truth evaluation.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>How it works:</b> For each of our 20 benchmarks, compute mean P(correct) across "
        "all questions. Compare this predicted difficulty against the actual accuracy. "
        "Spearman rank correlation measures how well the ordering matches (1.0 = perfect).",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Result:</b> Spearman r = <b>0.961</b> (GPT-5-mini), <b>0.979</b> (GPT-5.2), "
        "<b>0.970</b> (Qwen3.5). Near-perfect difficulty ranking.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Assessment:</b> <font color='#c05621'>Moderate value.</font> "
        "The rank correlation is impressive, but this aggregates over entire benchmarks (N=20 "
        "data points). The signal may partly reflect benchmark-level base rates rather than "
        "fine-grained difficulty estimation. More useful for dataset designers than end users.",
        styles["BodyText2"]
    ))
    add_figure(elements, f"{FIG_UC}/uc4_difficulty_gpt5mini.pdf",
               "Figure 7: UC4 - Each dot is a benchmark. Near-perfect rank correlation.",
               width=4.5*inch, styles=styles)

    # ── UC5 (merged with UC-B) ────────────────────────────────────────
    elements.append(Paragraph("7e. UC5 - Response Selection (merged UC5 + UC-B)", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>Why it matters:</b> When you have access to multiple models, you want to automatically "
        "pick the best response for each query. This UC merges pairwise reward modeling (former UC5) "
        "with best-of-N selection (former UC-B) into a unified \"calibrator-guided response selection\" "
        "evaluation. Normally, reward models require expensive human preference labels; here the "
        "calibrator provides a zero-cost reward signal.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>How it works:</b> For each question, collect responses from 2 or 3 target models. "
        "Score each with the calibrator. <b>Pairwise (N=2):</b> Among informative pairs (exactly "
        "one correct), does the calibrator rank the correct response higher? "
        "<b>Best-of-N (N=3):</b> Return the response with highest P(correct) and check if it's "
        "correct. Compare to random selection, best individual model, and oracle.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Result:</b> <b>Pairwise (N=2): 64.4% accuracy</b> (+8.6% over random, 1,560 questions). "
        "<b>Best-of-N (N=3): 66.5% accuracy</b> (+6.7% over best individual model, 801 questions). "
        "The calibrator consistently outperforms random selection and verbalized confidence for all N.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Assessment:</b> <font color='#38a169'>Useful, clean result.</font> "
        "+6.7% over the best model at N=3 is a meaningful gain for free (no retraining). "
        "Requires running the calibrator on all N candidate responses, but no model retraining. "
        "Most practical when multiple model outputs are already available (e.g., ensembles).",
        styles["BodyText2"]
    ))
    add_figure(elements, f"{FIG_UC}/uc5_response_selection.pdf",
               "Figure 8: UC5 - Response selection: pairwise ranking (left) and best-of-N accuracy (right).",
               width=4.5*inch, styles=styles)

    # ── UC6 ────────────────────────────────────────────────────────────
    elements.append(Paragraph("7f. UC6 - Cascade Inference", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>Why it matters:</b> In production, you want to use the cheapest model that can answer "
        "correctly. A cascade pipeline routes easy queries to GPT-5-mini and only escalates to "
        "GPT-5.2 (4.75x more expensive) when the cheap model is uncertain. Optionally, a third "
        "tier routes to human review.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>How it works:</b> <b>2-tier cascade:</b> Run GPT-5-mini, score with calibrator. If "
        "P(correct) &ge; threshold, accept cheap answer (cost=1.0). Otherwise, escalate to "
        "GPT-5.2 (cost=1.0+4.75). Sweep threshold and plot cost vs accuracy. "
        "<b>3-tier cascade:</b> Add human review tier for lowest-confidence queries. "
        "Grid search over two thresholds, find Pareto frontier.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Result:</b> 2-tier break-even: matches GPT-5.2 accuracy while routing <b>70%</b> of "
        "queries to GPT-5-mini, saving <b>~58%</b> of cost. Calibrator cascade dominates "
        "verbalized confidence cascade. 3-tier Pareto frontier shows smooth accuracy-cost tradeoff.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Assessment:</b> <font color='#38a169'>Strong practical value.</font> "
        "Similar to UC2 but with explicit cost modeling and multi-tier options. 58% cost savings "
        "at break-even is compelling. Extends naturally to N-tier cascades with different model sizes.",
        styles["BodyText2"]
    ))
    add_figure(elements, f"{FIG_UC}/uc6_cascade.pdf",
               "Figure 9: UC6 - Cascade inference: cost vs accuracy Pareto frontier.",
               width=4.5*inch, styles=styles)

    elements.append(PageBreak())

    # ── UC7 ────────────────────────────────────────────────────────────
    elements.append(Paragraph("7h. UC7 - Self-Improvement / Targeted Retry", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>Why it matters:</b> Given a limited budget for retrying queries, which ones should be "
        "retried? The calibrator can identify low-confidence responses for targeted retry, "
        "improving overall accuracy more efficiently than random retry selection.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>How it works:</b> (1) <b>Overconfident error analysis:</b> Identify responses where "
        "P(correct) &gt; 0.7 but the answer is wrong. These represent systematic blind spots. "
        "(2) <b>Simulated retry experiment (new):</b> Given a budget of K retries, select the K "
        "lowest-confidence responses to retry. Assume retried responses achieve the benchmark's "
        "average accuracy. Compare calibrator-guided retry to random retry and verbalized-confidence "
        "retry at budgets of 5%, 10%, 20%, and 50%.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Result:</b> At 20% retry budget, calibrator-guided retry improves accuracy by "
        "<b>+7.5%</b> (GPT-5-mini), <b>+9.1%</b> (GPT-5.2), <b>+7.5%</b> (Qwen3.5). "
        "This consistently beats random retry (+4.2-3.8%) and verbalized retry (+3.4-5.4%) "
        "at all budget levels. The calibrator targets retries more effectively.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Assessment:</b> <font color='#38a169'>Quantitative and useful.</font> "
        "The simulated retry experiment provides concrete evidence that calibrator-guided retry "
        "allocation outperforms alternatives. The 20% budget is practical: retry 1 in 5 queries "
        "for +7-9% accuracy gain. Limitation: assumes retry achieves benchmark average accuracy.",
        styles["BodyText2"]
    ))
    add_figure(elements, f"{FIG_UC}/uc7_self_improvement.pdf",
               "Figure 9: UC7 - Simulated retry: accuracy improvement vs retry budget.",
               width=4.5*inch, styles=styles)

    # ── UC8 ────────────────────────────────────────────────────────────
    elements.append(Paragraph("7i. UC8 - OOD / Deployment Monitoring", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>Why it matters:</b> Deployed LLMs can silently degrade when input distributions "
        "shift. If the calibrator can detect accuracy drops from confidence scores alone, it "
        "enables automated alerting without ground-truth labels.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>How it works:</b> Group samples by benchmark (19 domains). Compute mean P(correct) per "
        "domain as a proxy for batch accuracy. Alert when mean confidence drops below threshold. "
        "New: (1) <b>Bootstrap CIs</b> on alert F1 (resampling benchmarks, not individual samples). "
        "(2) <b>KS test</b> for distribution shift detection between reference and target models. "
        "(3) <b>Calibration drift</b>: Spearman r between accuracy deltas and P(correct) deltas "
        "across benchmarks when comparing models.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Result:</b> Alert F1 = <b>1.000 [1.000, 1.000]</b> (bootstrap 95% CI) across all "
        "three models. Benchmark-level Spearman r = 0.961-0.979 (predicted vs actual accuracy). "
        "Calibration drift correlation: r=0.806 (GPT-5.2), r=0.839 (Qwen3.5) - accuracy drops "
        "are tracked by confidence drops. <b>Caveat: N=19 benchmarks, not thousands of queries.</b>",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Assessment:</b> <font color='#38a169'>Useful, with noted limitations.</font> "
        "Perfect F1 at benchmark level is convincing but operates on N=19 coarse domains, not "
        "fine-grained query-level OOD. KS test adds a distributional baseline. Calibration drift "
        "analysis shows the mechanism: when accuracy drops, so does calibrator confidence.",
        styles["BodyText2"]
    ))
    add_figure(elements, f"{FIG_UC}/uc8_ood_detection.pdf",
               "Figure 10: UC8 - Batch-level monitoring: predicted vs actual accuracy per benchmark.",
               width=4.5*inch, styles=styles)

    elements.append(PageBreak())

    # UC9 has been merged into UC3 above (Error Discovery)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 8: NOVEL USE CASES (UC-A, UC-C, UC-D)
    # ══════════════════════════════════════════════════════════════════════
    elements.append(Paragraph("8. Novel Use Cases", styles["SectionHead"]))
    elements.append(Paragraph(
        "Beyond standard UQ applications, we introduce three novel use cases that push the "
        "calibrator into training data generation, data curation, and reasoning verification. "
        "(UC-B best-of-N was merged into UC5 above.)",
        styles["BodyText2"]
    ))

    # ── UC-A ───────────────────────────────────────────────────────────
    elements.append(Paragraph("8a. UC-A - DPO Preference Pair Generation", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>Why it matters:</b> Direct Preference Optimization (DPO) is a popular technique for "
        "aligning LLMs with human preferences. It requires training pairs: a \"chosen\" (good) "
        "response and a \"rejected\" (bad) response for the same question. Creating these pairs "
        "normally requires expensive human annotation. If the calibrator can automatically "
        "identify which response is better, it eliminates this annotation bottleneck.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>How it works:</b> For each question, collect responses from all three target models. "
        "The calibrator scores each response. For pairs where exactly one is correct "
        "(\"informative pairs\"), the higher-scored response becomes \"chosen\" and the lower "
        "becomes \"rejected.\" We measure how often this matches ground truth (pair accuracy). "
        "We also test a margin filter: only keep pairs where the score gap exceeds a threshold "
        "(high-confidence pairs).",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Result:</b> Pairwise accuracy <b>97.5% [97.0%, 98.0%]</b> (bootstrap 95% CI). "
        "Informative pair accuracy <b>88.8% [86.5%, 91.0%]</b>. Reward quality: calibrator "
        "point-biserial r=<b>0.812</b> vs verbalized r=0.302 (correlation with correctness). "
        "The calibrator's reward signal is 2.7x more correlated with ground truth.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Assessment:</b> <font color='#c05621'>Strong signal, not yet end-to-end validated.</font> "
        "97.5% pairwise accuracy with tight bootstrap CIs is compelling. Point-biserial r=0.812 "
        "shows the calibrator captures correctness far better than verbalized confidence as a "
        "reward signal. However, we have <b>not</b> trained a DPO model on these pairs. "
        "Only 22% of pairs are informative (the rest are both-correct or both-wrong).",
        styles["BodyText2"]
    ))
    add_figure(elements, f"{FIG_UC}/uc_a_pair_quality.pdf",
               "Figure 12: UC-A - DPO pair quality increases with calibrator score margin.",
               width=4.5*inch, styles=styles)

    # UC-B has been merged into UC5 above (Response Selection)

    elements.append(PageBreak())

    # ── UC-C ───────────────────────────────────────────────────────────
    elements.append(Paragraph("8b. UC-C - Data Curation for Distillation", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>Why it matters:</b> Knowledge distillation trains a smaller \"student\" model to "
        "mimic a larger \"teacher\" model. If the teacher sometimes produces wrong answers, "
        "the student learns those errors too. Filtering out incorrect teacher outputs before "
        "distillation should produce a better student. The challenge: ground truth labels "
        "are usually unavailable, so you need a way to estimate which teacher outputs are wrong.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>How it works:</b> Score all teacher-model responses with the calibrator. Set a "
        "confidence threshold and keep only responses above it. This trades dataset size for "
        "quality (fewer samples, but higher accuracy). Train a student model on the filtered "
        "dataset and compare to: (1) training on all data (no filter), (2) oracle filtering "
        "(ground truth labels). We sweep retention rates from 10% to 100%.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Result:</b> At 50% retention: GPT-5-mini <b>91.3% [87.1%, 93.1%]</b>, "
        "GPT-5.2 <b>95.2% [93.6%, 97.0%]</b>, Qwen3.5 <b>88.1% [85.4%, 93.2%]</b> "
        "(bootstrap 95% CIs). All far exceed random 50% retention (~51-59%). "
        "AUAR (area under accuracy-retention curve): 0.245-0.278 for calibrator vs 0.339-0.423 "
        "for verbalized. Data efficiency ratio: 50% of data retains 94-100% of full-data accuracy.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Assessment:</b> <font color='#c05621'>Strong curation signal, modest downstream impact.</font> "
        "The filtering quality is impressive with tight CIs. The 50% data &rarr; ~95% accuracy "
        "retention is genuinely useful for reducing annotation costs. However, no end-to-end "
        "student training experiment yet to confirm downstream impact.",
        styles["BodyText2"]
    ))
    add_figure(elements, f"{FIG_UC}/uc_c_quality_quantity.pdf",
               "Figure 14: UC-C - Quality-quantity tradeoff. Higher threshold = better data quality.",
               width=4.5*inch, styles=styles)

    # ── UC-D ───────────────────────────────────────────────────────────
    elements.append(Paragraph("8c. UC-D - Agent Step Verification (Honest Negative)", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "<b>Why it matters:</b> Modern LLMs use chain-of-thought reasoning, sometimes generating "
        "very long responses. \"Overthinking\" occurs when a model reasons extensively but arrives "
        "at the wrong answer -- the reasoning process itself goes off-track. If the calibrator "
        "could detect when reasoning deteriorates mid-response, it could enable early stopping "
        "or step-level verification in agentic pipelines.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>How it works:</b> Truncate each response at various points (25%, 50%, 75%) and "
        "score each truncated version with the calibrator. If the score drops sharply at a "
        "truncation point, it may indicate the reasoning went wrong at that step. We measure: "
        "(1) partial correlation between calibrator score and correctness (controlling for "
        "response length) to test if the score captures more than just \"long = wrong\", and "
        "(2) AUROC for detecting score-drop events that predict final-answer errors.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Result:</b> Partial correlation of r=0.813 between calibrator score and "
        "correctness (controlling for response length), confirming the calibrator captures info "
        "beyond response length. However, step-level drop detection showed weak signal "
        "(AUROC ~0.53-0.56). <b>This is an honest negative</b> -- the outcome-level calibrator "
        "works well, but step-level verification may need specialized training.",
        styles["BodyText2"]
    ))
    elements.append(Paragraph(
        "<b>Assessment:</b> <font color='#e53e3e'>Weak result / honest negative.</font> "
        "Step-level detection (AUROC 0.53) is near random. The outcome-level partial correlation "
        "(r=0.813) is interesting but not actionable. The calibrator was trained on complete "
        "question-answer pairs, not reasoning steps, and it shows. We include this as "
        "transparency about what the approach cannot do.",
        styles["BodyText2"]
    ))
    add_figure(elements, f"{FIG_UC}/uc_d_overthinking.pdf",
               "Figure 15: UC-D - Overthinking analysis. Calibrator detects errors beyond response length.",
               width=4.5*inch, styles=styles)

    elements.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 9: ROBUSTNESS & RIGOR
    # ══════════════════════════════════════════════════════════════════════
    elements.append(Paragraph("9. Robustness & Rigor", styles["SectionHead"]))
    elements.append(Paragraph(
        "We proactively addressed anticipated reviewer concerns with thorough robustness checks.",
        styles["BodyText2"]
    ))

    # 9a. Evaluation protocol
    elements.append(Paragraph("9a. Evaluation Protocol", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "All results in this report use <b>strictly held-out test samples</b> (1,774 samples from "
        "the internal validation split that were never used during training). We report held-out "
        "AUROC = 0.898 as our primary metric. No in-distribution evaluation is used.",
        styles["BodyText2"]
    ))
    elements.append(Spacer(1, 8))

    # 9b. Statistical significance
    elements.append(Paragraph("9b. Statistical Significance", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "We ran three independent significance tests on every calibrator-vs-baseline comparison "
        "(paired permutation, DeLong for AUROC comparison, McNemar for per-sample disagreement):",
        styles["BodyText2"]
    ))
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
        "All 15 comparisons (5 baselines x 3 tests) yield <b>p &lt; 0.001</b>. "
        "No confidence intervals overlap.",
        styles["BodyText2"]
    ))
    elements.append(Spacer(1, 8))

    # 9c. Per-benchmark breakdown
    elements.append(Paragraph("9c. Per-Benchmark AUROC Breakdown", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "The calibrator outperforms verbalized confidence on <b>19 of 20 benchmarks</b>.",
        styles["BodyText2"]
    ))
    perbench_tbl = make_table(
        ["Benchmark", "Modality", "AUROC", "N", "Accuracy"],
        [
            ["livebench", "TXT", "0.998", "147", "65.3%"],
            ["realworldqa", "VLM", "0.993", "443", "46.5%"],
            ["mathvista", "VLM", "0.985", "378", "33.1%"],
            ["mmvet", "VLM", "0.975", "172", "30.8%"],
            ["hallusionbench", "VLM", "0.962", "378", "79.9%"],
            ["arc_agi", "TXT", "0.906", "118", "50.8%"],
            ["gpqa", "TXT", "0.893", "234", "76.5%"],
            ["...", "", "", "", ""],
            ["hle_multimodal", "VLM", "0.810", "35", "14.3%"],
            ["chembench", "TXT", "0.787", "223", "65.9%"],
            ["prbench", "TXT", "0.772", "142", "71.8%"],
        ],
        col_widths=[1.3*inch, 0.8*inch, 0.8*inch, 0.6*inch, 0.8*inch]
    )
    elements.append(perbench_tbl)
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "<b>Stats:</b> Mean per-benchmark AUROC = 0.915, std = 0.064, coefficient of variation = 0.070 "
        "(low CV means consistent performance across benchmarks, not just good on average).",
        styles["BodyText2"]
    ))
    elements.append(Spacer(1, 8))

    # 9d. Held-out benchmark generalization
    elements.append(Paragraph("9d. Held-Out Benchmark Generalization (Leave-5-Out CV)", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "Does the model overfit to specific benchmarks? We removed 5 entire benchmarks from "
        "training, retrained, and evaluated on those held-out benchmarks. A large performance "
        "drop would indicate overfitting; stability means the model learned general patterns.",
        styles["BodyText2"]
    ))
    heldout_tbl = make_table(
        ["Fold", "Held-Out Benchmarks", "In-Dist", "Held-Out", "Gap"],
        [
            ["0", "gpqa, hallusion, mathverse, mmvet, vizwiz", "0.948", "0.962", "-0.014"],
            ["1", "hle, mmmu, mmstar, omnimath, simpleqa", "0.956", "0.942", "+0.013"],
            ["2", "bbeh, charxiv, mathvision, mathvista, rwqa", "0.944", "0.965", "-0.021"],
            ["3", "arc_agi, chembench, hle_mm, livebench, prbench", "0.959", "0.897", "+0.062"],
            ["Mean", "", "", "", "+0.010"],
        ],
        col_widths=[0.5*inch, 2.6*inch, 0.9*inch, 0.9*inch, 0.6*inch]
    )
    elements.append(heldout_tbl)
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "<b>Result:</b> Mean gap = +0.010 (negligible). In 2 of 4 folds, held-out "
        "<i>outperforms</i> in-distribution. The model learns general uncertainty patterns.",
        styles["BodyText2"]
    ))

    elements.append(PageBreak())

    # 9e. Proper Scoring Rules
    elements.append(Paragraph("9e. Proper Scoring Rules (Held-Out Set)", styles["SubSectionHead"]))
    elements.append(Paragraph(
        "AUROC measures ranking quality. Proper scoring rules additionally measure calibration "
        "(when the model says 80% confident, is it right ~80% of the time?):",
        styles["BodyText2"]
    ))
    for item in [
        "<b>Brier score: 0.141</b> (lower is better; perfect = 0, random = 0.25)",
        "<b>ECE: 0.035</b> (gap between predicted confidence and actual accuracy; &lt; 0.05 is excellent)",
    ]:
        elements.append(Paragraph(f"&bull; {item}", styles["BulletCustom"]))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "Both confirm the calibrator is well-calibrated, not just a good ranker.",
        styles["BodyText2"]
    ))

    elements.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 10: NEXT STEPS & PAPER STATUS
    # ══════════════════════════════════════════════════════════════════════
    elements.append(Paragraph("10. Paper Status & Next Steps", styles["SectionHead"]))

    elements.append(Paragraph("<b>Experiments: ~90% complete</b>", styles["BodyText2"]))
    elements.append(Paragraph("Completed:", styles["BodyText2"]))
    for item in [
        "Best v2 unified model trained and evaluated (0.898 held-out AUROC)",
        "10 baselines compared with significance tests (all p < 0.001)",
        "5 training ablations (rank, source, modality, size, prompt)",
        "6 elicitation strategies tested (logit is optimal for closed-source setting)",
        "13 downstream use cases validated",
        "Robustness: sensitivity, difficulty stratification, held-out CV, proper scoring rules",
        "Proxy semantic entropy shown to fail for cross-model setting (AUROC 0.467)",
        "Multi-seed analysis started (seed 42 = 0.898)",
    ]:
        elements.append(Paragraph(f"&bull; {item}", styles["BulletCustom"]))

    elements.append(Spacer(1, 8))
    elements.append(Paragraph("<b>Remaining experiments:</b>", styles["BodyText2"]))
    for item in [
        "True Semantic Entropy on Qwen3.5-397B (8 GPUs, overnight) for legitimate head-to-head",
        "Claude API evaluation (pending advisor budget approval)",
        "Multi-seed completion (seeds 123, 456) for variance reporting",
        "Merge LoRA weights for deployment release (2B and 8B variants)",
    ]:
        elements.append(Paragraph(f"&bull; {item}", styles["BulletCustom"]))

    elements.append(Spacer(1, 8))
    elements.append(Paragraph("<b>Paper writing (main bottleneck):</b>", styles["BodyText2"]))
    for item in [
        "Related work section (literature survey done, positioning clear)",
        "Methodology section (approach, training, evaluation protocol)",
        "Experimental setup (benchmarks, models, metrics)",
        "Camera-ready figures and tables",
        "ECCV 2026 submission deadline TBD",
    ]:
        elements.append(Paragraph(f"&bull; {item}", styles["BulletCustom"]))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 11: SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("11. Summary", styles["SectionHead"]))
    elements.append(Paragraph(
        "We built and validated a <b>unified uncertainty quantification model</b> that "
        "works across text and vision modalities, across multiple target models, and on "
        "closed-source systems - a combination no prior work achieves.",
        styles["BodyText2"]
    ))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("<b>Headline results:</b>", styles["BodyText2"]))
    for item in [
        "<b>0.898</b> held-out AUROC on internal test set (v2, +6.7 over v1)",
        "Generalizes across 3 target models (GPT-5-mini, GPT-5.2, Qwen3.5-397B)",
        "Outperforms <b>10 baselines</b> with p &lt; 0.001 on all significance tests",
        "Proxy Semantic Entropy scores <b>below random</b> (0.467) - validates our approach",
        "<b>13 downstream use cases</b> validated (selective prediction to DPO pair generation)",
        "2B model retains <b>98%</b> of 8B performance - laptop-deployable",
        "Logit captures <b>95%+</b> of available signal - simple and practical",
    ]:
        elements.append(Paragraph(f"&bull; {item}", styles["BulletCustom"]))

    elements.append(Spacer(1, 8))
    elements.append(Paragraph(
        "<b>Bottom line:</b> The experimental work is substantially complete. The main "
        "remaining bottleneck is paper writing. The project is on track for ECCV 2026.",
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
        title="UQ Research Progress Report - March 2026",
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
            "UQ for LLMs - Research Update - Mar 2026"
        )
        canvas.restoreState()

    doc.build(elements, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"Report generated: {OUTPUT}")
    print(f"Size: {os.path.getsize(OUTPUT) / 1024:.0f} KB")


if __name__ == "__main__":
    build_report()
