"""Pinocchio Demo — Hugging Face Spaces Gradio App.

Try Pinocchio: score any LLM response for correctness in real time.
"""

import gradio as gr
import torch
import time

# Global model (loaded once at startup)
judge = None


def load_judge():
    """Load the Pinocchio model at startup."""
    global judge
    from pinocchio_local import Pinocchio

    judge = Pinocchio(device_map="cpu", torch_dtype=torch.float32)


# --- Example data -----------------------------------------------------------

EXAMPLES = [
    # (question, answer, source_model, benchmark)
    # Correct: GPQA physics (high confidence)
    [
        "The value of sqrt(3^3 + 3^3 + 3^3) is what?",
        '{"reasoning": "Compute 3^3 = 27. Then 3^3 + 3^3 + 3^3 = 27 + 27 + 27 = 81. Finally, sqrt(81) = 9.", "answer": 9, "confidence": 1.0}',
        "gpt-5-mini",
        "omnimath",
    ],
    # Incorrect: GPQA chemistry
    [
        "Which of the following molecules has C3h symmetry?\n(A) triisopropyl borate\n(B) quinuclidine\n(C) benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone\n(D) triphenylene trifuran hexaone",
        '{"answer": "B", "confidence": 0.6}',
        "gpt-5-mini",
        "gpqa",
    ],
    # Correct: SimpleQA factual
    [
        "What is the birth name of the Mexican artist Peso Pluma?",
        '{"answer": "Hassan Emilio Kabande Laija", "confidence": 0.8}',
        "gpt-5-mini",
        "simpleqa",
    ],
    # Incorrect: SimpleQA factual
    [
        "In TWD Season 11, Episode 10, we learn that Connie got whose uncle kicked out of Congress before the apocalypse?",
        '{"answer": "Yumiko", "confidence": 0.4}',
        "gpt-5-mini",
        "simpleqa",
    ],
    # Correct: PRBench finance (long-form)
    [
        "Our revolver is almost maxed, we're burning cash, and our EBITDA is down 20% YoY. We have a $50mm term loan maturing in 9 months. What are our options to avoid default?",
        '{"reasoning": "You have a near-maxed revolver, burning cash, and EBITDA down 20% YoY with a $50M term loan maturing in 9 months. Priority 1: negotiate a maturity extension or amend-and-extend with existing lenders. Priority 2: explore asset sales or sale-leaseback for near-term liquidity. Priority 3: seek subordinated or mezzanine capital. Priority 4: if all else fails, consider a prepackaged restructuring.", "answer": "Negotiate extension, asset sales, mezzanine capital, or prepack restructuring"}',
        "gpt-5-mini",
        "prbench",
    ],
    # Correct: Livebench logic puzzle
    [
        "In this question, assume each person either always tells the truth or always lies. Charlie is at the barbershop. The person at the museum says the person at the ice skating rink lies. Is Charlie a truth-teller or a liar?",
        '{"reasoning": "Assign truth values. The person at the museum says the person at the ice rink lies. If the museum person is truthful, then the rink person lies. Cross-referencing other constraints, Charlie at the barbershop is a truth-teller.", "answer": "Charlie is a truth-teller"}',
        "gpt-5-mini",
        "livebench",
    ],
    # Incorrect: HLE probability (hard)
    [
        "Four bikes are racing. The true probabilities of each bike winning are: Bike 1: 1/2, Bike 2: 1/4, Bike 3: 1/8, Bike 4: 1/8. A bookmaker offers odds of 4-for-1 on Bike 1, 3-for-1 on Bike 2, and 7-for-1 on Bikes 3 and 4. What is the optimal Kelly bet allocation?",
        '{"reasoning": "Interpret odds m-for-1 as net profit m per unit stake, so gross return a_i = m+1. Thus a1=5, a2=4, a3=a4=8. Optimize expected log growth under Kelly criterion. f1=0.375, f2=0.0625, f3=f4=0.0156.", "answer": "f1=37.5%, f2=6.25%, f3=f4=1.56%"}',
        "gpt-5-mini",
        "hle",
    ],
    # Correct: ChemBench
    [
        "Which of the following is an example of a weak base?\n(A) NH3\n(B) NaOH\n(C) KOH\n(D) Ca(OH)2",
        '{"answer": "A", "confidence": 1.0}',
        "gpt-5-mini",
        "chembench",
    ],
]


# --- Scoring function -------------------------------------------------------


def score_response(question: str, answer: str, source_model: str, benchmark: str):
    """Score a single Q&A pair and return formatted results."""
    if not question.strip():
        return "", "", ""
    if not answer.strip():
        return "", "", ""

    start = time.time()
    p_correct = judge.score(
        question=question.strip(),
        answer=answer.strip(),
        source_model=source_model.strip(),
        benchmark=benchmark.strip(),
    )
    elapsed = time.time() - start

    # Color-coded confidence label
    if p_correct >= 0.8:
        label = "High confidence"
        color = "#22c55e"  # green
        emoji = "&#x2705;"
    elif p_correct >= 0.5:
        label = "Medium confidence"
        color = "#f59e0b"  # amber
        emoji = "&#x26A0;&#xFE0F;"
    else:
        label = "Low confidence"
        color = "#ef4444"  # red
        emoji = "&#x274C;"

    # Main score display
    score_html = f"""
    <div style="text-align: center; padding: 24px;">
        <div style="font-size: 64px; font-weight: 700; color: {color}; line-height: 1;">
            {p_correct:.1%}
        </div>
        <div style="font-size: 18px; color: {color}; margin-top: 8px; font-weight: 500;">
            {emoji} {label}
        </div>
        <div style="font-size: 13px; color: #888; margin-top: 12px;">
            Scored in {elapsed:.2f}s on CPU
        </div>
    </div>
    """

    # Gauge bar
    gauge_html = f"""
    <div style="padding: 16px;">
        <div style="display: flex; justify-content: space-between; font-size: 12px; color: #888; margin-bottom: 4px;">
            <span>Likely wrong</span>
            <span>Likely correct</span>
        </div>
        <div style="background: linear-gradient(to right, #ef4444, #f59e0b, #22c55e);
                    border-radius: 8px; height: 16px; position: relative;">
            <div style="position: absolute; left: {p_correct * 100}%;
                        top: -4px; transform: translateX(-50%);
                        width: 4px; height: 24px; background: white;
                        border: 2px solid #333; border-radius: 2px;">
            </div>
        </div>
        <div style="text-align: center; margin-top: 8px; font-size: 13px; color: #aaa;">
            P(correct) = {p_correct:.4f}
        </div>
    </div>
    """

    # Interpretation
    if p_correct >= 0.8:
        advice = "Pinocchio is confident this answer is correct. Safe to use as-is."
    elif p_correct >= 0.5:
        advice = "Pinocchio is uncertain. Consider verifying this answer or asking a stronger model."
    else:
        advice = "Pinocchio thinks this answer is likely wrong. Do not trust it without verification."

    interpretation_html = f"""
    <div style="padding: 12px 16px; background: #1a1a2e; border-radius: 8px;
                border-left: 4px solid {color}; font-size: 14px; color: #ccc;">
        {advice}
    </div>
    """

    return score_html, gauge_html, interpretation_html


# --- Build the Gradio interface ---------------------------------------------

HEADER_HTML = """
<div style="text-align: center; padding: 20px 0 10px 0;">
    <h1 style="font-size: 2.5em; margin: 0; font-weight: 700;">
        &#x1F9E5; Pinocchio
    </h1>
    <p style="font-size: 1.15em; color: #aaa; margin: 8px 0 0 0;">
        Estimate the uncertainty of <em>any</em> LLM in a single forward pass
    </p>
    <p style="font-size: 0.9em; color: #666; margin: 8px 0 0 0;">
        <a href="https://pypi.org/project/pinocchio-uq/" target="_blank">pip install pinocchio-uq</a>
        &nbsp;&bull;&nbsp;
        <a href="https://github.com/KevinDavidHayes/pinocchio" target="_blank">GitHub</a>
        &nbsp;&bull;&nbsp;
        <a href="https://huggingface.co/KevinDavidHayes/pinocchio-0.8b" target="_blank">Model</a>
    </p>
</div>
"""

HOW_IT_WORKS_HTML = """
<div style="padding: 16px; font-size: 14px; color: #bbb; line-height: 1.6;">
    <h3 style="margin-top: 0;">How it works</h3>
    <p>
        Pinocchio is a <strong>0.8B parameter</strong> language model fine-tuned to predict whether
        an LLM's response is correct. Given any (question, answer) pair, it outputs a calibrated
        probability P(correct) in a single forward pass.
    </p>
    <ul style="padding-left: 20px;">
        <li><strong>Works on any model</strong> &mdash; GPT-5, Claude, Gemini, LLaMA, Qwen, etc.</li>
        <li><strong>No internals needed</strong> &mdash; just the question and answer text.</li>
        <li><strong>Single pass</strong> &mdash; no sampling, no multiple generations.</li>
        <li><strong>Calibrated</strong> &mdash; returns real probabilities, not just yes/no.</li>
    </ul>
    <h3>Use cases</h3>
    <ul style="padding-left: 20px;">
        <li><strong>Selective prediction</strong> &mdash; abstain when uncertain</li>
        <li><strong>Model routing</strong> &mdash; escalate to stronger models on hard questions</li>
        <li><strong>Error detection</strong> &mdash; flag likely-wrong answers for human review</li>
        <li><strong>Benchmark evaluation</strong> &mdash; estimate accuracy without ground truth</li>
    </ul>
    <h3>Quick install</h3>
    <pre style="background: #111; padding: 12px; border-radius: 6px; overflow-x: auto;">pip install pinocchio-uq

from pinocchio import Pinocchio
judge = Pinocchio()
score = judge.score(question="What is 2+2?", answer="4")
print(f"P(correct) = {score:.3f}")  # 0.97</pre>
</div>
"""

CSS = """
.main-container { max-width: 900px; margin: 0 auto; }
footer { display: none !important; }
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="emerald"),
    css=CSS,
    title="Pinocchio - LLM Uncertainty Estimation",
) as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown(
                "*Pinocchio is trained on challenging benchmarks (GPQA, MMLU, HLE, etc.) and works best "
                "on non-trivial questions. Paste the **full LLM response** (not just the final answer) for best results.*"
            )
            question_input = gr.Textbox(
                label="Question",
                placeholder="Paste a challenging question (e.g. from GPQA, MMLU, SimpleQA, math competitions...)",
                lines=3,
            )
            answer_input = gr.Textbox(
                label="LLM's Answer",
                placeholder="Paste the full LLM response including reasoning, not just the final answer...",
                lines=5,
            )
            with gr.Row():
                model_input = gr.Textbox(
                    label="Source Model (optional)",
                    placeholder="e.g. gpt-5, claude-4-sonnet",
                    scale=1,
                )
                benchmark_input = gr.Textbox(
                    label="Benchmark (optional)",
                    placeholder="e.g. mmlu, math, triviaqa",
                    scale=1,
                )
            score_btn = gr.Button("Score Response", variant="primary", size="lg")

        with gr.Column(scale=2):
            score_output = gr.HTML(
                value='<div style="text-align:center; padding:40px; color:#666;">Enter a question and answer, then click Score</div>'
            )
            gauge_output = gr.HTML()
            interpretation_output = gr.HTML()

    gr.Examples(
        examples=EXAMPLES,
        inputs=[question_input, answer_input, model_input, benchmark_input],
        outputs=[score_output, gauge_output, interpretation_output],
        fn=score_response,
        cache_examples=False,
        label="Try these examples",
    )

    score_btn.click(
        fn=score_response,
        inputs=[question_input, answer_input, model_input, benchmark_input],
        outputs=[score_output, gauge_output, interpretation_output],
    )

    with gr.Accordion("How it works & Installation", open=False):
        gr.HTML(HOW_IT_WORKS_HTML)

# Load model at startup
print("Loading Pinocchio model...")
load_judge()
print("Model loaded!")

demo.launch()
