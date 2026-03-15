"""Pinocchio model loading and scoring — inlined for HF Spaces deployment."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError:
    Qwen3_5ForConditionalGeneration = None

logger = logging.getLogger(__name__)

DEFAULT_HUB_REPO = "KevinDavidHayes/pinocchio-0.8b"
DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-0.8B"

# --- Prompt templates -------------------------------------------------------

PROMPT_BASELINE = """Question: {question}

Answer: {response}

Is the answer correct? (i) No (ii) Yes"""

PROMPT_COMBINED = """Benchmark: {benchmark}
Source model: {source_model}

Question: {question}

Answer: {response}

Analyze whether the answer above is correct. Consider:
- Does the answer address the question?
- Are there factual errors or logical flaws?
- Is the answer complete?

Based on your analysis, is the answer correct? (i) No (ii) Yes"""

TRUNCATION = {
    "baseline": (500, 300),
    "combined": (1500, 800),
}


def format_prompt(question, response, variant="combined", benchmark="", source_model=""):
    q_len, r_len = TRUNCATION.get(variant, TRUNCATION["combined"])
    template = PROMPT_COMBINED if variant == "combined" else PROMPT_BASELINE
    return template.format(
        question=question[:q_len],
        response=response[:r_len],
        benchmark=benchmark,
        source_model=source_model,
    )


# --- Model loading -----------------------------------------------------------


def load_model(adapter=DEFAULT_HUB_REPO, base_model=DEFAULT_BASE_MODEL,
               device_map="auto", torch_dtype=None):
    if torch_dtype is None:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32  # CPU: use float32 for compatibility

    logger.info("Loading base model: %s", base_model)
    if Qwen3_5ForConditionalGeneration is not None and "Qwen3.5" in base_model:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            base_model, torch_dtype=torch_dtype, device_map=device_map,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch_dtype, device_map=device_map,
            trust_remote_code=True,
        )

    logger.info("Loading LoRA adapter: %s", adapter)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    device = next(model.parameters()).device
    logger.info("Model loaded on device: %s", device)

    return model, tokenizer, device


def score_single(model, tokenizer, device, question, response,
                 prompt_variant="combined", benchmark="", source_model=""):
    prompt = format_prompt(
        question=question, response=response, variant=prompt_variant,
        benchmark=benchmark, source_model=source_model,
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]
    token_no = tokenizer.encode("(i", add_special_tokens=False)[-1]
    token_yes = tokenizer.encode("(ii", add_special_tokens=False)[-1]
    probs = torch.softmax(logits[[token_no, token_yes]], dim=0)
    return probs[1].item()


# --- High-level API ----------------------------------------------------------


class Pinocchio:
    def __init__(self, adapter=DEFAULT_HUB_REPO, base_model=DEFAULT_BASE_MODEL,
                 device_map="auto", torch_dtype=None):
        self._model, self._tokenizer, self._device = load_model(
            adapter=adapter, base_model=base_model,
            device_map=device_map, torch_dtype=torch_dtype,
        )

    def score(self, question="", answer="", source_model="", benchmark="",
              prompt_variant="combined"):
        if not answer:
            raise ValueError("No answer provided.")
        return score_single(
            self._model, self._tokenizer, self._device,
            question=question, response=answer,
            prompt_variant=prompt_variant,
            benchmark=benchmark, source_model=source_model,
        )
