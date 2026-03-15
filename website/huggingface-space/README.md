---
title: Pinocchio - LLM Uncertainty Estimation
emoji: "\U0001F9E5"
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: 5.12.0
python_version: "3.12"
app_file: app.py
pinned: true
license: apache-2.0
models:
  - KevinDavidHayes/pinocchio-0.8b
tags:
  - uncertainty-quantification
  - llm-evaluation
  - calibration
short_description: "Estimate the uncertainty of any LLM in a single forward pass"
---

# Pinocchio Demo

Estimate the uncertainty of **any** LLM response in a single forward pass.

Paste a question and an LLM's answer, and Pinocchio will tell you how likely the answer is to be correct.

## Links

- **Paper**: [Pinocchio: Estimating the Uncertainty of Black-Box Language Models](https://arxiv.org/abs/TODO) (ICML 2026)
- **Package**: `pip install pinocchio-uq`
- **Model**: [KevinDavidHayes/pinocchio-0.8b](https://huggingface.co/KevinDavidHayes/pinocchio-0.8b)
- **Code**: [GitHub](https://github.com/KevinDavidHayes/pinocchio)
