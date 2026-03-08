# Rebuttal Draft — Pinocchio

## Issue 1: Metadata Leakage (benchmark/model in prompt)
**Status: NEEDS EXPERIMENT (Agent 1)**

*Planned response once we have the no-metadata AUROC:*

> We appreciate this concern. We ran the calibrator with benchmark and source model fields blanked out (empty strings) on the full test set. The AUROC is [X.XXX] without metadata vs. 0.953 with metadata, a drop of [Y] points. This confirms that the calibrator's primary signal comes from the question-response content, not from benchmark/model identity. We will add this ablation to Table 4 and discuss it in the text.

*If the drop is small (<3 pts):* This is our strongest rebuttal point. Emphasize that metadata provides marginal improvement, not the core signal.

*If the drop is large (>5 pts):* Acknowledge honestly. Add discussion about what this means—the calibrator partially leverages knowledge of which benchmark/model produced the response. Note that in real deployment, this information IS typically known (the user knows which model they're querying and what domain the question is from). Reframe: metadata is a feature, not a bug.

---

## Issue 2: Per-benchmark split / Leave-K-Out detail
**Status: FIXED IN PAPER**

> We have added Table 5 (leave-K-out cross-validation) with full per-fold results, including held-out benchmark identities, in-distribution and held-out AUROC, and gap for each fold. Three of four folds show negative gaps (held-out performs *better* than in-distribution), and even the worst fold (fold 3, gap = +0.062, held-out benchmarks: ARC-AGI, ChemBench, HLE-Multimodal, LiveBench, PRBench) maintains 0.897 AUROC. The mean gap of 0.010 ± 0.032 is negligible.

---

## Issue 3: Weak baselines / Missing GPT-5-as-judge
**Status: NEEDS EXPERIMENT (Agent 2)**

*Argument for verbalized confidence being weak:*

> Our verbalized confidence AUROC (0.607) is consistent with the literature when evaluated on *hard* benchmarks. Prior work reporting 0.70–0.85 (Xiong et al., Tian et al.) primarily evaluates on MMLU and similar benchmarks with 70–90% accuracy. Our benchmark suite deliberately targets the 30–70% accuracy range where uncertainty estimation matters most—this is precisely the regime where verbalized confidence is known to degrade (Groot & Valdenegro-Toro, 2024; Xuan et al., 2025). We cite both works in our related work section.

*On GPT-5-as-judge baseline:*

> [Will add once Agent 2 runs this experiment. If we beat GPT-5-as-judge, this becomes a headline result. If GPT-5-as-judge is competitive, we reframe: our method achieves similar discrimination at 10x lower cost (8B inference vs. GPT-5 API call).]

---

## Issue 4: No truly unseen target model
**Status: NEEDS EXPERIMENT (Agents 1/2)**

> [Will add once we have LLaMA-3 or Claude results. Key framing: even moderate performance (>0.75) on a fully unseen model family would demonstrate transfer. The single-model results (0.67–0.76) are for models trained on ONE source; a model trained on THREE sources should generalize better to a fourth.]

---

## Issue 5: Interpretability / What does the calibrator learn?
**Status: NEEDS EXPERIMENT (Agent 1)**

> [Will add scramble test results. If scrambling destroys performance, this proves the calibrator reads content. If not, we have a problem.]

*Pre-emptive argument:* We now include an error analysis (Appendix E) showing systematic failure patterns: confident-but-wrong failures cluster in visual counting/spatial tasks (where textual cues are uninformative), while unconfident-but-right failures cluster in math benchmarks with terse correct answers. These patterns are consistent with content-based reasoning, not surface shortcuts.

---

## Issue 6: Model size ablation — hyperparameter tuning artifact
**Status: ARGUMENT ONLY**

> The reviewer correctly notes that LoRA hyperparameters (r=32, α=64, lr=1e-4) were tuned for the 8B model and applied uniformly across model sizes. The 4B and 9B models' overfitting may partly reflect suboptimal hyperparameters rather than a fundamental model-size effect.
>
> However, two observations support the core finding that small models suffice for this task: (1) The Qwen3-VL family ablation (2B, 4B, 8B), which used a different hyperparameter configuration (r=16, α=32), shows all three sizes within 1.4 AUROC points of each other (0.816–0.830). (2) The 0.8B Qwen3.5 model achieves 0.852 AUROC with the same hyperparameters that cause overfitting at 4B/9B, suggesting that for this binary classification task, model capacity beyond ~2B provides diminishing returns.
>
> A full per-model-size hyperparameter sweep would strengthen this finding and is planned for the camera-ready version.

---

## Issue 7: Healthcare/finance regulatory claims
**Status: FIXED IN PAPER**

> We have removed all regulatory standard references (FDA, Reg BI) from the main text and appendix. The domain deployment sections now report error rates and cost reductions without regulatory framing. We add a caveat: "these are illustrative simulations on benchmark data; deployment in regulated domains would require validation on domain-specific datasets and compliance review."

---

## Issue 8: Multi-seed results
**Status: NEEDS EXPERIMENT (Agent 1)**

> [Will add once seed 456 is complete and seed 123 is rerun for full epochs. Will report mean ± std across 3 seeds.]

---

## Issue 9: Truncation at 800 characters
**Status: NEEDS EXPERIMENT (Agent 1)**

> [Will add truncation length ablation: 400, 800, 1200, 1600 chars. If performance is stable across lengths, this addresses the concern. If not, we adjust the default.]

---

## Issue 10: Comparison to Kapoor et al.
**Status: ARGUMENT ONLY**

> Our work builds directly on the finding of Kapoor et al. (2024) that fine-tuned judges outperform baselines for uncertainty estimation. We position Pinocchio as extending their approach along three axes:
>
> 1. **Cross-architecture transfer:** Kapoor et al. fine-tune judges within the same model family (e.g., LLaMA judging LLaMA). This requires open-weight access to a model in the same family as the target. Pinocchio uses a different architecture (Qwen3-VL) to judge models from different families (GPT-5, Qwen3.5), enabling true black-box deployment.
>
> 2. **Multimodality:** Kapoor et al. evaluate on text-only benchmarks (primarily MMLU variants). Pinocchio handles both text and vision-language tasks in a unified model.
>
> 3. **Benchmark diversity:** Kapoor et al. focus on MMLU variants with a single difficulty level. We evaluate on 20 benchmarks spanning 7 domains and difficulty levels from 14% to 85% accuracy.
>
> A direct reproduction of Kapoor et al.'s method on our benchmarks would require training a same-family judge for each target model (GPT-5 family → fine-tune a GPT-5-mini judge, etc.), which is infeasible for closed-source models—precisely the setting our work targets. We have updated Table 2 and its footnote to clarify this distinction.

---

## Issue 11: "First multimodal" claim
**Status: FIXED IN PAPER**

> We have softened the claim from "providing the first multimodal uncertainty quantifier" to "providing a unified multimodal uncertainty quantifier that handles both modalities in a single model." We acknowledge VL-Uncertainty and Uncertainty-o in the related work section.

---

## Issue 12: Table 2 Kapoor marking
**Status: FIXED IN PAPER**

> We have changed the black-box marking for Kapoor et al. from ✗ to "partial" and added a footnote explaining the distinction: Kapoor et al. do not use the target model's logits directly, but require open-weight access to a model in the same family. Pinocchio uses a separate architecture with no same-family requirement.

---

## Issue 13: AUROC numbers across tables
**Status: FIXED IN PAPER**

> Table 4 (elicitation ablation) now includes a caption note: "on internal validation split (distinct from the test set in Table 1)." The two evaluation sets differ because elicitation strategies were compared during development using a validation split, while main results use the final held-out test set.

---

## Issue 14: ECE and Brier score
**Status: FIXED IN PAPER**

> Table 1 now includes Brier score (0.084) alongside AUROC. We report ECE = 0.023 in the main results text, with per-model Brier scores in the range 0.078–0.089. The reliability diagram shows the calibrator is well-calibrated across the full confidence range, with the largest calibration gaps in the 0.3–0.6 range (where sample counts are lowest).

---

## Issue 15: Exact training count
**Status: FIXED IN PAPER**

> Replaced "approximately 10,000" with the exact count: 10,047 training examples and 4,447 test examples (total: 14,494 from 11,821 unique question-model pairs).

---

## Issue 16: Error analysis / failure cases
**Status: FIXED IN PAPER**

> We have added Appendix E (Error Analysis and Failure Cases) with:
> - Quantification: 112 hard errors (1.4% confident-but-wrong, 1.1% unconfident-but-right)
> - Per-benchmark failure distribution table
> - Systematic pattern analysis: CW failures cluster in visual/spatial tasks; UR failures cluster in math with terse correct answers
