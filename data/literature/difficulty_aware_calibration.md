# Literature Review: Training Calibrated Models Across Difficulty Levels

**Date:** 2026-03-11
**Context:** Pinocchio gives ~0.7 for everything on trivial questions because it was trained only on hard benchmarks (40-60% accuracy). It cannot output near-0 or near-1 probabilities. This review identifies approaches to fix this.

---

## 1. The Core Problem: Difficulty-Homogeneous Training Data

### Entropy Sentinel (2026)
- **Paper:** "Entropy Sentinel: Continuous LLM Accuracy Monitoring from Decoding Entropy Traces in STEM"
- **Authors:** (arXiv:2601.09001)
- **Venue:** arXiv preprint, Jan 2026
- **Key finding directly relevant to us:** "Training sets that span difficulty (mixing easy and hard tasks) generalize substantially better than difficulty-homogeneous sets." Groups with intermediate weighted accuracy (0.4-0.6) achieve optimal performance, but difficulty-homogeneous groups at either extreme degrade generalization. Easy-only training underrepresents high-entropy failure patterns; hard-only training underrepresents low-entropy success patterns.
- **Training approach:** Lightweight classifier on entropy profile features (from top-k logprobs). Trains on just 2 benchmarks, generalizes to 8 others --- but only when training benchmarks span the difficulty spectrum.
- **Actionable for us:** Our training data is almost entirely in the 0.4-0.6 accuracy range. We need to add easy benchmarks (>80% accuracy) and very hard benchmarks (<20% accuracy) to the training mix so the model sees the full spectrum of correct/incorrect confidence signals.

---

## 2. Training Loss Functions for Difficulty-Aware Calibration

### Inverse Focal Loss / AdaFocal (NeurIPS 2022, CVPR 2025)
- **Paper:** "AdaFocal: Calibration-aware Adaptive Focal Loss" (Ghosh et al.)
- **Venue:** NeurIPS 2022
- **Key insight:** Standard cross-entropy and focal loss compress confidence scores, making easy and hard samples indistinguishable. Inverse focal loss (gamma < 0) does the opposite: it up-weights easy samples to push their confidence higher, helping separate easy from hard.
- **AdaFocal** dynamically adjusts gamma per sample group based on validation calibration feedback. When the model is underconfident on easy samples, it uses inverse focal loss; when overconfident on hard samples, it uses standard focal loss.
- **Actionable for us:** Even if we add easy data, standard BCE may still compress our score range. Consider replacing BCE with AdaFocal or a calibration-aware loss that explicitly rewards high confidence on correct-easy and low confidence on incorrect-hard.

### Uncertainty Weighted Gradients (CVPR 2025)
- **Paper:** "Uncertainty Weighted Gradients for Model Calibration" (Lin et al.)
- **Venue:** CVPR 2025
- **Approach:** Reweights gradient contributions based on per-sample uncertainty, effectively giving more training signal to samples where the model is miscalibrated.
- **Actionable for us:** Could be applied during fine-tuning to emphasize samples where our model's P(correct) is most wrong.

### Optimizing Calibration by Gaining Aware of Prediction Correctness (2024)
- **Paper:** "Optimizing Calibration by Gaining Aware of Prediction Correctness" (arXiv:2404.13016)
- **Venue:** arXiv 2024
- **Key insight:** Standard CE loss always pushes confidence up on the ground-truth class, even for misclassified samples. This paper proposes a loss that DECREASES confidence on wrongly-predicted samples and INCREASES it on correctly-predicted ones. Uses data augmentation (rotation, color jitter) to generate correctness-varying versions of the same input.
- **Critical warning:** "Most confidence calibration methods are useless or harmful for failure prediction" --- popular calibration methods often worsen the separation between correct and incorrect samples.
- **Actionable for us:** Our task IS failure prediction (predicting P(correct)). Standard calibration methods (temperature scaling, Platt scaling) may actually hurt our AUROC even if they improve ECE. We should optimize for AUROC/separation first, calibration second.

---

## 3. RL-Based Calibration Training

### Rewarding Doubt (ICLR 2026)
- **Paper:** "Rewarding Doubt: A Reinforcement Learning Approach to Calibrated Confidence Expression of Large Language Models"
- **Authors:** Stangel, Bani-Harouni et al.
- **Venue:** ICLR 2026 (poster)
- **Approach:** Fine-tunes LLMs with RL using logarithmic scoring rule (proper scoring rule) as reward. Models confidence as a betting game --- high confidence on correct answers yields high reward; high confidence on wrong answers yields large penalty.
- **Key result:** After RL fine-tuning, "the model's confidence scores span a wider range, including lower values, indicating a more nuanced expression of uncertainty." The model learns to express doubt on hard questions.
- **Theoretical guarantee:** Optimal policy under log scoring rule = perfectly calibrated.
- **Generalization:** Improved calibration generalizes to new tasks without retraining.
- **Actionable for us:** After SFT on mixed-difficulty data, a second-stage RL pass with Brier score or log scoring rule reward could expand our score range. This is the most principled fix for the narrow-range problem.

### Calibration-Aware RL (CalRL, 2026)
- **Paper:** "Balancing Classification and Calibration Performance in Decision-Making LLMs via Calibration Aware Reinforcement Learning"
- **Venue:** arXiv:2601.13284, Jan 2026
- **Key finding:** Standard RLVR produces extremely overconfident models. SFT yields substantially better calibration, even under distribution shift. CalRL adjusts decision-token probabilities to improve calibration while preserving RLVR performance gains.
- **Actionable for us:** If we ever move to RL-based training, we must include calibration-aware rewards, not just correctness rewards.

### CoCA: Confidence Before Answering (2026)
- **Paper:** "Confidence Before Answering: A Paradigm Shift for Efficient LLM Uncertainty Estimation"
- **Authors:** Changcheng Li et al.
- **Venue:** arXiv:2603.05881, March 2026
- **Approach:** GRPO RL framework that jointly optimizes confidence calibration and answer accuracy via segmented credit assignment. Model outputs confidence BEFORE answering.
- **Actionable for us:** Different paradigm (confidence-first), but the segmented reward idea (separate rewards for confidence accuracy vs answer accuracy) could inspire a two-headed training objective.

---

## 4. Post-Hoc Calibration Under Distribution Shift

### Temperature Scaling Limitations
- **Paper:** "Post-hoc Uncertainty Calibration for Domain Drift Scenarios" (CVPR 2021)
- **Key finding:** Temperature scaling deteriorates under distribution shift. A single temperature learned on hard benchmarks will NOT work for easy benchmarks.
- **Fix:** "Domain-robust calibration" via perturbation-based strategies --- calibrate on validation sets augmented with controlled input noise.
- **Actionable for us:** If we use temperature scaling, we need difficulty-stratified calibration: fit separate temperatures for different difficulty bins. Or use input-conditional temperature (Local Temperature Scaling, ICCV 2021).

### DACA: Disagreement-Aware Confidence Alignment (NeurIPS 2025)
- **Paper:** "Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator" (Luo & Wang)
- **Venue:** NeurIPS 2025
- **Approach:** Uses disagreement between pre-trained (base) and post-trained (instruction-tuned) model to selectively calibrate. Only uses agreement examples for temperature optimization.
- **Improved ECE by up to 15.08%** on common benchmarks.
- **Actionable for us:** We could compare our Pinocchio scores against a base Qwen3-VL model's raw logit confidence, and use agreement/disagreement to adjust calibration.

---

## 5. Alignment-Based Confidence Training

### Know When You're Wrong (2026)
- **Paper:** "Know When You're Wrong: Aligning Confidence with Correctness for LLM Error Detection"
- **Venue:** arXiv:2603.06604, March 2026
- **Key finding:** SFT yields well-calibrated confidence via MLE. RL methods (PPO, GRPO, DPO) induce overconfidence via reward exploitation. SFT improved AUROC from 0.806 to 0.879 and reduced ECE from 0.163 to 0.034 on Qwen3-4B.
- **Approach:** Uses normalized confidence from anchor token probabilities (Yes/No logits for correctness assessment --- very similar to our approach).
- **Actionable for us:** Validates that our SFT approach is sound. The key is the TRAINING DATA must cover the full difficulty range, not the training method itself.

---

## 6. Conformal and Distribution-Free Methods

### Conditional Conformal Prediction
- **Paper:** "Conformal Prediction for NLP: A Survey" (TACL)
- **Key insight:** Standard conformal prediction gives marginal coverage guarantees but NOT conditional (per-difficulty-level) guarantees. Mondrian conformal predictors can give guarantees within data subgroups (e.g., difficulty bins).
- **Actionable for us:** After fixing the training data, we could apply Mondrian conformal prediction to give per-difficulty-level coverage guarantees. This is a principled way to handle the fact that our model may be better calibrated at some difficulty levels than others.

---

## 7. Reward Model Training (Related Literature)

### Beyond Correctness: Confidence-Aware Reward Modeling (EMNLP 2025)
- **Paper:** "Beyond Correctness: Confidence-Aware Reward Modeling"
- **Venue:** EMNLP 2025
- **Approach:** Reward models that incorporate confidence awareness, not just binary correctness.
- **Relevance:** Our Pinocchio model IS a reward/correctness model. The same data curation principles apply.

### Edge-of-Competence Data Selection
- **Finding from reward model literature:** "Design RL data around the model's edge of competence --- filtering for tasks where the model fails at pass@1 but succeeds at pass@k. This avoids redundancy on easy tasks while preventing reward sparsity on impossible ones."
- **Actionable for us:** Our current training data is already at the edge of competence (40-60% accuracy). We need to ALSO include data away from the edge (easy + very hard) so the model learns the full score range.

---

## Summary: Actionable Fixes for Pinocchio's Narrow Score Range

### Tier 1: Most Impactful and Easiest to Implement

1. **Add easy benchmark data to training** (from Entropy Sentinel findings)
   - Include benchmarks where source models get >85% correct (e.g., BoolQ, HellaSwag, WinoGrande, ARC-Easy, MMLU easy subsets)
   - Include benchmarks where models get <20% correct (already have HLE at ~25%)
   - Target: training data should span 10-95% accuracy range, not just 40-60%
   - Expected effect: Model sees many "obviously correct" and "obviously wrong" examples, learns to output near-1 and near-0

2. **Stratified sampling during training**
   - Don't just dump in easy data --- balance it. Aim for ~1/3 easy (>70% acc), ~1/3 medium (30-70%), ~1/3 hard (<30%)
   - Prevents the model from being overwhelmed by easy examples

3. **Difficulty-conditional evaluation**
   - Report AUROC and ECE separately for easy/medium/hard bins
   - Current aggregate AUROC hides the per-difficulty failure

### Tier 2: Moderate Effort, High Impact

4. **Replace BCE with calibration-aware loss** (from AdaFocal)
   - Use focal loss with adaptive gamma, or add a calibration penalty term
   - Specifically: penalize the model when P(correct) is far from 0 or 1 on easy samples

5. **RL fine-tuning with proper scoring rule** (from Rewarding Doubt)
   - After SFT, run RL with log scoring rule or Brier score reward
   - Directly incentivizes the model to output extreme probabilities when warranted
   - Most principled fix but requires RL infrastructure

6. **Difficulty-stratified temperature scaling** (post-hoc)
   - Fit separate temperature parameters for different difficulty bins
   - Or use input-conditional temperature (predict temperature from input features)
   - Quick fix that doesn't require retraining

### Tier 3: Lower Priority / Supplementary

7. **Synthetic easy examples**
   - Generate trivially correct/incorrect QA pairs and add to training
   - E.g., "What is 2+2?" with correct answer "4" (label=1) and wrong answer "7" (label=0)
   - Risk: model might learn superficial features from synthetic data

8. **Mondrian conformal prediction**
   - Apply per-difficulty-bin conformal calibration for deployment guarantees
   - Complementary to training fixes, not a replacement

9. **Multi-head / mixture-of-experts calibration**
   - Train separate calibration heads for different difficulty levels
   - More complex architecture, harder to maintain

---

## Key Takeaway

The literature strongly supports one conclusion: **the root cause is training data composition, not model architecture or loss function**. The Entropy Sentinel paper directly shows that difficulty-homogeneous training sets degrade generalization. The fix is straightforward: add easy and very hard benchmark data to training so the model sees the full difficulty spectrum. Loss function changes (AdaFocal, RL with proper scoring rules) can further help but are secondary to fixing the data.

---

## References

- Entropy Sentinel (arXiv:2601.09001) - https://arxiv.org/abs/2601.09001
- AdaFocal (arXiv:2211.11838) - https://arxiv.org/abs/2211.11838
- Rewarding Doubt (arXiv:2503.02623, ICLR 2026) - https://arxiv.org/abs/2503.02623
- CalRL (arXiv:2601.13284) - https://arxiv.org/abs/2601.13284
- CoCA (arXiv:2603.05881) - https://arxiv.org/abs/2603.05881
- Know When You're Wrong (arXiv:2603.06604) - https://arxiv.org/abs/2603.06604
- DACA / Unsupervised Confidence Calibrator (arXiv:2505.16690, NeurIPS 2025) - https://arxiv.org/abs/2505.16690
- Optimizing Calibration via Prediction Correctness (arXiv:2404.13016) - https://arxiv.org/abs/2404.13016
- Post-hoc Calibration for Domain Drift (CVPR 2021) - https://openaccess.thecvf.com/content/CVPR2021/papers/Tomani_Post-Hoc_Uncertainty_Calibration_for_Domain_Drift_Scenarios_CVPR_2021_paper.pdf
- Uncertainty Weighted Gradients (CVPR 2025) - https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_Uncertainty_Weighted_Gradients_for_Model_Calibration_CVPR_2025_paper.pdf
- Beyond Correctness: Confidence-Aware Reward Modeling (EMNLP 2025) - https://aclanthology.org/2025.emnlp-main.1385.pdf
- Rethinking Confidence Calibration for Failure Prediction (arXiv:2303.02970) - https://arxiv.org/abs/2303.02970
- On Calibration of Modern Neural Networks (arXiv:1706.04599) - https://arxiv.org/abs/1706.04599
- UQ and Confidence Calibration in LLMs: A Survey (arXiv:2503.15850) - https://arxiv.org/abs/2503.15850
- Conformal Prediction for NLP: A Survey (TACL) - https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00715/125278
