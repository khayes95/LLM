# TASK: Multimodal Uncertainty Quantification for Molmo-72B

**Status:** Active  
**Last Updated:** 2024-12-24  
**Planning Agent:** Claude (planning session)  
**Execution:** Coding agent  

---

## Quick Reference (Read First Every Session)

```
MODEL:      allenai/Molmo-72B-0924 (bf16, NO quantization)
GPUS:       8x A100-80GB available
APPROACH:   Probe-based (frozen model → extract features → train MLP)
BENCHMARKS: MMMU-Pro, VSR, HallusionBench (MC/binary only)
GOAL:       Train classifier to predict correctness from hidden states
METRIC:     AUROC > 0.75 indicates success
```

---

## 1. Project Overview

We're extending text-based uncertainty quantification to vision-language models. The hypothesis: a VLM's internal representations contain signals that predict whether its answer is correct.

**Pipeline:**
```
[Image + Question] → Molmo-72B → [Hidden States + Probs] → MLP → P(correct)
```

**This is a feasibility study.** Get results first, optimize later.

---

## 2. Critical Implementation Details

### 2.1 Molmo API (NON-STANDARD - READ CAREFULLY)

Molmo does NOT use standard HuggingFace API. Use this pattern:

```python
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

# Loading
processor = AutoProcessor.from_pretrained(
    "allenai/Molmo-72B-0924",
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "allenai/Molmo-72B-0924",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Processing inputs (NOTE: processor.process, not processor())
inputs = processor.process(
    images=[image],  # Must be a list
    text=prompt,
)
# Add batch dimension and move to device
inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

# Generation
output = model.generate_from_batch(
    inputs,
    GenerationConfig(max_new_tokens=64, stop_strings=["<|endoftext|>"]),
    tokenizer=processor.tokenizer,
)
response = processor.tokenizer.decode(output[0], skip_special_tokens=True)
```

### 2.2 Extracting Hidden States

For UQ features, we need hidden states during the forward pass:

```python
def extract_features(model, inputs):
    """Extract UQ features from model internals."""
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Last layer, last token position
    last_hidden = outputs.hidden_states[-1][:, -1, :]  # [batch, 8192]
    
    # Probability features from logits
    logits = outputs.logits[:, -1, :]  # [batch, vocab_size]
    probs = torch.softmax(logits, dim=-1)
    top_prob = probs.max(dim=-1).values  # [batch]
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # [batch]
    
    return {
        "hidden_state": last_hidden.cpu(),
        "top_prob": top_prob.cpu(),
        "entropy": entropy.cpu(),
    }
```

### 2.3 Image Handling

Molmo requires RGB images:

```python
from PIL import Image

image = Image.open(path)
if image.mode != "RGB":
    image = image.convert("RGB")
```

---

## 3. Benchmarks

### 3.1 Priority Order

| Priority | Benchmark | Type | Samples | Expected Acc | Scoring |
|----------|-----------|------|---------|--------------|---------|
| 1 | VSR | True/False | ~10K | ~55% | Exact match |
| 2 | HallusionBench | Yes/No | ~1.1K | ~35% | Exact match |
| 3 | MMMU-Pro | MC (A/B/C/D) | ~1.7K | ~28% | Exact match |

Start with VSR - it's simplest and gives good class balance.

### 3.2 Scoring Logic

```python
def score_response(response: str, ground_truth: str, task_type: str) -> bool:
    """Score model response against ground truth."""
    response = response.strip().lower()
    ground_truth = ground_truth.strip().lower()
    
    if task_type == "true_false":  # VSR
        # Extract true/false from response
        if "true" in response and "false" not in response:
            pred = "true"
        elif "false" in response:
            pred = "false"
        else:
            pred = response[:10]  # Fallback
        return pred == ground_truth
    
    elif task_type == "yes_no":  # HallusionBench
        if "yes" in response and "no" not in response:
            pred = "yes"
        elif "no" in response:
            pred = "no"
        else:
            pred = response[:10]
        return pred == ground_truth
    
    elif task_type == "multiple_choice":  # MMMU-Pro
        # Extract letter from response
        for letter in ["a", "b", "c", "d", "e"]:
            if letter in response[:20]:
                return letter == ground_truth.lower()
        return False
    
    return False
```

---

## 4. Data Pipeline

### 4.1 Output Format

Save features incrementally to avoid memory issues:

```python
# Per-sample format
sample = {
    "benchmark": "vsr",
    "question_id": "val_001",
    "prompt": "Is the cat to the left of the dog?",
    "response": "True, the cat is positioned to the left.",
    "ground_truth": "true",
    "is_correct": True,  # <- Label for UQ training
    "features": {
        "hidden_state": torch.tensor([...]),  # [8192]
        "top_prob": 0.73,
        "entropy": 2.41,
    }
}

# Save incrementally
torch.save(sample, f"data/features/{benchmark}_{question_id}.pt")
```

### 4.2 Batching Strategy

Due to memory constraints, use batch_size=1 for feature extraction:

```python
for sample in tqdm(dataset):
    features = extract_features(model, sample)
    save_features(features, sample["id"])
    torch.cuda.empty_cache()  # Prevent memory buildup
```

---

## 5. UQ Classifier

### 5.1 Architecture

```python
class UQClassifier(nn.Module):
    """Predicts P(correct) from VLM hidden states."""
    
    def __init__(self, hidden_dim=8192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 2, 1024),  # +2 for prob and entropy
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
    
    def forward(self, hidden_state, top_prob, entropy):
        x = torch.cat([
            hidden_state,
            top_prob.unsqueeze(-1),
            entropy.unsqueeze(-1)
        ], dim=-1)
        return torch.sigmoid(self.net(x))
```

### 5.2 Training

```python
def train_uq_classifier(train_loader, val_loader, epochs=20):
    model = UQClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    
    best_auroc = 0
    for epoch in range(epochs):
        # Training
        model.train()
        for batch in train_loader:
            pred = model(batch["hidden_state"], batch["top_prob"], batch["entropy"])
            loss = criterion(pred.squeeze(), batch["is_correct"].float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Validation
        auroc = evaluate_auroc(model, val_loader)
        if auroc > best_auroc:
            best_auroc = auroc
            torch.save(model.state_dict(), "uq_classifier_best.pt")
        
        print(f"Epoch {epoch}: AUROC = {auroc:.4f}")
    
    return best_auroc
```

---

## 6. Evaluation Metrics

```python
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_uq(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch["hidden_state"], batch["top_prob"], batch["entropy"])
            all_preds.extend(pred.squeeze().tolist())
            all_labels.extend(batch["is_correct"].tolist())
    
    results = {
        "auroc": roc_auc_score(all_labels, all_preds),
        "auprc": average_precision_score(all_labels, all_preds),
        "n_samples": len(all_labels),
        "n_correct": sum(all_labels),
        "n_incorrect": len(all_labels) - sum(all_labels),
    }
    return results
```

**Success criteria:**
- AUROC > 0.75: Strong signal, method works
- AUROC 0.60-0.75: Weak signal, may need more features
- AUROC ~0.50: No signal, fundamental rethink needed

---

## 7. Experiment Plan

### Phase 1: Smoke Test (1-2 hours)
- [ ] Load Molmo-72B (bf16, device_map="auto")
- [ ] Run on 10 VSR samples, verify outputs look correct
- [ ] Extract hidden states, verify shapes
- [ ] Save features to disk, verify loadable

### Phase 2: VSR Full Run (4-6 hours)
- [ ] Run inference on full VSR dataset (~10K samples)
- [ ] Extract and save features for each sample
- [ ] Calculate overall accuracy (expect ~55%)

### Phase 3: Train UQ Classifier (1 hour)
- [ ] Load saved features
- [ ] 80/20 train/test split
- [ ] Train MLP classifier
- [ ] Evaluate AUROC on test set

### Phase 4: Expand to Other Benchmarks (optional)
- [ ] Repeat Phase 2-3 for HallusionBench
- [ ] Repeat Phase 2-3 for MMMU-Pro
- [ ] Test cross-benchmark generalization

---

## 8. File Structure

```
project/
├── TASK_MULTIMODAL_UQ.md      # This file (task spec)
├── research_log.md             # Progress tracking
├── src/
│   ├── molmo_client.py         # Model loading + inference
│   ├── feature_extraction.py   # Hidden state extraction
│   ├── benchmark_loaders.py    # Dataset loading
│   ├── scoring.py              # Response scoring
│   ├── uq_classifier.py        # MLP classifier
│   └── evaluate.py             # Metrics
├── scripts/
│   ├── run_inference.py        # Main inference script
│   ├── train_uq.py             # Train classifier
│   └── eval_uq.py              # Evaluate classifier
├── data/
│   ├── features/               # Saved features (.pt files)
│   └── results/                # Evaluation results
└── slurm/
    └── run_inference.slurm     # SLURM job script
```

---

## 9. Common Pitfalls

| Problem | Solution |
|---------|----------|
| OOM during inference | Use batch_size=1, call torch.cuda.empty_cache() |
| Wrong Molmo API | Use processor.process() and generate_from_batch() |
| Image format errors | Always convert to RGB |
| Malformed model outputs | Add fallback logic in scoring |
| Memory leak over long runs | Save features to disk incrementally |
| Non-deterministic results | Set torch.manual_seed(), use greedy decoding |

---

## 10. Generation Settings

Use greedy decoding for reproducibility:

```python
gen_config = GenerationConfig(
    max_new_tokens=64,
    do_sample=False,  # Greedy
    temperature=1.0,
    stop_strings=["<|endoftext|>"],
)
```

---

## 11. Checkpoints & Resumability

For long runs, save progress:

```python
def run_with_checkpointing(dataset, checkpoint_path="checkpoint.json"):
    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            completed = set(json.load(f)["completed_ids"])
    else:
        completed = set()
    
    for sample in dataset:
        if sample["id"] in completed:
            continue
        
        # Process sample...
        features = extract_features(model, sample)
        save_features(features, sample["id"])
        
        # Update checkpoint
        completed.add(sample["id"])
        with open(checkpoint_path, "w") as f:
            json.dump({"completed_ids": list(completed)}, f)
```

---

## 12. Questions for Planning Agent

If you hit blockers, document in research_log.md and flag for review:

- [ ] What if AUROC is below 0.6?
- [ ] Should we try different layers (not just last)?
- [ ] Should we try mean pooling over all tokens?
- [ ] What if a benchmark has <30% accuracy?

---

## Changelog

- 2024-12-24: Initial spec created by planning agent
