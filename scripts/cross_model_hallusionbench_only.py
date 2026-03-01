#!/usr/bin/env python3
"""Quick test: Just HallusionBench for cross-model transfer."""
import sys
import json
from pathlib import Path
from typing import Optional
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration

PROMPT_TEMPLATE = """Question: {question}
Answer: {response}
Is the answer correct? (i) No (ii) Yes"""

def load_qwen_model():
    model_name = "Qwen/Qwen2.5-VL-72B-Instruct"
    print(f"Loading {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    return model, processor

def load_vlm_judge(checkpoint_path: str = "data/vlm_judge_combined/checkpoint-788"):
    base_model = "Qwen/Qwen3-VL-8B-Instruct"
    print(f"Loading VLM judge from {checkpoint_path}...")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()
    return model, processor

def generate_response(model, processor, image, prompt):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated_ids, skip_special_tokens=True).strip()

def get_p_correct(model, processor, image, question, response):
    prompt = PROMPT_TEMPLATE.format(question=question, response=response)
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True,
                      min_pixels=256*28*28, max_pixels=512*28*28)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    token_i = processor.tokenizer.encode("i", add_special_tokens=False)[-1]
    token_ii = processor.tokenizer.encode("ii", add_special_tokens=False)[-1]
    probs = torch.softmax(logits[[token_i, token_ii]], dim=0)
    return probs[1].item()

def main():
    print("=" * 60)
    print("HALLUSIONBENCH CROSS-MODEL TRANSFER")
    print("=" * 60)
    
    qwen_model, qwen_processor = load_qwen_model()
    judge_model, judge_processor = load_vlm_judge()
    
    print("\nLoading HallusionBench...")
    ds = load_dataset("lmms-lab/HallusionBench", split="image")
    max_samples = 200
    fallback = Image.new("RGB", (336, 336), color="gray")
    
    predictions, labels = [], []
    n_correct = 0
    
    for idx in tqdm(range(min(max_samples, len(ds))), desc="Processing"):
        row = ds[idx]
        image = row.get("image")
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            image = fallback
        question = row.get("question", "")
        gt_answer = str(row.get("gt_answer", "")).strip().lower()
        
        prompt = f"{question}\nAnswer with Yes or No."
        response = generate_response(qwen_model, qwen_processor, image, prompt)
        
        # Check correctness
        resp_lower = response.lower()
        resp_yes = "yes" in resp_lower and "no" not in resp_lower
        resp_no = "no" in resp_lower and "yes" not in resp_lower
        gt_yes = gt_answer in ["yes", "1", "true"]
        
        if resp_yes:
            is_correct = gt_yes
        elif resp_no:
            is_correct = not gt_yes
        else:
            is_correct = False
        
        if is_correct:
            n_correct += 1
        
        p_correct = get_p_correct(judge_model, judge_processor, image, prompt[:500], response[:300])
        predictions.append(p_correct)
        labels.append(float(is_correct))
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    accuracy = n_correct / len(labels)
    if len(np.unique(labels)) > 1:
        auroc = roc_auc_score(labels, predictions)
        auprc = average_precision_score(labels, predictions)
    else:
        auroc, auprc = 0.5, labels.mean()
    
    print(f"\n{'='*60}")
    print("HALLUSIONBENCH RESULTS")
    print(f"{'='*60}")
    print(f"Qwen accuracy: {accuracy:.1%} ({n_correct}/{len(labels)})")
    print(f"Judge AUROC:   {auroc:.3f}")
    print(f"Judge AUPRC:   {auprc:.3f}")
    print(f"P(correct):    {predictions.mean():.3f} ± {predictions.std():.3f}")
    
    # Save
    results = {"benchmark": "hallusionbench", "n_samples": len(labels), 
               "qwen_accuracy": accuracy, "judge_auroc": auroc, "judge_auprc": auprc}
    with open("data/vlm_judge_combined/cross_model_hallusionbench.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/vlm_judge_combined/cross_model_hallusionbench.json")

if __name__ == "__main__":
    main()
