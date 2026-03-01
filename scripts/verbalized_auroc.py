#!/usr/bin/env python3
"""Compute AUROC for verbalized confidence from existing predictions."""
import os
import json
from sklearn.metrics import roc_auc_score, brier_score_loss

confs = []
correct = []
per_bench = {}

for d in sorted(os.listdir("runs")):
    if not d.startswith("qwen35_397b_") or "backup" in d:
        continue
    bench = d.replace("qwen35_397b_", "")
    p = f"runs/{d}/predictions.jsonl"
    if not os.path.exists(p):
        continue
    b_confs = []
    b_correct = []
    with open(p) as f:
        for line in f:
            s = json.loads(line)
            c = s.get("prediction", {}).get("confidence")
            sc = s.get("score", {}).get("correct")
            if c is not None and sc is not None:
                # Ensure binary labels
                label = 1 if sc else 0
                confs.append(float(c))
                correct.append(label)
                b_confs.append(float(c))
                b_correct.append(label)
    if len(set(b_correct)) > 1 and len(b_confs) >= 10:
        auroc = roc_auc_score(b_correct, b_confs)
        per_bench[bench] = {"auroc": auroc, "n": len(b_confs), "mean_conf": sum(b_confs)/len(b_confs)}

print(f"Total samples with confidence: {len(confs)}")
print(f"Correct: {sum(correct)} ({sum(correct)/len(correct)*100:.1f}%)")
print(f"Mean confidence: {sum(confs)/len(confs):.3f}")
print(f"Unique confidence values: {len(set(confs))}")
print()

if len(set(correct)) > 1:
    auroc = roc_auc_score(correct, confs)
    brier = brier_score_loss(correct, confs)
    print(f"Verbalized AUROC: {auroc:.4f}")
    print(f"Verbalized Brier: {brier:.4f}")
else:
    auroc = None
    print("Cannot compute AUROC (single class)")

print()
print("Per-benchmark:")
for bench in sorted(per_bench, key=lambda b: per_bench[b]["auroc"], reverse=True):
    r = per_bench[bench]
    print(f"  {bench:<20} AUROC={r['auroc']:.3f} (n={r['n']}, mean_conf={r['mean_conf']:.2f})")

# Compare to calibrators
print()
print("COMPARISON:")
for label, path in [
    ("v3 (mini)", "data/cross_model/text_v3_on_qwen35_full.json"),
    ("Combined", "data/cross_model/text_combined_on_qwen35_full.json"),
    ("GPT5.2-cal", "data/cross_model/text_gpt52cal_on_qwen35_full.json"),
    ("VLM Judge", "data/cross_model/vlm_judge_vsr_fixed_on_qwen35_full.json"),
]:
    if os.path.exists(path):
        with open(path) as f:
            cal = json.load(f)
        cal_auroc = cal["auroc"]
        delta = cal_auroc - auroc if auroc else 0
        print(f"  {label:<15} AUROC={cal_auroc:.4f}  (delta vs verbalized: {delta:+.4f})")

# Save results
output = {
    "model": "Qwen3.5-397B-A17B-FP8",
    "method": "verbalized",
    "n_samples": len(confs),
    "overall_auroc": auroc,
    "overall_brier": brier if auroc else None,
    "mean_confidence": sum(confs)/len(confs),
    "n_unique_values": len(set(confs)),
    "per_benchmark": per_bench,
}
os.makedirs("data/baselines", exist_ok=True)
with open("data/baselines/verbalized_qwen35.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: data/baselines/verbalized_qwen35.json")
