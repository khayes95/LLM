#!/bin/bash
# Monitor size ablation SLURM 7911 and write results to research log when done
cd /scratch/khayes/LLM

while true; do
    # Check if job is still running
    if ! squeue -u khayes -h | grep -q 7911; then
        echo "$(date): Job 7911 finished or disappeared from queue"
        break
    fi
    
    # Check 4B status
    if [ -f uq_models/unified_size_ablation/4b/results.json ]; then
        echo "$(date): 4B done"
        if [ -f uq_models/unified_size_ablation/8b/results.json ]; then
            echo "$(date): 8B done — all complete"
            break
        else
            pct=$(grep -o '[0-9]\+%' logs/size_ablation_8b.log 2>/dev/null | tail -1)
            echo "$(date): 8B training at ${pct:-not started}"
        fi
    else
        pct=$(grep -o '[0-9]\+%' logs/size_ablation_4b.log 2>/dev/null | tail -1)
        echo "$(date): 4B training at ${pct:-loading...}"
    fi
    
    sleep 300
done

# Collect results
echo ""
echo "=== Collecting results ==="

RESULTS=""
for size in 2b 4b 8b; do
    rfile="uq_models/unified_size_ablation/${size}/results.json"
    if [ -f "$rfile" ]; then
        line=$(/scratch/khayes/.conda/envs/uq_eval/bin/python3 -c "
import json
d = json.load(open('$rfile'))
auroc = d.get('auroc', 0)
vlm = d.get('vlm_auroc', 0)
txt = d.get('text_auroc', 0)
brier = d.get('brier', 0)
t = d.get('training_time_min', 0)
print(f'| Qwen3-VL-{\"$size\".upper()} | {auroc:.4f} | {vlm:.4f} | {txt:.4f} | {brier:.4f} | {t:.0f} min |')
")
        RESULTS="${RESULTS}${line}\n"
        echo "$line"
    else
        echo "${size}: FAILED (no results.json)"
        RESULTS="${RESULTS}| Qwen3-VL-${size^^} | FAILED | — | — | — | — |\n"
    fi
done

# Check exit code
exit_code=$(sacct -j 7911 --format=ExitCode -n 2>/dev/null | head -1 | tr -d ' ')
echo "Job exit code: $exit_code"

# Write to research log
echo ""
echo "=== Writing to RESEARCH_LOG.md ==="

# Build the log entry
ENTRY="
### $(date '+%Y-%m-%d %H:%M') — Unified VLM size ablation complete
Why: Test if smaller Qwen3-VL models (2B, 4B) retain UQ performance for laptop deployment
Script: \`scripts/unified_size_ablation.py\` | SLURM 7911 | Output: \`uq_models/unified_size_ablation/\`

| Model | AUROC | VLM | Text | Brier | Time |
|-------|-------|-----|------|-------|------|
$(echo -e "$RESULTS")
Reference: Qwen3-VL-8B standalone = 0.831 AUROC (from best_unified)
"

# Insert after the first "## 2026-02-" line (most recent date section)
/scratch/khayes/.conda/envs/uq_eval/bin/python3 << 'PYEOF'
import re

entry = """
### TIMESTAMP — Unified VLM size ablation complete
Why: Test if smaller Qwen3-VL models (2B, 4B) retain UQ performance for laptop deployment
Script: `scripts/unified_size_ablation.py` | SLURM 7911 | Output: `uq_models/unified_size_ablation/`

RESULTS_TABLE
Reference: Qwen3-VL-8B standalone = 0.831 AUROC (from best_unified)
"""

import json, datetime

# Build results table
table_lines = ["| Model | AUROC | VLM | Text | Brier | Time |", "|-------|-------|-----|------|-------|------|"]
for size in ["2b", "4b", "8b"]:
    rfile = f"uq_models/unified_size_ablation/{size}/results.json"
    try:
        d = json.load(open(rfile))
        auroc = d.get("auroc", 0)
        vlm = d.get("vlm_auroc", 0)
        txt = d.get("text_auroc", 0)
        brier = d.get("brier", 0)
        t = d.get("training_time_min", 0)
        table_lines.append(f"| Qwen3-VL-{size.upper()} | {auroc:.4f} | {vlm:.4f} | {txt:.4f} | {brier:.4f} | {t:.0f} min |")
    except:
        table_lines.append(f"| Qwen3-VL-{size.upper()} | FAILED | — | — | — | — |")

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
entry = entry.replace("TIMESTAMP", now).replace("RESULTS_TABLE", "\n".join(table_lines))

with open("RESEARCH_LOG.md") as f:
    content = f.read()

# Insert after "## 2026-02-27" if it exists, else after "## 2026-02-26"
import re
match = re.search(r"(## 2026-02-2[67]\n)", content)
if match:
    pos = match.end()
    content = content[:pos] + entry + "\n" + content[pos:]
else:
    # Insert after first "---" after the header
    pos = content.find("---", content.find("## Current"))
    if pos > 0:
        pos = content.find("\n", pos) + 1
        content = content[:pos] + "\n## 2026-02-27\n" + entry + "\n" + content[pos:]

with open("RESEARCH_LOG.md", "w") as f:
    f.write(content)

print("Research log updated successfully")
PYEOF

echo "=== Monitor script done at $(date) ==="
