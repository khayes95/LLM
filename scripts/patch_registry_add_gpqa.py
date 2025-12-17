from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "uq_eval" / "registry.py"

if not REG.exists():
    print(f"[error] registry not found: {REG}", file=sys.stderr)
    sys.exit(1)

text = REG.read_text(encoding="utf-8")

import_line = "from .benchmarks.gpqa_diamond import GPQADiamondBenchmark\n"
if import_line not in text:
    # insert after last 'from .benchmarks....' import if exists
    bench_imports = list(re.finditer(r"^from \.benchmarks\.[^\n]+\n", text, flags=re.M))
    if bench_imports:
        last = bench_imports[-1]
        insert_at = last.end()
        text = text[:insert_at] + import_line + text[insert_at:]
    else:
        # fallback: after future import
        m = re.search(r"from __future__ import annotations\n\n", text)
        insert_at = m.end() if m else 0
        text = text[:insert_at] + import_line + text[insert_at:]

if '"gpqa_diamond"' not in text:
    # find the benchmark registry dict
    m = re.search(r"_BENCH_REGISTRY[^=]*=\s*\{\n", text)
    if not m:
        print("[error] cannot find _BENCH_REGISTRY = { ... } block in registry.py", file=sys.stderr)
        sys.exit(1)

    start = m.end()
    end = text.find("}", start)
    if end == -1:
        print("[error] cannot find closing '}' for _BENCH_REGISTRY block", file=sys.stderr)
        sys.exit(1)

    insertion = '    "gpqa_diamond": GPQADiamondBenchmark,\n'
    text = text[:end] + insertion + text[end:]

REG.write_text(text, encoding="utf-8")
print("[ok] patched:", REG)
