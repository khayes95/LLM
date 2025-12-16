from __future__ import annotations

import json
import re
from typing import Any


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def extract_first_json_obj(text: str) -> dict | None:
    """Best-effort: parse whole text as JSON dict, else find first {...} block."""
    if not text:
        return None

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None

    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def extract_choice_letter(text: str, choices: str = "ABCD") -> str | None:
    if not text:
        return None
    # Find standalone A/B/C/D
    m = re.findall(r"\b([A-D])\b", text.upper())
    if m:
        c = m[-1]
        return c if c in choices else None
    return None


def extract_last_number(text: str) -> str | None:
    if not text:
        return None
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text.strip())
    return nums[-1] if nums else None
