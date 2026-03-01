from __future__ import annotations

import base64
import io
import json
import re
from typing import Any

from PIL import Image


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


# ============================================================================
# Image utilities for vision benchmarks
# ============================================================================

def encode_image_to_base64(img: Image.Image, format: str = "PNG", max_size: int = 1024) -> str:
    """Encode a PIL Image to base64 string, optionally resizing."""
    # Resize if too large (saves tokens)
    if max(img.size) > max_size:
        img = img.copy()
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    # Convert to RGB if necessary (for JPEG)
    if img.mode in ("RGBA", "P") and format.upper() == "JPEG":
        img = img.convert("RGB")

    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def pil_to_data_url(img: Image.Image, format: str = "PNG", max_size: int = 1024) -> str:
    """Convert PIL Image to data URL for OpenAI vision API."""
    b64 = encode_image_to_base64(img, format=format, max_size=max_size)
    mime = f"image/{format.lower()}"
    return f"data:{mime};base64,{b64}"


def build_vision_content(text: str, images: list[Image.Image], max_size: int = 1024) -> list[dict]:
    """Build OpenAI vision message content with text and images.

    Returns a content array suitable for the 'content' field of a message.
    Uses 'input_image' type for OpenAI Responses API (gpt-5-mini, o1, etc.)
    and 'image_url' type for Chat Completions API (gpt-4o, etc.).

    The OpenAI Responses API expects:
    [
        {"type": "input_image", "image_url": "data:image/png;base64,..."},
        {"type": "input_text", "text": "..."},
    ]

    The Chat Completions API expects:
    [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        {"type": "text", "text": "..."},
    ]

    We use Responses API format since that's what gpt-5-mini uses.
    """
    content = []

    # Add images first (common practice for vision models)
    for img in images:
        data_url = pil_to_data_url(img, max_size=max_size)
        # Use Responses API format: input_image with direct image_url
        content.append({
            "type": "input_image",
            "image_url": data_url
        })

    # Add text - use input_text for Responses API
    content.append({"type": "input_text", "text": text})

    return content
