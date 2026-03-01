from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import requests

from ..types import ModelRequest, ModelResponse
from .base import BaseModelClient


_DEFAULT_BASE_URL = "https://api.key77qiqi.cn/v1"  # from your provided script: /v1/chat/completions

_ROLE_MAP = {
    # Responses API uses "developer", ChatCompletions uses "system"
    "developer": "system",
}


def _convert_content_item(item: dict[str, Any]) -> dict[str, Any]:
    """Convert Responses API content items to Chat Completions format.

    Responses API: {"type": "input_image", "image_url": "data:..."}
    Chat Completions: {"type": "image_url", "image_url": {"url": "data:..."}}
    """
    t = item.get("type", "")
    if t == "input_image":
        return {"type": "image_url", "image_url": {"url": item["image_url"]}}
    if t == "input_text":
        return {"type": "text", "text": item["text"]}
    return item


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        mm = dict(m)
        role = mm.get("role")
        if isinstance(role, str):
            mm["role"] = _ROLE_MAP.get(role, role)
        # Convert multimodal content from Responses API format to Chat Completions
        content = mm.get("content")
        if isinstance(content, list):
            mm["content"] = [_convert_content_item(c) if isinstance(c, dict) else c for c in content]
        out.append(mm)
    return out


def _extract_thinking(text: str) -> tuple[str, str | None]:
    """Extract and strip <think>...</think> blocks from model output.

    Returns (stripped_text, thinking_content).
    """
    if "</think>" in text:
        parts = text.split("</think>", 1)
        thinking = parts[0]
        # Remove opening <think> tag if present
        if "<think>" in thinking:
            thinking = thinking.split("<think>", 1)[-1]
        stripped = parts[-1].strip()
        return stripped, thinking.strip()
    return text, None


@dataclass(slots=True)
class ChatCompletionsHTTPClient(BaseModelClient):
    """A minimal HTTP client for OpenAI-compatible /v1/chat/completions endpoints.

    It follows the same pattern as the provided script:
    - POST /v1/chat/completions
    - Authorization: Bearer <key>
    - Parse choices[0].message.content
    """

    model_name: str
    api_key: str | None = None
    base_url: str | None = None  # can be '.../v1' or full '.../v1/chat/completions'
    timeout_s: float = 30.0
    max_try: int = 5
    sleep_s: float = 1.0
    disable_thinking: bool = False  # Disable thinking for Qwen3-style models (faster)
    strip_thinking: bool = True  # Strip <think>...</think> from response text

    def __post_init__(self) -> None:
        if not self.api_key:
            # Prefer UQ_API_KEY; fall back to OPENAI_API_KEY for convenience.
            self.api_key = os.environ.get("UQ_API_KEY") or os.environ.get("OPENAI_API_KEY")

        if not self.base_url:
            self.base_url = os.environ.get("UQ_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or _DEFAULT_BASE_URL

    def _resolve_url(self) -> str:
        assert self.base_url is not None
        url = self.base_url.rstrip("/")
        # Allow passing either base '/v1' or full '/v1/chat/completions'
        if url.endswith("/chat/completions"):
            return url
        return url + "/chat/completions"

    def generate(self, req: ModelRequest) -> ModelResponse:
        url = self._resolve_url()

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": self.model_name,
            "temperature": float(req.temperature),
            "messages": _normalize_messages(req.messages),
            "max_tokens": int(req.max_output_tokens),
        }

        # Optional tool calling (if your endpoint supports it)
        if req.tools is not None:
            payload["tools"] = req.tools
        if req.tool_choice is not None:
            payload["tool_choice"] = req.tool_choice

        # Optional logprobs (not all proxies support)
        if req.logprobs:
            payload["logprobs"] = True
            if req.top_logprobs:
                payload["top_logprobs"] = int(req.top_logprobs)

        # Disable thinking for Qwen3-style models (much faster inference)
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        last_err: str | None = None
        last_resp_text: str | None = None

        for i in range(int(self.max_try)):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=float(self.timeout_s))
                last_resp_text = r.text
                r.raise_for_status()
                result = r.json()

                # OpenAI-style response parsing
                text = ""
                try:
                    text = result["choices"][0]["message"]["content"]
                except Exception:
                    # fallback
                    text = str(result)

                # Extract and strip thinking tokens (Qwen3-style <think>...</think>)
                thinking = None
                if self.strip_thinking and text:
                    text, thinking = _extract_thinking(text)

                usage = result.get("usage", None) if isinstance(result, dict) else None
                return ModelResponse(text=text or "", raw=result, usage=usage, thinking=thinking)

            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                # simple retry/backoff
                if i < int(self.max_try) - 1:
                    time.sleep(float(self.sleep_s))

        # Do not crash the whole run; return a marked response.
        return ModelResponse(
            text=f"API_FAILED after {self.max_try} tries: {last_err}",
            raw={"error": last_err, "url": url, "last_response_text": last_resp_text},
            usage=None,
        )
