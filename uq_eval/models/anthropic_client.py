from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import anthropic

from ..types import ModelRequest, ModelResponse
from .base import BaseModelClient


# Claude models that support extended thinking
_THINKING_MODELS = {"claude-opus-4-6", "claude-sonnet-4-6"}

# Map from OpenAI-style roles to Anthropic roles
_ROLE_MAP = {
    "developer": "user",  # Claude doesn't have "developer"; fold into system or user
    "system": "user",     # System messages handled separately
}


def _extract_system_and_messages(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Separate system/developer messages from user/assistant messages.

    Claude expects system as a top-level parameter, not in the messages array.
    """
    system_parts: list[str] = []
    claude_messages: list[dict[str, Any]] = []

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        if role in ("system", "developer"):
            # Collect system content
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                        system_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        system_parts.append(part)
            continue

        # Convert content to Claude format
        claude_content = _convert_content(content)
        claude_messages.append({"role": role, "content": claude_content})

    # Claude requires messages to start with "user" role
    # If first message is assistant, prepend a minimal user message
    if claude_messages and claude_messages[0].get("role") == "assistant":
        claude_messages.insert(0, {"role": "user", "content": "Please continue."})

    # Merge consecutive same-role messages (Claude doesn't allow them)
    merged: list[dict[str, Any]] = []
    for msg in claude_messages:
        if merged and merged[-1]["role"] == msg["role"]:
            # Merge content
            prev = merged[-1]["content"]
            curr = msg["content"]
            if isinstance(prev, str) and isinstance(curr, str):
                merged[-1]["content"] = prev + "\n\n" + curr
            elif isinstance(prev, list) and isinstance(curr, list):
                merged[-1]["content"] = prev + curr
            elif isinstance(prev, str) and isinstance(curr, list):
                merged[-1]["content"] = [{"type": "text", "text": prev}] + curr
            elif isinstance(prev, list) and isinstance(curr, str):
                merged[-1]["content"] = prev + [{"type": "text", "text": curr}]
        else:
            merged.append(msg)

    system = "\n\n".join(system_parts) if system_parts else None
    return system, merged


def _make_image_block(url_or_data_url: str) -> dict[str, Any] | None:
    """Convert a URL or data URL to an Anthropic image content block.

    Handles:
    - data:image/...;base64,... → base64 image block
    - https://... or http://... → URL-based image block
    - Anything else → logs a warning and returns None
    """
    if not url_or_data_url:
        return None

    # Try parsing as a data URL first
    media_type, b64_data = _parse_data_url(url_or_data_url)
    if b64_data:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64_data,
            },
        }

    # Check if it's an HTTP(S) URL — Anthropic supports URL-based images
    if url_or_data_url.startswith(("http://", "https://")):
        return {
            "type": "image",
            "source": {
                "type": "url",
                "url": url_or_data_url,
            },
        }

    # Neither a data URL nor an HTTP URL — image will be dropped
    print(
        f"[anthropic] WARNING: dropping image — unrecognized format "
        f"(starts with {url_or_data_url[:40]!r}...)"
    )
    return None


def _convert_content(content: Any) -> Any:
    """Convert OpenAI content format to Anthropic content format.

    OpenAI Responses API uses:
      {"type": "input_image", "image_url": "data:image/png;base64,..."}
      {"type": "input_text", "text": "..."}

    OpenAI Chat Completions uses:
      {"type": "image_url", "image_url": {"url": "data:..."}}
      {"type": "text", "text": "..."}

    Anthropic uses:
      {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
      {"type": "text", "text": "..."}
    """
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return str(content)

    claude_parts: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            claude_parts.append({"type": "text", "text": str(item)})
            continue

        item_type = item.get("type", "")

        # OpenAI Responses API: input_text
        if item_type == "input_text":
            claude_parts.append({"type": "text", "text": item.get("text", "")})

        # OpenAI Responses API: input_image with data URL
        elif item_type == "input_image":
            data_url = item.get("image_url", "")
            img_block = _make_image_block(data_url)
            if img_block:
                claude_parts.append(img_block)

        # OpenAI Chat Completions: image_url
        elif item_type == "image_url":
            url_obj = item.get("image_url", {})
            url = url_obj.get("url", "") if isinstance(url_obj, dict) else str(url_obj)
            img_block = _make_image_block(url)
            if img_block:
                claude_parts.append(img_block)

        # Standard text
        elif item_type == "text":
            claude_parts.append({"type": "text", "text": item.get("text", "")})

        else:
            # Unknown type, try to extract text
            text = item.get("text", "") or str(item)
            claude_parts.append({"type": "text", "text": text})

    return claude_parts


def _parse_data_url(data_url: str) -> tuple[str, str]:
    """Parse a data URL into (media_type, base64_data).

    Input: "data:image/png;base64,iVBOR..."
    Output: ("image/png", "iVBOR...")

    Returns ("", "") if the string is not a valid data URL.
    """
    match = re.match(r"data:([^;]+);base64,(.+)", data_url, re.DOTALL)
    if match:
        return match.group(1), match.group(2)
    return "", ""


@dataclass(slots=True)
class AnthropicClient(BaseModelClient):
    """Anthropic backend via the Messages API."""

    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # accepted for CLI compatibility, ignored
    timeout_s: float = 600.0
    max_retries: int = 5
    retry_sleep_s: float = 2.0

    def __post_init__(self) -> None:
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "Anthropic API key required. Pass api_key= or set ANTHROPIC_API_KEY env var."
            )
        client_kwargs: dict[str, Any] = {
            "api_key": key,
            "timeout": float(self.timeout_s),
            "max_retries": 0,  # We handle retries ourselves in generate()
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self._client = anthropic.Anthropic(**client_kwargs)

    def generate(self, req: ModelRequest) -> ModelResponse:
        system, messages = _extract_system_and_messages(req.messages)

        # Build request kwargs
        create_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": req.max_output_tokens or 4096,
        }

        # Add system message if present
        if system:
            create_kwargs["system"] = system

        # Add temperature (Claude supports it on all models)
        if req.temperature is not None:
            create_kwargs["temperature"] = float(req.temperature)

        last_err: str | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.messages.create(**create_kwargs)

                # Extract text from response
                text_parts = []
                for block in resp.content:
                    if block.type == "text":
                        text_parts.append(block.text)
                text = "\n".join(text_parts)

                # Build usage dict
                usage = None
                if resp.usage:
                    usage = {
                        "input_tokens": resp.usage.input_tokens,
                        "output_tokens": resp.usage.output_tokens,
                        "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
                    }

                # Serialize raw response
                raw = resp.model_dump() if hasattr(resp, "model_dump") else None

                return ModelResponse(text=text, raw=raw, usage=usage)

            except anthropic.RateLimitError as e:
                last_err = f"RateLimitError: {e}"
                wait = self.retry_sleep_s * (2 ** attempt)
                print(f"[anthropic] rate limited, waiting {wait:.0f}s (attempt {attempt+1}/{self.max_retries})")
                time.sleep(wait)

            except anthropic.APIStatusError as e:
                last_err = f"APIStatusError({e.status_code}): {e.message}"
                if e.status_code >= 500:
                    # Server error, retry
                    wait = self.retry_sleep_s * (2 ** attempt)
                    print(f"[anthropic] server error {e.status_code}, retrying in {wait:.0f}s")
                    time.sleep(wait)
                else:
                    # Client error (400, 401, etc.), don't retry
                    break

            except anthropic.APIConnectionError as e:
                last_err = f"APIConnectionError: {e}"
                wait = self.retry_sleep_s * (2 ** attempt)
                print(f"[anthropic] connection error, retrying in {wait:.0f}s")
                time.sleep(wait)

            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                break

        # All retries exhausted — return error response (don't crash the run)
        return ModelResponse(
            text=f"API_FAILED after {self.max_retries} tries: {last_err}",
            raw={"error": last_err},
            usage=None,
        )
