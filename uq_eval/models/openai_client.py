from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from openai import OpenAI

from ..types import ModelRequest, ModelResponse
from .base import BaseModelClient


_ROLE_MAP = {
    # If your prompts use "system", map it to the Responses API "developer" role.
    "system": "developer",
}


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        mm = dict(m)
        role = mm.get("role")
        if isinstance(role, str):
            mm["role"] = _ROLE_MAP.get(role, role)
        out.append(mm)
    return out


# Models that don't support temperature parameter
_NO_TEMPERATURE_MODELS = {"gpt-5-mini", "gpt-5", "gpt-5.2", "o1", "o1-mini", "o1-preview", "o3", "o3-mini"}

# Reasoning models need higher max_output_tokens because reasoning tokens count against the limit
# Default CLI uses 256 which gets consumed by reasoning, leaving no visible output
_REASONING_MODELS = {"gpt-5-mini", "gpt-5", "gpt-5.2", "o1", "o1-mini", "o1-preview", "o3", "o3-mini"}
_REASONING_MODEL_MIN_OUTPUT_TOKENS = 16384  # High limit to ensure reasoning + visible output; analyze usage after to optimize


@dataclass(slots=True)
class OpenAIResponsesClient(BaseModelClient):
    """OpenAI backend via the Responses API."""

    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout_s: float | None = 600.0  # default is 10 min in SDK; customize if needed

    def __post_init__(self) -> None:
        # Official SDK reads OPENAI_API_KEY from env if api_key=None.
        kwargs: dict[str, Any] = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout_s is not None:
            kwargs["timeout"] = float(self.timeout_s)
        self._client = OpenAI(**kwargs)

    def generate(self, req: ModelRequest) -> ModelResponse:
        include: list[str] = []
        if req.logprobs:
            # Include logprobs in output message blocks (if supported by the model)
            include.append("message.output_text.logprobs")

        client = self._client
        if self.timeout_s is not None and hasattr(self._client, "with_options"):
            client = self._client.with_options(timeout=float(self.timeout_s))

        # Ensure reasoning models have sufficient output tokens
        max_output_tokens = req.max_output_tokens
        if any(self.model_name.startswith(m) for m in _REASONING_MODELS):
            max_output_tokens = max(max_output_tokens or 256, _REASONING_MODEL_MIN_OUTPUT_TOKENS)

        # Build request kwargs, excluding temperature for models that don't support it
        create_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "input": _normalize_messages(req.messages),
            "max_output_tokens": max_output_tokens,
            "tools": req.tools,
            "tool_choice": req.tool_choice,
            "include": include or None,
            "metadata": req.metadata or None,
        }

        # Only include temperature if model supports it
        if not any(self.model_name.startswith(m) for m in _NO_TEMPERATURE_MODELS):
            create_kwargs["temperature"] = req.temperature

        # Add reasoning effort for reasoning models (low/medium/high)
        if req.reasoning_effort and any(self.model_name.startswith(m) for m in _REASONING_MODELS):
            create_kwargs["reasoning"] = {"effort": req.reasoning_effort}

        resp = client.responses.create(**create_kwargs)

        text = getattr(resp, "output_text", None)
        if text is None:
            text = str(resp)

        # Keep JSON-serializable raw payload when possible
        if hasattr(resp, "model_dump"):
            raw = resp.model_dump()
        elif hasattr(resp, "to_dict"):
            raw = resp.to_dict()
        else:
            raw = resp

        usage = getattr(resp, "usage", None)
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()

        return ModelResponse(text=text, raw=raw, usage=usage)
