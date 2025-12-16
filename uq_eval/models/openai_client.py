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

        resp = client.responses.create(
            model=self.model_name,
            input=_normalize_messages(req.messages),
            temperature=req.temperature,
            max_output_tokens=req.max_output_tokens,
            tools=req.tools,
            tool_choice=req.tool_choice,
            include=include or None,
            metadata=req.metadata or None,
        )

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
