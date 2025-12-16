from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


JsonDict = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """A backend-agnostic request object.

    For OpenAI Responses API, `messages` will be sent as the `input` array.
    """
    messages: list[JsonDict]
    temperature: float = 0.0
    max_output_tokens: int = 1024

    # Optional fields for tool use / structured output
    tools: Optional[list[JsonDict]] = None
    tool_choice: Optional[Any] = None

    # Uncertainty hooks
    logprobs: bool = False
    top_logprobs: int = 0

    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class ModelResponse:
    text: str
    raw: Any | None = None
    usage: JsonDict | None = None
    logprobs: Any | None = None
    tool_calls: list[JsonDict] | None = None


@dataclass(frozen=True, slots=True)
class Example:
    id: str
    input: Any
    target: Any
    meta: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class Prediction:
    example_id: str
    answer: Any
    confidence: float | None
    raw_text: str
    extra: JsonDict = field(default_factory=dict)
