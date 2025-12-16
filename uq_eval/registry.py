from __future__ import annotations

from typing import Any

from .benchmarks.base import BaseBenchmark
from .models.base import BaseModelClient

from .benchmarks.dummy_qa import DummyQABenchmark
from .benchmarks.sanity_mcq import SanityMCQBenchmark
from .benchmarks.sanity_math import SanityMathBenchmark
from .benchmarks.sanity_unanswerable import SanityUnanswerableBenchmark
from .benchmarks.sanity_long_context import SanityLongContextBenchmark
from .benchmarks.jsonl_qa import JsonlQABenchmark

from .models.openai_client import OpenAIResponsesClient
from .models.chat_completions_http_client import ChatCompletionsHTTPClient


_BENCH_REGISTRY: dict[str, type[BaseBenchmark]] = {
    "dummy_qa": DummyQABenchmark,
    "sanity_mcq": SanityMCQBenchmark,
    "sanity_math": SanityMathBenchmark,
    "sanity_unanswerable": SanityUnanswerableBenchmark,
    "sanity_long_context": SanityLongContextBenchmark,
    "jsonl_qa": JsonlQABenchmark,
}

_MODEL_REGISTRY: dict[str, type[BaseModelClient]] = {
    "openai": OpenAIResponsesClient,
    "chat_http": ChatCompletionsHTTPClient,
}


def load_benchmark(name: str, **kwargs: Any) -> BaseBenchmark:
    name = name.strip()
    if name not in _BENCH_REGISTRY:
        raise KeyError(f"Unknown benchmark: {name}. Available: {sorted(_BENCH_REGISTRY)}")
    return _BENCH_REGISTRY[name](**kwargs)


def load_model_client(backend: str, **kwargs: Any) -> BaseModelClient:
    backend = backend.strip()
    if backend not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model backend: {backend}. Available: {sorted(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[backend](**kwargs)
