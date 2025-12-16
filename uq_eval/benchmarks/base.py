from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from ..types import Example, ModelRequest, ModelResponse, Prediction


class BaseBenchmark(ABC):
    """Each benchmark implements: load examples, build requests, parse predictions, and scoring."""

    name: str
    mode: str = "text"  # "text", "text+tool", "longtext", "text+image", ...

    @abstractmethod
    def iter_examples(self, split: str) -> Iterable[Example]:
        raise NotImplementedError

    @abstractmethod
    def build_request(self, ex: Example) -> ModelRequest:
        raise NotImplementedError

    @abstractmethod
    def parse_prediction(self, ex: Example, resp: ModelResponse) -> Prediction:
        raise NotImplementedError

    @abstractmethod
    def score(self, ex: Example, pred: Prediction) -> dict:
        raise NotImplementedError
