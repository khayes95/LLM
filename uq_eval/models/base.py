from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import ModelRequest, ModelResponse


class BaseModelClient(ABC):
    """Interface for closed-source / open-source model backends."""

    @abstractmethod
    def generate(self, req: ModelRequest) -> ModelResponse:
        raise NotImplementedError
