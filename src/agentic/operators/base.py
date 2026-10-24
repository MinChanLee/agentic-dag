"""Base operator interface."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class BaseOperator(ABC):
    """Abstract base for operators.

    Each operator class carries a ``VERSION`` string used as part of the
    cache key so that upgrading the operator logic automatically invalidates
    stale caches.
    """

    VERSION: str = "1.0"

    @abstractmethod
    async def execute(
        self,
        rendered_prompt: str,
        output_schema: type[BaseModel] | None = None,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute the operator and return a result."""
        ...
