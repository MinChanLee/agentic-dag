"""Operator registry – maps operator name strings to classes."""

from __future__ import annotations

from ..llm.base import BaseLLMClient
from ..operators.base import BaseOperator
from ..operators.generic_prompt import GenericPromptOperator

OPERATOR_CLASSES: dict[str, type[BaseOperator]] = {
    "GenericPromptOperator": GenericPromptOperator,
}


def get_operator(name: str, llm_client: BaseLLMClient) -> BaseOperator:
    """Instantiate an operator by name."""
    cls = OPERATOR_CLASSES.get(name)
    if cls is None:
        available = ", ".join(sorted(OPERATOR_CLASSES.keys()))
        raise ValueError(f"Unknown operator '{name}'. Available: [{available}]")
    return cls(llm_client=llm_client)  # type: ignore[call-arg]


def register_operator(name: str, cls: type[BaseOperator]) -> None:
    """Register a custom operator for use in node specs."""
    OPERATOR_CLASSES[name] = cls
