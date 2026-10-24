from .base import BaseLLMClient
from .mock import MockLLMClient

__all__ = ["BaseLLMClient", "GeminiClient", "MockLLMClient", "OpenAIClient"]


def __getattr__(name: str):
    # Provider SDKs are optional extras ([gemini] / [openai]); import them only on use
    # so `pip install -e ".[dev]"` + MockLLMClient works without any provider installed.
    if name == "GeminiClient":
        from .gemini import GeminiClient

        return GeminiClient
    if name == "OpenAIClient":
        from .openai import OpenAIClient

        return OpenAIClient
    raise AttributeError(name)
