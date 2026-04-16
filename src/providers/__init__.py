"""Model provider factory."""

from __future__ import annotations

from .base import ModelProvider
from .ollama import OllamaProvider
from .kimi import KimiProvider

__all__ = ["ModelProvider", "OllamaProvider", "KimiProvider", "get_provider"]


def get_provider(provider_name: str | None = None) -> ModelProvider:
    from config import Config

    name = (provider_name or Config.MODEL_PROVIDER).lower()
    if name == "local":
        return OllamaProvider()
    if name == "kimi":
        return KimiProvider()
    raise ValueError(f"Unknown model provider: {name}")
