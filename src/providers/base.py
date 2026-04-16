"""Abstract model provider interface."""

from abc import ABC, abstractmethod


class ModelProvider(ABC):
    """Unified interface for text generation backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Currently active model identifier."""

    @abstractmethod
    async def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate text from *prompt* with optional *system* instructions.

        Returns the generated text or an empty string on failure.
        """
