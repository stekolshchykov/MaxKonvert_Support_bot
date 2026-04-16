"""Ollama (local) model provider."""

import logging

import httpx

from config import Config
from .base import ModelProvider

logger = logging.getLogger(__name__)


class OllamaProvider(ModelProvider):
    def __init__(self, url: str | None = None, model: str | None = None, timeout: float | None = None):
        self.url = (url or Config.OLLAMA_URL).rstrip("/")
        self.model = model or Config.OLLAMA_MODEL
        self.timeout = timeout or Config.OLLAMA_TIMEOUT_SECONDS

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model_id(self) -> str:
        return self.model

    async def generate(self, prompt: str, system: str | None = None) -> str:
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": 1536,
                "num_predict": 180,
            },
        }
        if system:
            payload["system"] = system
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.url}/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("response", "").strip()
                return answer[: Config.MAX_ANSWER_CHARS]
        except Exception:
            logger.exception("Ollama generation error")
            return ""
