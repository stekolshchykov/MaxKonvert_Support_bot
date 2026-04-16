"""Kimi (Moonshot) API provider via OpenAI-compatible client."""

import logging

from openai import AsyncOpenAI, APIError, APITimeoutError

from config import Config
from .base import ModelProvider

logger = logging.getLogger(__name__)


class KimiProvider(ModelProvider):
    def __init__(
        self,
        api_token: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ):
        self.api_token = api_token or Config.KIMI_API_TOKEN
        self.base_url = (base_url or Config.KIMI_BASE_URL).rstrip("/")
        self.model = model or Config.KIMI_MODEL
        self.timeout = timeout or Config.KIMI_TIMEOUT_SECONDS
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_token,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    @property
    def name(self) -> str:
        return "kimi"

    @property
    def model_id(self) -> str:
        return self.model

    async def generate(self, prompt: str, system: str | None = None) -> str:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            client = self._get_client()
            resp = await client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.3,
                max_tokens=512,
            )
            content = resp.choices[0].message.content or ""
            return content.strip()[: Config.MAX_ANSWER_CHARS]
        except APITimeoutError:
            logger.exception("Kimi API timeout")
            return ""
        except APIError as e:
            logger.exception("Kimi API error: %s", e)
            return ""
        except Exception:
            logger.exception("Kimi generation error")
            return ""
