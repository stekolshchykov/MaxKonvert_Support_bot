"""Kimi (Moonshot) API provider via OpenAI-compatible or Anthropic-compatible client."""

import logging

import httpx
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
        # Kimi Code tokens (sk-kimi-...) use the Anthropic-compatible endpoint
        # under api.kimi.com/coding/v1, while regular Moonshot API keys use
        # the OpenAI-compatible endpoint (api.moonshot.cn/v1 or api.moonshot.ai/v1).
        self._is_kimi_code = (
            "api.kimi.com/coding" in self.base_url
            or self.api_token.startswith("sk-kimi-")
        )

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
        if self._is_kimi_code:
            return await self._generate_anthropic(prompt, system)
        return await self._generate_openai(prompt, system)

    async def _generate_openai(self, prompt: str, system: str | None = None) -> str:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            client = self._get_client()
            resp = await client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.7,
                max_tokens=1024,
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

    async def _generate_anthropic(self, prompt: str, system: str | None = None) -> str:
        """Use Anthropic-compatible Messages API (required for Kimi Code tokens)."""
        url = f"{self.base_url}/messages"
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.7,
        }
        if system:
            payload["system"] = system
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                content_blocks = data.get("content", [])
                text_parts = [
                    block.get("text", "")
                    for block in content_blocks
                    if block.get("type") == "text"
                ]
                answer = "\n".join(text_parts).strip()
                return answer[: Config.MAX_ANSWER_CHARS]
        except httpx.TimeoutException:
            logger.exception("Kimi Code API timeout")
            return ""
        except httpx.HTTPStatusError as e:
            logger.exception(
                "Kimi Code API HTTP error: %s - %s", e.response.status_code, e.response.text
            )
            return ""
        except Exception:
            logger.exception("Kimi Code generation error")
            return ""
