"""FastAPI HTTP API interface for the assistant.

Exposes:
    POST /api/chat        — send a message, get assistant reply
    GET  /health          — health check
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from assistant import Assistant
from config import Config

logger = logging.getLogger(__name__)

app = FastAPI(title="MaxKonvert Assistant API")
assistant = Assistant()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message text")
    session_id: str | None = Field(
        None, description="Optional session/conversation identifier"
    )
    user_id: str | None = Field(None, description="Optional user identifier")
    metadata: dict[str, Any] | None = Field(None, description="Optional extra metadata")


class ChatResponse(BaseModel):
    answer: str
    route: str
    best_score: float
    elapsed_ms: int
    model: str
    provider: str
    timestamp: str


@app.on_event("startup")
async def startup():
    errors = Config.validate()
    if errors:
        for err in errors:
            logger.error("Config validation error: %s", err)
        raise RuntimeError("Configuration invalid: " + "; ".join(errors))
    assistant.ensure_index()
    logger.info("API server started; provider=%s model=%s", assistant.provider.name, assistant.provider.model_id)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "maxkonvert-assistant-api",
        "provider": assistant.provider.name,
        "model": assistant.provider.model_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    key = req.session_id or req.user_id or "api:anonymous"
    metadata = {"channel": "http_api", **(req.metadata or {})}
    if req.user_id:
        metadata["user_id"] = req.user_id

    try:
        result = await assistant.process_message(
            user_text=req.message.strip(),
            conversation_key=key,
            channel="http_api",
            metadata=metadata,
        )
    except Exception as exc:
        logger.exception("API processing error")
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}")

    return ChatResponse(
        answer=result["answer"],
        route=result["route"],
        best_score=result["best_score"],
        elapsed_ms=result["elapsed_ms"],
        model=assistant.provider.model_id,
        provider=assistant.provider.name,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
