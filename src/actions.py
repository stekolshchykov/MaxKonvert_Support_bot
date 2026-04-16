"""Structured outbound HTTP action layer.

Actions are defined in a JSON config file (default: kb/http_actions.json).
The assistant can reference them by name; this module validates and executes them.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HttpAction:
    name: str
    description: str
    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    default_body: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ActionRegistry:
    """Loads and stores HTTP action definitions."""

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path
        self.actions: dict[str, HttpAction] = {}
        if config_path:
            self.load(config_path)

    def load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            logger.info("Action config not found at %s; no actions loaded.", path)
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            actions = data.get("actions", []) if isinstance(data, dict) else []
            for raw in actions:
                action = HttpAction(
                    name=raw["name"],
                    description=raw.get("description", ""),
                    method=raw.get("method", "GET").upper(),
                    url=raw["url"],
                    headers=raw.get("headers", {}),
                    default_body=raw.get("default_body"),
                )
                self.actions[action.name] = action
            logger.info("Loaded %s HTTP actions from %s", len(self.actions), path)
        except Exception:
            logger.exception("Failed to load action config from %s", path)

    def get(self, name: str) -> HttpAction | None:
        return self.actions.get(name)

    def list_descriptions(self) -> list[str]:
        return [f"- {a.name}: {a.description}" for a in self.actions.values()]


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class ActionExecutor:
    """Safe executor for HTTP actions with predictable result formatting."""

    def __init__(self, registry: ActionRegistry, timeout: float = 30.0):
        self.registry = registry
        self.timeout = timeout

    async def execute(
        self,
        name: str,
        *,
        query_params: dict[str, Any] | None = None,
        body: dict[str, Any] | str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        action = self.registry.get(name)
        if action is None:
            return {
                "success": False,
                "error": f"Action '{name}' is not defined.",
                "status_code": None,
                "body": None,
            }

        if action.method not in {"GET", "POST", "PUT", "DELETE"}:
            return {
                "success": False,
                "error": f"Method {action.method} is not supported.",
                "status_code": None,
                "body": None,
            }

        url = action.url
        headers = dict(action.headers)
        if extra_headers:
            headers.update(extra_headers)

        request_body = body if body is not None else action.default_body

        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                kwargs: dict[str, Any] = {"headers": headers}
                if query_params:
                    kwargs["params"] = query_params
                if request_body is not None and action.method in ("POST", "PUT"):
                    if isinstance(request_body, dict):
                        kwargs["json"] = request_body
                    else:
                        kwargs["content"] = str(request_body).encode("utf-8")

                resp = await client.request(action.method, url, **kwargs)
                parsed_body = _try_parse_json(resp.text)
                result = {
                    "success": 200 <= resp.status_code < 300,
                    "status_code": resp.status_code,
                    "body": parsed_body,
                    "error": None,
                }
                logger.info(
                    "Action %s %s -> %s (success=%s)",
                    action.method,
                    url,
                    resp.status_code,
                    result["success"],
                )
                return result
        except httpx.TimeoutException:
            logger.exception("Action timeout: %s %s", action.method, url)
            return {
                "success": False,
                "error": "Request timed out.",
                "status_code": None,
                "body": None,
            }
        except Exception as exc:
            logger.exception("Action execution error: %s %s", action.method, url)
            return {
                "success": False,
                "error": str(exc),
                "status_code": None,
                "body": None,
            }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _try_parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return text


# Regex to detect action requests inside model output.
_ACTION_RE = re.compile(
    r"ACTION\s*[:：]\s*([A-Z]+)\s*\|\s*(.+?)\s*(?:\||\n|$)",
    re.IGNORECASE,
)
_TOOL_RE = re.compile(
    r"TOOL\s*[:：]\s*([^\n]+)",
    re.IGNORECASE,
)


def parse_action_requests(text: str) -> list[dict[str, Any]]:
    """Parse explicit action/tool requests from assistant raw output.

    Supports two notation styles:
    1. ACTION:METHOD|url|headers|body
    2. TOOL:{"name":"...","params":{...},"body":{...}}
    """
    requests: list[dict[str, Any]] = []

    # Style 2 (structured JSON) — preferred
    for match in _TOOL_RE.finditer(text):
        try:
            payload = json.loads(match.group(1))
            requests.append({
                "type": "tool",
                "name": payload.get("name"),
                "params": payload.get("params", {}),
                "body": payload.get("body"),
            })
        except Exception:
            continue

    # Style 1 (legacy inline)
    for match in _ACTION_RE.finditer(text):
        method = match.group(1).upper()
        rest = match.group(2).strip()
        parts = [p.strip() for p in rest.split("|")]
        url = parts[0] if parts else ""
        headers = {}
        body = None
        if len(parts) >= 2:
            try:
                headers = json.loads(parts[1])
            except Exception:
                pass
        if len(parts) >= 3:
            body = parts[2]
        requests.append({
            "type": "raw_http",
            "method": method,
            "url": url,
            "headers": headers,
            "body": body,
        })

    return requests
