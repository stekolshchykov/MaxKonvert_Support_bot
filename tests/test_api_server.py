import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from fastapi.testclient import TestClient
from config import Config


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setattr(Config, "QUESTIONS_DIR", str(tmp_path / "questions"))
    monkeypatch.setattr(Config, "QUESTIONS_LOG_FILE", str(tmp_path / "questions" / "questions.ndjson"))
    monkeypatch.setattr(Config, "NEW_QUESTIONS_LOG_FILE", str(tmp_path / "questions" / "new_questions.ndjson"))
    monkeypatch.setattr(Config, "UNANSWERED_LOG_FILE", str(tmp_path / "questions" / "unanswered.ndjson"))
    monkeypatch.setattr(Config, "QUESTIONS_STATE_FILE", str(tmp_path / "questions" / "state.json"))
    monkeypatch.setattr(Config, "INDEX_PATH", str(tmp_path / "index"))
    monkeypatch.setattr(Config, "DOCS_PATH", str(tmp_path / "docs"))
    monkeypatch.setattr(Config, "ACTIONS_CONFIG_PATH", str(tmp_path / "actions.json"))
    Path(tmp_path / "questions").mkdir(parents=True, exist_ok=True)
    Path(tmp_path / "docs").mkdir(parents=True, exist_ok=True)

    import api_server
    # Reset assistant index to avoid building from real docs
    api_server.assistant.index = type("I", (), {"chunks": [], "search": lambda *a, **k: [], "build": lambda *a, **k: None})()
    api_server.assistant.index_signature = (0, 0, 0, 0)
    return TestClient(api_server.app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "maxkonvert-assistant-api"


def test_chat_requires_session_id(client):
    resp = client.post("/api/chat", json={"message": "Привет"})
    assert resp.status_code == 422


def test_chat_creates_session(client):
    resp = client.post(
        "/api/chat",
        json={
            "session_id": "sess_001",
            "message": "Привет",
            "user_id": "user_123",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "sess_001"
    assert "answer" in data
    assert "route" in data


def test_chat_restores_session_history(client):
    # First message
    client.post(
        "/api/chat",
        json={"session_id": "sess_002", "message": "Меня зовут Анна"},
    )
    # Check history endpoint
    resp = client.get("/api/session/sess_002")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "sess_002"
    assert any(t["text"] == "Меня зовут Анна" for t in data["turns"])


def test_clear_session(client):
    client.post("/api/chat", json={"session_id": "sess_003", "message": "Привет"})
    resp = client.delete("/api/session/sess_003")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cleared"

    resp = client.get("/api/session/sess_003")
    assert resp.json()["turns"] == []


def test_swagger_ui_accessible(client):
    resp = client.get("/docs")
    assert resp.status_code == 200
    assert "swagger-ui" in resp.text.lower() or "openapi" in resp.text.lower()
