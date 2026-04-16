import asyncio
import pytest
from unittest.mock import AsyncMock

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from assistant import Assistant, UserProfileExtractor, load_profile, save_profile
from config import Config


class FakeProvider:
    def __init__(self):
        self.name = "fake"
        self.model_id = "fake-model"
        self.generate = AsyncMock(return_value="")


@pytest.fixture
def assistant(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "USER_PROFILES_DIR", str(tmp_path / "profiles"))
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
    fake = FakeProvider()
    ast = Assistant(provider=fake, actions_config_path=str(tmp_path / "actions.json"))
    ast.provider = fake
    return ast


# ---------------------------------------------------------------------------
# UserProfileExtractor
# ---------------------------------------------------------------------------

def test_extract_name():
    assert UserProfileExtractor.extract("Меня зовут Виталий")["имя"] == "Виталий"
    assert UserProfileExtractor.extract("я — Анна")["имя"] == "Анна"
    assert UserProfileExtractor.extract("имя: Сергей")["имя"] == "Сергей"


def test_extract_geo():
    # extractor returns the matched keyword capitalized
    assert UserProfileExtractor.extract("Работаю с трафиком из России")["geo"] == "России"


def test_extract_format():
    assert UserProfileExtractor.extract("Интересует wap")["формат"] == "WAP"


# ---------------------------------------------------------------------------
# Profile persistence
# ---------------------------------------------------------------------------

def test_profile_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "USER_PROFILES_DIR", str(tmp_path / "profiles"))
    save_profile("chat:user", {"имя": "Виталий"})
    assert load_profile("chat:user") == {"имя": "Виталий"}


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

def test_session_memory_persisted_across_instances(tmp_path, monkeypatch):
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

    fake1 = FakeProvider()
    ast1 = Assistant(provider=fake1, actions_config_path=str(tmp_path / "actions.json"))
    ast1.add_dialog_turn("session_abc", "user", "Привет")
    ast1.add_dialog_turn("session_abc", "assistant", "Привет!")

    # Simulate new process / new Assistant instance
    fake2 = FakeProvider()
    ast2 = Assistant(provider=fake2, actions_config_path=str(tmp_path / "actions.json"))
    hist = ast2.get_dialog_history_text("session_abc")
    assert "User: Привет" in hist
    assert "Assistant: Привет!" in hist

    ast2.clear_memory("session_abc")
    hist_after = ast2.get_dialog_history_text("session_abc")
    assert hist_after == "(история пуста)"


# ---------------------------------------------------------------------------
# Fallback builders
# ---------------------------------------------------------------------------

def test_sales_fallback_never_dumps_raw_snippets(assistant):
    fb = assistant.build_sales_manager_fallback("foo", [(0.1, {"file": "doc.md", "text": "foo bar"})], low_match=True)
    assert "[" not in fb
    assert "doc.md" not in fb
    assert "По вашему запросу вижу" not in fb


def test_docs_grounded_fallback_is_human(assistant):
    fb = assistant.build_docs_grounded_fallback("foo", [])
    assert "В базе знаний пока нет" not in fb
    assert "точной цифры" in fb


# ---------------------------------------------------------------------------
# Routing: social / memory queries should force model
# ---------------------------------------------------------------------------

def test_should_use_model_for_social_query(assistant):
    key = "1:1"
    assert assistant.should_use_model("Привет", key, 0.05) is True
    assert assistant.should_use_model("Как моё имя?", key, 0.05) is True
    assert assistant.should_use_model("Как меня зовут?", key, 0.05) is True


def test_should_use_model_when_history_exists_and_short(assistant):
    key = "1:1"
    assistant.add_dialog_turn(key, "user", "Привет")
    assistant.add_dialog_turn(key, "assistant", "Привет!")
    assert assistant.should_use_model("А сколько стоит?", key, 0.10) is True


def test_should_skip_model_when_no_history_and_very_low_score(assistant):
    key = "2:2"
    assert assistant.should_use_model("asdfghjkl", key, 0.05) is False


# ---------------------------------------------------------------------------
# Integration: prompt includes profile
# ---------------------------------------------------------------------------

def test_prompt_includes_profile(assistant):
    prompt = assistant.build_prompt(
        user_text="Как моё имя?",
        history_text="User: Меня зовут Виталий\nAssistant: Привет, Виталий!",
        docs_text="",
        profile_text="- имя: Виталий",
    )
    assert "Что мы знаем о пользователе" in prompt
    assert "- имя: Виталий" in prompt


# ---------------------------------------------------------------------------
# Direct definition filters markdown headings
# ---------------------------------------------------------------------------

def test_direct_definition_skips_markdown_headings(assistant):
    results = [
        (0.5, {"text": "# WAP\n\nWAP — это мобильный протокол для доступа к интернет-контенту."}),
    ]
    answer = assistant.build_direct_definition_answer("Что такое WAP", results)
    assert answer
    assert "#" not in answer
    assert "WAP — это" in answer


# ---------------------------------------------------------------------------
# Async integration: social query with history uses model route
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_social_query_uses_model_when_history_exists(assistant):
    key = "1:1"
    assistant.provider.generate.return_value = "Ты Виталий, мы уже здоровались 😊"
    assistant.add_dialog_turn(key, "user", "Меня зовут Виталий")
    assistant.add_dialog_turn(key, "assistant", "Привет, Виталий!")
    # mock empty index so similarity is zero
    class FakeIndex:
        chunks = []
        def search(self, *a, **k):
            return []
        def build(self, *a, **k):
            pass
    assistant.index = FakeIndex()
    result = await assistant.process_message("Как моё имя?", key, channel="test")
    assert result["route"] == "model"
    assert "Виталий" in result["answer"]
