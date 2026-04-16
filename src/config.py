"""Centralized configuration loaded from environment."""

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
APP_ENV_FILE = os.getenv("APP_ENV_FILE", "").strip()
if APP_ENV_FILE:
    load_dotenv(APP_ENV_FILE, override=True)


class Config:
    """Immutable-like config container."""

    # Paths
    APP_ROOT = Path(os.getenv("APP_ROOT", str(ROOT_DIR))).resolve()
    DOCS_PATH = os.getenv("DOCS_PATH", str(ROOT_DIR / "docs")).strip()
    INDEX_PATH = os.getenv("INDEX_PATH", str(ROOT_DIR / "data" / "index")).strip()
    QUESTIONS_DIR = os.getenv("QUESTIONS_DIR", str(ROOT_DIR / "questions")).strip()
    LOG_PATH = os.getenv("LOG_PATH", str(Path(QUESTIONS_DIR) / "logs")).strip()
    ACTIONS_CONFIG_PATH = os.getenv("ACTIONS_CONFIG_PATH", str(ROOT_DIR / "kb" / "http_actions.json")).strip()

    # Telegram
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
    TELEGRAM_BOT_URL = os.getenv("TELEGRAM_BOT_URL", "").strip()

    # Model provider selection
    MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "local").strip().lower()

    # Ollama (local/current)
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").strip()
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b").strip()
    OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "5.0").strip())

    # Kimi (Moonshot)
    KIMI_API_TOKEN = os.getenv("KIMI_API_TOKEN", "").strip()
    KIMI_BASE_URL = os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1").strip()
    KIMI_MODEL = os.getenv("KIMI_MODEL", "moonshot-v1-8k").strip()
    KIMI_TIMEOUT_SECONDS = float(os.getenv("KIMI_TIMEOUT_SECONDS", "30.0").strip())

    # Embedding / retrieval
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ).strip()
    TOP_K = int(os.getenv("TOP_K", "5").strip())
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.30").strip())
    LOW_CONFIDENCE_MODEL_SCORE = float(os.getenv("LOW_CONFIDENCE_MODEL_SCORE", "0.40").strip())
    CONTEXT_TURNS = int(os.getenv("CONTEXT_TURNS", "6").strip())
    RETRIEVAL_HISTORY_TURNS = int(os.getenv("RETRIEVAL_HISTORY_TURNS", "2").strip())
    MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "1800").strip())
    MAX_DOC_SNIPPET_CHARS = int(os.getenv("MAX_DOC_SNIPPET_CHARS", "450").strip())
    MAX_DOC_FRAGMENTS = int(os.getenv("MAX_DOC_FRAGMENTS", "3").strip())
    MAX_ANSWER_CHARS = int(os.getenv("MAX_ANSWER_CHARS", "1600").strip())

    # HTTP API server
    API_HOST = os.getenv("API_HOST", "0.0.0.0").strip()
    API_PORT = int(os.getenv("API_PORT", "8082").strip())
    BOT_HEALTH_HOST = os.getenv("BOT_HEALTH_HOST", "0.0.0.0").strip()
    BOT_HEALTH_PORT = int(os.getenv("BOT_HEALTH_PORT", "8081").strip())

    # Bot persona
    BOT_ROLE = os.getenv(
        "BOT_ROLE",
        (
            "Ты — опытный sales-менеджер MaxKonvert в Telegram. "
            "Ты ведёшь диалог как живой человек и мягко доводишь пользователя до следующего шага."
        ),
    ).strip()
    BOT_EXTRA_RULES = os.getenv(
        "BOT_EXTRA_RULES",
        (
            "Отвечай на языке пользователя. "
            "Пиши естественно, без канцелярита. "
            "Не используй формулировки вида «нет в документации», «я не знаю». "
            "Если данных мало, дай полезный ориентир и задай один уточняющий вопрос. "
            "Не выдумывай конкретные цифры и факты."
        ),
    ).strip()
    SALES_DEFAULT_CTA = os.getenv(
        "SALES_DEFAULT_CTA",
        (
            "Чтобы дать максимально точный и выгодный вариант, уточните, пожалуйста: GEO, "
            "примерный объём трафика в сутки и какой формат вам ближе (MT, WAP, pseudo, premium SMS)."
        ),
    ).strip()

    # Logging files
    QUESTIONS_LOG_FILE = os.getenv(
        "QUESTIONS_LOG_FILE", str(Path(QUESTIONS_DIR) / "questions.ndjson")
    ).strip()
    NEW_QUESTIONS_LOG_FILE = os.getenv(
        "NEW_QUESTIONS_LOG_FILE", str(Path(QUESTIONS_DIR) / "new_questions.ndjson")
    ).strip()
    UNANSWERED_LOG_FILE = os.getenv(
        "UNANSWERED_LOG_FILE", str(Path(QUESTIONS_DIR) / "unanswered_questions.ndjson")
    ).strip()
    QUESTIONS_STATE_FILE = os.getenv(
        "QUESTIONS_STATE_FILE", str(Path(QUESTIONS_DIR) / "questions_state.json")
    ).strip()

    # Reindex / watcher
    REINDEX_LOCK_WAIT_SECONDS = int(os.getenv("REINDEX_LOCK_WAIT_SECONDS", "180").strip())
    REINDEX_INTERVAL_SEC = int(os.getenv("REINDEX_INTERVAL_SEC", "300").strip())
    DEBOUNCE_SEC = int(os.getenv("DEBOUNCE_SEC", "3").strip())
    PYTHON_BIN = os.getenv("PYTHON_BIN", "python3").strip()
    REINDEX_SCRIPT = os.getenv("REINDEX_SCRIPT", str(ROOT_DIR / "scripts" / "reindex.py")).strip()

    @classmethod
    def validate(cls) -> list[str]:
        """Return list of validation errors (empty if ok)."""
        errors: list[str] = []
        if cls.MODEL_PROVIDER not in ("local", "kimi"):
            errors.append(f"Invalid MODEL_PROVIDER='{cls.MODEL_PROVIDER}'. Use 'local' or 'kimi'.")
        if cls.MODEL_PROVIDER == "kimi" and not cls.KIMI_API_TOKEN:
            errors.append("KIMI_API_TOKEN is required when MODEL_PROVIDER=kimi.")
        return errors
