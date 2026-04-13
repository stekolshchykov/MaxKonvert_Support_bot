import json
import logging
import os
import re
import sys
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from indexer import DocIndex

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
APP_ENV_FILE = os.getenv("APP_ENV_FILE", "").strip()
if APP_ENV_FILE:
    load_dotenv(APP_ENV_FILE, override=True)

TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b").strip()
DOCS_PATH = os.getenv("DOCS_PATH", str(ROOT_DIR / "docs")).strip()
INDEX_PATH = os.getenv("INDEX_PATH", str(ROOT_DIR / "data" / "index")).strip()
QUESTIONS_DIR = os.getenv("QUESTIONS_DIR", str(ROOT_DIR / "questions")).strip()
LOG_PATH = os.getenv("LOG_PATH", str(Path(QUESTIONS_DIR) / "logs")).strip()
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
).strip()
TOP_K = int(os.getenv("TOP_K", "5").strip())
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.30").strip())
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "5.0").strip())
CONTEXT_TURNS = int(os.getenv("CONTEXT_TURNS", "6").strip())
RETRIEVAL_HISTORY_TURNS = int(os.getenv("RETRIEVAL_HISTORY_TURNS", "2").strip())
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "1800").strip())
MAX_DOC_SNIPPET_CHARS = int(os.getenv("MAX_DOC_SNIPPET_CHARS", "450").strip())
MAX_DOC_FRAGMENTS = int(os.getenv("MAX_DOC_FRAGMENTS", "3").strip())
BOT_ROLE = os.getenv(
    "BOT_ROLE",
    (
        "Ты — поддержка партнёрской программы MaxKonvert. "
        "Отвечай профессионально и кратко, опираясь только на локальную документацию."
    ),
).strip()
BOT_EXTRA_RULES = os.getenv(
    "BOT_EXTRA_RULES",
    (
        "Если в документации нет подтверждения, прямо скажи об этом. "
        "Отвечай на языке пользователя. Не выдумывай факты."
    ),
).strip()

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

Path(LOG_PATH).mkdir(parents=True, exist_ok=True)
Path(QUESTIONS_DIR).mkdir(parents=True, exist_ok=True)
Path(QUESTIONS_LOG_FILE).touch(exist_ok=True)
Path(NEW_QUESTIONS_LOG_FILE).touch(exist_ok=True)
Path(UNANSWERED_LOG_FILE).touch(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(Path(LOG_PATH) / "bot.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

index = DocIndex(INDEX_PATH, EMBEDDING_MODEL)
conversation_memory = defaultdict(lambda: deque(maxlen=max(2, CONTEXT_TURNS * 2)))
state_lock = threading.Lock()


def ensure_index():
    if not index.chunks:
        logger.info("Index empty, building...")
        index.build(DOCS_PATH)


def read_json(path: str) -> dict[str, Any]:
    try:
        p = Path(path)
        if not p.exists():
            return {}
        data = json.loads(p.read_text(encoding="utf-8") or "{}")
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


question_state = read_json(QUESTIONS_STATE_FILE)


def write_json_atomic(path: str, payload: dict[str, Any]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def append_jsonl(path: str, payload: dict[str, Any]):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        logger.exception("Failed to write JSONL: %s", path)


def normalize_question(text: str) -> str:
    s = (text or "").lower().strip()
    s = s.replace("ё", "е")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-zа-я0-9 ]", "", s)
    return s.strip()


def conversation_key(update: Update) -> str:
    user = update.effective_user
    chat = update.effective_chat
    return f"{getattr(chat, 'id', 'none')}:{getattr(user, 'id', 'none')}"


def add_dialog_turn(key: str, role: str, text: str):
    text = (text or "").strip()
    if not text:
        return
    conversation_memory[key].append({"role": role, "text": text})


def get_recent_user_context(key: str) -> str:
    history = conversation_memory.get(key)
    if not history:
        return ""
    user_turns = [t["text"] for t in history if t["role"] == "user"]
    if not user_turns:
        return ""
    return "\n".join(user_turns[-max(1, RETRIEVAL_HISTORY_TURNS):])


def get_dialog_history_text(key: str) -> str:
    history = conversation_memory.get(key)
    if not history:
        return "(история пуста)"
    lines = []
    for turn in history:
        prefix = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{prefix}: {turn['text']}")
    text = "\n".join(lines)
    return text[-MAX_CONTEXT_CHARS:] if len(text) > MAX_CONTEXT_CHARS else text


def is_followup_query(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    followup_starts = (
        "а ",
        "и ",
        "это ",
        "а это",
        "а как",
        "а где",
        "а что",
        "а какой",
        "а какие",
        "а когда",
        "а сколько",
        "по этому",
        "по нему",
        "и это",
    )
    if lowered.startswith(followup_starts):
        return True
    pronouns = {"это", "этот", "эта", "эти", "они", "он", "она", "его", "ее", "её", "их"}
    tokens = [t for t in re.split(r"\W+", lowered) if t]
    return len(tokens) <= 7 and any(t in pronouns for t in tokens)


def build_question_event(
    update: Update, user_text: str, status: str, best_score: float, key: str
) -> dict[str, Any]:
    user = update.effective_user
    chat = update.effective_chat
    message = update.effective_message
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "conversation_key": key,
        "status": status,
        "best_score": round(best_score, 6),
        "question": user_text,
        "normalized_question": normalize_question(user_text),
        "user": {
            "id": getattr(user, "id", None),
            "username": getattr(user, "username", None),
            "full_name": getattr(user, "full_name", None),
            "language_code": getattr(user, "language_code", None),
        },
        "chat": {
            "id": getattr(chat, "id", None),
            "type": getattr(chat, "type", None),
        },
        "message_id": getattr(message, "message_id", None),
    }


def record_question_if_new(update: Update, question: str, key: str, best_score: float):
    normalized = normalize_question(question)
    if not normalized:
        return
    with state_lock:
        count = int(question_state.get(normalized, 0)) + 1
        question_state[normalized] = count
        write_json_atomic(QUESTIONS_STATE_FILE, question_state)
    if count == 1:
        event = build_question_event(update, question, "new_question", best_score, key)
        event["occurrence_count"] = count
        append_jsonl(NEW_QUESTIONS_LOG_FILE, event)


def build_prompt(user_text: str, history_text: str, docs_text: str) -> str:
    return (
        f"{BOT_ROLE}\n"
        f"{BOT_EXTRA_RULES}\n\n"
        "[Chat history with this user]\n"
        f"{history_text}\n\n"
        "[Knowledge base fragments]\n"
        f"{docs_text}\n\n"
        f"[Current question]\n{user_text}\n\n"
        "[Answer]"
    )


def build_docs_text(results: list[tuple[float, dict[str, Any]]]) -> str:
    parts = []
    for _, payload in results[: max(1, MAX_DOC_FRAGMENTS)]:
        file_name = payload.get("file", "unknown")
        text = (payload.get("text", "") or "")[:MAX_DOC_SNIPPET_CHARS]
        parts.append(f"[{file_name}]\n{text}")
    return "\n\n---\n\n".join(parts)


def is_unknown_answer(text: str) -> bool:
    lowered = (text or "").lower()
    markers = [
        "к сожалению",
        "нет информации",
        "не найден",
        "не указ",
        "not found in documentation",
        "no information in the documentation",
    ]
    return any(marker in lowered for marker in markers)


async def ask_ollama(prompt: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_ctx": 1536,
                        "num_predict": 220,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
    except Exception:
        logger.exception("Ollama error")
        return ""


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "Привет! Я бот поддержки MaxKonvert.\n"
        "Задайте вопрос, и я отвечу по документации."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = (update.effective_message.text or "").strip()
    if not user_text:
        return

    key = conversation_key(update)
    history_for_retrieval = get_recent_user_context(key)
    retrieval_query = user_text
    if history_for_retrieval and is_followup_query(user_text):
        retrieval_query = f"{history_for_retrieval}\n{user_text}"

    ensure_index()
    results = index.search(retrieval_query, top_k=TOP_K)
    best_score = results[0][0] if results else 0.0

    append_jsonl(
        QUESTIONS_LOG_FILE,
        build_question_event(update, user_text, "received", best_score, key),
    )
    record_question_if_new(update, user_text, key, best_score)

    if best_score < SIMILARITY_THRESHOLD:
        fallback = (
            "К сожалению, в документации нет подтверждённой информации по этому вопросу. "
            "Обратитесь в официальную поддержку MaxKonvert: https://t.me/MaxKonvert"
        )
        append_jsonl(
            UNANSWERED_LOG_FILE,
            build_question_event(update, user_text, "no_docs_match", best_score, key),
        )
        add_dialog_turn(key, "user", user_text)
        add_dialog_turn(key, "assistant", fallback)
        await update.effective_message.reply_text(fallback)
        return

    prompt = build_prompt(
        user_text=user_text,
        history_text=get_dialog_history_text(key),
        docs_text=build_docs_text(results),
    )
    answer = await ask_ollama(prompt)
    if not answer:
        answer = (
            "Не удалось сформировать ответ в заданный лимит времени. "
            "Повторите запрос или обратитесь в поддержку: https://t.me/MaxKonvert"
        )
        append_jsonl(
            UNANSWERED_LOG_FILE,
            build_question_event(update, user_text, "model_timeout_or_error", best_score, key),
        )
    elif is_unknown_answer(answer):
        append_jsonl(
            UNANSWERED_LOG_FILE,
            build_question_event(update, user_text, "model_no_answer", best_score, key),
        )

    add_dialog_turn(key, "user", user_text)
    add_dialog_turn(key, "assistant", answer)
    await update.effective_message.reply_text(answer)


def main():
    ensure_index()
    if not TOKEN:
        logger.error("TELEGRAM_TOKEN not set")
        sys.exit(1)
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
