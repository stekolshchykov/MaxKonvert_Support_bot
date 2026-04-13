import json
import logging
import os
import re
import sys
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
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
LOW_CONFIDENCE_MODEL_SCORE = float(os.getenv("LOW_CONFIDENCE_MODEL_SCORE", "0.40").strip())
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "5.0").strip())
CONTEXT_TURNS = int(os.getenv("CONTEXT_TURNS", "6").strip())
RETRIEVAL_HISTORY_TURNS = int(os.getenv("RETRIEVAL_HISTORY_TURNS", "2").strip())
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "1800").strip())
MAX_DOC_SNIPPET_CHARS = int(os.getenv("MAX_DOC_SNIPPET_CHARS", "450").strip())
MAX_DOC_FRAGMENTS = int(os.getenv("MAX_DOC_FRAGMENTS", "3").strip())
MAX_ANSWER_CHARS = int(os.getenv("MAX_ANSWER_CHARS", "1600").strip())
BOT_HEALTH_HOST = os.getenv("BOT_HEALTH_HOST", "0.0.0.0").strip()
BOT_HEALTH_PORT = int(os.getenv("BOT_HEALTH_PORT", "8081").strip())
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
STARTED_AT_UTC = datetime.now(timezone.utc)

index: DocIndex | None = None
index_signature: tuple[int, int, int, int] | None = None
conversation_memory = defaultdict(lambda: deque(maxlen=max(2, CONTEXT_TURNS * 2)))
state_lock = threading.Lock()
index_lock = threading.Lock()


def start_health_server():
    if BOT_HEALTH_PORT <= 0:
        return

    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path != "/health":
                self.send_response(404)
                self.end_headers()
                return
            payload = {
                "status": "ok",
                "service": "maxkonvert-bot",
                "model": OLLAMA_MODEL,
                "started_at_utc": STARTED_AT_UTC.isoformat(),
            }
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args):
            logger.debug("health_http %s", fmt % args)

    server = HTTPServer((BOT_HEALTH_HOST, BOT_HEALTH_PORT), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Health server started at %s:%s", BOT_HEALTH_HOST, BOT_HEALTH_PORT)


def current_index_signature() -> tuple[int, int, int, int]:
    index_dir = Path(INDEX_PATH)
    files = [index_dir / "index.faiss", index_dir / "meta.pkl"]
    values: list[int] = []
    for f in files:
        try:
            st = f.stat()
            values.extend([int(st.st_mtime_ns), int(st.st_size)])
        except FileNotFoundError:
            values.extend([0, 0])
    return (values[0], values[1], values[2], values[3])


def maybe_reload_index(force: bool = False) -> DocIndex:
    global index, index_signature
    sig = current_index_signature()
    if force or index is None:
        logger.info("Initializing index...")
        index = DocIndex(INDEX_PATH, EMBEDDING_MODEL)
        index_signature = current_index_signature()
        return index

    # Reload in-memory index when on-disk artifacts changed after reindex.
    if index_signature is not None and sig != index_signature and any(sig):
        logger.info("Detected index update on disk, reloading in-memory index")
        previous_chunks = len(index.chunks)
        candidate = DocIndex(INDEX_PATH, EMBEDDING_MODEL)
        if candidate.chunks:
            index = candidate
            index_signature = current_index_signature()
            logger.info(
                "Index reloaded successfully (%s -> %s chunks)",
                previous_chunks,
                len(candidate.chunks),
            )
        else:
            logger.warning("Index reload produced empty chunks; keeping current in-memory index")
    return index


def get_index() -> DocIndex:
    with index_lock:
        return maybe_reload_index(force=False)


def ensure_index() -> DocIndex:
    with index_lock:
        idx = maybe_reload_index(force=False)
        if not idx.chunks:
            logger.info("Index empty, building...")
            idx.build(DOCS_PATH)
            global index_signature
            index_signature = current_index_signature()
        return idx


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
        "Стиль ответа: 3-6 коротких предложений, дружелюбно и по делу.\n"
        "Если есть факты во фрагментах, вплетай их в текст естественно.\n"
        "Не называй конкретные способы, цифры, проценты и API, если этого нет во фрагментах.\n"
        "Если факт не подтвержден во фрагментах, не отрицай это напрямую, а переведи в менеджерский шаг: уточни вводные и предложи быстро подобрать вариант.\n"
        "В конце добавляй один конкретный уточняющий вопрос, чтобы продвинуть диалог к запуску.\n\n"
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


def rerank_results_by_query_tokens(
    user_text: str, results: list[tuple[float, dict[str, Any]]]
) -> list[tuple[float, dict[str, Any]]]:
    tokens = query_tokens(user_text)
    if not tokens or not results:
        return results
    rescored: list[tuple[float, int, float, dict[str, Any]]] = []
    for score, payload in results:
        text = (payload.get("text", "") or "").lower().replace("ё", "е")
        overlap = sum(1 for tok in tokens if tok in text)
        adjusted = float(score) + (0.08 * overlap)
        if overlap and len(tokens) == 1 and tokens[0] in text:
            adjusted += 0.12
        rescored.append((adjusted, overlap, float(score), payload))
    rescored.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    return [(row[2], row[3]) for row in rescored]


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


def contains_unbacked_claims(answer: str, docs_text: str) -> bool:
    answer_tokens = {
        t
        for t in re.findall(r"[a-zа-я0-9]+", (answer or "").lower().replace("ё", "е"))
        if len(t) >= 4
    }
    docs_tokens = {
        t
        for t in re.findall(r"[a-zа-я0-9]+", (docs_text or "").lower().replace("ё", "е"))
        if len(t) >= 4
    }
    if not answer_tokens or not docs_tokens:
        return False

    ignore = {
        "maxkonvert",
        "можно",
        "нужно",
        "лучше",
        "вариант",
        "уточните",
        "подскажите",
        "подобрать",
        "запуск",
        "запуска",
        "трафик",
        "трафика",
        "условия",
        "формат",
        "форматы",
        "geo",
    }
    core = {t for t in answer_tokens if t not in ignore}
    if not core:
        return False
    unsupported = {t for t in core if t not in docs_tokens}
    ratio = len(unsupported) / max(1, len(core))
    # Be conservative: only flag clearly unsupported answers, avoid false positives.
    return len(unsupported) >= 6 and ratio >= 0.55


def query_tokens(text: str) -> list[str]:
    parts = re.findall(r"[a-zа-я0-9]+", (text or "").lower().replace("ё", "е"))
    blacklist = {
        "что",
        "как",
        "какой",
        "какие",
        "есть",
        "это",
        "а",
        "и",
        "по",
        "в",
        "на",
        "ли",
        "за",
        "к",
        "у",
        "из",
        "про",
        "подскажите",
        "скажите",
        "пожалуйста",
    }
    return [p for p in parts if len(p) >= 3 and p not in blacklist]


def build_extractive_fallback(user_text: str, results: list[tuple[float, dict[str, Any]]]) -> str:
    if not results:
        return ""
    tokens = query_tokens(user_text)

    def collect_lines(require_tokens: bool) -> list[tuple[str, str]]:
        lines: list[tuple[str, str]] = []
        seen = set()
        for _, payload in results[: max(1, MAX_DOC_FRAGMENTS)]:
            file_name = payload.get("file", "unknown")
            text = payload.get("text", "") or ""
            for raw in re.split(r"[\n\r]+", text):
                line = raw.strip(" -*\t")
                if len(line) < 18:
                    continue
                low = line.lower().replace("ё", "е")
                if require_tokens and tokens and not any(tok in low for tok in tokens):
                    continue
                key = normalize_question(line)
                if key in seen:
                    continue
                seen.add(key)
                lines.append((line, file_name))
                if len(lines) >= 4:
                    break
            if len(lines) >= 4:
                break
        return lines

    # Primary pass: token-aware lines.
    lines = collect_lines(require_tokens=True)
    # Secondary pass: tolerate typos/short greetings by taking strongest lines without token filter.
    if not lines:
        lines = collect_lines(require_tokens=False)
    if not lines:
        return ""

    snippets = [line for line, _ in lines[:3]]
    if not snippets:
        return ""
    lead = "По вашему запросу вижу рабочие варианты:"
    body = " ".join(snippets)
    body = re.sub(r"\s+", " ", body).strip()
    return f"{lead} {body} {SALES_DEFAULT_CTA}"


def build_sales_manager_fallback(
    user_text: str, results: list[tuple[float, dict[str, Any]]], *, low_match: bool = False
) -> str:
    extractive = build_extractive_fallback(user_text, results)
    if extractive:
        return extractive
    if low_match:
        return (
            "Спасибо за вопрос. "
            "Подберу для вас оптимальный сценарий запуска под ваш трафик и цели. "
            f"{SALES_DEFAULT_CTA}"
        )
    return (
        "Могу быстро собрать для вас понятный план запуска и монетизации. "
        f"{SALES_DEFAULT_CTA}"
    )


def build_direct_definition_answer(user_text: str, results: list[tuple[float, dict[str, Any]]]) -> str:
    tokens = query_tokens(user_text)
    if not results or not tokens:
        return ""
    candidates: list[str] = []
    for _, payload in results:
        text = payload.get("text", "") or ""
        for raw in text.splitlines():
            line = raw.strip(" \t-*")
            if len(line) < 8:
                continue
            low = line.lower().replace("ё", "е")
            if not any(tok in low for tok in tokens):
                continue
            if (" это " in low) or ("- это" in low) or ("— это" in low):
                candidates.append(line)
                break
        if not candidates:
            compact = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
            if compact:
                low = compact.lower().replace("ё", "е")
                if any(tok in low for tok in tokens) and (
                    (" это " in low) or ("- это" in low) or ("— это" in low)
                ):
                    candidates.append(compact)
        if candidates:
            break
    if not candidates:
        return ""
    line = re.sub(r"^[#]+\s*", "", candidates[0]).strip()
    line = re.sub(r"\s+", " ", line).strip()
    line = line.rstrip(" .") + "."
    return (
        f"{line} "
        "Если хотите, подскажу, как это применить в вашей связке и что запускать первым."
    )


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
                        "temperature": 0.1,
                        "num_ctx": 1536,
                        "num_predict": 180,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("response", "").strip()
            return answer[:MAX_ANSWER_CHARS]
    except Exception:
        logger.exception("Ollama error")
        return ""


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "Привет! Я менеджер MaxKonvert и помогу подобрать оптимальный вариант монетизации.\n"
        "Опишите ваш трафик или задайте вопрос по условиям — разберём по шагам."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    started = datetime.now(timezone.utc)
    user_text = (update.effective_message.text or "").strip()
    if not user_text:
        return

    key = conversation_key(update)
    history_for_retrieval = get_recent_user_context(key)
    retrieval_query = user_text
    if history_for_retrieval and is_followup_query(user_text):
        retrieval_query = f"{history_for_retrieval}\n{user_text}"

    idx = ensure_index()
    raw_results = idx.search(retrieval_query, top_k=max(TOP_K, 8))
    best_score = raw_results[0][0] if raw_results else 0.0
    results = rerank_results_by_query_tokens(user_text, raw_results)[:TOP_K]

    append_jsonl(
        QUESTIONS_LOG_FILE,
        build_question_event(update, user_text, "received", best_score, key),
    )
    record_question_if_new(update, user_text, key, best_score)

    if best_score < SIMILARITY_THRESHOLD:
        fallback = build_sales_manager_fallback(user_text, results, low_match=True)
        append_jsonl(
            UNANSWERED_LOG_FILE,
            build_question_event(update, user_text, "needs_kb_update_low_match", best_score, key),
        )
        add_dialog_turn(key, "user", user_text)
        add_dialog_turn(key, "assistant", fallback)
        await update.effective_message.reply_text(fallback)
        return

    if best_score < LOW_CONFIDENCE_MODEL_SCORE:
        fallback = build_sales_manager_fallback(user_text, results, low_match=True)
        append_jsonl(
            UNANSWERED_LOG_FILE,
            build_question_event(update, user_text, "needs_kb_update_low_confidence", best_score, key),
        )
        add_dialog_turn(key, "user", user_text)
        add_dialog_turn(key, "assistant", fallback)
        await update.effective_message.reply_text(fallback)
        return

    answer = build_direct_definition_answer(user_text, results)
    if answer:
        append_jsonl(
            QUESTIONS_LOG_FILE,
            build_question_event(update, user_text, "direct_definition_from_docs", best_score, key),
        )
    else:
        prompt = build_prompt(
            user_text=user_text,
            history_text=get_dialog_history_text(key),
            docs_text=build_docs_text(results),
        )
        answer = await ask_ollama(prompt)
        if not answer:
            answer = build_sales_manager_fallback(user_text, results)
            if answer:
                append_jsonl(
                    QUESTIONS_LOG_FILE,
                    build_question_event(
                        update, user_text, "sales_fallback_after_model_error", best_score, key
                    ),
                )
            append_jsonl(
                UNANSWERED_LOG_FILE,
                build_question_event(update, user_text, "model_timeout_or_error", best_score, key),
            )
        elif is_unknown_answer(answer):
            answer = build_sales_manager_fallback(user_text, results, low_match=best_score < 0.35)
            append_jsonl(
                QUESTIONS_LOG_FILE,
                build_question_event(
                    update, user_text, "sales_override_model_no_answer", best_score, key
                ),
            )
            append_jsonl(
                UNANSWERED_LOG_FILE,
                build_question_event(
                    update, user_text, "needs_kb_update_model_no_answer", best_score, key
                ),
            )
        elif contains_unbacked_claims(answer, build_docs_text(results)):
            # Keep model answer (manager-style UX priority), but record a quality signal.
            append_jsonl(
                QUESTIONS_LOG_FILE,
                build_question_event(
                    update, user_text, "model_answer_with_unbacked_tokens", best_score, key
                ),
            )

    add_dialog_turn(key, "user", user_text)
    add_dialog_turn(key, "assistant", answer)
    elapsed_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
    logger.info(
        "Answered message key=%s score=%.4f elapsed_ms=%s model=%s",
        key,
        best_score,
        elapsed_ms,
        OLLAMA_MODEL,
    )
    await update.effective_message.reply_text(answer)


def main():
    start_health_server()
    if not TOKEN:
        logger.error("TELEGRAM_TOKEN not set")
        sys.exit(1)
    ensure_index()
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
