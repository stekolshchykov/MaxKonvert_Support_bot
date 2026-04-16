import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from assistant import Assistant
from config import Config

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
APP_ENV_FILE = os.getenv("APP_ENV_FILE", "").strip()
if APP_ENV_FILE:
    load_dotenv(APP_ENV_FILE, override=True)

# Telegram-specific env (kept here for backward compatibility)
TOKEN = (os.getenv("TELEGRAM_TOKEN", "") or os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b").strip()
BOT_HEALTH_HOST = os.getenv("BOT_HEALTH_HOST", "0.0.0.0").strip()
BOT_HEALTH_PORT = int(os.getenv("BOT_HEALTH_PORT", "8081").strip())

# Logging setup
Path(Config.LOG_PATH).mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(Path(Config.LOG_PATH) / "bot.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
STARTED_AT_UTC = datetime.now(timezone.utc)

# Shared assistant instance
assistant = Assistant()


def start_health_server():
    if BOT_HEALTH_PORT <= 0:
        return

    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                payload = {
                    "status": "ok",
                    "service": "maxkonvert-bot",
                    "model": assistant.provider.model_id,
                    "provider": assistant.provider.name,
                    "started_at_utc": STARTED_AT_UTC.isoformat(),
                }
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"MaxKonvert Bot is running")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, fmt: str, *args):
            logger.debug("health_http %s", fmt % args)

    server = HTTPServer((BOT_HEALTH_HOST, BOT_HEALTH_PORT), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Health server started at %s:%s", BOT_HEALTH_HOST, BOT_HEALTH_PORT)


def conversation_key(update: Update) -> str:
    user = update.effective_user
    chat = update.effective_chat
    return f"{getattr(chat, 'id', 'none')}:{getattr(user, 'id', 'none')}"


def build_telegram_metadata(update: Update) -> dict[str, Any]:
    user = update.effective_user
    chat = update.effective_chat
    message = update.effective_message
    return {
        "telegram_user_id": getattr(user, "id", None),
        "telegram_username": getattr(user, "username", None),
        "telegram_chat_id": getattr(chat, "id", None),
        "telegram_chat_type": getattr(chat, "type", None),
        "telegram_message_id": getattr(message, "message_id", None),
    }


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "Привет! Я менеджер MaxKonvert и помогу подобрать оптимальный вариант монетизации.\n"
        "Опишите ваш трафик или задайте вопрос по условиям — разберём по шагам."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = (update.effective_message.text or "").strip()
    if not user_text:
        return

    key = conversation_key(update)
    metadata = build_telegram_metadata(update)

    try:
        result = await assistant.process_message(
            user_text=user_text,
            conversation_key=key,
            channel="telegram",
            metadata=metadata,
        )
        await update.effective_message.reply_text(result["answer"])
    except Exception:
        logger.exception("Error handling message from key=%s", key)
        await update.effective_message.reply_text(
            "Извините, не удалось обработать запрос. Попробуйте ещё раз через пару секунд."
        )


def main():
    errors = Config.validate()
    if errors:
        for err in errors:
            logger.error("Config validation error: %s", err)
        sys.exit(1)

    start_health_server()
    if not TOKEN:
        logger.error("TELEGRAM_TOKEN not set")
        sys.exit(1)

    assistant.ensure_index()
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Bot started (provider=%s model=%s)", assistant.provider.name, assistant.provider.model_id)
    app.run_polling()


if __name__ == "__main__":
    main()
