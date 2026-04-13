#!/usr/bin/env python3
import fcntl
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
APP_ENV_FILE = os.getenv("APP_ENV_FILE", "").strip()
if APP_ENV_FILE:
    load_dotenv(APP_ENV_FILE, override=True)

sys.path.insert(0, str(ROOT_DIR / "src"))
from indexer import DocIndex  # noqa: E402

LOCK_FILE = Path("/tmp/maxkonvert-bot-reindex.lock")
INDEX_PATH = os.getenv("INDEX_PATH", str(ROOT_DIR / "data" / "index"))
DOCS_PATH = os.getenv("DOCS_PATH", str(ROOT_DIR / "docs"))
QUESTIONS_DIR = os.getenv("QUESTIONS_DIR", str(ROOT_DIR / "questions"))
LOG_PATH = os.getenv("LOG_PATH", str(Path(QUESTIONS_DIR) / "logs"))
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

Path(LOG_PATH).mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(Path(LOG_PATH) / "reindex.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("reindex")


def main():
    lock = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.info("Another reindex is already running, skipping")
        return
    try:
        logger.info("Starting reindex...")
        idx = DocIndex(INDEX_PATH, EMBEDDING_MODEL)
        idx.build(DOCS_PATH)
        logger.info("Reindex completed successfully")
    except Exception:
        logger.exception("Reindex failed")
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
