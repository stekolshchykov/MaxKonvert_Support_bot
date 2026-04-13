#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${APP_ROOT:-/app}"
QUESTIONS_DIR="${QUESTIONS_DIR:-${APP_ROOT}/kb/questions}"
LOG_PATH="${LOG_PATH:-${QUESTIONS_DIR}/logs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REINDEX_SCRIPT="${REINDEX_SCRIPT:-${APP_ROOT}/scripts/reindex.py}"
REINDEX_INTERVAL_SEC="${REINDEX_INTERVAL_SEC:-300}"
LOG_FILE="${LOG_PATH}/reindex-periodic.log"

mkdir -p "$LOG_PATH"
touch "$LOG_FILE"

echo "$(date '+%Y-%m-%d %H:%M:%S') [periodic] Started periodic reindex loop" >> "$LOG_FILE"

while true; do
  if "$PYTHON_BIN" "$REINDEX_SCRIPT" >> "$LOG_FILE" 2>&1; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [periodic] Reindex OK" >> "$LOG_FILE"
  else
    echo "$(date '+%Y-%m-%d %H:%M:%S') [periodic] Reindex failed" >> "$LOG_FILE"
  fi
  sleep "$REINDEX_INTERVAL_SEC"
done
