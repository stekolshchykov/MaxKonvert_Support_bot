#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${APP_ROOT:-/app}"
DOCS_DIR="${DOCS_PATH:-${APP_ROOT}/kb/docs}"
QUESTIONS_DIR="${QUESTIONS_DIR:-${APP_ROOT}/kb/questions}"
LOG_PATH="${LOG_PATH:-${QUESTIONS_DIR}/logs}"
DEBOUNCE_SEC="${DEBOUNCE_SEC:-3}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REINDEX_SCRIPT="${REINDEX_SCRIPT:-${APP_ROOT}/scripts/reindex.py}"
LOG_FILE="${LOG_PATH}/watcher.log"

mkdir -p "$LOG_PATH"
touch "$LOG_FILE"

echo "$(date '+%Y-%m-%d %H:%M:%S') [watcher] Started watching $DOCS_DIR" >> "$LOG_FILE"

inotifywait -m -r -e create,modify,delete,move --format '%w%f %e %T' --timefmt '%Y-%m-%d %H:%M:%S' "$DOCS_DIR" | while read -r file event time; do
    echo "$time [watcher] Event: $event on $file" >> "$LOG_FILE"
    while IFS= read -r -t "$DEBOUNCE_SEC" line; do
        read -r f e t <<< "$line"
        echo "$t [watcher] Debounced event: $e on $f" >> "$LOG_FILE"
    done
    echo "$(date '+%Y-%m-%d %H:%M:%S') [watcher] Triggering reindex after debounce" >> "$LOG_FILE"
    if "$PYTHON_BIN" "$REINDEX_SCRIPT" >> "$LOG_FILE" 2>&1; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [watcher] Reindex finished OK" >> "$LOG_FILE"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') [watcher] Reindex finished with error" >> "$LOG_FILE"
    fi
done
