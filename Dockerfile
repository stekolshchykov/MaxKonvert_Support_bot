FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ROOT=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    inotify-tools \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY src /app/src
COPY scripts /app/scripts

RUN chmod +x /app/scripts/watcher.sh /app/scripts/reindex_loop.sh

CMD ["python3", "/app/src/bot.py"]
