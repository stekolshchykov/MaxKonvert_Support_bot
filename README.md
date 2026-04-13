# MaxKonvert Support Bot (All-in-Docker)

## Что внутри
- `bot` (`src/bot.py`): Telegram-бот в роли sales-менеджера, с памятью диалога по пользователю и логированием вопросов.
- `editor` (`src/docs_editor.py`): web-редактор документации + вкладка `Questions`.
- `watcher` + `reindex-periodic`: автопереиндексация базы знаний.
- `ollama` + `ollama-init`: локальная LLM полностью в Docker, модель подтягивается автоматически.

## Web UI возможности
- `Docs`:
  - create/save документов;
  - delete документа (раздела);
  - `Reindex Now`.
- `Questions`:
  - список `new/unanswered/all`;
  - удаление вопроса из логов (если уже обработан, чтобы не засорять список).

## Быстрый запуск
1. Скопировать шаблон:
```bash
cp .env.example .env.docker
```
2. В `.env.docker` заполнить минимум:
- `TELEGRAM_TOKEN`
- при необходимости порты: `BOT_HEALTH_PORT`, `DOCS_EDITOR_PORT`
- при необходимости host paths для volumes: `HOST_KB_DOCS_PATH`, `HOST_KB_QUESTIONS_PATH`, `HOST_KB_INDEX_PATH`, `HOST_OLLAMA_DATA_PATH`
3. Запустить:
```bash
docker compose --env-file .env.docker up -d --build
```

## Конфигурация (основное)
- `TELEGRAM_TOKEN` — токен Telegram-бота.
- `OLLAMA_MODEL` — модель Ollama (`qwen2.5:3b` по умолчанию).
- `BOT_ROLE`, `BOT_EXTRA_RULES`, `SALES_DEFAULT_CTA` — постоянный промпт роли менеджера.
- `BOT_HEALTH_PORT` — порт health endpoint бота (`/health`).
- `DOCS_EDITOR_PORT` — порт web-редактора базы знаний.
- `HOST_KB_DOCS_PATH` — папка документации на хосте.
- `HOST_KB_QUESTIONS_PATH` — папка логов вопросов на хосте.
- `HOST_KB_INDEX_PATH` — папка индекса на хосте.
- `HOST_OLLAMA_DATA_PATH` — папка моделей Ollama на хосте.

## Проверка
```bash
docker compose ps
curl http://127.0.0.1:${BOT_HEALTH_PORT:-8081}/health
curl http://127.0.0.1:${DOCS_EDITOR_PORT:-8090}/health
```

## Smoke-тест менеджерского стиля
```bash
docker compose exec -T bot python3 /app/scripts/sales_smoke_test.py
```

## Логи
```bash
docker compose logs -f ollama
docker compose logs -f bot
docker compose logs -f editor
docker compose logs -f watcher
```

## Docker Hub
Если нужно публиковать образ в Docker Hub:
```bash
docker login -u <your_username>
```
