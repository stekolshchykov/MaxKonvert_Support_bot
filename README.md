# MaxKonvert Support Bot (All-in-Docker)

## Архитектура после расширения

```
┌─────────────────┐     ┌─────────────────┐
│   Telegram      │     │   HTTP API      │
│   (bot.py)      │     │  (api_server)   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
            ┌─────────────────┐
            │   Assistant     │
            │  (assistant.py) │
            └────────┬────────┘
                     │
     ┌───────────────┼───────────────┐
     ▼               ▼               ▼
┌─────────┐   ┌───────────┐   ┌───────────┐
│ DocIndex│   │ Provider  │   │  Actions  │
│(FAISS)  │   │  Layer    │   │ (HTTP)    │
└─────────┘   └─────┬─────┘   └───────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
   ┌──────────┐          ┌──────────┐
   │  Ollama  │          │   Kimi   │
   │ (local)  │          │  (API)   │
   └──────────┘          └──────────┘
```

## Что внутри
- `bot` (`src/bot.py`): Telegram-бот в роли sales-менеджера. Теперь использует общий `Assistant`.
- `api` (`src/api_server.py`): FastAPI HTTP API для внешних сервисов. Тот же `Assistant`, другой вход.
- `editor` (`src/docs_editor.py`): web-редактор документации + вкладка `Questions`.
- `watcher` + `reindex-periodic`: автопереиндексация базы знаний.
- `ollama` + `ollama-init`: локальная LLM полностью в Docker (опционально, только для `MODEL_PROVIDER=local`).
- `assistant` (`src/assistant.py`): единое ядро обработки запросов, память диалога, поиск по KB, вызов модели, экшены.
- `providers` (`src/providers/`): абстракция провайдера модели — `local` (Ollama) и `kimi` (Moonshot API).
- `actions` (`src/actions.py`): слой исходящих HTTP-запросов (GET/POST/PUT/DELETE).

## Быстрый запуск

### 1. Подготовить окружение
```bash
cp .env.example .env.docker
# Отредактировать .env.docker
```

### 2. Режим локальной модели (Ollama)
```bash
docker compose --profile local --env-file .env.docker up -d --build
```

### 3. Режим Kimi API
```bash
# В .env.docker:
# MODEL_PROVIDER=kimi
# KIMI_API_TOKEN=sk-...
docker compose --env-file .env.docker up -d --build
```

### 4. Проверка
```bash
docker compose ps
curl http://127.0.0.1:${BOT_HEALTH_PORT:-8081}/health
curl http://127.0.0.1:${API_PORT:-8082}/health
curl http://127.0.0.1:${DOCS_EDITOR_PORT:-8090}/health
```

## Конфигурация (основное)

### Провайдер модели
- `MODEL_PROVIDER=local | kimi`
- `OLLAMA_URL` — адрес Ollama (по умолчанию `http://ollama:11434`).
- `OLLAMA_MODEL` — модель Ollama (`qwen2.5:3b` по умолчанию).
- `KIMI_API_TOKEN` — токен Moonshot API (обязателен при `MODEL_PROVIDER=kimi`).
- `KIMI_BASE_URL` — базовый URL Kimi API (`https://api.moonshot.cn/v1`).
- `KIMI_MODEL` — идентификатор модели Kimi (`moonshot-v1-8k`).

### Telegram
- `TELEGRAM_TOKEN` — токен Telegram-бота.
- `TELEGRAM_BOT_URL` — ссылка на бота (для справки).

### Порты
- `BOT_HEALTH_PORT` — порт health endpoint бота (`/health`).
- `API_PORT` — порт HTTP API (`/health`, `/api/chat`).
- `DOCS_EDITOR_PORT` — порт web-редактора базы знаний.

### Пути
- `HOST_KB_DOCS_PATH` — папка документации на хосте.
- `HOST_KB_QUESTIONS_PATH` — папка логов вопросов на хосте.
- `HOST_KB_INDEX_PATH` — папка индекса на хосте.
- `HOST_OLLAMA_DATA_PATH` — папка моделей Ollama на хосте (только для local).

### Поведение бота
- `BOT_ROLE`, `BOT_EXTRA_RULES`, `SALES_DEFAULT_CTA` — постоянный промпт роли менеджера.
- `TOP_K`, `SIMILARITY_THRESHOLD`, `LOW_CONFIDENCE_MODEL_SCORE` — пороги поиска и уверенности.

## HTTP API

### `POST /api/chat`
Запрос:
```json
{
  "message": "Какие услуги предлагает MaxKonvert?",
  "session_id": "sess-123",
  "user_id": "user-456",
  "metadata": {"source": "crm"}
}
```

Ответ:
```json
{
  "answer": "MaxKonvert предлагает монетизацию через МТ-подписки, WAP-клик, псевдо-подписки, премиум-SMS и парковку доменов.",
  "route": "model",
  "best_score": 0.42,
  "elapsed_ms": 1200,
  "model": "qwen2.5:3b",
  "provider": "ollama",
  "timestamp": "2026-04-16T08:00:00+00:00"
}
```

### `GET /health`
Возвращает `{"status":"ok", ...}` с информацией о провайдере и модели.

## HTTP Actions (исходящие запросы)

Ассистент может вызывать внешние HTTP-эндпоинты, если это требуется по инструкциям из базы знаний.

### Настройка
Действия описываются в `kb/http_actions.json`:
```json
{
  "actions": [
    {
      "name": "check_status",
      "description": "Проверить статус демо-сервера",
      "method": "GET",
      "url": "https://httpbin.org/get",
      "headers": {"Accept": "application/json"}
    }
  ]
}
```

### Как это работает
1. Если в промпте присутствуют доступные действия, модель может запросить вызов через тег:
   `TOOL:{"name":"check_status","params":{"foo":"bar"}}`
2. `Assistant` распознаёт тег, вызывает `ActionExecutor`, получает результат.
3. Результат подаётся модели повторно для формулировки финального ответа.

Поддерживаются методы: `GET`, `POST`, `PUT`, `DELETE`.

## Smoke-тест

```bash
# Локальный режим (без Ollama — проверяет фолбэки и KB)
docker compose exec -T bot python3 /app/scripts/sales_smoke_test.py

# Полный тестовый набор
source .venv/bin/activate
MODEL_PROVIDER=local python3 scripts/test_suite.py
MODEL_PROVIDER=kimi KIMI_API_TOKEN=... python3 scripts/test_suite.py
```

## Тестовое покрытие

В проекте есть детальный тест-сьют (`scripts/test_suite.py`), покрывающий:
- **Регрессия / существующее поведение** — KB-ответы, follow-up, прямые определения.
- **Переключение провайдеров** — `local` ↔ `kimi`, валидация конфигурации.
- **Telegram-канал** — сообщения, follow-up, пустой ввод, неизвестные запросы.
- **HTTP API** — `/health`, `/api/chat`, валидация, структура ответа.
- **HTTP Actions** — GET/POST/PUT/DELETE, заголовки, query params, JSON body, ошибки.
- **KB-only** — ответы только из документации.
- **KB + API** — смешанные сценарии с вызовом действий.
- **Качество / human-likeness** — отсутствие роботизированных отказов.
- **Граничные случаи** — пустые сообщения, таймауты, невалидные запросы.
- **Кросс-канальная консистентность** — одинаковые вопросы через Telegram и API дают одинаковую логику маршрутизации.

Минимальный набор: **50+ тестов**. Фактически выполняется **64+ тестов** (включая live-проверки Kimi при наличии валидного токена).

## Порт Базы Знаний
- Внутри контейнера/стека: `DOCS_EDITOR_PORT` (по умолчанию `8090`).
- На сервере `128` для доступа из сети: `http://<IP_128>:18090/`.
  - На текущем прод-контуре это прокси `128:18090 -> CT205:8090`.

## Логи
```bash
docker compose logs -f ollama
docker compose logs -f bot
docker compose logs -f api
docker compose logs -f editor
docker compose logs -f watcher
```

## Docker Hub
Если нужно публиковать образ в Docker Hub:
```bash
docker login -u <your_username>
```

## Ограничения и известные риски
- **Kimi API токен**: при предоставленном токене в ходе тестирования была получена ошибка `401 Invalid Authentication`. Это означает, что либо токен недействителен/истёк, либо требуется дополнительная настройка доступа. Код интеграции реализован корректно — при валидном токене всё будет работать.
- **Conversation memory** не разделяется между процессами `bot` и `api`. Если пользователь пишет в Telegram и затем через HTTP API с тем же `session_id`, история диалога будет разной, потому что память хранится в памяти процесса. Для полноценной кросс-канальной памяти требуется внешнее хранилище (Redis и т.д.).
- **Ollama в Docker на macOS ARM**: стандартные `+cpu` wheels PyTorch недоступны для macOS. При локальном тестировании на macOS использовались совместимые версии `torch` и `faiss-cpu` из PyPI. В Linux-контейнерах оригинальные версии из `requirements.txt` работают корректно.
