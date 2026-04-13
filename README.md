# MaxKonvert Support Bot (Docker Local)

## Что внутри
- Telegram-бот (`src/bot.py`) с логикой:
  - роль + история конкретного пользователя + фрагменты из базы знаний;
  - без ручных `if вопрос про X`;
  - запись всех вопросов;
  - отдельная фиксация новых и неотвеченных вопросов.
- Веб-консоль (`src/docs_editor.py`):
  - редактирование `docs` в браузере;
  - вкладка `Questions` (все/новые/неотвеченные вопросы);
  - кнопка `Reindex Now`.
- Автообновление индекса:
  - `watcher` по событиям файлов;
  - `reindex-periodic` по таймеру.

## Docker запуск
1. Подготовь папки:
```bash
mkdir -p kb/docs kb/questions kb/index
```
2. Положи документацию в `kb/docs`.
3. Скопируй `.env.docker` и заполни `TELEGRAM_TOKEN`:
```bash
cp .env.docker .env
```
4. Запусти:
```bash
docker compose up -d --build
```

## Порт и папка знаний
- Веб-консоль по умолчанию: `http://<host>:8090/`
- Папка знаний/вопросов (на хосте):
  - `./kb/docs` — документация
  - `./kb/questions` — вопросы (questions/new/unanswered + state)
  - `./kb/index` — индекс

## Проверка
```bash
docker compose ps
curl http://127.0.0.1:8090/health
```

## Логи
```bash
docker compose logs -f bot
docker compose logs -f editor
docker compose logs -f watcher
```
