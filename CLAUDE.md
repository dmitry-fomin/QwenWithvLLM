# QwenWithvLLM

Система для запуска мультимодальных VLM (Qwen3-VL, InternVL, Pixtral и др.) на локальном GPU-сервере через vLLM. Включает OpenAI-совместимый API, Python-клиент, Web UI, Telegram-бот, мониторинг и кэширование.

## Стек
- Python 3.10+
- vLLM (nightly ≥0.9) — inference server
- FastAPI — Web UI backend
- OpenAI SDK — клиент к vLLM
- python-telegram-bot — Telegram интеграция
- mss + Pillow — захват скриншотов
- pytest + pytest-asyncio — тесты

## Структура проекта
- `client/` — VLMClient, AsyncVLMClient, CachedVLMClient
- `configs/` — env-конфигурации для каждой модели
- `scripts/` — setup, download, start, switch_model (bash)
- `web/` — FastAPI Web UI (порт 7860)
- `bot/` — Telegram бот
- `tools/` — утилиты (screenshot OCR)
- `monitoring/` — логирование запросов, CLI dashboard
- `tests/` — pytest (99 тестов)

## Документация
- [docs/client.md](docs/client.md) — VLM клиент
- [docs/server.md](docs/server.md) — vLLM сервер и конфигурации
- [docs/web-ui.md](docs/web-ui.md) — Web интерфейс
- [docs/telegram-bot.md](docs/telegram-bot.md) — Telegram бот
- [docs/monitoring.md](docs/monitoring.md) — Мониторинг и логирование

## Агенты
- [agents/python.md](agents/python.md) — Python backend
- [agents/database.md](agents/database.md) — Кэш и хранилище

## Правила
- Нативный Python, **без Docker** — пользователь хочет прямой доступ к GPU
- Модели ограничены **64 GB VRAM** (2x RTX 5090): до 26B BF16 или до 72B INT4
- Изображения идут **до текста** в массиве `content` (требование InternVL/Qwen-VL)
- vLLM всегда запускается с `--trust-remote-code`
- Переключение моделей — рестарт vLLM через `switch_model.sh`

## Обновление
- Обновляй этот файл при значимых изменениях в архитектуре
- Обновляй docs/*.md при изменении логики модулей
- Обновляй agents/*.md при изменении паттернов
