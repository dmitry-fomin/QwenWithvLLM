# Агент: Python Backend

## Окружение
- Python: 3.10+
- Пакетный менеджер: pip / uv (nightly vLLM через uv)
- Виртуальное окружение: venv (создаётся в `scripts/setup_server.sh`)
- Фреймворк: FastAPI (Web UI), python-telegram-bot (бот), vLLM (inference)

## Архитектура
- `client/` — клиентская библиотека (VLMClient, AsyncVLMClient, CachedVLMClient)
- `web/` — FastAPI Web UI с Jinja2 шаблонами
- `bot/` — Telegram бот (polling mode)
- `tools/` — CLI утилиты (screenshot OCR)
- `monitoring/` — логирование и дашборд
- `scripts/` — bash скрипты для серверных операций
- `configs/` — env-файлы с конфигурациями моделей
- `tests/` — pytest

## FastAPI-специфичное
- Pydantic: v2 (через FastAPI)
- Роутеры: в `web/app.py` (один файл)
- Зависимости: `get_client()` — синглтон VLMClient
- Async: частично (health check — async, analyze — sync через VLMClient)
- Templates: Jinja2 в `web/templates/`

## Тестирование
- pytest + pytest-asyncio
- Fixtures: `tests/conftest.py` (mock OpenAI, sample images, env configs)
- 99 тестов: клиент (sync/async), конфиги, скрипты (синтаксис bash)
- Запуск: `python -m pytest tests/ -v`

## Code Style
- Типы: type hints в публичных методах
- Docstrings: Google style
- Стандарт: PEP 8
- Без линтера (пока), без форматтера

## Асинхронность
- AsyncVLMClient для параллельной обработки изображений
- `asyncio.gather()` в batch_requests.py
- Telegram бот полностью async (python-telegram-bot)

## Запрещено
- `import *`
- Мутабельные аргументы по умолчанию
- Bare except
- Docker (пользователь явно просил нативный Python)
- Хранение секретов в коде
