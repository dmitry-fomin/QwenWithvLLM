# Модуль: bot/

## Назначение
Telegram бот для отправки изображений и получения OCR/анализа от VLM модели.

## Ключевые сущности

### Telegram бот (`bot/telegram_bot.py`)
Команды:
- `/start` — приветствие
- `/help` — справка
- `/ocr` — переключить в OCR режим
- `/describe` — переключить в режим описания
- `/elements` — переключить в режим UI элементов
- `/model` — показать текущую модель

Обработчики:
- `handle_photo` — фото с подписью или без
- `handle_document` — изображения отправленные как файл
- `handle_text` — текстовые сообщения (без изображения)

## Бизнес-логика
- Режим (`ocr`/`describe`/`elements`) сохраняется в `context.user_data` и сбрасывается после обработки
- Фото скачивается через Telegram Bot API, кодируется в base64
- Длинные ответы (>4000 символов) разбиваются на части
- ACL через `ALLOWED_USERS` (список user_id через запятую)

## API
```bash
TELEGRAM_BOT_TOKEN=xxx python3 bot/telegram_bot.py
# Опционально:
VLLM_SERVER=http://localhost:8000/v1 VLLM_MODEL=qwen3-vl-8b
ALLOWED_USERS=123456,789012
```

## Зависимости
- python-telegram-bot ≥21.0
- client.vllm_client (AsyncVLMClient)
