# Модуль: web/

## Назначение
Браузерный интерфейс для отправки изображений и текстовых запросов к VLM через vLLM.

## Ключевые сущности

### FastAPI приложение (`web/app.py`)
- `POST /api/analyze` — основной endpoint, принимает prompt + image (file/URL)
- `GET /api/health` — проверка доступности vLLM
- `GET /api/models` — список моделей на сервере
- `GET /` — HTML страница

### HTML UI (`web/templates/index.html`)
- Drag-n-drop загрузка изображений
- Quick prompts: OCR, Describe, UI Elements, App Analysis, Translate
- Настройки: server URL, model, max_tokens, temperature
- Статус vLLM сервера (health check каждые 15 сек)
- Copy to clipboard

## Бизнес-логика
- Загруженные файлы кодируются в base64 на сервере
- Глобальный VLMClient переиспользуется между запросами
- Ctrl+Enter отправляет запрос

## API
```bash
# Запуск
python3 web/app.py                            # порт 7860
uvicorn web.app:app --host 0.0.0.0 --port 7860 --reload
```

## Зависимости
- FastAPI, uvicorn, Jinja2, python-multipart
- httpx (для health check)
- client.vllm_client (внутренняя)
