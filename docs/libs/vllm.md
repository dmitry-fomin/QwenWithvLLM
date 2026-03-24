# vLLM (nightly ≥0.9)

## Назначение
Высокопроизводительный inference-сервер для LLM/VLM моделей. Предоставляет OpenAI-совместимый API.

## Основные методы/API
```bash
# Запуск сервера
python3 -m vllm.entrypoints.openai.api_server \
    --model MODEL_PATH \
    --served-model-name NAME \
    --tensor-parallel-size N \
    --trust-remote-code \
    --dtype bfloat16

# API endpoints (OpenAI-совместимые)
GET  /health
GET  /v1/models
POST /v1/chat/completions
```

## Паттерны использования в проекте
- Запускается через `scripts/start_vllm.sh` с env-конфигом
- Клиенты общаются через OpenAI SDK (не напрямую)
- Модели переключаются рестартом (одна модель за раз)
- `--trust-remote-code` обязателен для всех поддерживаемых моделей
- `--limit-mm-per-prompt image=N` ограничивает изображения на запрос

## Частые ошибки
- Забыть `--trust-remote-code` → модель не загрузится
- `max-model-len` слишком большой → CUDA OOM
- Nightly нужен для Qwen3-VL (≥0.11.0), стабильный не поддерживает
- AWQ квантизация Qwen2.5-VL-32B может деградировать качество в vLLM
