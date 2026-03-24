# OpenAI Python SDK (≥1.30.0)

## Назначение
Клиент для OpenAI-совместимого API. Используется для общения с vLLM сервером.

## Основные методы/API
```python
from openai import OpenAI, AsyncOpenAI

# Синхронный
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.chat.completions.create(
    model="qwen3-vl-8b",
    messages=[{"role": "user", "content": "..."}],
    max_tokens=2048,
    temperature=0.3,
    stream=False,  # или True для стриминга
)

# Асинхронный
client = AsyncOpenAI(base_url="...", api_key="dummy")
response = await client.chat.completions.create(...)
```

## Паттерны использования в проекте
- `api_key="dummy"` — vLLM не проверяет ключ
- `base_url` указывает на vLLM сервер (не на api.openai.com)
- Мультимодальный контент передаётся как список в `content`:
  ```python
  content = [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
      {"type": "text", "text": "Describe this."},
  ]
  ```
- Стриминг через `stream=True` + context manager

## Частые ошибки
- Не указать `base_url` → запросы уйдут на api.openai.com
- Изображения после текста → некоторые модели работают хуже (InternVL)
- Timeout по умолчанию может быть мал для больших моделей → ставим 300 сек
