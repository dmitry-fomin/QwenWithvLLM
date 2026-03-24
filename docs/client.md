# Модуль: client/

## Назначение
Python-клиент для взаимодействия с VLM моделями через vLLM OpenAI-совместимый API. Поддерживает текст, изображения (URL/файл/base64), стриминг, async и кэширование.

## Ключевые сущности

### VLMClient (`client/vllm_client.py`)
Синхронный клиент. Основной метод — `chat()`.
- `_build_image_content()` — автодетект типа изображения (URL/файл/base64)
- `_stream_response()` — генератор для стриминга
- `multi_turn_chat()` — многоходовой диалог

### AsyncVLMClient (`client/vllm_client.py`)
Асинхронная версия для параллельной обработки через `asyncio.gather()`.

### CachedVLMClient (`client/cached_client.py`)
VLMClient + двухуровневый кэш:
- `LRUCache` — in-memory, потокобезопасный
- `DiskCache` — shelve, персистентный
- TTL, cache_stats(), clear_cache()

## Бизнес-логика
- Изображения всегда идут **до текста** в `content` массиве (требование InternVL/Qwen-VL)
- Локальные файлы автоматически кодируются в base64 с определением MIME
- Стриминг возвращает генератор, не кэшируется
- Кэш-ключ = hash(model + prompt + images_hash + params)

## API
```python
client = VLMClient(base_url="http://localhost:8000/v1", model_name="qwen3-vl-8b")
result = client.chat(prompt="Describe", images=["img.jpg"], max_tokens=2048)
```

## Зависимости
- openai SDK (внешняя)
- Pillow (для определения MIME)
- shelve (стандартная библиотека, для DiskCache)
