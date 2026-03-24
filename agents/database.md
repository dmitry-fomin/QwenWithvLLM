# Агент: Database / Cache

## СУБД
Нет традиционной СУБД. Проект использует:
- **shelve** — дисковый кэш для CachedVLMClient (`cache/vlm_cache`)
- **JSON Lines** — лог запросов (`logs/requests.jsonl`)
- **HuggingFace Hub** — хранилище моделей (S3-like, remote)

## ORM
Не используется. Доступ к данным:
- `shelve.open()` для дискового кэша
- `json.loads()` / `json.dumps()` для логов
- `huggingface_hub.snapshot_download()` для моделей

## Кэширование (`client/cached_client.py`)
- **L1**: LRUCache в памяти (OrderedDict, потокобезопасный)
- **L2**: DiskCache через shelve (персистентный)
- **TTL**: настраиваемое время жизни записи (по умолчанию 3600 сек)
- **Ключ**: SHA256 hash от (model + prompt + images_hash + params)

## Логирование (`monitoring/logger.py`)
- Формат: JSON Lines (append-only)
- Поля: timestamp, model, prompt_length, has_images, latency_ms, error
- Потокобезопасность: threading.Lock

## Соглашения
- Файлы кэша: `cache/` директория, gitignored
- Файлы логов: `logs/` директория, gitignored
- Модели: `models/` директория, gitignored (~50-150 GB)
