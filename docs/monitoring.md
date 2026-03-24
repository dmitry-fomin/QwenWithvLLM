# Модуль: monitoring/

## Назначение
Логирование запросов к VLM, сбор метрик, CLI дашборд с GPU статусом.

## Ключевые сущности

### RequestLogger (`monitoring/logger.py`)
Потокобезопасный логгер в JSON Lines формат.
- Пишет в `logs/requests.jsonl`
- In-memory метрики: requests, errors, latency, tokens (оценка)

### LoggedVLMClient (`monitoring/logger.py`)
VLMClient-обёртка с автоматическим логированием.
- Логирует: model, prompt_length, has_images, latency_ms, error
- `get_stats()` — агрегированная статистика

### CLI Dashboard (`monitoring/dashboard.py`)
Терминальный дашборд:
- Статус vLLM (health check)
- GPU загрузка (nvidia-smi: VRAM, utilization, temp, power)
- Статистика запросов (total, errors, p50/p95 latency)
- Watch mode с автообновлением

## Бизнес-логика
- Лог-файл — append-only JSON Lines (одна строка = один запрос)
- Стриминговые запросы не логируются (сложно измерить)
- GPU info парсится из `nvidia-smi --query-gpu`

## API
```bash
# Dashboard
python3 monitoring/dashboard.py                    # одноразовый
python3 monitoring/dashboard.py --watch 5          # каждые 5 сек

# Логирование в коде
from monitoring.logger import LoggedVLMClient
client = LoggedVLMClient(log_file="logs/requests.jsonl")
```

## Зависимости
- httpx (для health check)
- nvidia-smi (системная утилита)
- client.vllm_client (внутренняя)
