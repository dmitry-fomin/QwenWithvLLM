# Модуль: scripts/ + configs/

## Назначение
Управление vLLM inference сервером: установка, загрузка моделей, запуск, переключение.

## Ключевые сущности

### Конфигурации (`configs/*.env`)
Каждый файл — конфиг для одной модели. Обязательные ключи:
- `MODEL_PATH` — HuggingFace repo_id или локальный путь
- `MODEL_NAME` — имя для `--served-model-name`
- `TENSOR_PARALLEL_SIZE` — количество GPU
- `MAX_MODEL_LEN` — максимальный контекст
- `GPU_MEMORY_UTILIZATION` — доля VRAM (0-1)
- `DTYPE` — bfloat16/float16/auto
- `EXTRA_ARGS` — дополнительные флаги vLLM

### Скрипты (`scripts/`)
- `setup_server.sh` — установка Python venv, uv, vLLM nightly
- `download_model.sh` — HF snapshot_download с resume
- `start_vllm.sh` — запуск vLLM, читает env-конфиг
- `switch_model.sh` — pkill текущего + restart с новым конфигом

## Бизнес-логика
- vLLM запускается как OpenAI-совместимый API на порту 8000
- `--trust-remote-code` обязателен для всех моделей
- Tensor parallelism=2 для 2x RTX 5090
- Модели не работают одновременно — только переключение

## Поддерживаемые модели
| Конфиг | Модель | VRAM |
|--------|--------|------|
| qwen3vl_8b.env | Qwen3-VL-8B | ~17 GB BF16 |
| qwen25vl_7b.env | Qwen2.5-VL-7B | ~15 GB BF16 |
| qwen25vl_72b_awq.env | Qwen2.5-VL-72B | ~40 GB INT4 |
| internvl25_26b.env | InternVL2.5-26B | ~52 GB BF16 |
| minicpmv_2_6.env | MiniCPM-V 2.6 | ~17 GB BF16 |
| pixtral_12b.env | Pixtral-12B | ~25 GB BF16 |

## Зависимости
- vLLM (≥0.9 nightly для Qwen3-VL)
- CUDA 12.1+
- nvidia-smi (для проверки GPU)
