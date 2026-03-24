# QwenWithvLLM — Vision-Language Models через vLLM

Полнофункциональная система для запуска мультимодальных LLM (Qwen3-VL, InternVL и других) на локальном GPU сервере с 2x RTX 5090 или любом другом GPU.

**Особенности:**
- 🚀 Поддержка 6+ моделей VLM под 64 GB VRAM
- 🔄 Быстрое переключение между моделями (рестарт сервера ~30 сек)
- 📸 Отличная поддержка OCR, распознавания UI, анализа скриншотов
- ⚡ Асинхронный клиент для параллельной обработки изображений
- 🛠️ OpenAI-совместимый API (работает с OpenAI SDK)
- 🐳 Нативный Python (без Docker)
- 📚 Полные примеры кода для всех сценариев

---

## Поддерживаемые модели

### BF16 (максимальное качество)

| Модель | VRAM | Лучшее для |
|---|---|---|
| **Qwen3-VL-8B** | ~17 GB | ⭐ OCR, UI-анализ, скриншоты — **рекомендуется** |
| **Qwen2.5-VL-7B** | ~15 GB | OCR, документы, UI-понимание |
| **MiniCPM-V 2.6** | ~17 GB | Быстрый OCR до 1.8M пикселей |
| **Pixtral-12B** | ~25 GB | Чарты, диаграммы, документы |
| **InternVL2.5-26B** | ~52 GB | Тяжелая OCR, мультиизображения |

### INT4 Quantized (больше моделей, немного медленнее)

| Модель | VRAM | Лучшее для |
|---|---|---|
| **Qwen2.5-VL-72B AWQ** | ~40 GB | Максимальное качество OCR/UI |
| **Qwen3-VL-32B AWQ** | ~18-20 GB | Баланс размера и качества |

**Рекомендация для начинающих:** Qwen3-VL-8B BF16 — лучший выбор.

---

## Требования

### Железо
- **GPU:** 2x RTX 5090 (64 GB VRAM) или аналог (4x A100 80GB, 8x H100 40GB и т.д.)
- **CPU RAM:** 256 GB
- **Диск:** ~160 GB для моделей
- **ОС:** Linux (Ubuntu 22.04+), macOS, Windows (через WSL2)

### ПО
- Python 3.10+
- NVIDIA CUDA 12.1+ (для GPU)
- pip / uv

---

## Быстрый старт

### 1️⃣ Клонируем репозиторий и входим в директорию

```bash
cd /path/to/QwenWithvLLM
```

### 2️⃣ Запускаем установку

```bash
bash scripts/setup_server.sh
```

Это установит:
- Python виртуальное окружение
- vLLM (nightly для поддержки Qwen3-VL)
- Все зависимости
- Проверит GPU

### 3️⃣ Загружаем модель

```bash
# Qwen3-VL-8B (рекомендуется)
HF_TOKEN=your_token bash scripts/download_model.sh Qwen/Qwen3-VL-8B-Instruct

# Или другую модель
HF_TOKEN=your_token bash scripts/download_model.sh OpenGVLab/InternVL2_5-26B
```

**Получить HF токен:** https://huggingface.co/settings/tokens

Это займет 30-90 минут в зависимости от размера модели и скорости сети.

### 4️⃣ Запускаем vLLM сервер

```bash
bash scripts/start_vllm.sh configs/qwen3vl_8b.env
```

Ждёте ~5 минут загрузки модели в GPU.

Когда видите:
```
INFO:     Application startup complete
Uvicorn running on http://0.0.0.0:8000
```

Сервер готов к работе.

### 5️⃣ В другом терминале — устанавливаем клиентские зависимости

```bash
pip install -r requirements-client.txt
```

### 6️⃣ Запускаем примеры

```bash
# OCR скриншота
python3 client/examples/screenshot_ocr.py

# Анализ UI элементов
python3 client/examples/ui_describe.py

# Работа с несколькими изображениями
python3 client/examples/multi_image.py

# Параллельная обработка (async)
python3 client/examples/batch_requests.py
```

---

## Использование API

### Синхронный клиент

```python
from client import VLMClient

client = VLMClient(
    base_url="http://localhost:8000/v1",
    model_name="qwen3-vl-8b"
)

# Текстовый запрос
response = client.chat(
    prompt="What is 2+2?"
)
print(response)

# Запрос с изображением
response = client.chat(
    prompt="Describe this image.",
    images=["https://example.com/image.jpg"],
    max_tokens=512,
    temperature=0.3,
)
print(response)

# Несколько изображений
response = client.chat(
    prompt="Compare these images.",
    images=[
        "/path/to/local/image.jpg",
        "https://example.com/image2.jpg",
    ]
)

# Со стримингом
for token in client.chat(
    prompt="Tell me a story.",
    stream=True,
    max_tokens=1024
):
    print(token, end="", flush=True)
```

### Асинхронный клиент (параллельная обработка)

```python
import asyncio
from client import AsyncVLMClient

async def main():
    client = AsyncVLMClient(model_name="qwen3-vl-8b")

    # Параллельная обработка нескольких изображений
    tasks = [
        client.chat(
            prompt="Extract text.",
            images=[f"image_{i}.jpg"]
        )
        for i in range(10)
    ]

    results = await asyncio.gather(*tasks)
    for result in results:
        print(result)

asyncio.run(main())
```

### OpenAI SDK (совместимость)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",  # vLLM не требует реального ключа
)

response = client.chat.completions.create(
    model="qwen3-vl-8b",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image.jpg"
                }
            },
            {
                "type": "text",
                "text": "What is in this image?"
            }
        ]
    }],
    max_tokens=512
)

print(response.choices[0].message.content)
```

---

## Переключение между моделями

### Способ 1: Рестарт вручную

1. Остановите текущий сервер (Ctrl+C)
2. Загрузите новую модель (если её ещё нет):
   ```bash
   HF_TOKEN=xxx bash scripts/download_model.sh OpenGVLab/InternVL2_5-26B
   ```
3. Запустите с новым конфигом:
   ```bash
   bash scripts/start_vllm.sh configs/internvl25_26b.env
   ```

### Способ 2: Автоматический скрипт

```bash
bash scripts/switch_model.sh configs/internvl25_26b.env
```

Занимает ~30 секунд.

---

## Примеры использования

### Пример 1: OCR документа

```python
from client import VLMClient

client = VLMClient()

text = client.chat(
    prompt="Extract all text from this document.",
    images=["document.pdf_page1.png"],
    max_tokens=2048,
    temperature=0.1,  # низкая для точности
)

print(text)
```

### Пример 2: Описание UI элементов

```python
description = client.chat(
    prompt="""List all interactive elements:
    - Buttons with labels
    - Text fields
    - Dropdowns
    - Links
    Format as markdown list.""",
    images=["screenshot.png"],
    temperature=0.2,
)
print(description)
```

### Пример 3: Сравнение изображений

```python
comparison = client.chat(
    prompt="""Compare these two UI designs:
    1. What are the differences?
    2. Which is better and why?
    3. What was improved?""",
    images=["before.png", "after.png"],
    max_tokens=1024,
)
```

### Пример 4: Параллельная OCR (100 изображений)

```python
import asyncio
from client import AsyncVLMClient

async def batch_ocr(image_paths):
    client = AsyncVLMClient()

    tasks = [
        client.chat(
            prompt="Extract all text.",
            images=[img],
            max_tokens=1024
        )
        for img in image_paths
    ]

    return await asyncio.gather(*tasks)

# Обработка 100 изображений параллельно
results = asyncio.run(batch_ocr([f"doc{i:03d}.png" for i in range(100)]))
```

---

## Конфигурация vLLM

Каждый конфиг в `configs/*.env` содержит параметры запуска:

```bash
MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct      # HF hub ID или локальный путь
MODEL_NAME=qwen3-vl-8b                    # Имя модели для API
TENSOR_PARALLEL_SIZE=2                    # Распределение по GPU
MAX_MODEL_LEN=16384                       # Максимальный контекст
GPU_MEMORY_UTILIZATION=0.90               # % использования VRAM
DTYPE=bfloat16                            # bfloat16, float16, auto
EXTRA_ARGS="..."                          # Дополнительные флаги vLLM
```

### Tuning параметров

**Если вам не хватает VRAM:**
```bash
GPU_MEMORY_UTILIZATION=0.85               # снизьте с 0.90
MAX_MODEL_LEN=8192                        # уменьшите контекст
DTYPE=float16                             # может быть чуть экономнее
```

**Если нужна скорость:**
```bash
GPU_MEMORY_UTILIZATION=0.95               # поднимите
DTYPE=bfloat16                            # быстрее всего
TENSOR_PARALLEL_SIZE=2                    # 2 GPU = 2x параллелизм
```

---

## Решение проблем

### ❌ "CUDA out of memory"

- Снизьте `GPU_MEMORY_UTILIZATION` (0.90 → 0.85)
- Уменьшите `MAX_MODEL_LEN` (16384 → 8192)
- Используйте меньшую модель (Qwen3-VL-8B вместо InternVL2.5-26B)
- Запустите только одну задачу за раз

### ❌ "Model not found"

```bash
# Проверьте что модель загружена
ls -la models/

# Или загрузите её
HF_TOKEN=xxx bash scripts/download_model.sh Qwen/Qwen3-VL-8B-Instruct
```

### ❌ "Connection refused"

- Убедитесь что сервер запущен: `bash scripts/start_vllm.sh`
- Проверьте порт: `curl http://localhost:8000/health`
- Проверьте URL в клиенте: должно быть `http://localhost:8000/v1`

### ❌ "HuggingFace token error"

```bash
# Убедитесь что токен установлен
echo $HF_TOKEN

# Или установите его
export HF_TOKEN="hf_..."
```

### ❌ Медленная обработка

- Используйте параллельный клиент (AsyncVLMClient) вместо синхронного
- Запустите модель с BF16 вместо INT4
- Используйте меньшую модель

---

## Архитектура

```
┌─────────────────────────────────────┐
│   Клиент (Python/JavaScript/etc)   │
│  (client/*.py или OpenAI SDK)       │
└────────────┬────────────────────────┘
             │ HTTP
             │ OpenAI API compatible
             ▼
┌─────────────────────────────────────┐
│      vLLM Server                    │
│  (scripts/start_vllm.sh)            │
│  Port 8000                          │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│    Vision-Language Model            │
│  (Qwen3-VL, InternVL, etc)         │
│  На 2x RTX 5090 (64 GB VRAM)        │
└─────────────────────────────────────┘
```

---

## Производительность

Примерные скорости на 2x RTX 5090:

| Модель |速度 (токен/сек) | Задержка первого токена |
|---|---|---|
| Qwen3-VL-8B | ~15-20 | 2-3 сек |
| Qwen2.5-VL-7B | ~18-25 | 2-3 сек |
| InternVL2.5-26B | ~8-12 | 3-5 сек |
| Qwen2.5-VL-72B AWQ | ~5-10 | 4-6 сек |

> Скорость зависит от разрешения изображения, размера контекста и батча.
> С parallel processing (asyncio) можно обрабатывать несколько изображений одновременно.

---

## Лицензия

MIT

---

## Контрибьюшены

Issues и PRs приветствуются!

---

## Дополнительные ресурсы

- [vLLM Документация](https://docs.vllm.ai)
- [Qwen Model Hub](https://huggingface.co/Qwen)
- [InternVL](https://huggingface.co/OpenGVLab)
- [OpenAI Python SDK](https://github.com/openai/openai-python)

---

**Готовы начать?**

```bash
bash scripts/setup_server.sh
```
