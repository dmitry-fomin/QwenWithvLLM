# Быстрый старт (3 минуты)

## Шаг 1: Установка (требует интернета, ~5 мин)

```bash
bash scripts/setup_server.sh
```

Это установит Python зависимости и vLLM.

## Шаг 2: Загрузка модели (требует времени, ~1-2 часа)

```bash
export HF_TOKEN="your_huggingface_token"
bash scripts/download_model.sh Qwen/Qwen3-VL-8B-Instruct
```

**Получить токен:** https://huggingface.co/settings/tokens

## Шаг 3: Запуск сервера

В отдельном терминале:

```bash
bash scripts/start_vllm.sh configs/qwen3vl_8b.env
```

Ждите ~5 мин пока модель загрузится в GPU. Когда видите `Application startup complete` — готово!

## Шаг 4: Тест клиента

В третьем терминале:

```bash
pip install -r requirements-client.txt
python3 client/examples/screenshot_ocr.py
```

## ✅ Готово!

Теперь можно использовать:

```python
from client import VLMClient

client = VLMClient()
response = client.chat(
    prompt="Describe this image.",
    images=["image.jpg"]
)
print(response)
```

---

## Переключение моделей

Если захотите попробовать другую модель:

```bash
# Загружаем другую модель
export HF_TOKEN="your_token"
bash scripts/download_model.sh OpenGVLab/InternVL2_5-26B

# Переключаемся
bash scripts/switch_model.sh configs/internvl25_26b.env
```

---

## Доступные модели

- `qwen3vl_8b.env` — Qwen3-VL-8B (**рекомендуется**)
- `qwen25vl_7b.env` — Qwen2.5-VL-7B
- `qwen25vl_72b_awq.env` — Qwen2.5-VL-72B (лучше, но медленнее)
- `internvl25_26b.env` — InternVL2.5-26B
- `minicpmv_2_6.env` — MiniCPM-V 2.6
- `pixtral_12b.env` — Pixtral-12B

---

## Проблемы?

Читайте полный README.md в разделе "Решение проблем".
