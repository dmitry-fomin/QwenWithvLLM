# Запуск на Vast.ai

Vast.ai — площадка аренды GPU. Здесь описано как арендовать машину, настроить и запустить vLLM сервер.

---

## 1. Выбор GPU

Подбирай GPU под модель по VRAM:

| Модель                 | VRAM   | Что арендовать на Vast.ai     |
|------------------------|--------|-------------------------------|
| Qwen3-VL-8B (BF16)     | ~17 GB | 1x RTX 4090 (24 GB)           |
| Qwen2.5-VL-7B (BF16)   | ~15 GB | 1x RTX 4090 (24 GB)           |
| Pixtral-12B (BF16)     | ~25 GB | 1x A100 40GB SXM              |
| InternVL2.5-26B (BF16) | ~52 GB | 2x A100 40GB или 1x A100 80GB |
| Qwen2.5-VL-72B AWQ     | ~40 GB | 1x A100 80GB или 2x A100 40GB |

> Конфиги с `TENSOR_PARALLEL_SIZE=2` требуют 2 GPU. При аренде одной GPU измени этот параметр на `1` в соответствующем `.env` файле.

---

## 2. Выбор образа

При создании инстанса в поле **Docker Image** выбери:

```
pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
```

Почему именно этот:
- Содержит CUDA 12.4 + PyTorch — именно под эту версию собран vLLM
- Python 3.10+ включён
- Достаточно места для pip-установки зависимостей

Альтернатива:
```
nvidia/cuda:12.4.1-devel-ubuntu22.04
```

> [!CAUTION]
> **Не используй `vastai/pytorch_cuda-13.x`** — этот образ содержит CUDA 13, а vLLM nightly скомпилирован под CUDA 12 (`libcudart.so.12`). Запуск упадёт с `ImportError: libcudart.so.12: cannot open shared object file`.
>
> **Не используй** `vllm/vllm-openai` — там старая стабильная версия, несовместимая с Qwen3-VL.

---

## 3. Настройки инстанса

При аренде выстави:

- **Disk Space:** минимум 80 GB (модели весят 15–80 GB)
- **Exposed ports:** добавь порт `8000` — через него работает vLLM API

---

## 4. Первый запуск после подключения по SSH

### Клонируй репо

```bash
git clone https://github.com/dmitry-fomin/QwenWithvLLM
cd QwenWithvLLM
```

### Установка зависимостей

```bash
./scripts/setup_server.sh
```

Это создаст venv, установит vLLM nightly и все зависимости. Займёт 5–10 минут.

### Скачай модель

```bash
export HF_TOKEN="hf_..."

./scripts/download_model.sh Qwen/Qwen3-VL-8B-Instruct

./scripts/download_model.sh OpenGVLab/InternVL2_5-26B

./scripts/download_model.sh Qwen/Qwen2.5-VL-72B-Instruct-AWQ	
```

Токен получи на [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
Загрузка займёт 10–60 минут в зависимости от модели и канала.

### Запусти сервер в tmux

```bash
tmux new -s vllm

./scripts/start_vllm.sh configs/qwen3vl_8b.env

./scripts/start_vllm.sh configs/internvl25_26b.env

./scripts/start_vllm.sh configs/qwen25vl_72b_awq.env
```

Жди строчку:
```
INFO:     Application startup complete
```

Это значит сервер готов. Отсоединиться от tmux не убивая сервер: `Ctrl+B`, затем `D`.

---

## 5. Подключение с локальной машины

Vast.ai не пробрасывает порты напрямую — нужен SSH-туннель.

В интерфейсе Vast.ai нажми **Connect** на своём инстансе — там будет команда вида:
```
ssh -p 12345 root@ssh6.vast.ai
```

Добавь `-L 8000:localhost:8000` для туннеля:
```bash
ssh -p 12345 root@ssh6.vast.ai -L 8000:localhost:8000
```

Теперь на локальной машине `http://localhost:8000` → это vLLM на сервере.

### Проверка что сервер работает

```bash
curl http://localhost:8000/health
```

Должно вернуть `{"status":"ok"}` или просто HTTP 200.

### Подключение клиента

```python
from client import VLMClient

client = VLMClient(
    base_url="http://localhost:8000/v1",
    model_name="qwen3-vl-8b"
)

response = client.chat(
    prompt="Что на этом скриншоте?",
    images=["screenshot.png"]
)
print(response)
```

---

## 6. Если берёшь одну GPU вместо двух

Конфиги по умолчанию рассчитаны на `TENSOR_PARALLEL_SIZE=2`. Если у тебя одна GPU, создай копию конфига и измени параметр:

```bash
cp configs/qwen3vl_8b.env configs/qwen3vl_8b_single.env
# Открой файл и измени TENSOR_PARALLEL_SIZE=1
```

Или прямо в командной строке:
```bash
sed -i 's/TENSOR_PARALLEL_SIZE=2/TENSOR_PARALLEL_SIZE=1/' configs/qwen3vl_8b.env
./scripts/start_vllm.sh configs/qwen3vl_8b.env
```

---

## 7. Управление сессией

| Задача                   | Команда                                        |
|--------------------------|------------------------------------------------|
| Открыть tmux сессию vllm | `tmux attach -t vllm`                          |
| Остановить сервер        | В tmux: `Ctrl+C`                               |
| Переключить модель       | `./scripts/switch_model.sh configs/другой.env` |
| Посмотреть GPU нагрузку  | `watch -n1 nvidia-smi`                         |
| Проверить логи           | `tmux attach -t vllm`                          |

---

## 8. Экономия денег

- Останавливай инстанс когда не используешь (кнопка **Stop** в интерфейсе)
- Используй **On-Demand** для коротких задач, **Interruptible** на 30–50% дешевле для длинных batch-задач
- Для 8B модели хватает 1x RTX 4090 (~$0.3–0.5/час) вместо A100 (~$1–2/час)
- Диск тарифицируется отдельно — чисти модели которые не используешь

---

## 9. Типичные проблемы

### "No space left on device" при скачивании модели

Диска не хватает. В настройках инстанса выстави больше Disk Space или удали лишнее:
```bash
du -sh ~/.cache/huggingface/hub/*/
# Удали ненужные модели
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct
```

### Сервер запустился, но клиент не подключается

Проверь что SSH-туннель активен — соединение должно висеть в терминале. Если закрыл его, запусти снова.

### ImportError: libcudart.so.12: cannot open shared object file

CUDA Runtime не найдена в `LD_LIBRARY_PATH`. Это уже исправлено в `scripts/start_vllm.sh`, но если запускаешь vLLM вручную:

```bash
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-12/lib64:/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH"
python3 -m vllm.entrypoints.openai.api_server ...
```

Или найди где лежит библиотека:
```bash
find / -name "libcudart.so.12" 2>/dev/null
```

### "CUDA out of memory" при старте

- Снизь `GPU_MEMORY_UTILIZATION` до `0.85`
- Уменьши `MAX_MODEL_LEN` до `8192`
- Или выбери модель поменьше

### Инстанс завис / не отвечает

```bash
# Пересоздай tmux сессию
tmux kill-session -t vllm
tmux new -s vllm
./scripts/start_vllm.sh configs/qwen3vl_8b.env
```
