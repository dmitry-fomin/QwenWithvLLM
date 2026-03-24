#!/bin/bash
set -e

# Универсальный скрипт запуска vLLM с любым конфигом
# Использование: bash scripts/start_vllm.sh configs/qwen3vl_8b.env

CONFIG_FILE="${1:-configs/qwen3vl_8b.env}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    echo ""
    echo "Available configs:"
    ls -1 configs/*.env 2>/dev/null || echo "  (none found)"
    exit 1
fi

echo "========================================="
echo "Starting vLLM Server"
echo "========================================="
echo "Config: $CONFIG_FILE"
echo ""

# Загружаем конфиг
set -a
source "$CONFIG_FILE"
set +a

echo "Model: $MODEL_PATH"
echo "Name:  $MODEL_NAME"
echo "Tensor Parallel: $TENSOR_PARALLEL_SIZE"
echo "Max Context: $MAX_MODEL_LEN"
echo "GPU Memory: $GPU_MEMORY_UTILIZATION (of total)"
echo "Dtype: $DTYPE"
echo ""

# Проверка виртуального окружения
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "venv" ]; then
        echo "Activating virtual environment..."
        source venv/bin/activate
    else
        echo "❌ Virtual environment not found. Run: bash scripts/setup_server.sh"
        exit 1
    fi
fi

# Проверка GPU
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
echo ""

# Получаем полный путь к модели (поддержка HF hub ID или локальных путей)
if [[ "$MODEL_PATH" == /* ]]; then
    # Абсолютный путь
    RESOLVED_MODEL="$MODEL_PATH"
else
    # HuggingFace hub ID
    RESOLVED_MODEL="$MODEL_PATH"
fi

echo "Starting vLLM API server..."
echo "Base URL will be: http://0.0.0.0:8000"
echo ""

# Запуск vLLM с параметрами из конфига
python3 -m vllm.entrypoints.openai.api_server \
    --model "$RESOLVED_MODEL" \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --dtype $DTYPE \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level INFO \
    $EXTRA_ARGS

# Примечание: Скрипт остаётся в foreground. Для работы в background:
# bash scripts/start_vllm.sh configs/qwen3vl_8b.env &
# Или используйте tmux/screen для отдельной сессии
