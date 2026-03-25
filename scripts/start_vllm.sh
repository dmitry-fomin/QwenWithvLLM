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

# Настройка путей к CUDA библиотекам (решает: libcudart.so.12: cannot open shared object file)
CUDA_LIB_PATHS="/usr/local/cuda/lib64:/usr/local/cuda-12/lib64:/usr/local/cuda-12.4/lib64"
export LD_LIBRARY_PATH="${CUDA_LIB_PATHS}:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# Уменьшает фрагментацию GPU-памяти (рекомендация из PyTorch OOM ошибок)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Настройка vLLM движка (V1 всё еще экспериментальный для TP=4)
# Если VLLM_USE_V1=0 (по умолчанию или из конфига), используется стабильный движок V0.
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

if [ "$VLLM_USE_V1" == "1" ]; then
    echo "Engine: vLLM V1 (Experimental/New)"
else
    echo "Engine: vLLM V0 (Stable/Classic)"
fi
echo ""

echo "Starting vLLM API server..."
if [ "$VLLM_USE_V1" == "1" ]; then
    echo "Using experimental vLLM V1 engine"
else
    echo "Using stable vLLM V0 engine"
fi
echo "Base URL will be: http://0.0.0.0:8000"
echo ""

# Строим аргументы как bash-массив (решает проблему с JSON и кавычками в EXTRA_ARGS)
declare -a CMD_ARGS=(
    --model "$RESOLVED_MODEL"
    --served-model-name "$MODEL_NAME"
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE
    --max-model-len $MAX_MODEL_LEN
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION
    --dtype $DTYPE
    --host 0.0.0.0
    --port 8000
    --uvicorn-log-level info
)

# MM_IMAGE_LIMIT — отдельная переменная конфига (не в EXTRA_ARGS)
# Позволяет корректно передать JSON без проблем с экранированием
if [ -n "$MM_IMAGE_LIMIT" ]; then
    CMD_ARGS+=(--limit-mm-per-prompt "{\"image\": $MM_IMAGE_LIMIT}")
fi

# Остальные флаги из EXTRA_ARGS (простые флаги без JSON-значений)
if [ -n "$EXTRA_ARGS" ]; then
    read -ra EXTRA_ARGS_ARRAY <<< "$EXTRA_ARGS"
    CMD_ARGS+=("${EXTRA_ARGS_ARRAY[@]}")
fi

# Проверка на флаг фонового режима (--bg)
BACKGROUND=0
NEW_ARGS=()
for arg in "${CMD_ARGS[@]}"; do
    if [ "$arg" == "--bg" ]; then
        BACKGROUND=1
    else
        NEW_ARGS+=("$arg")
    fi
done

# Если передан аргумент скрипту напрямую (не через CMD_ARGS в массиве, а как $2, $3...)
# Но в текущей версии скрипта мы всё собираем в CMD_ARGS.
# Добавим обработку --bg как первого или второго аргумента самого скрипта bash
if [[ "$*" == *"--bg"* ]]; then
    BACKGROUND=1
fi

if [ "$BACKGROUND" == "1" ]; then
    LOG_FILE="vllm_${MODEL_NAME// /_}.log"
    echo "🚀 Starting vLLM in background..."
    echo "Log file: $LOG_FILE"
    
    # Рекурсивно вызываем тот же скрипт, но без флага --bg, через nohup
    # Или просто запускаем python через nohup
    nohup python3 -m vllm.entrypoints.openai.api_server "${CMD_ARGS[@]}" > "$LOG_FILE" 2>&1 &
    
    PID=$!
    echo "vLLM started with PID: $PID"
    echo "You can monitor logs with: tail -f $LOG_FILE"
    echo ""
    echo "To stop it, use: kill $PID"
    exit 0
fi

python3 -m vllm.entrypoints.openai.api_server "${CMD_ARGS[@]}"

# Примечание: Скрипт по умолчанию остаётся в foreground. 
# Для работы в фоне используйте флаг --bg:
# bash scripts/start_vllm.sh configs/qwen3vl_8b.env --bg
#
# Или используйте tmux (если установлен на сервере):
# tmux new -s vllm
# bash scripts/start_vllm.sh configs/qwen3vl_8b.env
