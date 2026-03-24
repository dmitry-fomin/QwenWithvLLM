#!/bin/bash

# Скрипт для переключения между моделями
# Останавливает текущий vLLM процесс и запускает новый с другой моделью
# Использование: bash scripts/switch_model.sh configs/internvl25_26b.env

CONFIG_FILE="${1:-configs/qwen3vl_8b.env}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

set -a
source "$CONFIG_FILE"
set +a

echo "========================================="
echo "Switching Model"
echo "========================================="
echo "Current model: $(curl -s http://localhost:8000/v1/models 2>/dev/null | head -1 || echo 'Unknown')"
echo "New model:     $MODEL_NAME from $CONFIG_FILE"
echo ""

# Проверяем запущен ли vLLM
if ! pgrep -f "vllm.entrypoints.openai" > /dev/null; then
    echo "⚠️  vLLM server is not running"
    echo "Starting $MODEL_NAME..."
    bash scripts/start_vllm.sh "$CONFIG_FILE"
    exit 0
fi

echo "Stopping vLLM server..."
pkill -f "vllm.entrypoints.openai" || true

# Даём время на graceful shutdown
echo "Waiting for shutdown (5 seconds)..."
sleep 5

# Если не остановился, убиваем harder
if pgrep -f "vllm.entrypoints.openai" > /dev/null; then
    echo "Force stopping..."
    pkill -9 -f "vllm.entrypoints.openai" || true
    sleep 2
fi

echo ""
echo "Starting new model: $MODEL_NAME..."
bash scripts/start_vllm.sh "$CONFIG_FILE"
