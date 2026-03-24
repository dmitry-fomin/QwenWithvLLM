#!/bin/bash
set -e

echo "========================================="
echo "vLLM Multi-Model Setup"
echo "========================================="

# Проверка OS
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "⚠️  Этот скрипт предназначен для Linux (Ubuntu/Debian)"
    echo "Для macOS и Windows требуется ручная установка"
    exit 1
fi

echo "✓ Detected: Linux"
uname -a

# Проверка GPU
echo ""
echo "=== GPU Check ==="
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA drivers not found. Install CUDA first."
    exit 1
fi
nvidia-smi

# Python версия
echo ""
echo "=== Python Setup ==="
PYTHON_VERSION=$(python3 --version 2>&1)
echo "Python version: $PYTHON_VERSION"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null; then
    echo "❌ Python 3.10+ required"
    exit 1
fi

# Виртуальное окружение
VENV_PATH="venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment at $VENV_PATH..."
    python3 -m venv "$VENV_PATH"
fi

echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Upgrade pip
echo ""
echo "=== Installing/Upgrading pip, setuptools, wheel ==="
pip install --upgrade pip setuptools wheel

# UV для быстрой установки
echo ""
echo "=== Installing uv (fast pip replacement) ==="
pip install uv

# vLLM nightly для поддержки Qwen3-VL и последних оптимизаций
echo ""
echo "=== Installing vLLM (nightly for Qwen3-VL support) ==="
echo "Выполняю: uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly"
uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly

# Или стабильная версия (если nightly не нужна):
# uv pip install -U vllm

# Зависимости
echo ""
echo "=== Installing dependencies ==="
uv pip install -r requirements.txt

echo ""
echo "========================================="
echo "✓ Setup Complete!"
echo "========================================="
echo ""
echo "Далее:"
echo "1. Скачайте модель:"
echo "   HF_TOKEN=xxx bash scripts/download_model.sh Qwen/Qwen3-VL-8B-Instruct"
echo ""
echo "2. Запустите vLLM сервер:"
echo "   bash scripts/start_vllm.sh configs/qwen3vl_8b.env"
echo ""
echo "3. В другом терминале, на клиентской машине:"
echo "   pip install -r requirements-client.txt"
echo "   python3 client/examples/screenshot_ocr.py"
echo ""
