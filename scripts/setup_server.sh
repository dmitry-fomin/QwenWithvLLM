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

# Шаг 1: PyTorch с явной версией CUDA 12.4
# ВАЖНО: не используй --torch-backend=auto — это может поставить torch под CUDA 13,
# которая несовместима с vLLM nightly (скомпилирован под CUDA 12, ищет libcudart.so.12).
# torch cu124 бандлит libcudart.so.12 внутри пакета — работает даже если в системе CUDA 13.
echo ""
echo "=== Installing PyTorch 2.5.1 with CUDA 12.4 (bundled libcudart.so.12) ==="
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Шаг 2: vLLM nightly для поддержки Qwen3-VL
echo ""
echo "=== Installing vLLM nightly (Qwen3-VL support) ==="
pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly

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
