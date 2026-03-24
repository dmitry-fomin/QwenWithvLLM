#!/bin/bash
set -e

# Использование: HF_TOKEN=xxx bash scripts/download_model.sh Qwen/Qwen3-VL-8B-Instruct
# или: bash scripts/download_model.sh OpenGVLab/InternVL2_5-26B

REPO_ID="${1:-Qwen/Qwen3-VL-8B-Instruct}"
MODEL_DIR="${2:-./models/${REPO_ID##*/}}"
HF_TOKEN="${HF_TOKEN:-}"

echo "========================================="
echo "Model Download"
echo "========================================="
echo "Repository: $REPO_ID"
echo "Local path: $MODEL_DIR"
echo ""

# Проверка Python окружения
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated"
    echo "Run: source venv/bin/activate"
    exit 1
fi

# Убеждаемся что huggingface_hub установлен
echo "Installing huggingface_hub..."
pip install --quiet huggingface-hub

# Создаём директорию
mkdir -p "$(dirname "$MODEL_DIR")"

echo ""
echo "Downloading $REPO_ID to $MODEL_DIR..."
echo "(This may take a while for large models ~50-150 GB)"
echo ""

python3 << EOF
import sys
from huggingface_hub import snapshot_download

repo_id = "$REPO_ID"
local_dir = "$MODEL_DIR"
hf_token = "$HF_TOKEN"

print(f"Downloading {repo_id}...")
print(f"Output directory: {local_dir}")
print()

try:
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=hf_token if hf_token else None,
        # Пропускаем ненужные форматы
        ignore_patterns=[
            '*.msgpack',
            '*.h5',
            'flax_model*',
            'tf_model*',
        ],
        resume_download=True,  # Продолжить при обрыве
        local_files_only=False,
    )
    print()
    print(f"✓ Download complete: {downloaded_path}")

    # Размер
    import os
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, dirnames, filenames in os.walk(local_dir)
        for filename in filenames
    )
    print(f"Total size: {total_size / (1024**3):.1f} GB")

except Exception as e:
    print(f"❌ Error: {e}", file=sys.stderr)
    sys.exit(1)
EOF

echo ""
echo "✓ Done!"
echo ""
echo "Next step:"
echo "  bash scripts/start_vllm.sh configs/qwen3vl_8b.env"
