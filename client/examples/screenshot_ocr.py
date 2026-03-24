"""
Пример: OCR скриншота рабочего стола или документа.

Использование:
    python3 client/examples/screenshot_ocr.py

Требует установки pyautogui для скриншота (опционально):
    pip install pyautogui
"""

import sys
from pathlib import Path

# Добавляем parent directory в path для импорта client
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from client import VLMClient


def ocr_screenshot(
    image_path: str = None,
    server_url: str = "http://localhost:8000/v1",
    model_name: str = "qwen3-vl-8b",
):
    """
    Выполнить OCR на скриншоте или изображении.

    Args:
        image_path: Путь к изображению (если None, использует пример)
        server_url: URL vLLM сервера
        model_name: Имя модели на сервере
    """
    client = VLMClient(
        base_url=server_url,
        model_name=model_name,
    )

    # Если путь не передан, используем пример URL
    if image_path is None:
        print("Using sample image from web...")
        image_path = "https://via.placeholder.com/600x400.png?text=Sample+Screenshot"

    print(f"Image: {image_path}")
    print()

    # Простой OCR
    print("=== Simple OCR ===")
    result = client.chat(
        prompt="Extract all text from this image. Return only the text without explanations.",
        images=[image_path],
        max_tokens=1024,
        temperature=0.1,
    )
    print(result)
    print()

    # Детальный анализ UI
    print("=== Detailed Analysis ===")
    result = client.chat(
        prompt="""Analyze this screenshot in detail:
1. What application or interface is shown?
2. List all visible UI elements (buttons, text fields, etc.)
3. What is the current state of the application?
4. Describe the layout and design.""",
        images=[image_path],
        max_tokens=2048,
        temperature=0.3,
    )
    print(result)
    print()

    # Обнаружение текста
    print("=== Text Detection ===")
    result = client.chat(
        prompt="List all visible text in this image with their approximate locations (top/center/bottom, left/center/right).",
        images=[image_path],
        max_tokens=1024,
        temperature=0.2,
    )
    print(result)


def ocr_local_file():
    """OCR локального файла если он существует."""
    import os

    test_file = "test_screenshot.png"
    if os.path.exists(test_file):
        print(f"\nFound local test file: {test_file}")
        ocr_screenshot(test_file)
    else:
        print(f"\nTo test with local file, save an image as '{test_file}' in project root.")


if __name__ == "__main__":
    print("=" * 50)
    print("VLM Screenshot OCR Demo")
    print("=" * 50)
    print()

    # Попробуем найти локальный файл
    try:
        ocr_local_file()
    except FileNotFoundError:
        print("⚠️  No local test file found")

    # Используем пример с URL
    try:
        ocr_screenshot()
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("Make sure vLLM server is running:")
        print("  bash scripts/start_vllm.sh configs/qwen3vl_8b.env")
