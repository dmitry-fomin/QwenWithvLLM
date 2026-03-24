"""
Пример: Описание и анализ UI элементов.

Идеально для автоматизации рабочего стола, тестирования приложений,
описания интерфейсов.

Использование:
    python3 client/examples/ui_describe.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from client import VLMClient


def analyze_ui(
    image_url: str = "https://via.placeholder.com/800x600.png?text=UI+Screenshot",
    server_url: str = "http://localhost:8000/v1",
    model_name: str = "qwen3-vl-8b",
):
    """
    Анализировать UI элементы на скриншоте.

    Args:
        image_url: URL или путь к изображению
        server_url: URL vLLM сервера
        model_name: Имя модели
    """
    client = VLMClient(
        base_url=server_url,
        model_name=model_name,
    )

    print(f"Analyzing UI: {image_url}")
    print()

    # Список элементов
    print("=== UI Elements ===")
    result = client.chat(
        prompt="""List all interactive UI elements visible in this screenshot:
- Buttons (with labels)
- Text inputs/fields
- Dropdowns/selectors
- Links
- Icons
- Other interactive elements

Format as a numbered list with location hints (top-left, center, bottom-right, etc).""",
        images=[image_url],
        max_tokens=1024,
        temperature=0.2,
    )
    print(result)
    print()

    # Распознавание назначения
    print("=== Purpose & Function ===")
    result = client.chat(
        prompt="""What is the purpose of this UI/application?
Describe:
1. Main function/goal
2. Target users
3. Key features visible
4. Workflow or user journey""",
        images=[image_url],
        max_tokens=1024,
        temperature=0.3,
    )
    print(result)
    print()

    # Рекомендации по улучшению
    print("=== Improvements ===")
    result = client.chat(
        prompt="""Analyze this UI and suggest improvements:
1. Usability issues
2. Design problems
3. Accessibility concerns
4. Layout recommendations

Be specific and actionable.""",
        images=[image_url],
        max_tokens=1024,
        temperature=0.3,
    )
    print(result)


def test_multiple_images():
    """Пример с несколькими изображениями (сравнение UI)."""
    client = VLMClient()

    print("=== Multi-Image Comparison ===")
    print()

    result = client.chat(
        prompt="""Compare these two UI screenshots:
1. What are the differences?
2. Which one is better designed and why?
3. What improvements were made from first to second?""",
        images=[
            "https://via.placeholder.com/400x300.png?text=Before",
            "https://via.placeholder.com/400x300.png?text=After",
        ],
        max_tokens=1024,
        temperature=0.3,
    )
    print(result)


if __name__ == "__main__":
    print("=" * 60)
    print("VLM UI Analysis Demo")
    print("=" * 60)
    print()

    try:
        analyze_ui()

        print("\n" + "=" * 60)
        print("Testing multi-image comparison...")
        print("=" * 60 + "\n")

        test_multiple_images()

    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("Make sure vLLM server is running:")
        print("  bash scripts/start_vllm.sh configs/qwen3vl_8b.env")
        print()
        print("Then try again:")
        print("  python3 client/examples/ui_describe.py")
