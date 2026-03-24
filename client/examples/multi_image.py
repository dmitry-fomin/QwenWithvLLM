"""
Пример: Работа с несколькими изображениями.

Модели Qwen и InternVL поддерживают несколько изображений в одном запросе.

Использование:
    python3 client/examples/multi_image.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from client import VLMClient


def compare_images():
    """Сравнение двух изображений."""
    client = VLMClient()

    print("=== Image Comparison ===")
    print()

    images = [
        "https://via.placeholder.com/300x300.png?text=Image+1",
        "https://via.placeholder.com/300x300.png?text=Image+2",
    ]

    result = client.chat(
        prompt="""Compare these two images:
1. What are the differences?
2. What is common?
3. If these are versions of the same thing, what changed?""",
        images=images,
        max_tokens=512,
        temperature=0.3,
    )
    print(result)
    print()


def analyze_document_pages():
    """Анализ документа из нескольких страниц."""
    client = VLMClient()

    print("=== Multi-Page Document Analysis ===")
    print()

    pages = [
        "https://via.placeholder.com/400x600.png?text=Page+1",
        "https://via.placeholder.com/400x600.png?text=Page+2",
        "https://via.placeholder.com/400x600.png?text=Page+3",
    ]

    result = client.chat(
        prompt="""You are analyzing a multi-page document. Review these pages and:
1. Extract the main topic/subject
2. List all key points across pages
3. Summarize the content in 3-5 sentences
4. Note any page numbers or sections mentioned""",
        images=pages,
        max_tokens=1024,
        temperature=0.3,
    )
    print(result)
    print()


def sequence_analysis():
    """Анализ последовательности событий/состояний."""
    client = VLMClient()

    print("=== Sequence/Timeline Analysis ===")
    print()

    sequence = [
        "https://via.placeholder.com/300x200.png?text=Step+1",
        "https://via.placeholder.com/300x200.png?text=Step+2",
        "https://via.placeholder.com/300x200.png?text=Step+3",
    ]

    result = client.chat(
        prompt="""These images show a sequence of states or steps. Describe:
1. What happens in each image?
2. What is the logical order/progression?
3. What is the overall process or workflow?
4. Are there any gaps or missing steps?""",
        images=sequence,
        max_tokens=1024,
        temperature=0.3,
    )
    print(result)
    print()


def grid_analysis():
    """Анализ изображений в сетке."""
    client = VLMClient()

    print("=== Grid/Gallery Analysis ===")
    print()

    grid = [
        f"https://via.placeholder.com/200x200.png?text=Item+{i+1}"
        for i in range(4)
    ]

    result = client.chat(
        prompt="""Analyze these items in a grid layout:
1. What is each item? (provide brief description)
2. Do they belong to the same category? Why?
3. How would you organize or categorize them?
4. Any patterns or relationships between items?""",
        images=grid,
        max_tokens=1024,
        temperature=0.3,
    )
    print(result)
    print()


def streaming_example():
    """Пример со стрингом длинного ответа."""
    client = VLMClient()

    print("=== Streaming Response Example ===")
    print()

    images = [
        "https://via.placeholder.com/400x300.png?text=Image+1",
        "https://via.placeholder.com/400x300.png?text=Image+2",
    ]

    prompt = """Analyze these images in detail:
1. Describe each image completely
2. Compare them
3. Discuss their relationship
4. Provide recommendations based on what you see"""

    print("Streaming response:")
    print("-" * 40)

    # Использаем стриминг для получения ответа по частям
    for chunk in client.chat(
        prompt=prompt,
        images=images,
        max_tokens=2048,
        stream=True,
    ):
        print(chunk, end="", flush=True)

    print("\n" + "-" * 40)
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("VLM Multi-Image Analysis Demo")
    print("=" * 60)
    print()

    try:
        compare_images()
        analyze_document_pages()
        sequence_analysis()
        grid_analysis()
        streaming_example()

    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("Make sure vLLM server is running:")
        print("  bash scripts/start_vllm.sh configs/qwen3vl_8b.env")
