"""
Пример: Параллельная обработка нескольких запросов (async).

Используется для обработки большого количества изображений/промптов одновременно.
Намного быстрее, чем последовательная обработка.

Использование:
    python3 client/examples/batch_requests.py
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from client import AsyncVLMClient


async def process_single_image(
    client: AsyncVLMClient,
    image_url: str,
    prompt: str = "Describe this image in detail.",
) -> Tuple[str, str]:
    """
    Асинхронно обработать одно изображение.

    Returns:
        Кортеж (image_url, result)
    """
    print(f"  Processing: {image_url[:50]}...")
    result = await client.chat(
        prompt=prompt,
        images=[image_url],
        max_tokens=512,
        temperature=0.3,
    )
    return image_url, result


async def batch_ocr(image_urls: List[str], model_name: str = "qwen3-vl-8b"):
    """
    Обработать множество изображений параллельно для OCR.

    Args:
        image_urls: Список URLs изображений
        model_name: Имя модели на сервере
    """
    client = AsyncVLMClient(model_name=model_name)

    prompt = "Extract all text from this image. Return only the text without explanations."

    print(f"Processing {len(image_urls)} images in parallel...")
    print()

    tasks = [
        process_single_image(client, url, prompt)
        for url in image_urls
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    print("\n=== Results ===\n")
    for image_url, result in results:
        if isinstance(result, Exception):
            print(f"❌ {image_url}: {result}")
        else:
            print(f"Image: {image_url[:50]}...")
            print(f"Text:\n{result}\n")
            print("-" * 60)
            print()


async def batch_classification(
    images_with_context: List[Tuple[str, str]],
    model_name: str = "qwen3-vl-8b",
):
    """
    Классифицировать изображения параллельно.

    Args:
        images_with_context: Список (image_url, description) пар
        model_name: Имя модели
    """
    client = AsyncVLMClient(model_name=model_name)

    print(f"Classifying {len(images_with_context)} images...")
    print()

    async def classify_image(url: str, context: str) -> Tuple[str, str]:
        result = await client.chat(
            prompt=f"Context: {context}\n\nClassify this image into one of these categories: A, B, C, or D. Explain your choice.",
            images=[url],
            max_tokens=256,
            temperature=0.2,
        )
        return url, result

    tasks = [
        classify_image(url, ctx)
        for url, ctx in images_with_context
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    print("=== Classifications ===\n")
    for (url, context), (_, result) in zip(images_with_context, results):
        if isinstance(result, Exception):
            print(f"❌ Error: {result}")
        else:
            print(f"Image: {url[:50]}...")
            print(f"Context: {context}")
            print(f"Classification: {result}")
            print()


async def benchmark_parallel_vs_sequential():
    """
    Сравнить скорость параллельной и последовательной обработки.
    """
    import time

    client = AsyncVLMClient()
    test_images = [
        "https://via.placeholder.com/300x300.png?text=Test+1",
        "https://via.placeholder.com/300x300.png?text=Test+2",
        "https://via.placeholder.com/300x300.png?text=Test+3",
    ]

    print("=== Performance Benchmark ===\n")

    # Последовательная обработка
    print("Sequential processing...")
    start = time.time()

    for img in test_images:
        await process_single_image(
            client, img, "Describe this image briefly."
        )

    sequential_time = time.time() - start
    print(f"Sequential time: {sequential_time:.2f}s\n")

    # Параллельная обработка
    print("Parallel processing...")
    start = time.time()

    tasks = [
        process_single_image(client, img, "Describe this image briefly.")
        for img in test_images
    ]
    await asyncio.gather(*tasks)

    parallel_time = time.time() - start
    print(f"Parallel time: {parallel_time:.2f}s\n")

    speedup = sequential_time / parallel_time
    print(f"Speedup: {speedup:.2f}x faster with parallelization")


async def main():
    """Основная функция примеров."""
    print("=" * 60)
    print("VLM Batch Processing Demo (Async)")
    print("=" * 60)
    print()

    try:
        # Пример 1: Параллельный OCR
        print("1. Batch OCR Example")
        print("-" * 60)
        test_images = [
            f"https://via.placeholder.com/400x300.png?text=Document+{i}"
            for i in range(1, 4)
        ]
        await batch_ocr(test_images)

        # Пример 2: Классификация
        print("\n2. Batch Classification Example")
        print("-" * 60)
        images_for_classification = [
            ("https://via.placeholder.com/200x200.png?text=Item+1", "Product photo"),
            ("https://via.placeholder.com/200x200.png?text=Item+2", "Product photo"),
            ("https://via.placeholder.com/200x200.png?text=Item+3", "Product photo"),
        ]
        await batch_classification(images_for_classification)

        # Пример 3: Бенчмарк
        print("\n3. Performance Benchmark")
        print("-" * 60)
        await benchmark_parallel_vs_sequential()

    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("Make sure vLLM server is running:")
        print("  bash scripts/start_vllm.sh configs/qwen3vl_8b.env")


if __name__ == "__main__":
    # Запуск async функции
    asyncio.run(main())
