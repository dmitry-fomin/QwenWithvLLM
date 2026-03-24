"""
Утилита для захвата скриншотов рабочего стола и OCR через VLM.

Зависимости:
    pip install Pillow mss

Поддерживает:
- Полный экран
- Конкретный монитор
- Область экрана (region)
- Конкретное окно (через название)
- Периодический захват (watch mode)

Использование:
    python3 tools/screenshot.py                        # Скриншот + OCR
    python3 tools/screenshot.py --mode describe        # Описание содержимого
    python3 tools/screenshot.py --mode elements        # Список UI элементов
    python3 tools/screenshot.py --region 0,0,800,600   # Область экрана
    python3 tools/screenshot.py --monitor 2            # Второй монитор
    python3 tools/screenshot.py --watch 5              # Каждые 5 секунд
    python3 tools/screenshot.py --save screenshot.png  # Сохранить скриншот
"""

import argparse
import base64
import io
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import mss
    import mss.tools
except ImportError:
    mss = None

try:
    from PIL import Image
except ImportError:
    Image = None

# Добавляем parent в path
sys.path.insert(0, str(Path(__file__).parent.parent))
from client.vllm_client import VLMClient


# Промпты для разных режимов анализа
PROMPTS = {
    "ocr": (
        "Extract ALL text visible on this screenshot. "
        "Return only the extracted text, preserving layout where possible. "
        "If text is in multiple columns, process left to right, top to bottom."
    ),
    "describe": (
        "Describe this desktop screenshot in detail:\n"
        "1. What application(s) are visible?\n"
        "2. What is the user currently doing?\n"
        "3. What content is shown on screen?\n"
        "4. Note any notifications, popups, or overlays."
    ),
    "elements": (
        "List ALL interactive UI elements visible on this screenshot:\n"
        "- Buttons (with labels and approximate location)\n"
        "- Text fields (with current content if visible)\n"
        "- Menus and menu items\n"
        "- Links and clickable text\n"
        "- Icons and toolbar items\n"
        "- Tabs\n"
        "- Checkboxes, radio buttons, sliders\n"
        "Format as a numbered list with location hints."
    ),
    "diff": (
        "Compare these two screenshots and describe what changed:\n"
        "1. What elements appeared or disappeared?\n"
        "2. What text changed?\n"
        "3. What is the likely user action between screenshots?\n"
        "4. Note any state changes (selections, focus, etc)."
    ),
    "custom": None,  # Пользователь передаёт свой промпт
}


def capture_screenshot(
    monitor: int = 0,
    region: Optional[tuple[int, int, int, int]] = None,
    save_path: Optional[str] = None,
) -> bytes:
    """
    Захватывает скриншот экрана.

    Args:
        monitor: Номер монитора (0 = все мониторы, 1 = первый, ...)
        region: Область захвата (left, top, width, height)
        save_path: Если указан, сохраняет скриншот в файл

    Returns:
        PNG данные в bytes
    """
    if mss is None:
        raise ImportError(
            "mss not installed. Run: pip install mss"
        )

    with mss.mss() as sct:
        if region:
            left, top, width, height = region
            area = {"left": left, "top": top, "width": width, "height": height}
        elif monitor == 0:
            # Все мониторы
            area = sct.monitors[0]
        else:
            if monitor >= len(sct.monitors):
                available = len(sct.monitors) - 1
                raise ValueError(
                    f"Monitor {monitor} not found. Available: 1-{available}"
                )
            area = sct.monitors[monitor]

        screenshot = sct.grab(area)

        # Конвертируем в PNG через PIL
        if Image is None:
            raise ImportError(
                "Pillow not installed. Run: pip install Pillow"
            )

        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

        # Сохраняем если нужно
        if save_path:
            img.save(save_path)
            print(f"Screenshot saved: {save_path}")

        # Возвращаем PNG bytes
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def screenshot_to_base64(png_data: bytes) -> str:
    """Конвертирует PNG bytes в data URL для API."""
    b64 = base64.b64encode(png_data).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def analyze_screenshot(
    client: VLMClient,
    png_data: bytes,
    mode: str = "ocr",
    custom_prompt: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.2,
) -> str:
    """
    Анализирует скриншот через VLM.

    Args:
        client: VLMClient экземпляр
        png_data: PNG данные скриншота
        mode: Режим анализа (ocr, describe, elements, custom)
        custom_prompt: Пользовательский промпт (для mode=custom)
        max_tokens: Максимум токенов ответа
        temperature: Температура (низкая для точности)

    Returns:
        Текст результата анализа
    """
    if mode == "custom" and custom_prompt:
        prompt = custom_prompt
    elif mode in PROMPTS:
        prompt = PROMPTS[mode]
    else:
        raise ValueError(f"Unknown mode: {mode}. Available: {list(PROMPTS.keys())}")

    # Передаём base64 изображение напрямую
    b64 = base64.b64encode(png_data).decode("utf-8")

    return client.chat(
        prompt=prompt,
        images=[b64],
        max_tokens=max_tokens,
        temperature=temperature,
    )


def watch_mode(
    client: VLMClient,
    interval: int = 5,
    mode: str = "ocr",
    monitor: int = 0,
    region: Optional[tuple[int, int, int, int]] = None,
):
    """
    Периодический захват и анализ экрана.

    Args:
        client: VLMClient
        interval: Интервал между захватами (секунды)
        mode: Режим анализа
        monitor: Номер монитора
        region: Область захвата
    """
    print(f"Watch mode: capturing every {interval}s (Ctrl+C to stop)")
    print("=" * 60)

    prev_result = None
    iteration = 0

    try:
        while True:
            iteration += 1
            timestamp = time.strftime("%H:%M:%S")

            print(f"\n[{timestamp}] Capture #{iteration}...")

            png_data = capture_screenshot(monitor=monitor, region=region)
            result = analyze_screenshot(client, png_data, mode=mode)

            # Если результат изменился, показываем
            if result != prev_result:
                if prev_result is not None:
                    print("--- CHANGED ---")
                print(result)
                prev_result = result
            else:
                print("(no change)")

            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\nStopped after {iteration} captures.")


def diff_screenshots(
    client: VLMClient,
    png_data_1: bytes,
    png_data_2: bytes,
    max_tokens: int = 2048,
) -> str:
    """
    Сравнить два скриншота и описать разницу.

    Args:
        client: VLMClient
        png_data_1: Первый скриншот (PNG bytes)
        png_data_2: Второй скриншот (PNG bytes)

    Returns:
        Описание различий
    """
    b64_1 = base64.b64encode(png_data_1).decode("utf-8")
    b64_2 = base64.b64encode(png_data_2).decode("utf-8")

    return client.chat(
        prompt=PROMPTS["diff"],
        images=[b64_1, b64_2],
        max_tokens=max_tokens,
        temperature=0.3,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Desktop Screenshot OCR/Analysis via VLM"
    )
    parser.add_argument(
        "--mode",
        choices=["ocr", "describe", "elements", "custom"],
        default="ocr",
        help="Analysis mode (default: ocr)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt (requires --mode custom)",
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=0,
        help="Monitor number (0=all, 1=first, ...)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Capture region: left,top,width,height",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save screenshot to file",
    )
    parser.add_argument(
        "--watch",
        type=int,
        default=None,
        help="Watch mode: capture every N seconds",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM server URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-vl-8b",
        help="Model name on server",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max response tokens",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Generation temperature (lower = more precise)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Analyze existing image file instead of capturing screenshot",
    )

    args = parser.parse_args()

    # Парсим region
    region = None
    if args.region:
        parts = [int(x) for x in args.region.split(",")]
        if len(parts) != 4:
            parser.error("--region must be: left,top,width,height")
        region = tuple(parts)

    # Инициализируем клиент
    client = VLMClient(
        base_url=args.server,
        model_name=args.model,
    )

    # Watch mode
    if args.watch:
        watch_mode(
            client,
            interval=args.watch,
            mode=args.mode,
            monitor=args.monitor,
            region=region,
        )
        return

    # Одиночный захват или анализ существующего файла
    if args.image:
        # Анализируем существующий файл
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"File not found: {args.image}")
            sys.exit(1)
        png_data = img_path.read_bytes()
        print(f"Analyzing: {args.image}")
    else:
        # Захватываем скриншот
        print("Capturing screenshot...")
        png_data = capture_screenshot(
            monitor=args.monitor,
            region=region,
            save_path=args.save,
        )

    print(f"Analyzing ({args.mode} mode)...")
    print("=" * 60)

    result = analyze_screenshot(
        client,
        png_data,
        mode=args.mode,
        custom_prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print(result)


if __name__ == "__main__":
    main()
