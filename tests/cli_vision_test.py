import sys
import os
from pathlib import Path

# Добавляем корень проекта в sys.path для импорта клиента
sys.path.insert(0, str(Path(__file__).parent.parent))
from client.vllm_client import VLMClient

def main():
    if len(sys.argv) < 2:
        print("Использование: python tests/cli_vision_test.py <путь_к_картинке> [промпт]")
        print("Пример: python tests/cli_vision_test.py sample.jpg 'Что на этой картинке?'")
        sys.exit(1)

    image_path = sys.argv[1]
    # Промпт по умолчанию, если не передан
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Опиши это изображение подробно."

    # Настройки из переменных окружения или значения по умолчанию
    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    model_name = os.getenv("VLLM_MODEL_NAME", "qwen25vl-7b")

    if not Path(image_path).exists():
        print(f"Ошибка: Файл {image_path} не найден.")
        sys.exit(1)

    # Инициализация клиента
    client = VLMClient(base_url=base_url, model_name=model_name)

    try:
        # Метод chat сам кодирует локальные файлы в base64
        response = client.chat(prompt=prompt, images=[image_path])
        # Вывод в stdout
        print(response)
    except Exception as e:
        print(f"Ошибка при запросе: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
