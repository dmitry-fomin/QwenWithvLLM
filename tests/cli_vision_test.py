import sys
import os
import traceback
from pathlib import Path
import httpx
from openai import OpenAI

# Добавляем корень проекта в sys.path, чтобы работал импорт из папки client
sys.path.insert(0, str(Path(__file__).parent.parent))
from client import VLMClient

def main():
    if len(sys.argv) < 2:
        print("Использование: python tests/cli_vision_test.py <путь_к_картинке>")
        sys.exit(1)

    image_path = sys.argv[1]
    
    # На macOS 'localhost' часто разрешается в IPv6 адрес [::1]. 
    # Судя по вашему curl, сервер vLLM слушает именно на нём, а не на 127.0.0.1.
    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    model_name = os.getenv("VLLM_MODEL_NAME", "qwen3-vl-8b")
    
    print(f"--- Тест VLM ---")
    print(f"Путь к картинке: {image_path}")
    print(f"Сервер: {base_url}")
    print(f"Модель: {model_name}")
    print(f"----------------")

    # ВАЖНО: Мы оставляем trust_env=False, чтобы игнорировать системный прокси,
    # но возвращаем 'localhost', так как сервер привязан к IPv6 (::1).
    http_client = httpx.Client(trust_env=False)

    # Инициализируем VLMClient
    client = VLMClient(base_url=base_url, model_name=model_name)
    client.client = OpenAI(
        base_url=base_url,
        api_key="dummy",
        http_client=http_client,
        timeout=300.0
    )

    try:
        response = client.chat(
            prompt="Что на этом изображении? Ответь кратко.",
            images=[image_path]
        )
        print("\nОТВЕТ МОДЕЛИ:")
        print(response)
    except Exception as e:
        print("\nОШИБКА:")
        print(f"Тип ошибки: {type(e).__name__}")
        print(f"Сообщение: {e}")
        print("\nПодробный трейсбек:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
