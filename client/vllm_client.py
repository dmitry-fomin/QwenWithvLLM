"""
Клиент для VLM моделей через vLLM OpenAI API.

Поддерживает:
- Текстовые запросы
- Запросы с изображениями (URL или base64)
- Мульти-изображения (до N на запрос в зависимости от модели)
- Стриминг ответов
- Асинхронные запросы
"""

import base64
from pathlib import Path
from typing import Optional, Union, Generator
from openai import OpenAI, AsyncOpenAI


class VLMClient:
    """Синхронный клиент для VLM моделей через vLLM."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "dummy",  # vLLM не требует реального ключа
        model_name: str = "qwen3-vl-8b",
        timeout: float = 300.0,
    ):
        """
        Инициализация клиента.

        Args:
            base_url: URL vLLM API (по умолчанию localhost:8000)
            api_key: API ключ (для vLLM можно передать любую строку)
            model_name: Имя модели (должно совпадать с --served-model-name в vLLM)
            timeout: Таймаут для запросов в секундах
        """
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Кодирует локальный файл в base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_mime_type(self, path: Union[str, Path]) -> str:
        """Определяет MIME тип по расширению."""
        suffix = Path(path).suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        return mime_map.get(suffix, "image/jpeg")

    def _build_image_content(
        self, image: Union[str, Path]
    ) -> dict:
        """
        Создаёт элемент контента для изображения.

        Поддерживает:
        - URL (http:// или https://)
        - Локальный файл (путь существует)
        - Raw base64 (строка без признаков URL/пути)
        """
        image_str = str(image)

        if image_str.startswith(("http://", "https://")):
            # URL
            return {
                "type": "image_url",
                "image_url": {"url": image_str},
            }
        elif Path(image_str).exists():
            # Локальный файл
            b64 = self._encode_image(image_str)
            mime = self._get_mime_type(image_str)
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{b64}"
                },
            }
        else:
            # Предполагаем что это уже base64
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_str}"
                },
            }

    def chat(
        self,
        prompt: str,
        images: Optional[list[Union[str, Path]]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
    ) -> str:
        """
        Отправить запрос к модели.

        Args:
            prompt: Текстовый запрос
            images: Список изображений (URL, пути или base64)
            system_prompt: Системный промпт
            max_tokens: Максимальное количество токенов в ответе
            temperature: Температура генерации (0.0-2.0)
            top_p: Top-p sampling (0.0-1.0)
            stream: Использовать стриминг (возвращает генератор)

        Returns:
            Если stream=False: полный текст ответа (str)
            Если stream=True: генератор токенов
        """
        messages = []

        # Системный промпт
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Формируем контент пользователя
        if images:
            content = []
            # Важно: изображения должны идти ДО текста в содержимом
            # (требуется для InternVL2.5 и других VLM)
            for img in images:
                content.append(self._build_image_content(img))
            content.append({"type": "text", "text": prompt})
        else:
            content = prompt

        messages.append({"role": "user", "content": content})

        if stream:
            return self._stream_response(
                messages, max_tokens, temperature, top_p
            )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response.choices[0].message.content

    def _stream_response(
        self, messages, max_tokens, temperature, top_p
    ) -> Generator[str, None, None]:
        """Генератор для стримингового ответа."""
        with self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        ) as stream:
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

    def multi_turn_chat(
        self,
        conversation: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        """
        Многоходовой диалог.

        Args:
            conversation: Список сообщений
                [
                    {"role": "user", "content": "Первое сообщение"},
                    {"role": "assistant", "content": "Ответ"},
                    {"role": "user", "content": "Второе сообщение"},
                    ...
                ]
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=conversation,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content


class AsyncVLMClient:
    """Асинхронный клиент для VLM моделей через vLLM."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "dummy",
        model_name: str = "qwen3-vl-8b",
        timeout: float = 300.0,
    ):
        self.model_name = model_name
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Кодирует локальный файл в base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_mime_type(self, path: Union[str, Path]) -> str:
        """Определяет MIME тип по расширению."""
        suffix = Path(path).suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        return mime_map.get(suffix, "image/jpeg")

    def _build_image_content(self, image: Union[str, Path]) -> dict:
        """Создаёт элемент контента для изображения."""
        image_str = str(image)

        if image_str.startswith(("http://", "https://")):
            return {
                "type": "image_url",
                "image_url": {"url": image_str},
            }
        elif Path(image_str).exists():
            b64 = self._encode_image(image_str)
            mime = self._get_mime_type(image_str)
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{b64}"
                },
            }
        else:
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_str}"
                },
            }

    async def chat(
        self,
        prompt: str,
        images: Optional[list[Union[str, Path]]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Асинхронный запрос к модели."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if images:
            content = []
            for img in images:
                content.append(self._build_image_content(img))
            content.append({"type": "text", "text": prompt})
        else:
            content = prompt

        messages.append({"role": "user", "content": content})

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response.choices[0].message.content
