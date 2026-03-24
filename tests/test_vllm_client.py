"""Тесты для VLMClient и AsyncVLMClient."""

import base64
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch, call

import pytest

from client.vllm_client import VLMClient, AsyncVLMClient


# ============================================================
# VLMClient — инициализация
# ============================================================


class TestVLMClientInit:
    def test_default_init(self):
        with patch("client.vllm_client.OpenAI") as mock_openai:
            client = VLMClient()
            mock_openai.assert_called_once_with(
                base_url="http://localhost:8000/v1",
                api_key="dummy",
                timeout=300.0,
            )
            assert client.model_name == "qwen3-vl-8b"

    def test_custom_init(self):
        with patch("client.vllm_client.OpenAI") as mock_openai:
            client = VLMClient(
                base_url="http://myserver:9000/v1",
                api_key="mykey",
                model_name="internvl2_5-26b",
                timeout=60.0,
            )
            mock_openai.assert_called_once_with(
                base_url="http://myserver:9000/v1",
                api_key="mykey",
                timeout=60.0,
            )
            assert client.model_name == "internvl2_5-26b"


# ============================================================
# VLMClient — вспомогательные методы
# ============================================================


class TestImageHelpers:
    @pytest.fixture(autouse=True)
    def setup(self):
        with patch("client.vllm_client.OpenAI"):
            self.client = VLMClient()

    def test_encode_image(self, sample_image_path):
        result = self.client._encode_image(sample_image_path)
        # Должен быть валидный base64
        decoded = base64.b64decode(result)
        assert decoded[:4] == b'\x89PNG'

    def test_get_mime_type_png(self):
        assert self.client._get_mime_type("test.png") == "image/png"

    def test_get_mime_type_jpg(self):
        assert self.client._get_mime_type("photo.jpg") == "image/jpeg"

    def test_get_mime_type_jpeg(self):
        assert self.client._get_mime_type("photo.jpeg") == "image/jpeg"

    def test_get_mime_type_gif(self):
        assert self.client._get_mime_type("anim.gif") == "image/gif"

    def test_get_mime_type_webp(self):
        assert self.client._get_mime_type("photo.webp") == "image/webp"

    def test_get_mime_type_bmp(self):
        assert self.client._get_mime_type("old.bmp") == "image/bmp"

    def test_get_mime_type_unknown_defaults_to_jpeg(self):
        assert self.client._get_mime_type("file.xyz") == "image/jpeg"

    def test_build_image_content_url(self):
        result = self.client._build_image_content("https://example.com/img.jpg")
        assert result == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/img.jpg"},
        }

    def test_build_image_content_http(self):
        result = self.client._build_image_content("http://example.com/img.png")
        assert result["image_url"]["url"] == "http://example.com/img.png"

    def test_build_image_content_local_file(self, sample_image_path):
        result = self.client._build_image_content(str(sample_image_path))
        assert result["type"] == "image_url"
        url = result["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        # Проверяем что base64 содержит наш PNG
        b64_part = url.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        assert decoded[:4] == b'\x89PNG'

    def test_build_image_content_local_jpeg(self, sample_jpeg_path):
        result = self.client._build_image_content(str(sample_jpeg_path))
        url = result["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")

    def test_build_image_content_raw_base64(self):
        raw_b64 = base64.b64encode(b"fake image data").decode()
        result = self.client._build_image_content(raw_b64)
        assert result["image_url"]["url"] == f"data:image/jpeg;base64,{raw_b64}"

    def test_build_image_content_path_object(self, sample_image_path):
        result = self.client._build_image_content(Path(sample_image_path))
        assert result["type"] == "image_url"
        assert "base64," in result["image_url"]["url"]


# ============================================================
# VLMClient — chat()
# ============================================================


class TestVLMClientChat:
    @pytest.fixture(autouse=True)
    def setup(self, mock_openai_client):
        with patch("client.vllm_client.OpenAI", return_value=mock_openai_client):
            self.client = VLMClient()
            self.mock_api = mock_openai_client

    def test_text_only_chat(self):
        result = self.client.chat(prompt="Hello!")
        assert result == "This is a test response"

        call_args = self.mock_api.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello!"

    def test_chat_with_system_prompt(self):
        self.client.chat(prompt="Hi", system_prompt="You are helpful.")

        call_args = self.mock_api.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["role"] == "user"

    def test_chat_with_image_url(self):
        self.client.chat(
            prompt="Describe this.",
            images=["https://example.com/img.jpg"],
        )

        call_args = self.mock_api.chat.completions.create.call_args
        content = call_args.kwargs["messages"][0]["content"]
        # content — это list, изображения перед текстом
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0]["type"] == "image_url"
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "Describe this."

    def test_chat_images_before_text(self):
        """Изображения должны идти ДО текста."""
        self.client.chat(
            prompt="Compare.",
            images=[
                "https://example.com/1.jpg",
                "https://example.com/2.jpg",
            ],
        )

        call_args = self.mock_api.chat.completions.create.call_args
        content = call_args.kwargs["messages"][0]["content"]
        assert len(content) == 3
        assert content[0]["type"] == "image_url"
        assert content[1]["type"] == "image_url"
        assert content[2]["type"] == "text"

    def test_chat_with_local_image(self, sample_image_path):
        self.client.chat(
            prompt="OCR this.",
            images=[str(sample_image_path)],
        )

        call_args = self.mock_api.chat.completions.create.call_args
        content = call_args.kwargs["messages"][0]["content"]
        assert content[0]["type"] == "image_url"
        assert "base64," in content[0]["image_url"]["url"]

    def test_chat_passes_parameters(self):
        self.client.chat(
            prompt="Test",
            max_tokens=512,
            temperature=0.3,
            top_p=0.8,
        )

        call_args = self.mock_api.chat.completions.create.call_args
        assert call_args.kwargs["max_tokens"] == 512
        assert call_args.kwargs["temperature"] == 0.3
        assert call_args.kwargs["top_p"] == 0.8
        assert call_args.kwargs["model"] == "qwen3-vl-8b"

    def test_chat_no_stream_by_default(self):
        result = self.client.chat(prompt="Test")
        # Должен вернуть строку, а не генератор
        assert isinstance(result, str)

    def test_chat_stream_returns_generator(self):
        # Настраиваем mock для стриминга
        mock_chunk1 = MagicMock()
        mock_chunk1.choices[0].delta.content = "Hello"
        mock_chunk2 = MagicMock()
        mock_chunk2.choices[0].delta.content = " world"

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=iter([mock_chunk1, mock_chunk2]))
        mock_stream.__exit__ = MagicMock(return_value=False)

        self.mock_api.chat.completions.create.return_value = mock_stream

        result = self.client.chat(prompt="Test", stream=True)
        tokens = list(result)
        assert tokens == ["Hello", " world"]


# ============================================================
# VLMClient — multi_turn_chat()
# ============================================================


class TestMultiTurnChat:
    @pytest.fixture(autouse=True)
    def setup(self, mock_openai_client):
        with patch("client.vllm_client.OpenAI", return_value=mock_openai_client):
            self.client = VLMClient()
            self.mock_api = mock_openai_client

    def test_multi_turn_passes_conversation(self):
        conversation = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = self.client.multi_turn_chat(conversation)
        assert result == "This is a test response"

        call_args = self.mock_api.chat.completions.create.call_args
        assert call_args.kwargs["messages"] == conversation

    def test_multi_turn_respects_parameters(self):
        self.client.multi_turn_chat(
            [{"role": "user", "content": "Hi"}],
            max_tokens=100,
            temperature=0.5,
        )

        call_args = self.mock_api.chat.completions.create.call_args
        assert call_args.kwargs["max_tokens"] == 100
        assert call_args.kwargs["temperature"] == 0.5


# ============================================================
# AsyncVLMClient
# ============================================================


class TestAsyncVLMClientInit:
    def test_default_init(self):
        with patch("client.vllm_client.AsyncOpenAI") as mock_async:
            client = AsyncVLMClient()
            mock_async.assert_called_once_with(
                base_url="http://localhost:8000/v1",
                api_key="dummy",
                timeout=300.0,
            )
            assert client.model_name == "qwen3-vl-8b"


class TestAsyncVLMClientChat:
    @pytest.fixture(autouse=True)
    def setup(self, mock_async_openai_client):
        with patch("client.vllm_client.AsyncOpenAI", return_value=mock_async_openai_client):
            self.client = AsyncVLMClient()
            self.mock_api = mock_async_openai_client

    @pytest.mark.asyncio
    async def test_async_text_chat(self):
        result = await self.client.chat(prompt="Hello async!")
        assert result == "Async test response"

        call_args = self.mock_api.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["content"] == "Hello async!"

    @pytest.mark.asyncio
    async def test_async_chat_with_images(self):
        await self.client.chat(
            prompt="Describe.",
            images=["https://example.com/img.jpg"],
        )

        call_args = self.mock_api.chat.completions.create.call_args
        content = call_args.kwargs["messages"][0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "image_url"
        assert content[1]["type"] == "text"

    @pytest.mark.asyncio
    async def test_async_chat_with_system_prompt(self):
        await self.client.chat(
            prompt="Hi",
            system_prompt="Be concise.",
        )

        call_args = self.mock_api.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_async_chat_with_local_image(self, sample_image_path):
        await self.client.chat(
            prompt="OCR",
            images=[str(sample_image_path)],
        )

        call_args = self.mock_api.chat.completions.create.call_args
        content = call_args.kwargs["messages"][0]["content"]
        assert "base64," in content[0]["image_url"]["url"]

    @pytest.mark.asyncio
    async def test_async_chat_parameters(self):
        await self.client.chat(
            prompt="Test",
            max_tokens=256,
            temperature=0.1,
            top_p=0.5,
        )

        call_args = self.mock_api.chat.completions.create.call_args
        assert call_args.kwargs["max_tokens"] == 256
        assert call_args.kwargs["temperature"] == 0.1
        assert call_args.kwargs["top_p"] == 0.5
