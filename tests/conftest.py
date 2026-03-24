"""Общие фикстуры для тестов."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


@pytest.fixture
def mock_openai_client():
    """Мок OpenAI клиента для синхронных запросов."""
    mock_client = MagicMock()

    # Стандартный ответ
    mock_choice = MagicMock()
    mock_choice.message.content = "This is a test response"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_async_openai_client():
    """Мок AsyncOpenAI клиента."""
    mock_client = MagicMock()

    mock_choice = MagicMock()
    mock_choice.message.content = "Async test response"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    return mock_client


@pytest.fixture
def sample_image_path(tmp_path):
    """Создаёт временный PNG файл для тестов."""
    # Минимальный 1x1 PNG
    png_data = (
        b'\x89PNG\r\n\x1a\n'
        b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02'
        b'\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f'
        b'\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    img_path = tmp_path / "test_image.png"
    img_path.write_bytes(png_data)
    return img_path


@pytest.fixture
def sample_jpeg_path(tmp_path):
    """Создаёт временный JPEG файл для тестов."""
    # Минимальный JPEG (не валидный, но достаточный для base64)
    jpeg_data = b'\xff\xd8\xff\xe0' + b'\x00' * 100 + b'\xff\xd9'
    img_path = tmp_path / "test_image.jpg"
    img_path.write_bytes(jpeg_data)
    return img_path


@pytest.fixture
def env_config_path(tmp_path):
    """Создаёт временный env-конфиг для тестов."""
    config = tmp_path / "test_model.env"
    config.write_text(
        'MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct\n'
        'MODEL_NAME=qwen3-vl-8b\n'
        'TENSOR_PARALLEL_SIZE=2\n'
        'MAX_MODEL_LEN=16384\n'
        'GPU_MEMORY_UTILIZATION=0.90\n'
        'DTYPE=bfloat16\n'
        'EXTRA_ARGS="--trust-remote-code --limit-mm-per-prompt image=10"\n'
    )
    return config
