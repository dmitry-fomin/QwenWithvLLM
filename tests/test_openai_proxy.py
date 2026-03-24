import sys
from pathlib import Path
from fastapi.testclient import TestClient
import unittest
from unittest.mock import patch, MagicMock
import json

# Добавляем корень проекта в sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from web.app import app

class TestOpenAIProxy(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("web.app.vllm_client.post")
    def test_api_generate_non_stream(self, mock_post):
        # Мокаем ответ от vLLM
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}]
        }
        mock_post.return_return_value = mock_response # Wait, mock_post is async
        
        # На самом деле vllm_client это AsyncClient, поэтому нужно мокать по-другому
        # Но TestClient работает с асинхронными эндпоинтами прозрачно.
        # Однако vllm_client.post внутри эндпоинта — это await vllm_client.post(...)
        
        async def async_mock(*args, **kwargs):
            return mock_response
        
        mock_post.side_effect = async_mock

        payload = {
            "model": "qwen3-vl-8b",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False
        }
        
        response = self.client.post("/api/generate", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["id"], "chatcmpl-123")
        self.assertEqual(data["choices"][0]["message"]["content"], "Hello!")

    @patch("web.app.vllm_client.stream")
    def test_api_generate_stream(self, mock_stream):
        # Мокаем стрим
        mock_response = MagicMock()
        
        async def async_aiter():
            yield b'data: {"id": "1", "choices": [{"delta": {"content": "He"}}]}\n\n'
            yield b'data: {"id": "1", "choices": [{"delta": {"content": "llo"}}]}\n\n'
            yield b'data: [DONE]\n\n'

        mock_response.aiter_bytes.return_value = async_aiter()
        
        class AsyncContextManager:
            async def __aenter__(self):
                return mock_response
            async def __aexit__(self, exc_type, exc, tb):
                pass
        
        mock_stream.return_value = AsyncContextManager()

        payload = {
            "model": "qwen3-vl-8b",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True
        }
        
        response = self.client.post("/api/generate", json=payload)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream")
        
        content = b"".join(response.iter_bytes())
        self.assertIn(b"He", content)
        self.assertIn(b"llo", content)
        self.assertIn(b"[DONE]", content)

if __name__ == "__main__":
    unittest.main()
