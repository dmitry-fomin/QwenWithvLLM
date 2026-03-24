"""
Middleware для логирования запросов к VLM клиенту.

Оборачивает VLMClient, логирует каждый запрос/ответ в JSON Lines файл.
Собирает метрики: latency, tokens, errors.

Использование:
    from monitoring.logger import LoggedVLMClient

    client = LoggedVLMClient(
        log_file="logs/requests.jsonl",
        model_name="qwen3-vl-8b",
    )
    result = client.chat(prompt="Hello", images=["img.jpg"])
    # Автоматически логируется в logs/requests.jsonl

    # Статистика
    stats = client.get_stats()
    print(stats)
"""

import json
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from client.vllm_client import VLMClient


class RequestLogger:
    """Потокобезопасный логгер запросов в JSON Lines формат."""

    def __init__(self, log_file: str = "logs/requests.jsonl"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # In-memory метрики
        self._total_requests = 0
        self._total_errors = 0
        self._total_latency = 0.0
        self._total_tokens_estimate = 0

    def log(self, entry: dict):
        """Записать лог в файл."""
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()

        with self._lock:
            self._total_requests += 1
            if entry.get("error"):
                self._total_errors += 1
            if entry.get("latency_ms"):
                self._total_latency += entry["latency_ms"]
            if entry.get("response_length"):
                # Грубая оценка: ~4 символа на токен
                self._total_tokens_estimate += entry["response_length"] // 4

            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_stats(self) -> dict:
        """Вернуть агрегированную статистику."""
        with self._lock:
            avg_latency = (
                self._total_latency / self._total_requests
                if self._total_requests > 0
                else 0
            )
            return {
                "total_requests": self._total_requests,
                "total_errors": self._total_errors,
                "error_rate": (
                    self._total_errors / self._total_requests
                    if self._total_requests > 0
                    else 0
                ),
                "avg_latency_ms": round(avg_latency, 1),
                "total_latency_ms": round(self._total_latency, 1),
                "estimated_tokens": self._total_tokens_estimate,
                "log_file": str(self.log_file),
            }


class LoggedVLMClient(VLMClient):
    """VLMClient с автоматическим логированием запросов."""

    def __init__(
        self,
        log_file: str = "logs/requests.jsonl",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._logger = RequestLogger(log_file)

    def chat(
        self,
        prompt: str,
        images: Optional[list[Union[str, Path]]] = None,
        **kwargs,
    ) -> str:
        start = time.monotonic()
        error = None
        result = ""

        try:
            # Если stream=True, не логируем (слишком сложно)
            if kwargs.get("stream"):
                return super().chat(prompt=prompt, images=images, **kwargs)

            result = super().chat(prompt=prompt, images=images, **kwargs)
            return result

        except Exception as e:
            error = str(e)
            raise

        finally:
            elapsed_ms = (time.monotonic() - start) * 1000

            entry = {
                "model": self.model_name,
                "prompt_length": len(prompt),
                "has_images": bool(images),
                "image_count": len(images) if images else 0,
                "response_length": len(result) if result else 0,
                "latency_ms": round(elapsed_ms, 1),
                "max_tokens": kwargs.get("max_tokens", 2048),
                "temperature": kwargs.get("temperature", 0.7),
                "error": error,
            }
            self._logger.log(entry)

    def get_stats(self) -> dict:
        """Статистика запросов."""
        return self._logger.get_stats()
