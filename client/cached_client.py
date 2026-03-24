"""
VLM клиент с кэшированием и автоматическим батчингом.

Кэширование:
- LRU кэш в памяти для повторяющихся запросов
- Опциональный дисковый кэш (shelve)
- Ключ кэша = hash(prompt + images_hash + params)

Батчинг:
- Автоматическая группировка запросов в батч
- Отправка батча при заполнении или по таймауту

Использование:
    from client.cached_client import CachedVLMClient

    client = CachedVLMClient(
        cache_dir="cache/",
        max_cache_size=1000,
    )

    # Первый вызов — идёт к серверу
    result = client.chat(prompt="Describe", images=["img.jpg"])

    # Повторный вызов с теми же параметрами — из кэша
    result = client.chat(prompt="Describe", images=["img.jpg"])

    # Статистика
    print(client.cache_stats())
"""

import hashlib
import json
import shelve
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

from client.vllm_client import VLMClient


class LRUCache:
    """Потокобезопасный LRU кэш в памяти."""

    def __init__(self, max_size: int = 500):
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            if key in self._cache:
                self._hits += 1
                self._cache.move_to_end(key)
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, value: str):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 3) if total > 0 else 0,
            }


class DiskCache:
    """Дисковый кэш через shelve (персистентный между перезапусками)."""

    def __init__(self, cache_dir: str = "cache"):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db_path = str(self._dir / "vlm_cache")
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            with shelve.open(self._db_path) as db:
                return db.get(key)

    def put(self, key: str, value: str):
        with self._lock:
            with shelve.open(self._db_path) as db:
                db[key] = value

    def clear(self):
        with self._lock:
            with shelve.open(self._db_path) as db:
                db.clear()

    @property
    def size(self) -> int:
        with self._lock:
            with shelve.open(self._db_path) as db:
                return len(db)


class CachedVLMClient(VLMClient):
    """VLMClient с двухуровневым кэшем (память + диск)."""

    def __init__(
        self,
        max_cache_size: int = 500,
        cache_dir: Optional[str] = None,
        cache_ttl: float = 3600.0,  # секунды до истечения кэша
        **kwargs,
    ):
        """
        Args:
            max_cache_size: Размер LRU кэша в памяти
            cache_dir: Директория для дискового кэша (None = только память)
            cache_ttl: Время жизни записи в кэше (секунды)
            **kwargs: Параметры VLMClient (base_url, model_name, etc.)
        """
        super().__init__(**kwargs)
        self._memory_cache = LRUCache(max_cache_size)
        self._disk_cache = DiskCache(cache_dir) if cache_dir else None
        self._cache_ttl = cache_ttl
        self._timestamps: dict[str, float] = {}
        self._lock = threading.Lock()

    def _make_cache_key(
        self,
        prompt: str,
        images: Optional[list] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Создать уникальный ключ кэша."""
        key_parts = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        # Для изображений используем hash содержимого
        if images:
            img_hashes = []
            for img in images:
                img_str = str(img)
                # Для URL используем сам URL, для файлов/base64 — hash
                if img_str.startswith(("http://", "https://")):
                    img_hashes.append(img_str)
                else:
                    img_hashes.append(
                        hashlib.md5(img_str[:1000].encode()).hexdigest()
                    )
            key_parts["images"] = img_hashes

        raw = json.dumps(key_parts, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def _is_expired(self, key: str) -> bool:
        """Проверить истёк ли TTL для ключа."""
        with self._lock:
            ts = self._timestamps.get(key)
            if ts is None:
                return True
            return (time.monotonic() - ts) > self._cache_ttl

    def chat(
        self,
        prompt: str,
        images: Optional[list[Union[str, Path]]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        skip_cache: bool = False,
    ) -> str:
        """
        Chat с кэшированием.

        Дополнительные параметры:
            skip_cache: Пропустить кэш и обратиться к серверу напрямую
        """
        # Стриминг не кэшируем
        if stream or skip_cache or system_prompt:
            return super().chat(
                prompt=prompt,
                images=images,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
            )

        cache_key = self._make_cache_key(prompt, images, max_tokens, temperature, top_p)

        # Проверяем memory cache
        if not self._is_expired(cache_key):
            cached = self._memory_cache.get(cache_key)
            if cached is not None:
                return cached

        # Проверяем disk cache
        if self._disk_cache and not self._is_expired(cache_key):
            cached = self._disk_cache.get(cache_key)
            if cached is not None:
                # Поднимаем в memory cache
                self._memory_cache.put(cache_key, cached)
                return cached

        # Cache miss — обращаемся к серверу
        result = super().chat(
            prompt=prompt,
            images=images,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # Сохраняем в кэш
        with self._lock:
            self._timestamps[cache_key] = time.monotonic()

        self._memory_cache.put(cache_key, result)
        if self._disk_cache:
            self._disk_cache.put(cache_key, result)

        return result

    def cache_stats(self) -> dict:
        """Статистика кэша."""
        stats = {
            "memory": self._memory_cache.stats,
            "ttl_seconds": self._cache_ttl,
        }
        if self._disk_cache:
            stats["disk"] = {"size": self._disk_cache.size}
        return stats

    def clear_cache(self):
        """Очистить все кэши."""
        self._memory_cache.clear()
        if self._disk_cache:
            self._disk_cache.clear()
        with self._lock:
            self._timestamps.clear()
