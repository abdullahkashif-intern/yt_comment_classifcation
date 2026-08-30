"""
cache.py — tiny in-memory TTL cache for classify-video responses.

Phase 1 of docs/scaling-roadmap.md. Deliberately minimal: a single process-wide
dict guarded by a lock, keyed by video_id, with a short TTL. This is NOT
persistent storage in the sense CLAUDE.md's guardrail is guarding against —
entries vanish on process restart and are never written to disk — it exists
purely to avoid re-spending YouTube quota and LLM cost on repeat requests for
the same video within a short window.

Single-process only: if a future phase (see docs/scaling-roadmap.md Phase 4)
introduces multiple worker processes, this cache stops being shared across
them and should be replaced with something out-of-process (e.g. Redis).
"""

import time
from threading import Lock
from typing import Generic, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    """A minimal thread-safe cache where each entry expires after a fixed TTL."""

    def __init__(self, ttl_seconds: float):
        self._ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[float, T]] = {}  # key -> (expires_at, value)
        self._lock = Lock()

    def get(self, key: str) -> T | None:
        """Return the cached value for `key`, or None if absent/expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None

            expires_at, value = entry
            if time.monotonic() > expires_at:
                del self._store[key]  # Evict lazily on access rather than via a background sweep
                return None

            return value

    def set(self, key: str, value: T) -> None:
        """Store `value` under `key`, resetting its TTL."""
        with self._lock:
            self._store[key] = (time.monotonic() + self._ttl_seconds, value)
