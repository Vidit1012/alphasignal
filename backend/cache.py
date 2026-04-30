"""Redis cache for sentiment scores with a 1-hour TTL."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import redis
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

TTL_SECONDS = 3_600  # 1 hour

_client: redis.Redis | None = None


def get_client() -> redis.Redis:
    global _client
    if _client is None:
        url = os.getenv("REDIS_URL")
        if not url:
            raise RuntimeError("REDIS_URL is not set — add it to your .env file")
        _client = redis.from_url(url, decode_responses=True, socket_connect_timeout=2)
    return _client


def _key(ticker: str) -> str:
    return f"alphasignal:sentiment:{ticker.upper()}"


def get_cached(ticker: str) -> dict[str, Any] | None:
    """Return cached sentiment dict, or None if missing / Redis unavailable."""
    try:
        raw = get_client().get(_key(ticker))
        return json.loads(raw) if raw else None
    except redis.RedisError:
        return None


def set_cached(ticker: str, data: dict[str, Any]) -> None:
    """Store *data* in Redis with TTL. Silently ignores Redis errors."""
    try:
        get_client().setex(_key(ticker), TTL_SECONDS, json.dumps(data))
    except redis.RedisError as exc:
        print(f"[cache] Redis write failed for {ticker}: {exc}")


def clear_cached(ticker: str) -> None:
    try:
        get_client().delete(_key(ticker))
    except redis.RedisError:
        pass
