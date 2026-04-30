"""Fetch recent news headlines for a ticker using yfinance."""
from __future__ import annotations

import yfinance as yf


def fetch_news(ticker: str, max_items: int = 20) -> list[str]:
    """Return a list of recent headline strings for *ticker*.

    Handles both the legacy yfinance news format (dict with 'title') and
    the newer format (dict with nested 'content.title').
    """
    try:
        t = yf.Ticker(ticker.upper())
        raw: list = t.news or []
        headlines: list[str] = []
        for item in raw[:max_items]:
            if not isinstance(item, dict):
                continue
            # Legacy format: item["title"]
            title: str = item.get("title") or ""
            # Newer format: item["content"]["title"]
            if not title:
                content = item.get("content") or {}
                title = content.get("title") or ""
            if title:
                headlines.append(str(title).strip())
        return headlines
    except Exception as exc:
        print(f"[news] Error fetching news for {ticker}: {exc}")
        return []


if __name__ == "__main__":
    # Run as: python -m backend.news
    for headline in fetch_news("NVDA"):
        print(headline)
