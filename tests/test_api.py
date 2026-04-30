"""Pytest tests for AlphaSignal FastAPI endpoints."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


# ---------------------------------------------------------------------------
# Shared mocks
# ---------------------------------------------------------------------------

def _mock_result(label: str = "positive", confidence: float = 0.92) -> MagicMock:
    r = MagicMock()
    r.label = label
    r.confidence = confidence
    return r


MOCK_HEADLINES = ["NVIDIA beats earnings estimates.", "AI chip demand surges globally."]

CACHED_RESPONSE = {
    "ticker": "AAPL",
    "overall_sentiment": "positive",
    "confidence": 0.88,
    "label_distribution": {"positive": 5, "negative": 1, "neutral": 2},
    "headlines_analyzed": 8,
    "cached": True,
}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /sentiment
# ---------------------------------------------------------------------------

class TestSentimentEndpoint:
    def test_fresh_sentiment_returns_200(self):
        with (
            patch("backend.main.get_cached", return_value=None),
            patch("backend.main.fetch_news", return_value=MOCK_HEADLINES),
            patch("backend.main.analyze", return_value=_mock_result("positive", 0.92)),
            patch("backend.main.set_cached"),
        ):
            response = client.post("/sentiment", json={"ticker": "NVDA"})

        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "NVDA"
        assert data["overall_sentiment"] in ("positive", "negative", "neutral")
        assert 0.0 <= data["confidence"] <= 1.0
        assert isinstance(data["headlines_analyzed"], int)
        assert data["cached"] is False

    def test_returns_cached_data(self):
        with patch("backend.main.get_cached", return_value=CACHED_RESPONSE):
            response = client.post("/sentiment", json={"ticker": "AAPL"})

        assert response.status_code == 200
        assert response.json()["cached"] is True
        assert response.json()["ticker"] == "AAPL"

    def test_404_when_no_headlines(self):
        with (
            patch("backend.main.get_cached", return_value=None),
            patch("backend.main.fetch_news", return_value=[]),
        ):
            response = client.post("/sentiment", json={"ticker": "FAKEXYZ"})

        assert response.status_code == 404

    def test_ticker_uppercased(self):
        with (
            patch("backend.main.get_cached", return_value=None),
            patch("backend.main.fetch_news", return_value=MOCK_HEADLINES),
            patch("backend.main.analyze", return_value=_mock_result()),
            patch("backend.main.set_cached"),
        ):
            response = client.post("/sentiment", json={"ticker": "nvda"})

        assert response.json()["ticker"] == "NVDA"

    def test_label_distribution_sums_to_headlines_analyzed(self):
        with (
            patch("backend.main.get_cached", return_value=None),
            patch("backend.main.fetch_news", return_value=MOCK_HEADLINES),
            patch("backend.main.analyze", return_value=_mock_result()),
            patch("backend.main.set_cached"),
        ):
            data = client.post("/sentiment", json={"ticker": "NVDA"}).json()

        dist = data["label_distribution"]
        assert sum(dist.values()) == data["headlines_analyzed"]


# ---------------------------------------------------------------------------
# POST /agent/query
# ---------------------------------------------------------------------------

class TestAgentEndpoint:
    def test_returns_200_with_answer(self):
        with patch("backend.main.run_agent", return_value="NVDA is bullish."):
            response = client.post(
                "/agent/query",
                json={"question": "Is NVDA bullish this week?"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "NVDA is bullish."

    def test_question_echoed_in_response(self):
        question = "What is the sentiment for TSLA right now?"
        with patch("backend.main.run_agent", return_value="Mixed signals."):
            data = client.post("/agent/query", json={"question": question}).json()

        assert data["question"] == question

    def test_500_on_agent_exception(self):
        with patch("backend.main.run_agent", side_effect=RuntimeError("LLM offline")):
            response = client.post("/agent/query", json={"question": "Test"})

        assert response.status_code == 500
