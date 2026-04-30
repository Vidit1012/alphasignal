"""LangGraph ReAct agent with 3 market intelligence tools powered by Ollama."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from .news import fetch_news
from .rag import index_headlines, semantic_search
from .sentiment import analyze

LLM_MODEL = "llama3.2"


# ---------------------------------------------------------------------------
# Tool 1 — Sentiment analysis over recent headlines
# ---------------------------------------------------------------------------

@tool
def get_sentiment(ticker: str, date_range: str = "1w") -> str:
    """Analyze the sentiment of recent news headlines for a stock ticker.

    Args:
        ticker: Stock ticker symbol (e.g. NVDA, AAPL, TSLA).
        date_range: Informal time range hint like '1d', '1w', '1m'. Used for
            context only — actual data depends on yfinance availability.

    Returns:
        JSON string with overall sentiment, label distribution, and per-headline
        details for the most recent headlines.
    """
    headlines = fetch_news(ticker, max_items=10)
    if not headlines:
        return json.dumps({"error": f"No news found for ticker {ticker}"})

    label_counts: dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0}
    details = []
    for h in headlines:
        r = analyze(h)
        label_counts[r.label] += 1
        details.append({"headline": h, "label": r.label, "confidence": r.confidence})

    dominant = max(label_counts, key=lambda k: label_counts[k])
    return json.dumps({
        "ticker": ticker.upper(),
        "overall_sentiment": dominant,
        "label_distribution": label_counts,
        "headlines_analyzed": len(headlines),
        "date_range": date_range,
        "details": details[:5],
    })


# ---------------------------------------------------------------------------
# Tool 2 — Semantic news search via ChromaDB / LlamaIndex
# ---------------------------------------------------------------------------

@tool
def search_news(ticker: str, topic: str) -> str:
    """Semantic search over indexed financial news for a ticker and topic.

    Refreshes the ChromaDB index with the latest headlines before searching,
    so results always reflect current data.

    Args:
        ticker: Stock ticker symbol.
        topic: Topic or question to search (e.g. 'earnings', 'AI chip demand').

    Returns:
        JSON string with the most semantically relevant news chunks.
    """
    # Refresh index with latest headlines before searching
    headlines = fetch_news(ticker, max_items=20)
    if headlines:
        index_headlines(headlines, ticker)

    query = f"{ticker.upper()} {topic}"
    chunks = semantic_search(query, top_k=5)
    if not chunks:
        return json.dumps({"error": "No relevant news found", "query": query})
    return json.dumps({"ticker": ticker.upper(), "topic": topic, "results": chunks})


# ---------------------------------------------------------------------------
# Tool 3 — Aggregate sentiment trend summary
# ---------------------------------------------------------------------------

@tool
def summarize_signals(ticker: str) -> str:
    """Aggregate sentiment across all recent headlines and return a market signal.

    Analyzes up to 20 recent headlines, computes bullish/bearish/neutral
    percentages, and returns a directional signal with a plain-English summary.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        JSON string with signal (BULLISH / BEARISH / MIXED), percentages, and
        a human-readable summary.
    """
    headlines = fetch_news(ticker, max_items=20)
    if not headlines:
        return json.dumps({"error": f"No news data available for {ticker}"})

    counts: dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0}
    for h in headlines:
        r = analyze(h)
        counts[r.label] += 1

    total = len(headlines)
    bull_pct = round(counts["positive"] / total * 100, 1)
    bear_pct = round(counts["negative"] / total * 100, 1)
    neutral_pct = round(counts["neutral"] / total * 100, 1)

    if bull_pct > 50:
        signal = "BULLISH"
    elif bear_pct > 50:
        signal = "BEARISH"
    else:
        signal = "MIXED/NEUTRAL"

    return json.dumps({
        "ticker": ticker.upper(),
        "signal": signal,
        "bullish_pct": bull_pct,
        "bearish_pct": bear_pct,
        "neutral_pct": neutral_pct,
        "headlines_analyzed": total,
        "summary": (
            f"{ticker.upper()} shows a {signal} signal based on {total} recent headlines. "
            f"Breakdown: {bull_pct}% positive, {bear_pct}% negative, {neutral_pct}% neutral."
        ),
    })


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def build_agent():
    """Build and return the LangGraph ReAct agent with all three tools."""
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    return create_react_agent(llm, [get_sentiment, search_news, summarize_signals])


_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def run_agent(question: str) -> str:
    """Invoke the agent with a natural-language *question* and return the answer."""
    result = get_agent().invoke({"messages": [HumanMessage(content=question)]})
    messages = result.get("messages", [])
    if messages:
        last = messages[-1]
        return last.content if hasattr(last, "content") else str(last)
    return "No response from agent."


if __name__ == "__main__":
    # Run as: python -m backend.agent
    print(run_agent("Is NVDA bullish this week and why?"))
