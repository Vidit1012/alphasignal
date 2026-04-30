"""FastAPI backend — exposes /sentiment and /agent/query endpoints."""
from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .agent import run_agent
from .cache import get_cached, set_cached
from .news import fetch_news
from .sentiment import analyze

load_dotenv()


app = FastAPI(title="AlphaSignal API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SentimentRequest(BaseModel):
    ticker: str = Field(..., examples=["NVDA"])


class SentimentResponse(BaseModel):
    ticker: str
    overall_sentiment: str
    confidence: float
    label_distribution: dict[str, int]
    headlines_analyzed: int
    cached: bool


class AgentRequest(BaseModel):
    question: str = Field(..., examples=["Is NVDA bullish this week and why?"])


class AgentResponse(BaseModel):
    question: str
    answer: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/sentiment", response_model=SentimentResponse)
async def sentiment_endpoint(req: SentimentRequest):
    """Return cached or freshly computed sentiment for a ticker."""
    ticker = req.ticker.upper().strip()

    cached = get_cached(ticker)
    if cached:
        cached["cached"] = True
        return cached

    headlines = fetch_news(ticker)
    if not headlines:
        raise HTTPException(status_code=404, detail=f"No news found for ticker {ticker}")

    counts: dict[str, int] = {"bearish": 0, "neutral": 0, "bullish": 0}
    confidences: list[float] = []
    for h in headlines:
        r = analyze(h)
        counts[r.label] += 1
        confidences.append(r.confidence)

    dominant = max(counts, key=lambda k: counts[k])
    result: dict = {
        "ticker": ticker,
        "overall_sentiment": dominant,
        "confidence": round(sum(confidences) / len(confidences), 4),
        "label_distribution": counts,
        "headlines_analyzed": len(headlines),
        "cached": False,
    }
    set_cached(ticker, result)
    return result


@app.post("/agent/query", response_model=AgentResponse)
async def agent_query_endpoint(req: AgentRequest):
    """Run the LangGraph agent on a natural-language market question."""
    try:
        answer = run_agent(req.question)
        return {"question": req.question, "answer": answer}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    # Run as: python -m backend.main  (from alphasignal/ root)
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
