import { useState } from "react";

const API = "http://127.0.0.1:8000";

const SENTIMENT_COLORS = {
  positive: "text-emerald-400 bg-emerald-950 border-emerald-700",
  negative: "text-red-400 bg-red-950 border-red-700",
  neutral: "text-slate-300 bg-slate-800 border-slate-600",
};

const SENTIMENT_ICONS = { positive: "▲", negative: "▼", neutral: "◆" };

function Badge({ label }) {
  const cls = SENTIMENT_COLORS[label] ?? SENTIMENT_COLORS.neutral;
  return (
    <span
      className={`inline-flex items-center gap-1 px-3 py-1 rounded-full border text-sm font-semibold uppercase tracking-wide ${cls}`}
    >
      {SENTIMENT_ICONS[label]} {label}
    </span>
  );
}

function DistBar({ label, count, total }) {
  const pct = total > 0 ? Math.round((count / total) * 100) : 0;
  const color =
    label === "positive"
      ? "bg-emerald-500"
      : label === "negative"
      ? "bg-red-500"
      : "bg-slate-500";
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-slate-400">
        <span className="capitalize">{label}</span>
        <span>
          {count} ({pct}%)
        </span>
      </div>
      <div className="h-2 rounded-full bg-slate-700">
        <div
          className={`h-2 rounded-full ${color} transition-all duration-500`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function Card({ title, children }) {
  return (
    <div className="rounded-2xl border border-slate-700 bg-slate-900 p-6 shadow-lg">
      <h2 className="mb-4 text-lg font-semibold text-slate-200">{title}</h2>
      {children}
    </div>
  );
}

export default function App() {
  // Sentiment state
  const [ticker, setTicker] = useState("");
  const [sentimentResult, setSentimentResult] = useState(null);
  const [sentimentLoading, setSentimentLoading] = useState(false);
  const [sentimentError, setSentimentError] = useState("");

  // Agent state
  const [question, setQuestion] = useState("");
  const [agentResult, setAgentResult] = useState(null);
  const [agentLoading, setAgentLoading] = useState(false);
  const [agentError, setAgentError] = useState("");

  async function handleSentiment(e) {
    e.preventDefault();
    if (!ticker.trim()) return;
    setSentimentLoading(true);
    setSentimentError("");
    setSentimentResult(null);
    try {
      const res = await fetch(`${API}/sentiment`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: ticker.trim().toUpperCase() }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      setSentimentResult(await res.json());
    } catch (err) {
      setSentimentError(err.message);
    } finally {
      setSentimentLoading(false);
    }
  }

  async function handleAgentQuery(e) {
    e.preventDefault();
    if (!question.trim()) return;
    setAgentLoading(true);
    setAgentError("");
    setAgentResult(null);
    try {
      const res = await fetch(`${API}/agent/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim() }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      setAgentResult(await res.json());
    } catch (err) {
      setAgentError(err.message);
    } finally {
      setAgentLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 px-4 py-10">
      <div className="mx-auto max-w-2xl space-y-8">
        {/* Header */}
        <header className="text-center space-y-1">
          <h1 className="text-4xl font-bold tracking-tight text-white">
            Alpha<span className="text-emerald-400">Signal</span>
          </h1>
          <p className="text-slate-400 text-sm">
            Financial news sentiment analysis &amp; market intelligence
          </p>
        </header>

        {/* ── Sentiment Analysis ── */}
        <Card title="Sentiment Analysis">
          <form onSubmit={handleSentiment} className="flex gap-2">
            <input
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="Ticker (e.g. NVDA)"
              maxLength={10}
              className="flex-1 rounded-lg bg-slate-800 border border-slate-600 px-4 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
            />
            <button
              type="submit"
              disabled={sentimentLoading || !ticker.trim()}
              className="rounded-lg bg-emerald-600 px-5 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {sentimentLoading ? "Analyzing…" : "Analyze"}
            </button>
          </form>

          {sentimentError && (
            <p className="mt-3 text-sm text-red-400">{sentimentError}</p>
          )}

          {sentimentResult && (
            <div className="mt-5 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">
                    {sentimentResult.ticker} · {sentimentResult.headlines_analyzed} headlines
                    {sentimentResult.cached && (
                      <span className="ml-2 text-yellow-500">(cached)</span>
                    )}
                  </p>
                  <Badge label={sentimentResult.overall_sentiment} />
                </div>
                <div className="text-right">
                  <p className="text-xs text-slate-500">Confidence</p>
                  <p className="text-2xl font-bold text-white">
                    {(sentimentResult.confidence * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
              <div className="space-y-2">
                {Object.entries(sentimentResult.label_distribution).map(([label, count]) => (
                  <DistBar
                    key={label}
                    label={label}
                    count={count}
                    total={sentimentResult.headlines_analyzed}
                  />
                ))}
              </div>
            </div>
          )}
        </Card>

        {/* ── Agent Query ── */}
        <Card title="Agent Query">
          <form onSubmit={handleAgentQuery} className="space-y-3">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              rows={3}
              placeholder='Ask a market question — e.g. "Is NVDA bullish this week and why?"'
              className="w-full rounded-lg bg-slate-800 border border-slate-600 px-4 py-3 text-sm text-white placeholder-slate-500 resize-none focus:outline-none focus:ring-2 focus:ring-emerald-500"
            />
            <button
              type="submit"
              disabled={agentLoading || !question.trim()}
              className="w-full rounded-lg bg-indigo-600 py-2 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {agentLoading ? "Thinking…" : "Ask Agent"}
            </button>
          </form>

          {agentError && (
            <p className="mt-3 text-sm text-red-400">{agentError}</p>
          )}

          {agentResult && (
            <div className="mt-5">
              <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">
                Agent Answer
              </p>
              <div className="rounded-lg bg-slate-800 border border-slate-700 px-4 py-3 text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">
                {agentResult.answer}
              </div>
            </div>
          )}
        </Card>

        <p className="text-center text-xs text-slate-600">
          Powered by FinBERT · LangGraph · Ollama llama3.2 · ChromaDB
        </p>
      </div>
    </div>
  );
}
