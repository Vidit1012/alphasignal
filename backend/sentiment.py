"""Sentiment inference using fine-tuned FinBERT checkpoint."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


CHECKPOINT_PATH = os.getenv("FINBERT_CHECKPOINT", "finetune/checkpoints/best")
DEFAULT_LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


@dataclass
class SentimentResult:
    label: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)


class SentimentAnalyzer:
    def __init__(self, checkpoint: str = CHECKPOINT_PATH):
        # Fall back to base FinBERT if fine-tuned checkpoint not found
        if not os.path.exists(checkpoint):
            checkpoint = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Use the model's label map if available, else fall back
        cfg = self.model.config
        self.label_map: dict[int, str] = (
            {int(k): v for k, v in cfg.id2label.items()}
            if hasattr(cfg, "id2label")
            else DEFAULT_LABEL_MAP
        )

    def predict(self, text: str) -> SentimentResult:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
        scores = {self.label_map[i]: round(float(p), 4) for i, p in enumerate(probs)}
        best_idx = int(torch.argmax(torch.tensor(probs)).item())
        return SentimentResult(
            label=self.label_map[best_idx],
            confidence=round(float(probs[best_idx]), 4),
            scores=scores,
        )


_analyzer: SentimentAnalyzer | None = None


def get_analyzer() -> SentimentAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer


def analyze(text: str) -> SentimentResult:
    """Module-level convenience function — lazily loads the model singleton."""
    return get_analyzer().predict(text)


if __name__ == "__main__":
    # Run as: python -m backend.sentiment
    result = analyze("NVIDIA reported record revenue, beating all analyst estimates.")
    print(result)
