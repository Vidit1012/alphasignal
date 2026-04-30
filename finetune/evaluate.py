"""Evaluate fine-tuned FinBERT checkpoint and log results to MLflow.

Usage (from alphasignal/ root):
    python -m finetune.evaluate
    FINBERT_CHECKPOINT=finetune/checkpoints/best python -m finetune.evaluate
"""
from __future__ import annotations

import os

import datasets
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from datasets import load_dataset
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


CHECKPOINT = os.getenv("FINBERT_CHECKPOINT", "finetune/checkpoints/best")
BASE_FALLBACK = "ProsusAI/finbert"
DATASET_NAME = "zeroshot/twitter-financial-news-sentiment"
LABELS = ["bearish", "neutral", "bullish"]
MAX_LENGTH = 128
EXPERIMENT_NAME = "alphasignal-finbert-eval"
ARTIFACTS_DIR = "finetune/eval_artifacts"


def main():
    checkpoint = CHECKPOINT if os.path.exists(CHECKPOINT) else BASE_FALLBACK
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_param("checkpoint", checkpoint)

        # ------------------------------------------------------------------ #
        # 1. Load the same validation split used during training              #
        # ------------------------------------------------------------------ #
        datasets.disable_caching()
        raw = load_dataset(DATASET_NAME)
        test_ds = raw["validation"]

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        def tokenize(batch):
            return tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
            )

        test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text"])
        test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        # ------------------------------------------------------------------ #
        # 2. Predict                                                          #
        # ------------------------------------------------------------------ #
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=len(LABELS), ignore_mismatched_sizes=True
        )
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=os.path.join(ARTIFACTS_DIR, "tmp"),
                per_device_eval_batch_size=32,
                report_to="none",
            ),
        )
        preds_out = trainer.predict(test_ds)
        preds = np.argmax(preds_out.predictions, axis=-1)
        true_labels = preds_out.label_ids

        # ------------------------------------------------------------------ #
        # 3. Classification report                                            #
        # ------------------------------------------------------------------ #
        report = classification_report(true_labels, preds, target_names=LABELS)
        print(report)
        report_path = os.path.join(ARTIFACTS_DIR, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        # ------------------------------------------------------------------ #
        # 4. Confusion matrix                                                 #
        # ------------------------------------------------------------------ #
        cm = confusion_matrix(true_labels, preds)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS).plot(
            ax=ax, colorbar=False
        )
        plt.title("FinBERT Confusion Matrix")
        plt.tight_layout()
        cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
        fig.savefig(cm_path, dpi=150)
        mlflow.log_artifact(cm_path)
        plt.close()

        # ------------------------------------------------------------------ #
        # 5. Scalar metrics                                                   #
        # ------------------------------------------------------------------ #
        mlflow.log_metrics({
            "accuracy": float(accuracy_score(true_labels, preds)),
            "f1_macro": float(f1_score(true_labels, preds, average="macro")),
            "f1_weighted": float(f1_score(true_labels, preds, average="weighted")),
        })

        print(f"\nArtifacts saved to {ARTIFACTS_DIR} and logged to MLflow.")


if __name__ == "__main__":
    main()
