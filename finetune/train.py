"""Fine-tune FinBERT on Twitter Financial News Sentiment with MLflow tracking.

Usage (from alphasignal/ root):
    python -m finetune.train
"""
from __future__ import annotations

import os

import datasets
import mlflow
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


BASE_MODEL = "ProsusAI/finbert"
DATASET_NAME = "zeroshot/twitter-financial-news-sentiment"
OUTPUT_DIR = "finetune/checkpoints"
BEST_DIR = "finetune/checkpoints/best"
NUM_LABELS = 3
MAX_LENGTH = 128
EXPERIMENT_NAME = "alphasignal-finbert-v2"

# Label mapping: 0=bearish (negative), 1=neutral, 2=bullish (positive)
ID2LABEL = {0: "bearish", 1: "neutral", 2: "bullish"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def tokenize_dataset(dataset, tokenizer):
    def _tok(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
    return dataset.map(_tok, batched=True, remove_columns=["text"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "f1_weighted": float(f1_score(labels, preds, average="weighted")),
    }


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(tags={"lr": "5e-5", "note": "higher lr experiment"}):
        # ------------------------------------------------------------------ #
        # 1. Load dataset                                                      #
        # ------------------------------------------------------------------ #
        datasets.disable_caching()
        raw = load_dataset(DATASET_NAME)
        # Dataset ships pre-split train/validation splits
        train_ds = raw["train"]
        val_ds = raw["validation"]

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        train_ds = tokenize_dataset(train_ds, tokenizer)
        val_ds = tokenize_dataset(val_ds, tokenizer)

        train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        # ------------------------------------------------------------------ #
        # 2. Model                                                            #
        # ------------------------------------------------------------------ #
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,   # FinBERT has 3 labels; safe to keep
        )

        # ------------------------------------------------------------------ #
        # 3. Training configuration                                           #
        # ------------------------------------------------------------------ #
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            logging_steps=10,
            report_to="none",   # MLflow logging handled manually below
            fp16=False,         # set True if GPU supports it
        )

        mlflow.log_params({
            "base_model": BASE_MODEL,
            "dataset": DATASET_NAME,
            "epochs": training_args.num_train_epochs,
            "lr": training_args.learning_rate,
            "train_batch_size": training_args.per_device_train_batch_size,
            "max_length": MAX_LENGTH,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
        })

        # ------------------------------------------------------------------ #
        # 4. Train                                                            #
        # ------------------------------------------------------------------ #
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )
        trainer.train()

        # ------------------------------------------------------------------ #
        # 5. Log final metrics                                                #
        # ------------------------------------------------------------------ #
        metrics = trainer.evaluate()
        mlflow.log_metrics({
            "eval_loss": metrics["eval_loss"],
            "eval_accuracy": metrics["eval_accuracy"],
            "eval_f1_macro": metrics["eval_f1_macro"],
            "eval_f1_weighted": metrics["eval_f1_weighted"],
        })

        # ------------------------------------------------------------------ #
        # 6. Save best checkpoint                                             #
        # ------------------------------------------------------------------ #
        os.makedirs(BEST_DIR, exist_ok=True)
        trainer.save_model(BEST_DIR)
        tokenizer.save_pretrained(BEST_DIR)
        mlflow.log_artifact(BEST_DIR, artifact_path="best_checkpoint")

        print(f"\nBest checkpoint saved to: {BEST_DIR}")
        print(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()
