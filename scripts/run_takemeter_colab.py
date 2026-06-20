"""Colab-ready training and evaluation pipeline for TakeMeter.

Usage in Colab:
1. Upload data/takemeter_nba_labeled.csv or put it at /content/takemeter_nba_labeled.csv.
2. Runtime -> Change runtime type -> T4 GPU.
3. Run:
   !pip -q install transformers scikit-learn matplotlib groq pandas
   !python run_takemeter_colab.py

Set GROQ_API_KEY in Colab Secrets or the environment before running the baseline.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DATA_PATHS = [
    Path("data/takemeter_nba_labeled.csv"),
    Path("takemeter_nba_labeled.csv"),
    Path("/content/takemeter_nba_labeled.csv"),
]
MODEL_NAME = "distilbert-base-uncased"
LABELS = ["analysis", "hot_take", "reaction"]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}
RANDOM_STATE = 42
MAX_LENGTH = 192
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5


class TakeDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer: AutoTokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def find_data_path() -> Path:
    for path in DATA_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find takemeter_nba_labeled.csv. Upload it to Colab or run from the repo root."
    )


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_path = find_data_path()
    df = pd.read_csv(data_path)
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(str)
    bad_labels = sorted(set(df["label"]) - set(LABELS))
    if bad_labels:
        raise ValueError(f"Unexpected labels found: {bad_labels}")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["label"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = TakeDataset(
        train_df["text"].tolist(),
        [LABEL2ID[label] for label in train_df["label"]],
        tokenizer,
    )
    val_dataset = TakeDataset(
        val_df["text"].tolist(),
        [LABEL2ID[label] for label in val_df["label"]],
        tokenizer,
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            total_loss += float(outputs.loss.item())
        val_metrics = evaluate_model(model, tokenizer, val_df, device=device, split_name="validation")
        print(
            f"epoch={epoch + 1} train_loss={total_loss / max(1, len(train_loader)):.4f} "
            f"val_accuracy={val_metrics['accuracy']:.4f}"
        )

    model.save_pretrained("takemeter_distilbert_model")
    tokenizer.save_pretrained("takemeter_distilbert_model")
    return model, tokenizer, device


def predict_model(model, tokenizer, texts: list[str], device: torch.device) -> tuple[list[str], list[float]]:
    model.eval()
    predictions: list[str] = []
    confidences: list[float] = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[start : start + BATCH_SIZE]
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encodings = {key: value.to(device) for key, value in encodings.items()}
            logits = model(**encodings).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            pred_ids = probs.argmax(axis=1)
            predictions.extend(ID2LABEL[int(pred_id)] for pred_id in pred_ids)
            confidences.extend(float(probs[index, pred_id]) for index, pred_id in enumerate(pred_ids))
    return predictions, confidences


def evaluate_predictions(y_true: list[str], y_pred: list[str]) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def evaluate_model(model, tokenizer, df: pd.DataFrame, device: torch.device, split_name: str) -> dict:
    predictions, confidences = predict_model(model, tokenizer, df["text"].tolist(), device)
    metrics = evaluate_predictions(df["label"].tolist(), predictions)
    print(f"{split_name} accuracy: {metrics['accuracy']:.4f}")
    return metrics | {"predictions": predictions, "confidences": confidences}


def groq_prompt(text: str) -> str:
    return f"""You are classifying r/nba comments into exactly one label.

Labels:
- analysis: structured basketball argument supported by specific reasoning, evidence, statistics, scheme observations, roster/cap context, matchup logic, or historical comparison.
- hot_take: bold evaluative claim or prediction about a player, team, coach, or front office with little or no supporting evidence.
- reaction: immediate emotion, joke, meme, celebration, complaint, or low-context response to a play, result, quote, or news item.

Decision rules:
- If specific evidence would still support the claim after removing emotional wording, choose analysis.
- If the comment makes a broader claim without enough support, choose hot_take.
- If it mainly reacts to the moment without a durable argument, choose reaction.

Return only one of these labels: analysis, hot_take, reaction.

Comment:
{text}
"""


def parse_label(response: str) -> str | None:
    cleaned = response.strip().lower()
    for label in LABELS:
        if cleaned == label:
            return label
    for label in LABELS:
        if label in cleaned:
            return label
    return None


def run_groq_baseline(test_df: pd.DataFrame) -> dict | None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not found; skipping Groq baseline.")
        return None

    from groq import Groq

    client = Groq(api_key=api_key)
    predictions: list[str] = []
    unparseable: list[dict] = []

    for index, row in test_df.iterrows():
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": groq_prompt(row["text"])}],
            temperature=0,
            max_tokens=8,
        )
        raw = response.choices[0].message.content or ""
        label = parse_label(raw)
        if label is None:
            label = "reaction"
            unparseable.append({"index": int(index), "response": raw, "text": row["text"]})
        predictions.append(label)
        time.sleep(0.2)

    metrics = evaluate_predictions(test_df["label"].tolist(), predictions)
    metrics["predictions"] = predictions
    metrics["unparseable"] = unparseable
    print(f"Groq baseline accuracy: {metrics['accuracy']:.4f}")
    print(f"Unparseable Groq responses: {len(unparseable)}")
    return metrics


def save_confusion_matrix(matrix: list[list[int]], output_path: str = "confusion_matrix.png") -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(LABELS)), labels=LABELS, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(LABELS)), labels=LABELS)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Fine-tuned DistilBERT Confusion Matrix")
    for row_index in range(len(LABELS)):
        for col_index in range(len(LABELS)):
            ax.text(col_index, row_index, matrix[row_index][col_index], ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)


def main() -> None:
    train_df, val_df, test_df = load_splits()
    print("Split sizes:", {"train": len(train_df), "validation": len(val_df), "test": len(test_df)})
    print("Train label counts:", train_df["label"].value_counts().to_dict())
    print("Validation label counts:", val_df["label"].value_counts().to_dict())
    print("Test label counts:", test_df["label"].value_counts().to_dict())

    baseline_metrics = run_groq_baseline(test_df)
    model, tokenizer, device = train_model(train_df, val_df)
    fine_tuned_metrics = evaluate_model(model, tokenizer, test_df, device, split_name="test")
    save_confusion_matrix(fine_tuned_metrics["confusion_matrix"])

    test_output = test_df.copy()
    test_output["fine_tuned_prediction"] = fine_tuned_metrics["predictions"]
    test_output["fine_tuned_confidence"] = fine_tuned_metrics["confidences"]
    if baseline_metrics:
        test_output["groq_prediction"] = baseline_metrics["predictions"]
    test_output.to_csv("test_predictions.csv", index=False)

    wrong = test_output[test_output["label"] != test_output["fine_tuned_prediction"]].head(10)
    sample = test_output.head(5)
    results = {
        "labels": LABELS,
        "base_model": MODEL_NAME,
        "hyperparameters": {
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "random_state": RANDOM_STATE,
        },
        "split_sizes": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "groq_baseline": baseline_metrics,
        "fine_tuned": fine_tuned_metrics,
        "wrong_examples_preview": wrong.to_dict(orient="records"),
        "sample_classifications": sample.to_dict(orient="records"),
    }
    Path("evaluation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote evaluation_results.json, confusion_matrix.png, and test_predictions.csv")


if __name__ == "__main__":
    main()
