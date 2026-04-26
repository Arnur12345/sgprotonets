"""Tier 1 sanity check: can we predict the CheXbert label from the report alone?

If AUC is very high (>~0.9), the report contains the label by construction —
which means feeding reports into SGProtoNet is leakage, not signal.

Trains a logistic regression head over frozen text embeddings:
  report -> [BERT/BiomedCLIP] -> embedding -> linear -> label
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.preprocessing import preprocess_report


@torch.no_grad()
def encode_reports(
    reports: list[str],
    model_name: str,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 256,
) -> np.ndarray:
    """Encode each report into a [CLS] embedding using a frozen text encoder."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    embeddings = []
    for i in range(0, len(reports), batch_size):
        batch = reports[i : i + batch_size]
        # Replace empty strings with a single space so the tokenizer doesn't choke
        batch = [t if t.strip() else " " for t in batch]
        toks = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        out = model(**toks)
        cls = out.last_hidden_state[:, 0, :]  # [CLS]
        embeddings.append(cls.cpu().numpy())
        if (i // batch_size) % 10 == 0:
            print(f"  encoded {i + len(batch)}/{len(reports)}")

    return np.concatenate(embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument(
        "--model_name",
        default="google-bert/bert-base-uncased",
        help="HF text encoder. Default matches configs/default.yaml.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=[
            "effusion", "cardiomegaly", "infiltrate", "pneumothorax", "emphysema",
            "mass", "nodule", "pneumonia", "atelectasis", "consolidation",
        ],
        help="Classes to include (defaults = train+val classes from default.yaml).",
    )
    parser.add_argument("--test_frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Text encoder: {args.model_name}")

    # Load metadata
    df = pd.read_csv(Path(args.data_dir) / "metadata.csv")
    df = df[df["label"].isin(args.classes)].reset_index(drop=True)
    df["report"] = df["report"].fillna("").apply(preprocess_report)

    print(f"\nDataset: {len(df)} samples across {df['label'].nunique()} classes")
    print(df["label"].value_counts().to_string())

    # Stratified train/test split (per-sample, not per-class — we want a held-out
    # AUC on samples whose reports the classifier has not seen).
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_frac,
        random_state=args.seed,
        stratify=df["label"],
    )
    print(f"\nTrain: {len(train_df)}, Test: {len(test_df)}")

    # Encode reports
    print("\nEncoding train reports...")
    X_train = encode_reports(train_df["report"].tolist(), args.model_name, device)
    print("Encoding test reports...")
    X_test = encode_reports(test_df["report"].tolist(), args.model_name, device)

    # Label encoding
    classes = sorted(df["label"].unique())
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y_train = np.array([cls_to_idx[l] for l in train_df["label"]])
    y_test = np.array([cls_to_idx[l] for l in test_df["label"]])

    # Train logistic regression
    print("\nFitting logistic regression head...")
    clf = LogisticRegression(
        max_iter=2000,
        C=1.0,
        multi_class="multinomial",
        random_state=args.seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    # One-vs-rest AUC per class + macro
    aucs = {}
    for i, c in enumerate(classes):
        y_true_bin = (y_test == i).astype(int)
        if y_true_bin.sum() == 0 or y_true_bin.sum() == len(y_true_bin):
            aucs[c] = float("nan")
        else:
            aucs[c] = roc_auc_score(y_true_bin, y_proba[:, i])
    macro_auc = float(np.nanmean(list(aucs.values())))

    print("\n" + "=" * 60)
    print("TIER 1 SANITY CHECK — Text-only label prediction")
    print("=" * 60)
    print(f"Accuracy:    {acc:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Macro AUC:   {macro_auc:.4f}")
    print("\nPer-class AUC (one-vs-rest):")
    for c in classes:
        print(f"  {c:<22s}  {aucs[c]:.4f}")
    print("=" * 60)
    print(
        "\nInterpretation:\n"
        "  Macro AUC > 0.95 -> SEVERE leakage (text is the label)\n"
        "  Macro AUC 0.80-0.95 -> Strong leakage; text-aware training is suspect\n"
        "  Macro AUC < 0.75 -> Reports carry related signal but not the label verbatim"
    )


if __name__ == "__main__":
    main()
