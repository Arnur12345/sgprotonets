"""Tier 1 sanity check (multi-label): can we predict CheXbert MULTI-HOT
labels from the report alone?

Pipeline:
  report -> [BERT/PubMedBERT] -> [CLS] -> per-label LogisticRegression -> multi-hot
Reports macro / micro AUC and per-label AUC. If macro AUC ~ 1.0 the report
carries the entire multi-hot label vector, which means feeding reports into
the multi-label SGProtoNet at training or inference is leakage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    out = []
    for i in range(0, len(reports), batch_size):
        batch = reports[i : i + batch_size]
        batch = [t if t.strip() else " " for t in batch]
        toks = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        cls = model(**toks).last_hidden_state[:, 0, :]
        out.append(cls.cpu().numpy())
        if (i // batch_size) % 10 == 0:
            print(f"  encoded {i + len(batch)}/{len(reports)}")
    return np.concatenate(out, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed_ml",
                        help="Multi-label processed dir (with metadata.csv 'labels' column).")
    parser.add_argument("--model_name", default="google-bert/bert-base-uncased")
    parser.add_argument("--test_frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min_pos", type=int, default=10,
        help="Skip labels with fewer than this many positive samples.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Text encoder: {args.model_name}")

    df = pd.read_csv(Path(args.data_dir) / "metadata.csv")
    df["report"] = df["report"].fillna("").apply(preprocess_report)

    # Parse pipe-delimited multi-label column
    df["label_list"] = df["labels"].apply(
        lambda x: [t for t in str(x).split("|") if t and t != "nan"]
    )
    print(f"Dataset: {len(df)} samples")

    # Build the multi-hot matrix
    all_labels = sorted({l for ll in df["label_list"] for l in ll})
    print(f"Found {len(all_labels)} distinct labels")
    cls_to_idx = {c: i for i, c in enumerate(all_labels)}
    Y = np.zeros((len(df), len(all_labels)), dtype=np.int8)
    for i, ll in enumerate(df["label_list"]):
        for l in ll:
            Y[i, cls_to_idx[l]] = 1

    # Drop labels with too few positives — AUC is undefined or unstable.
    pos_counts = Y.sum(axis=0)
    keep = pos_counts >= args.min_pos
    kept_labels = [all_labels[i] for i in range(len(all_labels)) if keep[i]]
    Y = Y[:, keep]
    print(f"Kept {len(kept_labels)} labels with >= {args.min_pos} positives:")
    for c, n in sorted(zip(kept_labels, Y.sum(axis=0)), key=lambda x: -x[1]):
        print(f"  {c:<22s} {n}")

    # Train/test split (random, NOT stratified — multi-label stratification is non-trivial
    # and 25% test on 3819 rows gives stable AUC even without stratification).
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(idx, test_size=args.test_frac, random_state=args.seed)
    print(f"\nTrain: {len(train_idx)}, Test: {len(test_idx)}")

    print("\nEncoding train reports...")
    X_train = encode_reports(df["report"].iloc[train_idx].tolist(), args.model_name, device)
    print("Encoding test reports...")
    X_test = encode_reports(df["report"].iloc[test_idx].tolist(), args.model_name, device)

    Y_train = Y[train_idx]
    Y_test = Y[test_idx]

    # Fit one binary logistic regression per label.
    print("\nFitting per-label logistic regressions...")
    aucs, aps, f1s = {}, {}, {}
    proba_all = np.zeros_like(Y_test, dtype=float)
    pred_all = np.zeros_like(Y_test, dtype=int)
    for i, lbl in enumerate(kept_labels):
        y_tr = Y_train[:, i]
        y_te = Y_test[:, i]
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            aucs[lbl] = float("nan"); aps[lbl] = float("nan"); f1s[lbl] = float("nan")
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=args.seed, n_jobs=-1)
        clf.fit(X_train, y_tr)
        p = clf.predict_proba(X_test)[:, 1]
        yhat = (p >= 0.5).astype(int)
        proba_all[:, i] = p
        pred_all[:, i] = yhat

        if y_te.sum() == 0 or y_te.sum() == len(y_te):
            aucs[lbl] = float("nan")
        else:
            aucs[lbl] = roc_auc_score(y_te, p)
        aps[lbl] = average_precision_score(y_te, p) if y_te.sum() > 0 else float("nan")
        f1s[lbl] = f1_score(y_te, yhat, zero_division=0)

    macro_auc = float(np.nanmean(list(aucs.values())))
    macro_ap = float(np.nanmean(list(aps.values())))
    macro_f1 = float(np.nanmean(list(f1s.values())))

    # Micro AUC: pool predictions/labels across all labels into one big vector.
    valid_cols = ~np.isnan([aucs[l] for l in kept_labels])
    if valid_cols.any():
        micro_auc = roc_auc_score(
            Y_test[:, valid_cols].ravel(), proba_all[:, valid_cols].ravel()
        )
    else:
        micro_auc = float("nan")

    # Sample-level metrics
    sample_f1 = f1_score(Y_test, pred_all, average="samples", zero_division=0)
    exact_match = (Y_test == pred_all).all(axis=1).mean()

    print("\n" + "=" * 70)
    print("TIER 1 (multi-label) — Text-only label prediction")
    print("=" * 70)
    print(f"  Macro AUC:        {macro_auc:.4f}")
    print(f"  Micro AUC:        {micro_auc:.4f}")
    print(f"  Macro AP:         {macro_ap:.4f}")
    print(f"  Macro F1:         {macro_f1:.4f}")
    print(f"  Sample F1:        {sample_f1:.4f}")
    print(f"  Exact match:      {exact_match:.4f}")
    print()
    print("  Per-label AUC / AP / F1 (sorted by AUC desc):")
    rows = sorted(kept_labels, key=lambda l: -aucs[l] if not np.isnan(aucs[l]) else 0)
    for l in rows:
        print(f"    {l:<22s}  AUC={aucs[l]:.4f}  AP={aps[l]:.4f}  F1={f1s[l]:.4f}")
    print("=" * 70)
    print(
        "\nInterpretation:\n"
        "  Macro AUC > 0.95 -> SEVERE leakage (report = multi-hot label)\n"
        "  Macro AUC 0.80-0.95 -> Strong leakage; multi-label SGProtoNet must avoid\n"
        "                          per-image reports at the prototype path.\n"
        "  Macro AUC < 0.75 -> Reports carry related signal but not the labels verbatim."
    )


if __name__ == "__main__":
    main()
