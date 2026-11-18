"""Train and evaluate two text classifiers on AG News.

Models
------
1. Baseline: scikit-learn TF-IDF + LogisticRegression.
2. PyTorch: embedding layer + mean pooling + small MLP classifier,
   trained with early stopping on a held-out validation split.

Methodology (no test leakage)
-----------------------------
- Train/val come from a stratified 20,000-article subsample of the
  official AG News training split (18,000 train / 2,000 validation).
- Hyperparameters and early stopping are selected on the validation
  split only.
- The test set is the full official 7,600-article AG News test split,
  evaluated exactly once per model after training.
- Both models see the same train/val/test data.

Artifacts are saved under ``models/`` so ``predict.py`` can load them
for inference.

Usage:
    python train.py [--train-size 20000] [--max-epochs 15] [--patience 2]
"""

import argparse
import copy
import json
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import LABELS, load_data

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"


def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


# ---------------------------------------------------------------------------
# scikit-learn baseline
# ---------------------------------------------------------------------------

def train_sklearn(X_train, y_train, X_test, y_test):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                                  max_features=50_000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ])
    pipe.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, pipe.predict(X_train))
    test_acc = accuracy_score(y_test, pipe.predict(X_test))
    return pipe, train_acc, test_acc


# ---------------------------------------------------------------------------
# PyTorch model: embedding + mean pooling + MLP
# ---------------------------------------------------------------------------

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128,
                 num_classes=4, dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len) token ids; 0 = padding
        emb = self.embedding(x)                       # (batch, seq, embed)
        mask = (x != 0).float().unsqueeze(-1)         # (batch, seq, 1)
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.mlp(pooled)


def build_vocab(texts, max_vocab=30_000, min_freq=2, max_len=64):
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, freq in counts.most_common(max_vocab):
        if freq < min_freq:
            break
        vocab[tok] = len(vocab)
    return vocab, max_len


def encode(texts, vocab, max_len):
    rows = []
    for text in texts:
        ids = [vocab.get(tok, 1) for tok in tokenize(text)][:max_len]
        ids += [0] * (max_len - len(ids))
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long)


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(1)
    return (preds == y).float().mean().item(), preds.tolist()


def train_torch(X_train, y_train, X_val, y_val, X_test, y_test,
                max_epochs=15, patience=2, seed=42):
    torch.manual_seed(seed)
    vocab, max_len = build_vocab(X_train)
    Xtr = encode(X_train, vocab, max_len)
    Xva = encode(X_val, vocab, max_len)
    Xte = encode(X_test, vocab, max_len)
    ytr = torch.tensor(y_train, dtype=torch.long)
    yva = torch.tensor(y_val, dtype=torch.long)
    yte = torch.tensor(y_test, dtype=torch.long)

    model = TextClassifier(len(vocab), num_classes=len(LABELS))
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)

    best_val, best_state, best_epoch, stale = -1.0, None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        val_acc, _ = evaluate(model, Xva, yva)
        print(f"  epoch {epoch:2d}: val accuracy {val_acc:.4f}")
        if val_acc > best_val:
            best_val, best_state, best_epoch, stale = (
                val_acc, copy.deepcopy(model.state_dict()), epoch, 0)
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stopping after epoch {epoch} "
                      f"(best val {best_val:.4f} at epoch {best_epoch})")
                break
    model.load_state_dict(best_state)
    train_acc, _ = evaluate(model, Xtr, ytr)
    test_acc, test_pred = evaluate(model, Xte, yte)
    return model, vocab, max_len, train_acc, test_acc, test_pred, best_epoch


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, default=20_000)
    ap.add_argument("--max-epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=2)
    args = ap.parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(
        train_size=args.train_size)
    print(f"Dataset: AG News (official split; "
          f"{args.train_size} train-subsample + 7,600 test)")
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}  "
          f"Classes: {LABELS}")

    # --- scikit-learn baseline ---
    sk_model, sk_train, sk_test = train_sklearn(
        X_train, y_train, X_test, y_test)
    sk_report = classification_report(
        y_test, sk_model.predict(X_test), target_names=LABELS,
        zero_division=0, output_dict=True)
    print("\n[scikit-learn] TF-IDF + LogisticRegression")
    print(f"  train accuracy: {sk_train:.4f}")
    print(f"  test  accuracy: {sk_test:.4f}")
    print(classification_report(
        y_test, sk_model.predict(X_test), target_names=LABELS,
        zero_division=0))

    # --- PyTorch model (early stopping on validation only) ---
    print("[PyTorch] Embedding + mean pooling + MLP "
          f"(max {args.max_epochs} epochs, patience {args.patience})")
    pt_model, vocab, max_len, pt_train, pt_test, pt_pred, best_epoch = \
        train_torch(X_train, y_train, X_val, y_val, X_test, y_test,
                    max_epochs=args.max_epochs, patience=args.patience)
    print(f"  best epoch (by val): {best_epoch}")
    print(f"  train accuracy: {pt_train:.4f}")
    print(f"  test  accuracy: {pt_test:.4f}")
    print(classification_report(y_test, pt_pred, target_names=LABELS,
                                zero_division=0))
    pt_report = classification_report(
        y_test, pt_pred, target_names=LABELS, zero_division=0,
        output_dict=True)

    # --- save artifacts ---
    MODELS_DIR.mkdir(exist_ok=True)
    with open(MODELS_DIR / "sklearn_model.pkl", "wb") as f:
        pickle.dump(sk_model, f)
    torch.save({
        "state_dict": pt_model.state_dict(),
        "vocab": vocab,
        "max_len": max_len,
        "vocab_size": len(vocab),
        "embed_dim": pt_model.embed_dim,
        "hidden_dim": pt_model.hidden_dim,
        "labels": LABELS,
    }, MODELS_DIR / "torch_model.pt")
    metrics = {
        "dataset": "AG News (Zhang et al., 2015), official train/test split",
        "train_subsample": len(X_train) + len(X_val),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "classes": LABELS,
        "sklearn": {"train_accuracy": round(sk_train, 4),
                    "test_accuracy": round(sk_test, 4),
                    "per_class_f1": {k: round(v["f1-score"], 4)
                                     for k, v in sk_report.items()
                                     if k in LABELS}},
        "pytorch": {"train_accuracy": round(pt_train, 4),
                    "test_accuracy": round(pt_test, 4),
                    "best_epoch": best_epoch,
                    "max_epochs": args.max_epochs,
                    "per_class_f1": {k: round(v["f1-score"], 4)
                                     for k, v in pt_report.items()
                                     if k in LABELS}},
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved models to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
