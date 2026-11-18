"""Classify new text with either (or both) trained models.

Loads the artifacts saved by ``train.py`` from ``models/``.

Usage:
    python predict.py "The striker scored twice in the final"
    python predict.py --model sklearn "Shares rose after strong earnings"
    python predict.py --model torch "The senate passed the new bill"
    echo "some text" | python predict.py          # reads stdin if no text given
"""

import argparse
import pickle
import sys
from pathlib import Path

import torch

from dataset import LABELS
from train import TextClassifier, encode

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"


def load_sklearn():
    with open(MODELS_DIR / "sklearn_model.pkl", "rb") as f:
        return pickle.load(f)


def load_torch():
    ckpt = torch.load(MODELS_DIR / "torch_model.pt", map_location="cpu", weights_only=False)
    model = TextClassifier(ckpt["vocab_size"],
                           embed_dim=ckpt.get("embed_dim", 100),
                           hidden_dim=ckpt.get("hidden_dim", 128),
                           num_classes=len(ckpt["labels"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def predict_sklearn(model, text):
    probs = model.predict_proba([text])[0]
    idx = int(probs.argmax())
    return LABELS[idx], float(probs[idx])


def predict_torch(model, ckpt, text):
    x = encode([text], ckpt["vocab"], ckpt["max_len"])
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    idx = int(probs.argmax())
    return LABELS[idx], float(probs[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="Text to classify")
    ap.add_argument("--model", choices=["sklearn", "torch", "both"], default="both")
    args = ap.parse_args()

    text = args.text
    if not text:
        text = sys.stdin.read().strip()
    if not text:
        ap.error("Provide text as an argument or via stdin.")

    if not (MODELS_DIR / "sklearn_model.pkl").exists():
        sys.exit("No trained models found. Run `python train.py` first.")

    if args.model in ("sklearn", "both"):
        model = load_sklearn()
        label, conf = predict_sklearn(model, text)
        print(f"[scikit-learn] {label}  (confidence {conf:.2f})")

    if args.model in ("torch", "both"):
        model, ckpt = load_torch()
        label, conf = predict_torch(model, ckpt, text)
        print(f"[PyTorch]      {label}  (confidence {conf:.2f})")


if __name__ == "__main__":
    main()
