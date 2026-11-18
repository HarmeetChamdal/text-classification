"""AG News dataset loader.

Uses the AG News corpus (Zhang et al., 2015): 120,000 training and 7,600
test news articles in 4 classes. Class labels in the CSV files are
1=World, 2=Sports, 3=Business, 4=Sci/Tech.

Source: the standard AG News CSVs mirrored at
https://github.com/mhjabreel/CharCnn_Keras (data/ag_news_csv/train.csv
and test.csv). The files are downloaded automatically on first use into
``data/`` (git-ignored, ~30 MB) if they are not already present.

The full 120k training set is large for a from-scratch embedding model
on CPU, so training uses a stratified 20,000-article subsample
(5,000 per class) of the official training split, and the model is
evaluated on the full official 7,600-article test split. The subsample
size is a documented, deliberate choice — the test set is never
subsampled and never touched during training or tuning.
"""

import csv
import urllib.request
from pathlib import Path

import numpy as np

LABELS = ["World", "Sports", "Business", "Sci/Tech"]

TRAIN_URL = (
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/"
    "master/data/ag_news_csv/train.csv"
)
TEST_URL = (
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/"
    "master/data/ag_news_csv/test.csv"
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

DEFAULT_TRAIN_SIZE = 20_000  # stratified subsample of the official train split
DEFAULT_VAL_SIZE = 2_000     # held out from the training subsample for tuning


def _download(url, dest):
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")


def _read_csv(path):
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) != 3:
                continue
            label = int(row[0]) - 1  # 1-based in the file -> 0-based
            texts.append(f"{row[1]} {row[2]}")
            labels.append(label)
    return texts, np.array(labels, dtype=int)


def _stratified_subsample(texts, labels, size, seed):
    rng = np.random.default_rng(seed)
    keep = []
    for cls in range(len(LABELS)):
        idx = np.where(labels == cls)[0]
        take = min(size // len(LABELS), len(idx))
        keep.extend(rng.choice(idx, size=take, replace=False).tolist())
    keep = np.array(sorted(keep))
    return [texts[i] for i in keep], labels[keep]


def load_data(train_size=DEFAULT_TRAIN_SIZE, val_size=DEFAULT_VAL_SIZE,
              seed=42, data_dir=None):
    """Return (X_train, y_train, X_val, y_val, X_test, y_test).

    - Train/val come from a stratified ``train_size`` subsample of the
      official AG News training split; ``val_size`` of it is held out
      for early stopping / hyperparameter selection.
    - Test is the full official 7,600-article AG News test split.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    data_dir.mkdir(exist_ok=True)
    train_path, test_path = data_dir / "train.csv", data_dir / "test.csv"
    if not train_path.exists():
        _download(TRAIN_URL, train_path)
    if not test_path.exists():
        _download(TEST_URL, test_path)

    train_texts, train_labels = _read_csv(train_path)
    test_texts, test_labels = _read_csv(test_path)

    sub_texts, sub_labels = _stratified_subsample(
        train_texts, train_labels, train_size, seed)

    # Stratified train/val split of the subsample (val used for tuning only).
    rng = np.random.default_rng(seed)
    val_idx, train_idx = [], []
    for cls in range(len(LABELS)):
        idx = np.where(sub_labels == cls)[0]
        rng.shuffle(idx)
        per_class_val = val_size // len(LABELS)
        val_idx.extend(idx[:per_class_val].tolist())
        train_idx.extend(idx[per_class_val:].tolist())
    val_idx, train_idx = np.array(val_idx), np.array(train_idx)

    X_train = [sub_texts[i] for i in train_idx]
    y_train = sub_labels[train_idx]
    X_val = [sub_texts[i] for i in val_idx]
    y_val = sub_labels[val_idx]
    return X_train, y_train, X_val, y_val, test_texts, test_labels
