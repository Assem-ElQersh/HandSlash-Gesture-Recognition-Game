"""
Model training entry point.

Handles:
  - Loading dataset from data/raw/
  - Train/val/test split
  - Dispatching to the correct model's fit() method
  - Saving checkpoints to checkpoints/<model_name>.pt (or .pkl)
  - Printing per-epoch val accuracy for neural models
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from data.dataset import GestureDataset, split_dataset, get_numpy_arrays


def _load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _make_model(model_name: str, gestures: list, cfg: dict):
    """Instantiate the requested model with config hyperparameters."""
    if model_name == "dtw":
        from models.dtw import DTWClassifier
        c = cfg["models"]["dtw"]
        return DTWClassifier(gestures, n_neighbors=c.get("n_neighbors", 1))

    if model_name == "hmm":
        from models.hmm import HMMClassifier
        c = cfg["models"]["hmm"]
        return HMMClassifier(
            gestures,
            n_states=c.get("n_states", 6),
            n_iter=c.get("n_iter", 100),
        )

    if model_name == "cnn":
        from models.cnn import CNNClassifier
        c = cfg["models"]["cnn"]
        return CNNClassifier(
            gestures,
            channels=tuple(c.get("channels", [64, 128])),
            dropout=c.get("dropout", 0.3),
        )

    if model_name == "lstm":
        from models.lstm import LSTMClassifier
        c = cfg["models"]["lstm"]
        return LSTMClassifier(
            gestures,
            hidden_size=c.get("hidden_size", 128),
            num_layers=c.get("num_layers", 2),
            dropout=c.get("dropout", 0.3),
            bidirectional=c.get("bidirectional", True),
        )

    if model_name == "transformer":
        from models.transformer import TransformerClassifier
        c = cfg["models"]["transformer"]
        return TransformerClassifier(
            gestures,
            d_model=c.get("d_model", 64),
            nhead=c.get("nhead", 4),
            num_encoder_layers=c.get("num_encoder_layers", 2),
            dim_feedforward=c.get("dim_feedforward", 128),
            dropout=c.get("dropout", 0.1),
        )

    raise ValueError(f"Unknown model: {model_name!r}")


def _checkpoint_path(model_name: str, ckpt_dir: str) -> Path:
    ext = ".pkl" if model_name in ("dtw", "hmm") else ".pt"
    return Path(ckpt_dir) / f"{model_name}{ext}"


def train(
    model_name: str,
    config_path: str = "configs/default.yaml",
    verbose: bool = True,
) -> None:
    cfg = _load_config(config_path)
    gestures: list = cfg["gestures"]
    raw_dir = cfg["paths"]["raw_data"]
    ckpt_dir = cfg["paths"]["checkpoints"]
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    dataset = GestureDataset(raw_dir, gestures)
    if len(dataset) == 0:
        print(f"No data found in '{raw_dir}'. Run: python main.py collect")
        return

    counts = dataset.class_counts()
    if verbose:
        print(f"Dataset: {len(dataset)} samples")
        for g, n in counts.items():
            print(f"  {g}: {n}")

    train_set, val_set, test_set = split_dataset(dataset)
    X_train, y_train = get_numpy_arrays(train_set.dataset)
    X_train = X_train[[i for i in train_set.indices]]
    y_train = y_train[[i for i in train_set.indices]]

    model = _make_model(model_name, gestures, cfg)

    if verbose:
        print(f"\nTraining {model_name} on {len(X_train)} samples...")

    t0 = time.time()
    mc = cfg["models"][model_name]

    if model_name in ("dtw", "hmm"):
        model.fit(X_train, y_train)
    else:
        model.fit(
            X_train, y_train,
            epochs=mc.get("epochs", 60),
            batch_size=mc.get("batch_size", 32),
            lr=mc.get("lr", 1e-3),
        )

    elapsed = time.time() - t0

    ckpt = _checkpoint_path(model_name, ckpt_dir)
    model.save(ckpt)

    if verbose:
        print(f"  Done in {elapsed:.1f}s  →  saved to {ckpt}")

    # Quick val accuracy
    X_val = get_numpy_arrays(val_set.dataset)[0][[i for i in val_set.indices]]
    y_val = get_numpy_arrays(val_set.dataset)[1][[i for i in val_set.indices]]
    if len(X_val) > 0:
        preds = model.predict(X_val)
        val_acc = (preds == y_val).mean() * 100
        if verbose:
            print(f"  Val accuracy: {val_acc:.1f}%  ({len(X_val)} samples)")


def train_all(config_path: str = "configs/default.yaml") -> None:
    """Train all 5 models sequentially."""
    for name in ("dtw", "hmm", "cnn", "lstm", "transformer"):
        print(f"\n{'='*50}")
        print(f"  Model: {name.upper()}")
        print(f"{'='*50}")
        try:
            train(name, config_path=config_path)
        except Exception as e:
            print(f"  ERROR training {name}: {e}")
