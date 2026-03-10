"""
Cross-model benchmark.

For each model that has a saved checkpoint:
  1. Load checkpoint
  2. Evaluate on test split → accuracy, macro F1
  3. Measure inference latency (1000 calls, single sample)
  4. Save confusion matrix PNG
  5. Print side-by-side comparison table

Usage:
  python main.py benchmark
  python benchmarks/compare_models.py           (direct run)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import yaml

from data.dataset import GestureDataset, split_dataset, get_numpy_arrays
from training.evaluator import evaluate, save_confusion_matrix, print_report


MODEL_NAMES = ["dtw", "hmm", "cnn", "lstm", "transformer"]


def _load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _ckpt_path(name: str, ckpt_dir: str) -> Path:
    ext = ".pkl" if name in ("dtw", "hmm") else ".pt"
    return Path(ckpt_dir) / f"{name}{ext}"


def _load_model(name: str, gestures: list, ckpt: Path, cfg: dict):
    if name == "dtw":
        from models.dtw import DTWClassifier
        return DTWClassifier(gestures).load(ckpt)
    if name == "hmm":
        from models.hmm import HMMClassifier
        return HMMClassifier(gestures).load(ckpt)
    if name == "cnn":
        from models.cnn import CNNClassifier
        c = cfg["models"]["cnn"]
        return CNNClassifier(
            gestures,
            channels=tuple(c.get("channels", [64, 128])),
            dropout=c.get("dropout", 0.3),
        ).load(ckpt)
    if name == "lstm":
        from models.lstm import LSTMClassifier
        c = cfg["models"]["lstm"]
        return LSTMClassifier(
            gestures,
            hidden_size=c.get("hidden_size", 128),
            num_layers=c.get("num_layers", 2),
            dropout=c.get("dropout", 0.3),
            bidirectional=c.get("bidirectional", True),
        ).load(ckpt)
    if name == "transformer":
        from models.transformer import TransformerClassifier
        c = cfg["models"]["transformer"]
        return TransformerClassifier(
            gestures,
            d_model=c.get("d_model", 64),
            nhead=c.get("nhead", 4),
            num_encoder_layers=c.get("num_encoder_layers", 2),
            dim_feedforward=c.get("dim_feedforward", 128),
            dropout=c.get("dropout", 0.1),
        ).load(ckpt)
    raise ValueError(f"Unknown model: {name}")


def _measure_latency(model, X_single: np.ndarray, n_calls: int = 1000) -> float:
    """Return median inference latency in milliseconds over n_calls."""
    times = []
    for _ in range(n_calls):
        t0 = time.perf_counter()
        model.predict(X_single)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def run_benchmark(config_path: str = "configs/default.yaml") -> None:
    cfg = _load_config(config_path)
    gestures: list = cfg["gestures"]
    raw_dir = cfg["paths"]["raw_data"]
    ckpt_dir = cfg["paths"]["checkpoints"]
    results_dir = Path(cfg["paths"]["benchmarks"])
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset = GestureDataset(raw_dir, gestures)
    if len(dataset) == 0:
        print("No data found. Run: python main.py collect  then  python main.py train --model all")
        return

    _, _, test_set = split_dataset(dataset)
    X_all, y_all = get_numpy_arrays(dataset)
    X_test = X_all[[i for i in test_set.indices]]
    y_test = y_all[[i for i in test_set.indices]]

    if len(X_test) == 0:
        print("Test split is empty.")
        return

    X_single = X_test[:1]

    rows = []
    for name in MODEL_NAMES:
        ckpt = _ckpt_path(name, ckpt_dir)
        if not ckpt.exists():
            print(f"  [{name}] No checkpoint at {ckpt} — skipping. Run: python main.py train --model {name}")
            continue

        print(f"\nEvaluating {name.upper()}...")
        try:
            model = _load_model(name, gestures, ckpt, cfg)
        except Exception as e:
            print(f"  Failed to load {name}: {e}")
            continue

        metrics = evaluate(model, X_test, y_test)
        latency_ms = _measure_latency(model, X_single)

        # Save confusion matrix
        cm_path = results_dir / f"confusion_{name}.png"
        save_confusion_matrix(
            y_test, metrics["predictions"], gestures, cm_path,
            title=f"Confusion Matrix — {name.upper()}"
        )

        print_report(metrics, name)
        print(f"  Inference latency (median): {latency_ms:.2f} ms")

        rows.append({
            "model": name,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "latency_ms": latency_ms,
        })

    if not rows:
        print("\nNo trained models found. Run: python main.py train --model all")
        return

    # Summary table
    print("\n" + "=" * 65)
    print(f"{'Model':<14} {'Accuracy':>9} {'Macro F1':>9} {'Latency (ms)':>13}")
    print("-" * 65)
    for r in rows:
        print(
            f"{r['model']:<14} "
            f"{r['accuracy']*100:>8.1f}%"
            f"{r['macro_f1']*100:>8.1f}%"
            f"{r['latency_ms']:>12.2f}"
        )
    print("=" * 65)
    print(f"\nConfusion matrices saved to: {results_dir}/")


if __name__ == "__main__":
    run_benchmark()
