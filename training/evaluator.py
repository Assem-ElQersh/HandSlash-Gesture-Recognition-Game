"""
Model evaluation utilities.

Provides:
  evaluate()          — accuracy, macro F1, per-class F1
  confusion_matrix()  — and save as PNG
  print_report()      — formatted table
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np


def evaluate(
    model,
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """
    Evaluate a GestureModel on (X, y).

    Returns
    -------
    dict with keys:
        accuracy    float
        macro_f1    float
        per_class   dict[gesture_name → f1]
        predictions np.ndarray
    """
    from sklearn.metrics import f1_score, accuracy_score

    preds = model.predict(X)
    accuracy = float(accuracy_score(y, preds))
    macro_f1 = float(f1_score(y, preds, average="macro", zero_division=0))
    per_class_f1 = f1_score(y, preds, average=None, zero_division=0)
    per_class = {
        model.gestures[i]: float(per_class_f1[i])
        for i in range(len(model.gestures))
    }

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "predictions": preds,
    }


def confusion_matrix_array(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Return (n_classes, n_classes) confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gestures: List[str],
    out_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    """Render and save a confusion matrix PNG using matplotlib."""
    import matplotlib.pyplot as plt

    n = len(gestures)
    cm = confusion_matrix_array(y_true, y_pred, n)

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n)))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(gestures, rotation=45, ha="right")
    ax.set_yticklabels(gestures)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=9)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=120)
    plt.close()


def print_report(results: dict, model_name: str) -> None:
    """Print a formatted evaluation report."""
    print(f"\n  Results — {model_name.upper()}")
    print(f"  Accuracy : {results['accuracy']*100:.1f}%")
    print(f"  Macro F1 : {results['macro_f1']*100:.1f}%")
    print("  Per-class F1:")
    for gesture, f1 in results["per_class"].items():
        bar = "█" * int(f1 * 20)
        print(f"    {gesture:<15} {f1*100:5.1f}%  {bar}")
