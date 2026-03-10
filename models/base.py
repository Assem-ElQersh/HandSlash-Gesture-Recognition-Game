"""
Abstract base class that every gesture model must implement.

This contract ensures that:
  - All models share the same fit / predict / predict_proba interface
  - The training loop, evaluator, and real-time pipeline can treat all
    models uniformly without knowing which one is loaded
  - Serialisation (save/load) is standardised

Input conventions
-----------------
X : np.ndarray  shape (N, 64, 4)   batch of feature trajectories
y : np.ndarray  shape (N,)  int    class indices

Single-sample inference uses X shape (1, 64, 4) or (64, 4).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class GestureModel(ABC):
    """
    Abstract base class for all gesture classifiers.

    Subclasses must implement: fit, predict, predict_proba, save, load.
    name and input_shape are informational class attributes.
    """

    #: Human-readable model name (set in each subclass)
    name: str = "base"

    def __init__(self, gestures: List[str]):
        """
        Parameters
        ----------
        gestures : list[str]  ordered class names; index = label integer
        """
        self.gestures = gestures
        self.n_classes = len(gestures)
        self._is_fitted = False

    # ------------------------------------------------------------------ #
    #  Required interface                                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GestureModel":
        """
        Train the model.

        Parameters
        ----------
        X : np.ndarray  (N, 64, 4)
        y : np.ndarray  (N,) int

        Returns
        -------
        self
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class indices.

        Parameters
        ----------
        X : np.ndarray  (N, 64, 4) or (64, 4) for single sample

        Returns
        -------
        np.ndarray  (N,) int
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (or log-likelihood scores).

        Parameters
        ----------
        X : np.ndarray  (N, 64, 4) or (64, 4)

        Returns
        -------
        np.ndarray  (N, n_classes) float  — rows should sum to 1 where possible
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Serialise model to *path*."""

    @abstractmethod
    def load(self, path: str | Path) -> "GestureModel":
        """Deserialise model from *path* and return self."""

    # ------------------------------------------------------------------ #
    #  Convenience helpers (shared implementation)                         #
    # ------------------------------------------------------------------ #

    def predict_label(self, X: np.ndarray) -> str:
        """Predict and return the gesture name (not the index)."""
        idx = int(self.predict(self._ensure_batch(X))[0])
        return self.gestures[idx]

    def predict_proba_dict(self, X: np.ndarray) -> Dict[str, float]:
        """Return {gesture_name: probability} for a single sample."""
        proba = self.predict_proba(self._ensure_batch(X))[0]
        return {g: float(p) for g, p in zip(self.gestures, proba)}

    @staticmethod
    def _ensure_batch(X: np.ndarray) -> np.ndarray:
        """Add batch dimension if X is a single sample (64, 4)."""
        if X.ndim == 2:
            return X[np.newaxis]
        return X

    def __repr__(self) -> str:
        fitted = "fitted" if self._is_fitted else "not fitted"
        return f"{self.__class__.__name__}(classes={self.gestures}, {fitted})"
