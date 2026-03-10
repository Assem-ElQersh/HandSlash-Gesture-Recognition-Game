"""
DTW-based 1-Nearest-Neighbour gesture classifier.

Dynamic Time Warping is the classical baseline for gesture recognition.
It computes the optimal alignment cost between two sequences, making it
naturally robust to speed variation without requiring resampling.

We use the resampled (x,y) part of the feature vector (columns 0:2),
since DTW already handles temporal distortion — adding velocity would
double-count speed information.

Library: dtaidistance  (pip install dtaidistance)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import numpy as np

from models.base import GestureModel


class DTWClassifier(GestureModel):
    """
    1-NN classifier using Dynamic Time Warping distance.

    Parameters
    ----------
    gestures    : list[str]  class names
    n_neighbors : int        k for k-NN vote (default 1)
    use_xy_only : bool       use only (x, y) columns for DTW (recommended)
    """

    name = "dtw"

    def __init__(
        self,
        gestures: List[str],
        n_neighbors: int = 1,
        use_xy_only: bool = True,
    ):
        super().__init__(gestures)
        self.n_neighbors = n_neighbors
        self.use_xy_only = use_xy_only
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def _prep(self, X: np.ndarray) -> np.ndarray:
        """Extract the columns used for DTW distance."""
        if self.use_xy_only:
            return X[:, :, :2]        # (N, 64, 2)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DTWClassifier":
        """
        Store training templates (DTW is non-parametric — no learning occurs).

        Parameters
        ----------
        X : (N, 64, 4)
        y : (N,) int
        """
        self._X_train = self._prep(X).astype(np.float64)
        self._y_train = np.asarray(y, dtype=int)
        self._is_fitted = True
        return self

    def _dtw_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        DTW distance between two sequences using dtaidistance if available,
        falling back to a pure-numpy implementation.

        a, b : (T, D)  float64
        """
        try:
            from dtaidistance import dtw_ndim
            return dtw_ndim.distance(a, b)
        except ImportError:
            return self._dtw_numpy(a, b)

    @staticmethod
    def _dtw_numpy(a: np.ndarray, b: np.ndarray) -> float:
        """Pure-numpy DTW — O(T^2), used only when dtaidistance is absent."""
        n, m = len(a), len(b)
        dtw_mat = np.full((n + 1, m + 1), np.inf)
        dtw_mat[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = float(np.sum((a[i - 1] - b[j - 1]) ** 2))
                dtw_mat[i, j] = cost + min(
                    dtw_mat[i - 1, j],
                    dtw_mat[i, j - 1],
                    dtw_mat[i - 1, j - 1],
                )
        return float(dtw_mat[n, m])

    def _compute_distances(self, query: np.ndarray) -> np.ndarray:
        """Compute DTW distance from *query* (64, D) to all training samples."""
        dists = np.array([
            self._dtw_distance(query, ref)
            for ref in self._X_train
        ])
        return dists

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._is_fitted, "Call fit() before predict()"
        X = self._ensure_batch(X)
        Xp = self._prep(X).astype(np.float64)
        preds = []
        for query in Xp:
            dists = self._compute_distances(query)
            nn_idx = np.argsort(dists)[: self.n_neighbors]
            nn_labels = self._y_train[nn_idx]
            # Majority vote
            counts = np.bincount(nn_labels, minlength=self.n_classes)
            preds.append(int(np.argmax(counts)))
        return np.array(preds, dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Soft probabilities via inverse-distance weighting over k neighbours.
        """
        assert self._is_fitted, "Call fit() before predict_proba()"
        X = self._ensure_batch(X)
        Xp = self._prep(X).astype(np.float64)
        proba_batch = []
        for query in Xp:
            dists = self._compute_distances(query)
            nn_idx = np.argsort(dists)[: self.n_neighbors]
            nn_dists = dists[nn_idx]
            nn_labels = self._y_train[nn_idx]
            # Inverse-distance weights
            weights = 1.0 / (nn_dists + 1e-8)
            scores = np.zeros(self.n_classes)
            for w, lbl in zip(weights, nn_labels):
                scores[lbl] += w
            proba_batch.append(scores / scores.sum())
        return np.array(proba_batch, dtype=np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "gestures": self.gestures,
                "n_neighbors": self.n_neighbors,
                "use_xy_only": self.use_xy_only,
                "X_train": self._X_train,
                "y_train": self._y_train,
            }, f)

    def load(self, path: str | Path) -> "DTWClassifier":
        with open(str(path), "rb") as f:
            d = pickle.load(f)
        self.gestures = d["gestures"]
        self.n_neighbors = d["n_neighbors"]
        self.use_xy_only = d["use_xy_only"]
        self._X_train = d["X_train"]
        self._y_train = d["y_train"]
        self._is_fitted = True
        return self
