"""
Hidden Markov Model gesture classifier.

One GaussianHMM is trained per gesture class.
Prediction is argmax over per-class log-likelihoods.

The observation sequence for each timestep is the full (x, y, vx, vy) feature
vector, modelled by a Gaussian emission per state.

Library: hmmlearn  (pip install hmmlearn)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import numpy as np

from models.base import GestureModel


class HMMClassifier(GestureModel):
    """
    Per-class Gaussian HMM classifier.

    Parameters
    ----------
    gestures    : list[str]
    n_states    : int    number of hidden states per HMM  (default 6)
    n_iter      : int    EM iterations  (default 100)
    covariance_type : str  'diag' or 'full' (default 'diag', faster)
    """

    name = "hmm"

    def __init__(
        self,
        gestures: List[str],
        n_states: int = 6,
        n_iter: int = 100,
        covariance_type: str = "diag",
    ):
        super().__init__(gestures)
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self._models: dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HMMClassifier":
        """
        Fit one GaussianHMM per class using all training samples for that class.

        Parameters
        ----------
        X : (N, 64, 4)
        y : (N,) int
        """
        from hmmlearn.hmm import GaussianHMM

        y = np.asarray(y)
        for cls_idx, gesture in enumerate(self.gestures):
            mask = y == cls_idx
            if mask.sum() == 0:
                continue
            # Concatenate all trajectories for this class into a single
            # observation stream; record lengths for hmmlearn
            X_cls = X[mask]                        # (n_samples, 64, 4)
            lengths = [64] * len(X_cls)
            obs = X_cls.reshape(-1, 4).astype(np.float64)

            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                verbose=False,
            )
            model.fit(obs, lengths)
            self._models[cls_idx] = model

        self._is_fitted = True
        return self

    def _log_likelihood(self, model, traj: np.ndarray) -> float:
        """Score a single (64, 4) trajectory under a GaussianHMM."""
        try:
            return model.score(traj.astype(np.float64))
        except Exception:
            return -np.inf

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._is_fitted, "Call fit() before predict()"
        X = self._ensure_batch(X)
        preds = []
        for traj in X:
            scores = [
                self._log_likelihood(self._models[i], traj)
                if i in self._models else -np.inf
                for i in range(self.n_classes)
            ]
            preds.append(int(np.argmax(scores)))
        return np.array(preds, dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Softmax over log-likelihoods to produce pseudo-probabilities.
        """
        assert self._is_fitted, "Call fit() before predict_proba()"
        X = self._ensure_batch(X)
        proba_batch = []
        for traj in X:
            scores = np.array([
                self._log_likelihood(self._models[i], traj)
                if i in self._models else -np.inf
                for i in range(self.n_classes)
            ], dtype=np.float64)
            # Stable softmax
            scores -= scores.max()
            exp_s = np.exp(scores)
            proba_batch.append((exp_s / exp_s.sum()).astype(np.float32))
        return np.array(proba_batch)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "gestures": self.gestures,
                "n_states": self.n_states,
                "n_iter": self.n_iter,
                "covariance_type": self.covariance_type,
                "models": self._models,
            }, f)

    def load(self, path: str | Path) -> "HMMClassifier":
        with open(str(path), "rb") as f:
            d = pickle.load(f)
        self.gestures = d["gestures"]
        self.n_states = d["n_states"]
        self.n_iter = d["n_iter"]
        self.covariance_type = d["covariance_type"]
        self._models = d["models"]
        self._is_fitted = True
        return self
