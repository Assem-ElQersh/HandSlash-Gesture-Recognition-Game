"""
1D Convolutional Neural Network gesture classifier.

Architecture
------------
Input: (batch, 4, 64)  — channels-first for Conv1d
  Conv1d(4 → 64, kernel=5, padding=2) + BN + ReLU
  Conv1d(64 → 128, kernel=3, padding=1) + BN + ReLU
  Dropout(p)
  GlobalAveragePooling1d  →  (batch, 128)
  Linear(128 → n_classes)

Strength: fast inference, captures local temporal patterns.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import GestureModel


class _CNNNet(nn.Module):
    def __init__(self, n_classes: int, channels=(64, 128), dropout: float = 0.3):
        super().__init__()
        c1, c2 = channels
        self.conv1 = nn.Conv1d(4, c1, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(c1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(c2)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, 64)  — channels first
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        x = x.mean(dim=-1)           # global average pool → (B, c2)
        return self.fc(x)


class CNNClassifier(GestureModel):
    """
    1D CNN gesture classifier backed by PyTorch.

    Parameters
    ----------
    gestures : list[str]
    channels : tuple     conv channel sizes (default (64, 128))
    dropout  : float
    device   : str       'cpu' or 'cuda'
    """

    name = "cnn"

    def __init__(
        self,
        gestures: List[str],
        channels: tuple = (64, 128),
        dropout: float = 0.3,
        device: str | None = None,
    ):
        super().__init__(gestures)
        self.channels = channels
        self.dropout = dropout
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._net: _CNNNet | None = None

    def _build_net(self) -> _CNNNet:
        return _CNNNet(self.n_classes, self.channels, self.dropout).to(self.device)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 60,
        batch_size: int = 32,
        lr: float = 1e-3,
        val_data=None,
    ) -> "CNNClassifier":
        """
        Parameters
        ----------
        X         : (N, 64, 4)
        y         : (N,) int
        val_data  : optional (X_val, y_val) tuple for loss monitoring
        """
        from torch.utils.data import TensorDataset, DataLoader

        self._net = self._build_net()
        optimiser = torch.optim.Adam(self._net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)

        # (N, 64, 4) → (N, 4, 64) for Conv1d
        Xt = torch.from_numpy(X).permute(0, 2, 1).to(self.device)
        yt = torch.from_numpy(y).long().to(self.device)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

        self._net.train()
        for _ in range(epochs):
            for xb, yb in loader:
                optimiser.zero_grad()
                loss = F.cross_entropy(self._net(xb), yb)
                loss.backward()
                optimiser.step()
            scheduler.step()

        self._is_fitted = True
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._is_fitted
        X = self._ensure_batch(X)
        Xt = torch.from_numpy(X).permute(0, 2, 1).to(self.device)
        self._net.eval()
        logits = self._net(Xt)
        return logits.argmax(dim=1).cpu().numpy().astype(int)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._is_fitted
        X = self._ensure_batch(X)
        Xt = torch.from_numpy(X).permute(0, 2, 1).to(self.device)
        self._net.eval()
        proba = F.softmax(self._net(Xt), dim=1)
        return proba.cpu().numpy().astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "gestures": self.gestures,
            "channels": self.channels,
            "dropout": self.dropout,
            "state_dict": self._net.state_dict(),
        }, path)

    def load(self, path: str | Path) -> "CNNClassifier":
        d = torch.load(str(path), map_location=self.device)
        self.gestures = d["gestures"]
        self.n_classes = len(self.gestures)
        self.channels = d["channels"]
        self.dropout = d["dropout"]
        self._net = self._build_net()
        self._net.load_state_dict(d["state_dict"])
        self._net.eval()
        self._is_fitted = True
        return self
