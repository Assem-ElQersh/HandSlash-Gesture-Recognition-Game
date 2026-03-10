"""
Bidirectional LSTM gesture classifier.

Architecture
------------
Input: (batch, 64, 4)  — sequence-first
  Bidirectional LSTM(hidden=128, num_layers=2, dropout=0.3)
  Take last hidden state (concat fwd + bwd) → (batch, 256)
  Dropout
  Linear(256 → n_classes)

Strength: captures long-range temporal dependencies in both directions.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import GestureModel


class _LSTMNet(nn.Module):
    def __init__(
        self,
        n_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=4,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        directions = 2 if bidirectional else 1
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * directions, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 64, 4)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * directions, B, hidden)
        # Grab last layer, both directions
        if self.lstm.bidirectional:
            fwd = h_n[-2]   # last fwd layer
            bwd = h_n[-1]   # last bwd layer
            h = torch.cat([fwd, bwd], dim=1)
        else:
            h = h_n[-1]
        return self.fc(self.drop(h))


class LSTMClassifier(GestureModel):
    """
    Bidirectional LSTM gesture classifier backed by PyTorch.

    Parameters
    ----------
    gestures      : list[str]
    hidden_size   : int   LSTM hidden units per direction (default 128)
    num_layers    : int   stacked LSTM layers (default 2)
    dropout       : float
    bidirectional : bool
    device        : str
    """

    name = "lstm"

    def __init__(
        self,
        gestures: List[str],
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        device: str | None = None,
    ):
        super().__init__(gestures)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._net: _LSTMNet | None = None

    def _build_net(self) -> _LSTMNet:
        return _LSTMNet(
            self.n_classes, self.hidden_size, self.num_layers,
            self.dropout, self.bidirectional,
        ).to(self.device)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 60,
        batch_size: int = 32,
        lr: float = 1e-3,
    ) -> "LSTMClassifier":
        from torch.utils.data import TensorDataset, DataLoader

        self._net = self._build_net()
        optimiser = torch.optim.Adam(self._net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)

        Xt = torch.from_numpy(X).to(self.device)     # (N, 64, 4)
        yt = torch.from_numpy(y).long().to(self.device)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

        self._net.train()
        for _ in range(epochs):
            for xb, yb in loader:
                optimiser.zero_grad()
                loss = F.cross_entropy(self._net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimiser.step()
            scheduler.step()

        self._is_fitted = True
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._is_fitted
        X = self._ensure_batch(X)
        Xt = torch.from_numpy(X).to(self.device)
        self._net.eval()
        return self._net(Xt).argmax(dim=1).cpu().numpy().astype(int)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._is_fitted
        X = self._ensure_batch(X)
        Xt = torch.from_numpy(X).to(self.device)
        self._net.eval()
        return F.softmax(self._net(Xt), dim=1).cpu().numpy().astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "gestures": self.gestures,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "state_dict": self._net.state_dict(),
        }, path)

    def load(self, path: str | Path) -> "LSTMClassifier":
        d = torch.load(str(path), map_location=self.device)
        self.gestures = d["gestures"]
        self.n_classes = len(self.gestures)
        self.hidden_size = d["hidden_size"]
        self.num_layers = d["num_layers"]
        self.dropout = d["dropout"]
        self.bidirectional = d["bidirectional"]
        self._net = self._build_net()
        self._net.load_state_dict(d["state_dict"])
        self._net.eval()
        self._is_fitted = True
        return self
