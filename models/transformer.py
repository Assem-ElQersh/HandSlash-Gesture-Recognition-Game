"""
Transformer encoder gesture classifier.

Architecture
------------
Input: (batch, 64, 4)
  Linear projection: 4 → d_model (64)
  Prepend a learnable [CLS] token  →  (batch, 65, d_model)
  Sinusoidal positional encoding
  N × TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
  Extract CLS token output  →  (batch, d_model)
  Linear(d_model → n_classes)

Strength: global attention over all timesteps; best for complex, longer gestures.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import GestureModel


class _SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class _TransformerNet(nn.Module):
    def __init__(
        self,
        n_classes: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(4, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_enc = _SinusoidalPE(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, n_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 64, 4)
        x = self.input_proj(x)                            # (B, 64, d_model)
        cls = self.cls_token.expand(x.size(0), -1, -1)   # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                   # (B, 65, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = self.norm(x[:, 0])                           # CLS token output
        return self.fc(x)


class TransformerClassifier(GestureModel):
    """
    Transformer encoder gesture classifier backed by PyTorch.

    Parameters
    ----------
    gestures           : list[str]
    d_model            : int    embedding dimension (default 64)
    nhead              : int    attention heads (default 4)
    num_encoder_layers : int    (default 2)
    dim_feedforward    : int    FFN inner dimension (default 128)
    dropout            : float
    device             : str
    """

    name = "transformer"

    def __init__(
        self,
        gestures: List[str],
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        device: str | None = None,
    ):
        super().__init__(gestures)
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._net: _TransformerNet | None = None

    def _build_net(self) -> _TransformerNet:
        return _TransformerNet(
            self.n_classes, self.d_model, self.nhead,
            self.num_encoder_layers, self.dim_feedforward, self.dropout,
        ).to(self.device)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 60,
        batch_size: int = 32,
        lr: float = 1e-3,
    ) -> "TransformerClassifier":
        from torch.utils.data import TensorDataset, DataLoader

        self._net = self._build_net()
        optimiser = torch.optim.Adam(self._net.parameters(), lr=lr, weight_decay=1e-4)
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
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_encoder_layers": self.num_encoder_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "state_dict": self._net.state_dict(),
        }, path)

    def load(self, path: str | Path) -> "TransformerClassifier":
        d = torch.load(str(path), map_location=self.device)
        self.gestures = d["gestures"]
        self.n_classes = len(self.gestures)
        self.d_model = d["d_model"]
        self.nhead = d["nhead"]
        self.num_encoder_layers = d["num_encoder_layers"]
        self.dim_feedforward = d["dim_feedforward"]
        self.dropout = d["dropout"]
        self._net = self._build_net()
        self._net.load_state_dict(d["state_dict"])
        self._net.eval()
        self._is_fitted = True
        return self
