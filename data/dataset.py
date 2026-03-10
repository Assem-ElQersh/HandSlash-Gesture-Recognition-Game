"""
PyTorch Dataset for gesture trajectories.

Expected on-disk layout (written by collector.py):
    data/raw/<label>/<timestamp>.npz
Each .npz contains:
    trajectory : np.ndarray  shape (64, 4)  float32  [x, y, vx, vy]
    label      : str
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split


class GestureDataset(Dataset):
    """
    Loads all .npz files under *data_dir* and exposes them as
    (trajectory_tensor, label_index) pairs.

    Parameters
    ----------
    data_dir  : str | Path   root of raw samples (contains per-class subdirs)
    gestures  : list[str]    ordered list of class names  (defines label ↔ index)
    transform : callable     optional transform applied to the trajectory tensor
    """

    def __init__(
        self,
        data_dir: str | Path,
        gestures: List[str],
        transform=None,
    ):
        self.data_dir = Path(data_dir)
        self.gestures = gestures
        self.label_to_idx = {g: i for i, g in enumerate(gestures)}
        self.transform = transform

        self.samples: List[Tuple[np.ndarray, int]] = []
        self._load()

    def _load(self) -> None:
        for label in self.gestures:
            label_dir = self.data_dir / label
            if not label_dir.exists():
                continue
            for f in sorted(label_dir.glob("*.npz")):
                data = np.load(f, allow_pickle=True)
                traj = data["trajectory"].astype(np.float32)   # (64, 4)
                idx = self.label_to_idx[label]
                self.samples.append((traj, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        traj, label_idx = self.samples[i]
        tensor = torch.from_numpy(traj)       # (64, 4)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, label_idx

    @property
    def num_classes(self) -> int:
        return len(self.gestures)

    def class_counts(self) -> dict:
        counts = {g: 0 for g in self.gestures}
        for _, idx in self.samples:
            counts[self.gestures[idx]] += 1
        return counts


def split_dataset(
    dataset: GestureDataset,
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """
    Split into train / val / test subsets.

    Parameters
    ----------
    dataset : GestureDataset
    train   : float  fraction for training
    val     : float  fraction for validation
    test    : float  fraction for testing
    seed    : int

    Returns
    -------
    (train_set, val_set, test_set)
    """
    assert abs(train + val + test - 1.0) < 1e-6, "Split fractions must sum to 1.0"
    n = len(dataset)
    n_train = int(n * train)
    n_val = int(n * val)
    n_test = n - n_train - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=generator)


def get_numpy_arrays(
    dataset: GestureDataset,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract (X, y) numpy arrays for sklearn-compatible models (DTW, HMM).

    Returns
    -------
    X : np.ndarray  shape (N, 64, 4)
    y : np.ndarray  shape (N,) int
    """
    if len(dataset) == 0:
        return np.empty((0, 64, 4), np.float32), np.empty(0, int)
    X = np.stack([s[0] for s in dataset.samples])
    y = np.array([s[1] for s in dataset.samples])
    return X, y
