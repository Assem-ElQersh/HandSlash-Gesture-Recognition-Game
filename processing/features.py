"""
Feature extraction for gesture trajectories.

Given a normalised (N, 2) trajectory, compute additional per-point features:

  vx, vy      finite-difference velocity (dx/dt, dy/dt)
  speed       ||(vx, vy)||
  curvature   direction change between consecutive segments (radians)

The primary output used by all neural models is the (N, 4) tensor:
  [x, y, vx, vy]

HMM additionally uses a direction histogram and quantised direction codes.
"""

import numpy as np


def compute_velocity(points: np.ndarray) -> np.ndarray:
    """
    Finite-difference velocity at each point.

    Parameters
    ----------
    points : np.ndarray  shape (N, 2)

    Returns
    -------
    np.ndarray  shape (N, 2)  (vx, vy) — first point is zero-padded
    """
    vel = np.zeros_like(points)
    vel[1:] = points[1:] - points[:-1]
    return vel.astype(np.float32)


def compute_curvature(points: np.ndarray) -> np.ndarray:
    """
    Signed curvature (direction change) at each point in radians.

    Returns
    -------
    np.ndarray  shape (N,)  — first two points are zero-padded
    """
    n = len(points)
    curv = np.zeros(n, dtype=np.float32)
    if n < 3:
        return curv
    d1 = points[1:-1] - points[:-2]
    d2 = points[2:] - points[1:-1]
    a1 = np.arctan2(d1[:, 1], d1[:, 0])
    a2 = np.arctan2(d2[:, 1], d2[:, 0])
    diff = a2 - a1
    # Wrap to [-pi, pi]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    curv[1:-1] = diff
    return curv


def direction_quantize(points: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """
    Quantise movement direction into *n_bins* equal angle sectors.
    Used as observation sequence for HMM.

    Returns
    -------
    np.ndarray  shape (N,) int  — bin index in [0, n_bins-1]
    """
    vel = compute_velocity(points)
    angles = np.arctan2(vel[:, 1], vel[:, 0])          # [-pi, pi]
    bins = np.floor((angles + np.pi) / (2 * np.pi) * n_bins).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)
    return bins


def direction_histogram(points: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """
    Normalised histogram of movement directions over the full trajectory.
    Compact global feature useful for DTW distance weighting.

    Returns
    -------
    np.ndarray  shape (n_bins,) float32
    """
    codes = direction_quantize(points, n_bins)
    hist, _ = np.histogram(codes, bins=n_bins, range=(0, n_bins))
    total = hist.sum()
    return (hist / max(total, 1)).astype(np.float32)


def extract_features(points: np.ndarray) -> np.ndarray:
    """
    Build the canonical (N, 4) feature matrix used by CNN / LSTM / Transformer.

    Layout: [x, y, vx, vy]

    Parameters
    ----------
    points : np.ndarray  shape (N, 2)  normalised trajectory

    Returns
    -------
    np.ndarray  shape (N, 4), float32
    """
    vel = compute_velocity(points)
    return np.concatenate([points, vel], axis=1).astype(np.float32)
