"""
Resample a variable-length (x, y) trajectory to a fixed number of points
using cumulative arc-length parameterisation and linear interpolation.

This makes all trajectories the same length regardless of how fast or slow
the gesture was performed, which is a prerequisite for every downstream model.
"""

import numpy as np


def path_length(points: np.ndarray) -> float:
    """Total Euclidean arc length of a trajectory."""
    diffs = np.diff(points, axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))


def resample(points: np.ndarray, n: int = 64) -> np.ndarray:
    """
    Resample a trajectory to exactly *n* evenly-spaced points.

    Parameters
    ----------
    points : np.ndarray  shape (T, 2), dtype float  raw (x, y) sequence
    n      : int         target number of points (default 64)

    Returns
    -------
    np.ndarray  shape (n, 2), dtype float32
    """
    points = np.asarray(points, dtype=np.float64)

    if len(points) < 2:
        # Degenerate: repeat the single point
        return np.tile(points[0] if len(points) == 1 else [0.0, 0.0], (n, 1)).astype(np.float32)

    # Cumulative arc lengths (parameter values)
    diffs = np.diff(points, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cum_lengths[-1]

    if total == 0.0:
        return np.tile(points[0], (n, 1)).astype(np.float32)

    # Target parameter values: n evenly-spaced samples in [0, total]
    target = np.linspace(0.0, total, n)

    resampled_x = np.interp(target, cum_lengths, points[:, 0])
    resampled_y = np.interp(target, cum_lengths, points[:, 1])

    return np.column_stack([resampled_x, resampled_y]).astype(np.float32)
