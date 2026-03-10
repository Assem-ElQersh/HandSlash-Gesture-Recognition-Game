"""
Normalization pipeline for gesture trajectories.

After resampling, raw pixel coordinates still vary with:
  - camera distance (scale)
  - hand position in frame (translation)
  - gesture orientation (rotation)

This module provides three transforms:
  translate_to_origin   zero-mean centroid
  scale_to_unit_box     divide by max bounding-box dimension
  pca_align             rotate so principal axis of variance → x-axis

The recommended pipeline: translate → scale → (optional) pca_align.
"""

import numpy as np


def translate_to_origin(points: np.ndarray) -> np.ndarray:
    """
    Shift so the centroid lies at (0, 0).

    Parameters
    ----------
    points : np.ndarray  shape (N, 2)

    Returns
    -------
    np.ndarray  shape (N, 2), same dtype
    """
    centroid = points.mean(axis=0)
    return points - centroid


def scale_to_unit_box(points: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Scale uniformly so the points fit inside [-1, 1] x [-1, 1].

    Parameters
    ----------
    points : np.ndarray  shape (N, 2)  (should already be centred)
    eps    : float       prevents division by zero for degenerate gestures

    Returns
    -------
    np.ndarray  shape (N, 2), float32
    """
    max_range = np.max(np.abs(points)) + eps
    return (points / max_range).astype(np.float32)


def pca_align(points: np.ndarray) -> np.ndarray:
    """
    Rotate so the direction of greatest variance aligns with the x-axis.

    This makes the representation orientation-invariant, which helps
    DTW and HMM but can harm CNN/LSTM/Transformer (they learn orientation
    on their own).  Toggle with the `use_pca` flag in normalize().

    Parameters
    ----------
    points : np.ndarray  shape (N, 2)  (should already be centred)

    Returns
    -------
    np.ndarray  shape (N, 2), float32
    """
    cov = np.cov(points.T)
    _, vecs = np.linalg.eigh(cov)
    # Eigenvector with largest eigenvalue is vecs[:, -1]
    principal = vecs[:, -1]
    angle = -np.arctan2(principal[1], principal[0])
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return (points @ rot.T).astype(np.float32)


def normalize(
    points: np.ndarray,
    use_pca: bool = False,
) -> np.ndarray:
    """
    Full normalization pipeline: translate → scale → (optional) PCA rotate.

    Parameters
    ----------
    points  : np.ndarray  shape (N, 2)
    use_pca : bool        apply PCA rotation alignment (default False)

    Returns
    -------
    np.ndarray  shape (N, 2), float32
    """
    pts = translate_to_origin(points.astype(np.float64))
    pts = scale_to_unit_box(pts)
    if use_pca:
        pts = pca_align(pts)
    return pts.astype(np.float32)
