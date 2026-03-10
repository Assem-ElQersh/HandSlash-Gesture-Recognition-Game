"""
External dataset loaders.

Converts external gesture datasets into the framework's canonical format:
  np.ndarray shape (64, 4) float32  [x, y, vx, vy]

Supported formats:
  - $N Multistroke Recognizer XML  (from depts.washington.edu/aimgroup/proj/dollar/ndollar.html)
  - Flat CSV:  label, x0, y0, x1, y1, ...
  - Directory of per-class .txt files: one (x, y) per line

All loaders return a list of (trajectory_array, label_str) tuples that can
be passed to collector._save_sample() or directly built into a GestureDataset.
"""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import numpy as np

from processing.resampling import resample
from processing.normalization import normalize
from processing.features import extract_features


# --------------------------------------------------------------------------- #
#  Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _to_canonical(raw_xy: np.ndarray) -> np.ndarray:
    """Resample → normalise → featurise a raw (T, 2) trajectory."""
    resampled = resample(raw_xy, n=64)
    normed = normalize(resampled)
    return extract_features(normed)               # (64, 4)


# --------------------------------------------------------------------------- #
#  $N Multistroke XML                                                          #
# --------------------------------------------------------------------------- #

def load_dollar_n(xml_path: str | Path) -> List[Tuple[np.ndarray, str]]:
    """
    Parse the $N Multistroke Recognizer XML format.

    Each <GestureSet> contains multiple <Gesture name="..."> elements.
    Each <Gesture> contains <Stroke> → <Point x="..." y="..."> elements.
    Points from all strokes are concatenated in order.

    Parameters
    ----------
    xml_path : path to the XML file

    Returns
    -------
    list of (canonical_trajectory, label) tuples
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    samples = []

    for gesture_el in root.iter("Gesture"):
        label = gesture_el.attrib.get("Name", gesture_el.attrib.get("name", "unknown"))
        points = []
        for stroke in gesture_el.iter("Stroke"):
            for pt in stroke.iter("Point"):
                x = float(pt.attrib.get("X", pt.attrib.get("x", 0)))
                y = float(pt.attrib.get("Y", pt.attrib.get("y", 0)))
                points.append([x, y])
        if len(points) >= 2:
            raw = np.array(points, dtype=np.float32)
            samples.append((_to_canonical(raw), label))

    return samples


# --------------------------------------------------------------------------- #
#  Flat CSV                                                                     #
# --------------------------------------------------------------------------- #

def load_csv(csv_path: str | Path) -> List[Tuple[np.ndarray, str]]:
    """
    Load from a flat CSV where each row is one gesture:
        label, x0, y0, x1, y1, x2, y2, ...

    Parameters
    ----------
    csv_path : path to .csv file

    Returns
    -------
    list of (canonical_trajectory, label) tuples
    """
    samples = []
    with open(str(csv_path), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            label = row[0].strip()
            coords = list(map(float, row[1:]))
            if len(coords) < 4:
                continue
            # Expect interleaved x, y pairs
            n_pts = len(coords) // 2
            xy = np.array(coords[:n_pts * 2]).reshape(n_pts, 2).astype(np.float32)
            samples.append((_to_canonical(xy), label))
    return samples


# --------------------------------------------------------------------------- #
#  Per-class text files in a directory                                          #
# --------------------------------------------------------------------------- #

def load_txt_directory(root_dir: str | Path) -> List[Tuple[np.ndarray, str]]:
    """
    Load a directory where each sub-folder is a class.
    Each .txt file contains one (x, y) per line (space or comma separated).

    Layout:
        root_dir/
          slash_left/
            001.txt
            002.txt
          circle/
            ...

    Returns
    -------
    list of (canonical_trajectory, label) tuples
    """
    root = Path(root_dir)
    samples = []
    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for txt_file in sorted(label_dir.glob("*.txt")):
            points = []
            with open(txt_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.replace(",", " ").split()
                    if len(parts) >= 2:
                        points.append([float(parts[0]), float(parts[1])])
            if len(points) >= 2:
                raw = np.array(points, dtype=np.float32)
                samples.append((_to_canonical(raw), label))
    return samples


# --------------------------------------------------------------------------- #
#  Save loaded external samples into the framework's raw/ layout               #
# --------------------------------------------------------------------------- #

def save_external_samples(
    samples: List[Tuple[np.ndarray, str]],
    raw_dir: str | Path,
) -> int:
    """
    Write externally loaded (trajectory, label) pairs into
    data/raw/<label>/<index>.npz so they appear alongside collected data.

    Returns
    -------
    int  number of samples written
    """
    raw_dir = Path(raw_dir)
    counts: dict = {}
    for traj, label in samples:
        label_dir = raw_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        idx = counts.get(label, 0)
        np.savez(label_dir / f"ext_{idx:06d}.npz", trajectory=traj, label=label)
        counts[label] = idx + 1
    return sum(counts.values())
