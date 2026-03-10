"""
Interactive gesture data collection tool.

Controls
--------
1-9 / a-z   Select gesture class (maps to label name from config)
SPACE       Toggle recording on/off  (starts / stops a sample)
D           Delete last saved sample for current class
Q / ESC     Quit

While recording, the fingertip trail is drawn in red.
A status bar at the bottom shows:
  current class | recording status | sample count per class
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from capture.webcam_tracker import CameraStream, HandTracker
from processing.resampling import resample
from processing.normalization import normalize
from processing.features import extract_features


def _load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _count_samples(raw_dir: Path, label: str) -> int:
    label_dir = raw_dir / label
    if not label_dir.exists():
        return 0
    return len(list(label_dir.glob("*.npz")))


def _save_sample(raw_dir: Path, label: str, trajectory: list) -> str:
    """Normalise, featurise, and save a collected trajectory."""
    label_dir = raw_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)

    pts = np.array(trajectory, dtype=np.float32)
    pts_resampled = resample(pts, n=64)
    pts_norm = normalize(pts_resampled)
    features = extract_features(pts_norm)      # (64, 4)

    timestamp = int(time.time() * 1000)
    path = label_dir / f"{timestamp}.npz"
    np.savez(path, trajectory=features, label=label)
    return str(path)


def run_collector(config_path: str = "configs/default.yaml") -> None:
    cfg = _load_config(config_path)
    gestures: list = cfg["gestures"]
    raw_dir = Path(cfg["paths"]["raw_data"])
    cam_src: int = cfg["capture"]["camera_index"]

    cam = CameraStream(src=cam_src)
    tracker = HandTracker(
        model_complexity=cfg["capture"].get("model_complexity", 0)
    )

    current_idx = 0
    recording = False
    trajectory: list = []
    message = ""
    message_until = 0.0

    def flash(msg: str, duration: float = 2.0) -> None:
        nonlocal message, message_until
        message = msg
        message_until = time.time() + duration

    print("Gesture Collector — press 1-9 to select class, SPACE to record, Q to quit")

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        results, fingertip, flipped = tracker.process(frame)
        display = tracker.draw(flipped.copy(), results)
        h, w = display.shape[:2]

        current_label = gestures[current_idx]

        if fingertip is not None:
            if recording:
                trajectory.append(list(fingertip))
                cv2.circle(display, fingertip, 10, (0, 0, 255), -1)
                # Draw trail
                if len(trajectory) >= 2:
                    pts = np.array(trajectory[-30:], np.int32).reshape(-1, 1, 2)
                    cv2.polylines(display, [pts], False, (0, 0, 255), 3)
            else:
                cv2.circle(display, fingertip, 10, (0, 255, 0), -1)

        # --- Status bar ---
        bar_h = 60
        bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
        rec_color = (0, 0, 200) if recording else (50, 50, 50)
        rec_text = "[ REC ]" if recording else "[     ]"
        counts = {g: _count_samples(raw_dir, g) for g in gestures}

        cv2.putText(bar, f"Class: {current_label}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(bar, rec_text, (220, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, rec_color, 2)
        cv2.putText(bar, f"Samples: {counts[current_label]}", (310, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 200), 1)

        # Show all class counts in a row
        summary = "  ".join(f"{g}:{counts[g]}" for g in gestures)
        cv2.putText(bar, summary, (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        display = np.vstack([display, bar])

        # Flash message overlay
        if time.time() < message_until:
            cv2.putText(display, message, (10, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # Gesture selector overlay (top)
        for i, g in enumerate(gestures):
            key_char = str(i + 1) if i < 9 else chr(ord("a") + i - 9)
            color = (0, 255, 0) if i == current_idx else (150, 150, 150)
            cv2.putText(display, f"{key_char}:{g}", (10 + i * 95, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.imshow("Gesture Collector", display)
        key = cv2.waitKey(1) & 0xFF

        # --- Key handling ---
        if key == ord("q") or key == 27:
            break

        elif key == ord(" "):
            if not recording:
                trajectory = []
                recording = True
                flash("Recording...", 9999)
            else:
                recording = False
                if len(trajectory) >= 10:
                    path = _save_sample(raw_dir, current_label, trajectory)
                    count = _count_samples(raw_dir, current_label)
                    flash(f"Saved! ({count} samples for '{current_label}')")
                    print(f"  Saved: {path}")
                else:
                    flash("Too short — not saved (< 10 points)")
                trajectory = []

        elif key == ord("d"):
            label_dir = raw_dir / current_label
            files = sorted(label_dir.glob("*.npz")) if label_dir.exists() else []
            if files:
                files[-1].unlink()
                flash(f"Deleted last sample for '{current_label}'")
            else:
                flash("No samples to delete")

        else:
            # Number keys 1-9 → select gesture
            for i in range(min(9, len(gestures))):
                if key == ord(str(i + 1)):
                    current_idx = i
                    recording = False
                    trajectory = []
                    flash(f"Selected: {gestures[i]}")
                    break
            else:
                # Letter keys a-z → gestures beyond index 9
                if ord("a") <= key <= ord("z"):
                    idx = key - ord("a") + 9
                    if idx < len(gestures):
                        current_idx = idx
                        recording = False
                        trajectory = []
                        flash(f"Selected: {gestures[idx]}")

    cam.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_collector()
