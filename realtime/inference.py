"""
Real-time gesture inference pipeline.

Wires together:
  CameraStream → HandTracker → GestureSegmenter → normalization → model → callback

Can be used in two ways:

1. Standalone (live overlay):
     from realtime.inference import run_live_inference
     run_live_inference(model, cfg)

2. Embedded in another loop (e.g. the game):
     engine = GestureEngine(model, cfg)
     while True:
         ret, frame = cam.read()
         result = engine.step(frame)
         if result is not None:
             label, confidence = result
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import yaml

from capture.webcam_tracker import CameraStream, HandTracker
from processing.resampling import resample
from processing.normalization import normalize
from processing.features import extract_features
from realtime.segmenter import GestureSegmenter
from models.base import GestureModel


def _load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _process_trajectory(raw_xy: np.ndarray, cfg: dict) -> np.ndarray:
    """Full normalization pipeline for a completed trajectory."""
    n = cfg["processing"]["n_points"]
    use_pca = cfg["processing"].get("use_pca", False)
    resampled = resample(raw_xy, n=n)
    normed = normalize(resampled, use_pca=use_pca)
    return extract_features(normed)                  # (n, 4)


class GestureEngine:
    """
    Embeddable inference engine for use inside any OpenCV loop.

    Parameters
    ----------
    model      : trained GestureModel
    cfg        : loaded config dict
    on_gesture : optional callback(label: str, confidence: float)
    """

    def __init__(
        self,
        model: GestureModel,
        cfg: dict,
        on_gesture: Optional[Callable[[str, float], None]] = None,
    ):
        self.model = model
        self.cfg = cfg
        self.on_gesture = on_gesture
        self.segmenter = GestureSegmenter.from_config(cfg)
        self.tracker = HandTracker(
            model_complexity=cfg["capture"].get("model_complexity", 0)
        )
        self._last_result: Optional[Tuple[str, float]] = None
        self._last_result_until: float = 0.0

    def step(self, bgr_frame: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Process one frame.

        Returns
        -------
        (label, confidence) if a gesture was just recognised, else None.
        The return value is also passed to on_gesture callback if set.
        """
        results, fingertip, flipped = self.tracker.process(bgr_frame)
        raw_trajectory = self.segmenter.update(fingertip)

        if raw_trajectory is not None:
            features = _process_trajectory(raw_trajectory, self.cfg)
            proba = self.model.predict_proba(features[np.newaxis])[0]
            label_idx = int(np.argmax(proba))
            label = self.model.gestures[label_idx]
            confidence = float(proba[label_idx])
            result = (label, confidence)
            self._last_result = result
            self._last_result_until = time.time() + 1.5
            if self.on_gesture:
                self.on_gesture(label, confidence)
            return result

        return None

    def draw_overlay(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Draw recording indicator and last result onto the frame.
        Does NOT call step() — call step() first, then draw_overlay().
        """
        h, w = bgr_frame.shape[:2]

        # Recording indicator
        if self.segmenter.is_recording:
            n = self.segmenter.buffer_length
            cv2.putText(
                bgr_frame,
                f"Recording... ({n} pts)",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )
            cv2.circle(bgr_frame, (w - 25, 25), 12, (0, 0, 255), -1)

        # Last recognised gesture
        if self._last_result is not None and time.time() < self._last_result_until:
            label, conf = self._last_result
            text = f"{label}  {conf*100:.0f}%"
            cv2.putText(
                bgr_frame, text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 100), 2,
            )

        return bgr_frame

    def close(self) -> None:
        self.tracker.close()


def run_live_inference(
    model: GestureModel,
    config_path: str = "configs/default.yaml",
) -> None:
    """
    Standalone live inference with a webcam overlay showing recognised gestures.
    Press Q to quit.
    """
    cfg = _load_config(config_path)
    cam = CameraStream(
        src=cfg["capture"]["camera_index"],
        width=cfg["capture"]["width"],
        height=cfg["capture"]["height"],
    )

    engine = GestureEngine(model, cfg)
    fps_counter = 0
    fps_time = time.time()
    fps_display = 0

    print(f"Live inference — model: {model.name}  |  classes: {model.gestures}")
    print("Perform a gesture to classify it. Press Q to quit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        engine.step(frame)

        results, fingertip, flipped = engine.tracker.process(frame)
        display = engine.tracker.draw(flipped.copy(), results)

        if fingertip:
            cv2.circle(display, fingertip, 8,
                       (0, 0, 255) if engine.segmenter.is_recording else (0, 255, 0),
                       -1)

        engine.draw_overlay(display)

        fps_counter += 1
        if time.time() - fps_time >= 0.5:
            fps_display = int(fps_counter / (time.time() - fps_time))
            fps_counter = 0
            fps_time = time.time()
        cv2.putText(display, f"FPS: {fps_display}", (display.shape[1] - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 1)

        cv2.imshow("Gesture Inference", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    engine.close()
    cv2.destroyAllWindows()
