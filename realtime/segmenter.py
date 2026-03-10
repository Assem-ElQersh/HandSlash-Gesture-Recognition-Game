"""
Real-time gesture segmentation.

Detects gesture start and end from a stream of fingertip (x, y) positions
using velocity thresholding:

  State: IDLE
    When speed > START_THRESHOLD  →  start buffering  →  State: RECORDING

  State: RECORDING
    Append each point to buffer
    When speed < END_THRESHOLD for N_IDLE_FRAMES consecutive frames  →  end gesture
    If buffer length >= MIN_GESTURE_LEN  →  emit gesture  →  State: IDLE
    If too short  →  discard  →  State: IDLE

Usage
-----
    seg = GestureSegmenter(cfg)
    for each frame:
        gesture_points = seg.update(fingertip_xy)
        if gesture_points is not None:
            # completed gesture — run inference
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import numpy as np


class GestureSegmenter:
    """
    Velocity-threshold gesture segmenter.

    Parameters
    ----------
    start_threshold : float  speed (px/frame) above which recording starts
    end_threshold   : float  speed below which a frame counts as "idle"
    n_idle_frames   : int    consecutive idle frames needed to end gesture
    min_gesture_len : int    minimum trajectory length to emit (reject noise)
    history_len     : int    sliding window for speed smoothing
    """

    IDLE = "idle"
    RECORDING = "recording"

    def __init__(
        self,
        start_threshold: float = 15.0,
        end_threshold: float = 8.0,
        n_idle_frames: int = 8,
        min_gesture_len: int = 10,
        history_len: int = 3,
    ):
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.n_idle_frames = n_idle_frames
        self.min_gesture_len = min_gesture_len

        self._state = self.IDLE
        self._buffer: List[Tuple[float, float]] = []
        self._prev_pos: Optional[Tuple[float, float]] = None
        self._idle_count = 0
        self._speed_history: deque = deque(maxlen=history_len)

    @classmethod
    def from_config(cls, cfg: dict) -> "GestureSegmenter":
        s = cfg.get("segmenter", {})
        return cls(
            start_threshold=s.get("start_threshold", 15.0),
            end_threshold=s.get("end_threshold", 8.0),
            n_idle_frames=s.get("n_idle_frames", 8),
            min_gesture_len=s.get("min_gesture_len", 10),
        )

    def update(
        self, pos: Optional[Tuple[float, float]]
    ) -> Optional[np.ndarray]:
        """
        Feed one fingertip position.

        Parameters
        ----------
        pos : (x, y) in pixels, or None if no hand detected

        Returns
        -------
        np.ndarray  shape (T, 2)  if a gesture just completed, else None
        """
        if pos is None:
            # Hand lost — abort any active recording
            if self._state == self.RECORDING and len(self._buffer) >= self.min_gesture_len:
                gesture = self._flush()
                self.reset()
                return gesture
            self.reset()
            return None

        speed = 0.0
        if self._prev_pos is not None:
            dx = pos[0] - self._prev_pos[0]
            dy = pos[1] - self._prev_pos[1]
            speed = float(np.hypot(dx, dy))

        self._speed_history.append(speed)
        smooth_speed = float(np.mean(self._speed_history))
        self._prev_pos = pos

        if self._state == self.IDLE:
            if smooth_speed > self.start_threshold:
                self._state = self.RECORDING
                self._buffer = [pos]
                self._idle_count = 0

        elif self._state == self.RECORDING:
            self._buffer.append(pos)
            if smooth_speed < self.end_threshold:
                self._idle_count += 1
                if self._idle_count >= self.n_idle_frames:
                    gesture = self._flush()
                    self.reset()
                    if gesture is not None:
                        return gesture
            else:
                self._idle_count = 0

        return None

    def _flush(self) -> Optional[np.ndarray]:
        if len(self._buffer) < self.min_gesture_len:
            return None
        return np.array(self._buffer, dtype=np.float32)

    def reset(self) -> None:
        self._state = self.IDLE
        self._buffer = []
        self._idle_count = 0
        self._prev_pos = None
        self._speed_history.clear()

    @property
    def is_recording(self) -> bool:
        return self._state == self.RECORDING

    @property
    def buffer_length(self) -> int:
        return len(self._buffer)
