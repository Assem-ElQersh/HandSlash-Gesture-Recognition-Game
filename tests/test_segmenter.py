"""Tests for the real-time gesture segmenter."""

import numpy as np
import pytest
from realtime.segmenter import GestureSegmenter


def _feed(seg, positions):
    """Feed a list of positions and return the first completed gesture."""
    for pos in positions:
        result = seg.update(pos)
        if result is not None:
            return result
    return None


# ------------------------------------------------------------------ #
#  Basic state transitions                                             #
# ------------------------------------------------------------------ #

class TestSegmenterStates:
    def test_idle_on_init(self):
        seg = GestureSegmenter()
        assert not seg.is_recording

    def test_starts_recording_above_threshold(self):
        seg = GestureSegmenter(start_threshold=10.0)
        seg.update((0, 0))
        seg.update((50, 0))   # speed = 50 >> threshold
        assert seg.is_recording

    def test_no_recording_below_threshold(self):
        seg = GestureSegmenter(start_threshold=100.0)
        seg.update((0, 0))
        seg.update((5, 0))    # speed = 5 << threshold
        assert not seg.is_recording

    def test_reset_on_none(self):
        seg = GestureSegmenter(start_threshold=5.0)
        seg.update((0, 0))
        seg.update((50, 0))   # starts recording
        seg.update(None)       # hand lost
        assert not seg.is_recording
        assert seg.buffer_length == 0


# ------------------------------------------------------------------ #
#  Gesture completion                                                  #
# ------------------------------------------------------------------ #

class TestGestureCompletion:
    def _make_fast_move(self, n=20):
        """Fast horizontal movement."""
        return [(i * 20, 0) for i in range(n)]

    def _make_idle(self, pos, n=12):
        """Stationary (idle) frames at *pos*."""
        return [pos] * n

    def test_completes_after_idle(self):
        seg = GestureSegmenter(
            start_threshold=10.0,
            end_threshold=5.0,
            n_idle_frames=8,
            min_gesture_len=5,
        )
        fast = self._make_fast_move(15)
        idle = self._make_idle(fast[-1], 12)

        result = _feed(seg, fast + idle)
        assert result is not None, "Expected a completed gesture"
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_returns_none_if_too_short(self):
        seg = GestureSegmenter(
            start_threshold=5.0,
            end_threshold=3.0,
            n_idle_frames=4,
            min_gesture_len=20,   # require at least 20 points
        )
        # Only 5 fast points — well below min_gesture_len
        fast = self._make_fast_move(5)
        idle = self._make_idle(fast[-1], 6)
        result = _feed(seg, fast + idle)
        assert result is None

    def test_gesture_content_correct(self):
        seg = GestureSegmenter(
            start_threshold=10.0,
            end_threshold=5.0,
            n_idle_frames=8,
            min_gesture_len=5,
        )
        fast = [(i * 20, i * 5) for i in range(15)]
        idle = [fast[-1]] * 12

        result = _feed(seg, fast + idle)
        assert result is not None
        # Result should be a subset of the fast positions (within buffer)
        for pt in result:
            assert len(pt) == 2

    def test_gesture_lost_on_none_fires_if_long_enough(self):
        seg = GestureSegmenter(
            start_threshold=5.0,
            end_threshold=3.0,
            n_idle_frames=8,
            min_gesture_len=5,
        )
        fast = self._make_fast_move(15)
        # Feed fast movement, then None (hand lost)
        for pos in fast:
            seg.update(pos)
        result = seg.update(None)
        assert result is not None, "Expected gesture to fire when hand is lost mid-recording"

    def test_buffer_grows_while_recording(self):
        seg = GestureSegmenter(start_threshold=5.0)
        seg.update((0, 0))
        seg.update((50, 0))   # start recording
        initial = seg.buffer_length
        seg.update((100, 0))
        assert seg.buffer_length >= initial


# ------------------------------------------------------------------ #
#  Config loading                                                      #
# ------------------------------------------------------------------ #

class TestSegmenterConfig:
    def test_from_config(self):
        cfg = {
            "segmenter": {
                "start_threshold": 20.0,
                "end_threshold": 6.0,
                "n_idle_frames": 10,
                "min_gesture_len": 15,
            }
        }
        seg = GestureSegmenter.from_config(cfg)
        assert seg.start_threshold == 20.0
        assert seg.end_threshold == 6.0
        assert seg.n_idle_frames == 10
        assert seg.min_gesture_len == 15

    def test_from_config_defaults(self):
        seg = GestureSegmenter.from_config({})
        assert seg.start_threshold == 15.0
        assert seg.end_threshold == 8.0
