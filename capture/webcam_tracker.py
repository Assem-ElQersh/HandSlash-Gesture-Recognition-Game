"""
Webcam capture + MediaPipe hand landmark extraction.

Provides two classes:
  CameraStream  — threaded frame reader (decouples capture from inference)
  HandTracker   — wraps MediaPipe Hands, returns per-frame fingertip (x, y)
"""

import threading
import cv2
import mediapipe as mp
import numpy as np


class CameraStream:
    """
    Reads frames in a background thread so the main loop never blocks
    waiting for the next camera frame.
    """

    def __init__(self, src: int = 0, width: int = 640, height: int = 480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        self.ret, self.frame = self.cap.read()
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self) -> None:
        while self._running:
            ret, frame = self.cap.read()
            with self._lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self._lock:
            return self.ret, self.frame.copy()

    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def release(self) -> None:
        self._running = False
        self._thread.join()
        self.cap.release()


class HandTracker:
    """
    Thin wrapper around MediaPipe Hands.

    Processes a BGR frame and returns the normalised (x, y) pixel position
    of the index fingertip (landmark 8) and the raw MediaPipe results object.

    Parameters
    ----------
    max_num_hands : int
    model_complexity : int  0 = LITE (faster), 1 = FULL
    min_detection_confidence : float
    min_tracking_confidence : float
    """

    INDEX_TIP = 8

    def __init__(
        self,
        max_num_hands: int = 1,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ):
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, bgr_frame: np.ndarray):
        """
        Parameters
        ----------
        bgr_frame : np.ndarray  Raw BGR frame from the camera.

        Returns
        -------
        results     MediaPipe results object (results.multi_hand_landmarks)
        fingertip   (x_px, y_px) tuple or None if no hand detected.
                    Coordinates are in pixel space of the *flipped* frame.
        """
        h, w = bgr_frame.shape[:2]
        flipped = cv2.flip(bgr_frame, 1)
        rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)

        fingertip = None
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark[self.INDEX_TIP]
            fingertip = (int(lm.x * w), int(lm.y * h))

        return results, fingertip, flipped

    def draw(self, bgr_frame: np.ndarray, results) -> np.ndarray:
        """Draw hand skeleton onto bgr_frame in-place and return it."""
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                self._mp_drawing.draw_landmarks(
                    bgr_frame,
                    hand_lm,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_drawing_styles.get_default_hand_landmarks_style(),
                    self._mp_drawing_styles.get_default_hand_connections_style(),
                )
        return bgr_frame

    def close(self) -> None:
        self._hands.close()
