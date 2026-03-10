"""
HandSlash — Fruit Slicing Demo

This file is the proof-of-concept demo for the Gesture Trajectory Engine.

Two operating modes (set USE_ENGINE = True/False):

  USE_ENGINE = False  (default when no trained model found)
    Classic mode: raw fingertip position is used for slicing.
    Works out of the box with no trained model.

  USE_ENGINE = True
    Engine mode: the GestureEngine classifies completed gestures.
    'slash_left', 'slash_right', or 'slash_down'  →  slice all fruits in path
    'circle'                                        →  shield (skip next miss)
    Any other gesture is ignored.
    Requires a trained model checkpoint.

Controls
--------
  Q / ESC   Quit
"""

from __future__ import annotations

import random
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml

# Make sure the project root is on sys.path when run directly
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from capture.webcam_tracker import CameraStream, HandTracker


# ------------------------------------------------------------------ #
#  Config                                                              #
# ------------------------------------------------------------------ #

CONFIG_PATH = str(_root / "configs" / "default.yaml")
PREFERRED_MODEL = "transformer"    # used when USE_ENGINE = True

with open(CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)

_ckpt_dir = _root / _cfg["paths"]["checkpoints"]
_gestures = _cfg["gestures"]


def _try_load_engine():
    """Attempt to load a trained model for engine mode."""
    for name in [PREFERRED_MODEL, "lstm", "cnn", "hmm", "dtw"]:
        ext = ".pkl" if name in ("dtw", "hmm") else ".pt"
        ckpt = _ckpt_dir / f"{name}{ext}"
        if ckpt.exists():
            try:
                if name == "dtw":
                    from models.dtw import DTWClassifier
                    model = DTWClassifier(_gestures).load(ckpt)
                elif name == "hmm":
                    from models.hmm import HMMClassifier
                    model = HMMClassifier(_gestures).load(ckpt)
                elif name == "cnn":
                    from models.cnn import CNNClassifier
                    model = CNNClassifier(_gestures).load(ckpt)
                elif name == "lstm":
                    from models.lstm import LSTMClassifier
                    model = LSTMClassifier(_gestures).load(ckpt)
                else:
                    from models.transformer import TransformerClassifier
                    model = TransformerClassifier(_gestures).load(ckpt)
                print(f"[demo] Loaded model: {name}")
                return model
            except Exception as e:
                print(f"[demo] Could not load {name}: {e}")
    return None


# ------------------------------------------------------------------ #
#  Game state                                                          #
# ------------------------------------------------------------------ #

class GameState:
    def __init__(self):
        self.score = 0
        self.lives = 15
        self.speed = [0, 3]
        self.spawn_rate = 1.0
        self.difficulty = 1
        self.last_milestone = 0
        self.game_over = False
        self.shield_active = False
        self.fruits: list = []
        self.next_spawn = 0.0
        self.slash_color = (255, 255, 255)

    def update_difficulty(self):
        milestone = self.score // 1000
        if milestone > self.last_milestone:
            self.last_milestone = milestone
            self.difficulty = milestone + 1
            self.spawn_rate = max(1.0, self.difficulty * 0.8)
            self.speed[1] = min(10, int(3 + 0.5 * self.difficulty))

    def spawn(self):
        if len(self.fruits) < 10 and time.time() > self.next_spawn:
            self.fruits.append({
                "color": (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                "pos": [random.randint(15, 600), 440],
            })
            self.next_spawn = time.time() + 1.0 / self.spawn_rate

    def move_fruits(self):
        kept = []
        for f in self.fruits:
            f["pos"][1] -= self.speed[1]
            f["pos"][0] += self.speed[0]
            if f["pos"][1] < 20 or f["pos"][0] > 650:
                if not self.shield_active:
                    self.lives -= 1
            else:
                kept.append(f)
        self.fruits = kept
        if self.shield_active and len(kept) < len(self.fruits):
            self.shield_active = False


FRUIT_R = 30
FRUIT_R_SQ = FRUIT_R ** 2


def _point_to_seg_dist_sq(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    seg = dx * dx + dy * dy
    if seg == 0:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg))
    return (px - (ax + t * dx)) ** 2 + (py - (ay + t * dy)) ** 2


def _slash_fruits(gs: GameState, prev_pos, curr_pos):
    """Slice fruits that intersect the finger segment."""
    if prev_pos is None or curr_pos is None:
        return
    ax, ay = prev_pos
    bx, by = curr_pos
    hit = [
        f for f in gs.fruits
        if _point_to_seg_dist_sq(f["pos"][0], f["pos"][1], ax, ay, bx, by) < FRUIT_R_SQ
    ]
    for f in hit:
        gs.score += 100
        gs.slash_color = f["color"]
        gs.fruits.remove(f)


# ------------------------------------------------------------------ #
#  Main game loop                                                      #
# ------------------------------------------------------------------ #

def run_game():
    model = _try_load_engine()
    use_engine = model is not None

    if use_engine:
        from realtime.inference import GestureEngine
        engine = GestureEngine(model, _cfg)
        print("[demo] Engine mode ON — perform slash gestures to slice fruits")
    else:
        engine = None
        print("[demo] Classic mode — use your index finger directly")

    cam = CameraStream(
        src=_cfg["capture"]["camera_index"],
        width=_cfg["capture"]["width"],
        height=_cfg["capture"]["height"],
    )
    tracker = HandTracker(model_complexity=_cfg["capture"].get("model_complexity", 0))

    gs = GameState()
    slash_trail = deque(maxlen=19)
    prev_pos = None
    fps = 0
    fps_t = time.time()
    fps_n = 0

    SLASH_GESTURES = {"slash_left", "slash_right", "slash_down"}

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        results, fingertip, flipped = tracker.process(frame)
        display = tracker.draw(flipped.copy(), results)
        h, w = display.shape[:2]

        if fingertip is not None:
            if use_engine:
                # Engine mode: segmenter fires, model classifies
                gesture_result = engine.step(frame)
                if gesture_result is not None:
                    label, conf = gesture_result
                    if label in SLASH_GESTURES and not gs.game_over:
                        # Slice ALL fruits (whole-gesture activation)
                        for f in list(gs.fruits):
                            gs.score += 100
                            gs.slash_color = f["color"]
                        gs.fruits.clear()
                    elif label == "circle":
                        gs.shield_active = True

                slash_trail.append(fingertip)
                cv2.circle(display, fingertip, 18, gs.slash_color, -1)

            else:
                # Classic mode: direct position collision
                cv2.circle(display, fingertip, 18, gs.slash_color, -1)
                slash_trail.append(fingertip)
                if not gs.game_over:
                    _slash_fruits(gs, prev_pos, fingertip)

            prev_pos = fingertip
        else:
            prev_pos = None
            if use_engine and engine:
                engine.segmenter.update(None)

        # Draw slash trail
        if len(slash_trail) >= 2:
            pts = np.array(list(slash_trail), np.int32).reshape(-1, 1, 2)
            cv2.polylines(display, [pts], False, gs.slash_color, 15)

        if not gs.game_over:
            gs.spawn()
            gs.move_fruits()
            gs.update_difficulty()

        # Draw fruits
        for f in gs.fruits:
            cv2.circle(display, tuple(int(v) for v in f["pos"]), FRUIT_R, f["color"], -1)

        # Shield indicator
        if gs.shield_active:
            cv2.putText(display, "SHIELD", (w // 2 - 50, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if gs.lives <= 0:
            gs.game_over = True

        if gs.game_over:
            cv2.putText(display, "GAME OVER",
                        (int(w * 0.1), int(h * 0.6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
            gs.fruits.clear()

        # HUD
        fps_n += 1
        if time.time() - fps_t >= 0.5:
            fps = int(fps_n / (time.time() - fps_t))
            fps_n = 0
            fps_t = time.time()

        cv2.putText(display, f"FPS: {fps}", (int(w * 0.82), 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 250, 0), 2)
        cv2.putText(display, f"Score: {gs.score}", (int(w * 0.35), 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
        cv2.putText(display, f"Level: {gs.difficulty}", (int(w * 0.01), 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 150), 5)
        cv2.putText(display, f"Lives: {gs.lives}", (200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        mode_label = f"Mode: {'Engine' if use_engine else 'Classic'}"
        cv2.putText(display, mode_label, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        if use_engine and engine:
            engine.draw_overlay(display)

        cv2.imshow("HandSlash", display)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cam.release()
    tracker.close()
    if engine:
        engine.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_game()
