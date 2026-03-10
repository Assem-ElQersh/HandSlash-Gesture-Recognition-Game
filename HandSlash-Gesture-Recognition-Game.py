import math
import random
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

# --- Threaded camera capture to decouple frame reading from inference ---
class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.ret, self.frame = self.cap.read()
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self):
        while self._running:
            ret, frame = self.cap.read()
            with self._lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self._lock:
            return self.ret, self.frame.copy()

    def release(self):
        self._running = False
        self._thread.join()
        self.cap.release()

# Mediapipe setup — model_complexity=0 uses the LITE model (~2x faster)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Global variables
curr_Frame = 0
prev_Frame = 0
FPS = 0

next_Time_to_Spawn = 0
Speed = [0, 3]
Fruit_Size = 30
Fruit_Size_sq = Fruit_Size ** 2  # avoid sqrt in distance check
Spawn_Rate = 1
Score = 0
last_scored_milestone = 0          # fixes difficulty re-triggering every frame
Lives = 15
Difficulty_level = 1
game_Over = False

slash = deque(maxlen=19)           # fixed-length deque replaces np.append/delete
slash_Color = (255, 255, 255)
prev_index_pos = None              # tracks last fingertip position for segment collision

w = h = 0

Fruits = []


def Spawn_Fruits():
    if len(Fruits) < 10:
        fruit = {
            "Color": (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            "Curr_position": [random.randint(15, 600), 440],
            "Next_position": [0, 0],
        }
        Fruits.append(fruit)


def Fruit_Movement(fruits, speed):
    global Lives
    updated = []
    for fruit in fruits:
        if fruit["Curr_position"][1] < 20 or fruit["Curr_position"][0] > 650:
            Lives -= 1
        else:
            fruit["Next_position"][0] = fruit["Curr_position"][0] + speed[0]
            fruit["Next_position"][1] = fruit["Curr_position"][1] - speed[1]
            fruit["Curr_position"] = list(fruit["Next_position"])
            updated.append(fruit)
    fruits[:] = updated


def distance_sq(a, b):
    """Squared Euclidean distance — avoids sqrt for hit detection."""
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def point_to_segment_dist_sq(px, py, ax, ay, bx, by):
    """
    Squared distance from point (px,py) to line segment (ax,ay)-(bx,by).
    Used so fast finger swipes that jump over a fruit are still detected.
    """
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0:
        return (px - ax) ** 2 + (py - ay) ** 2
    # Project point onto segment, clamp to [0,1]
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    closest_x = ax + t * dx
    closest_y = ay + t * dy
    return (px - closest_x) ** 2 + (py - closest_y) ** 2


# Video capture
cap = CameraStream(src=0)

while True:
    success, img = cap.read()
    if not success:
        continue

    h, w, _ = img.shape
    img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
    img.flags.writeable = False

    results = hands.process(img)

    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if results and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            lm8 = hand_landmarks.landmark[8]
            index_pos = (int(lm8.x * w), int(lm8.y * h))
            cv2.circle(img, index_pos, 18, slash_Color, -1)
            slash.append(index_pos)

            # Segment collision: check the full path traveled this frame,
            # not just the current point — catches fast swipes
            if prev_index_pos is not None:
                ax, ay = prev_index_pos
                bx, by = index_pos
                slashed = [
                    f for f in Fruits
                    if point_to_segment_dist_sq(
                        f["Curr_position"][0], f["Curr_position"][1],
                        ax, ay, bx, by
                    ) < Fruit_Size_sq
                ]
            else:
                slashed = [f for f in Fruits if distance_sq(index_pos, f["Curr_position"]) < Fruit_Size_sq]

            for fruit in slashed:
                Score += 100
                slash_Color = fruit["Color"]
                Fruits.remove(fruit)

            prev_index_pos = index_pos
    else:
        prev_index_pos = None   # reset when hand leaves frame

    # Difficulty scaling — only trigger once per 1000-point milestone
    milestone = Score // 1000
    if milestone > last_scored_milestone:
        last_scored_milestone = milestone
        Difficulty_level = milestone + 1
        Spawn_Rate = max(1, Difficulty_level * 0.8)
        Speed[1] = min(10, int(3 + 0.5 * Difficulty_level))

    if Lives <= 0:
        game_Over = True

    # Draw slash trail from deque
    if len(slash) >= 2:
        pts = np.array(list(slash), np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], False, slash_Color, 15, 0)

    # Draw fruits
    for fruit in Fruits:
        pos = tuple(fruit["Curr_position"])
        cv2.circle(img, pos, Fruit_Size, fruit["Color"], -1)

    # FPS counter
    curr_Frame = time.time()
    delta_Time = curr_Frame - prev_Frame
    if delta_Time >= 0.5:
        FPS = int(1 / delta_Time)
        prev_Frame = curr_Frame

    cv2.putText(img, f"FPS: {FPS}", (int(w * 0.82), 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 250, 0), 2)
    cv2.putText(img, f"Score: {Score}", (int(w * 0.35), 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
    cv2.putText(img, f"Level: {Difficulty_level}", (int(w * 0.01), 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 150), 5)
    cv2.putText(img, f"Lives remaining: {Lives}", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if not game_Over:
        if time.time() > next_Time_to_Spawn:
            Spawn_Fruits()
            next_Time_to_Spawn = time.time() + (1 / Spawn_Rate)
        Fruit_Movement(Fruits, Speed)
    else:
        cv2.putText(img, "GAME OVER", (int(w * 0.1), int(h * 0.6)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
        Fruits.clear()

    cv2.imshow("img", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
