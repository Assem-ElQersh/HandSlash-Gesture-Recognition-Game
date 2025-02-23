import math
import random
import threading
import time

import cv2
import mediapipe as mp
import numpy as np

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Global variables
curr_Frame = 0
prev_Frame = 0
delta_time = 0

next_Time_to_Spawn = 0
Speed = [0, 3]
Fruit_Size = 30
Spawn_Rate = 1
Score = 0
Lives = 15
Difficulty_level = 1
game_Over = False

slash = np.array([[]], np.int32)
slash_Color = (255, 255, 255)
slash_length = 19

w = h = 0

Fruits = []

# Functions
def Spawn_Fruits():
    if len(Fruits) < 10:  # Limit the number of active fruits
        fruit = {}
        random_x = random.randint(15, 600)
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        fruit["Color"] = random_color
        fruit["Curr_position"] = [random_x, 440]
        fruit["Next_position"] = [0, 0]
        Fruits.append(fruit)

def Fruit_Movement(Fruits, speed):
    global Lives
    updated_fruits = []
    for fruit in Fruits:
        if fruit["Curr_position"][1] < 20 or fruit["Curr_position"][0] > 650:
            Lives -= 1
        else:
            fruit["Next_position"][0] = fruit["Curr_position"][0] + speed[0]
            fruit["Next_position"][1] = fruit["Curr_position"][1] - speed[1]
            fruit["Curr_position"] = fruit["Next_position"]
            updated_fruits.append(fruit)

    Fruits[:] = updated_fruits  # Efficiently update the Fruits list

def distance(a, b):
    x1, y1 = a
    x2, y2 = b
    return int(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

def process_hands(img):
    global results
    return hands.process(img)

# Video capture setup
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Skipping frame")
        continue

    h, w, c = img.shape
    img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
    img.flags.writeable = False

    # Process hands directly
    results = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if results and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for id, lm in enumerate(hand_landmarks.landmark):
                if id == 8:
                    index_pos = (int(lm.x * w), int(lm.y * h))
                    cv2.circle(img, index_pos, 18, slash_Color, -1)
                    slash = np.append(slash, index_pos)

                    while len(slash) >= slash_length:
                        slash = np.delete(slash, len(slash) - slash_length, 0)

                    for fruit in Fruits:
                        d = distance(index_pos, fruit["Curr_position"])
                        cv2.putText(img, str(d), fruit["Curr_position"], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        if d < Fruit_Size:
                            Score += 100
                            slash_Color = fruit["Color"]
                            Fruits.remove(fruit)


    if Score % 1000 == 0 and Score != 0:
        Difficulty_level = (Score / 1000) + 1
        Difficulty_level = int(Difficulty_level)
        Spawn_Rate = max(1, Difficulty_level * 0.8)
        Speed[1] = min(10, int(3 + 0.5 * Difficulty_level))

    if Lives <= 0:
        game_Over = True

    slash = slash.reshape((-1, 1, 2))
    cv2.polylines(img, [slash], False, slash_Color, 15, 0)

    curr_Frame = time.time()
    delta_Time = curr_Frame - prev_Frame
    if curr_Frame - prev_Frame >= 0.5:  # Update FPS every 0.5 seconds
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

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()