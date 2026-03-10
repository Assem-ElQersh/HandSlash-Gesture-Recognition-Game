# HandSlash — Gesture Recognition Game

An AI-powered fruit-slicing game where you slice falling fruits in real-time using your index finger tracked by your webcam. Built with **MediaPipe Hand Tracking** and **OpenCV**.

---

## Gameplay

- Fruits spawn at the bottom of the screen and rise upward.
- Move your **index finger** in front of the camera to slash through them.
- Each slashed fruit adds **100 points** to your score.
- Miss a fruit and you lose a **life** (you start with 15).
- The game gets faster and spawns more fruits as your score climbs.
- Press **Q** to quit at any time.

---

## Requirements

- Python **3.10**
- conda (Anaconda or Miniconda)
- A webcam

---

## Setup

### 1. Create and activate a conda environment

```bash
conda create -n handslash python=3.10 -y
conda activate handslash
```

### 2. Install dependencies

The key constraint is that `mediapipe==0.10.21` (the last version with the `solutions` API) requires `numpy<2`, while newer OpenCV builds default to `numpy>=2`. Install in this exact order to avoid conflicts:

```bash
pip install "mediapipe==0.10.21" "numpy<2"
pip install --no-deps "opencv-contrib-python==4.11.0.86"
pip install "numpy<2"
```

> **Why this order?**
> - `mediapipe>=0.10.30` dropped the `solutions` API (`mp.solutions.hands`, `mp.solutions.drawing_utils`, etc.) — this game requires the old API, so we pin to `0.10.21`.
> - `mediapipe==0.10.21` requires `numpy<2`, but `opencv-contrib-python` by default pulls `numpy>=2`.
> - Using `--no-deps` on the OpenCV install prevents it from upgrading numpy, then we pin numpy back to `<2`.

---

## Running the Game

```bash
conda activate handslash
python "HandSlash-Gesture-Recognition-Game.py"
```

### Camera index

The script defaults to camera index `0` (built-in webcam). If you have multiple cameras and the wrong one opens, change line 79:

```python
cap = cv2.VideoCapture(0)  # change 0 to 1, 2, etc.
```

---

## Dependency Versions (pinned)

| Package                  | Version   |
|--------------------------|-----------|
| Python                   | 3.10      |
| mediapipe                | 0.10.21   |
| opencv-contrib-python    | 4.11.0.86 |
| numpy                    | < 2       |

---

## Project Structure

```
HandSlash-Gesture-Recognition-Game/
├── HandSlash-Gesture-Recognition-Game.py   # main game script
└── README.md
```

---

## How It Works

1. **Hand detection** — MediaPipe's `Hands` solution tracks hand landmarks in each webcam frame.
2. **Index finger tracking** — Landmark `8` (index fingertip) position is extracted and drawn as a colored circle.
3. **Slash trail** — A polyline of the last N fingertip positions is drawn to visualize the slash motion.
4. **Collision detection** — Euclidean distance between the fingertip and each fruit center is computed; a hit is registered when `distance < Fruit_Size`.
5. **Difficulty scaling** — Every 1000 points, spawn rate and fruit speed increase.
 
---

## License

MIT License. See the [LICENSE](LICENSE) file for details.
