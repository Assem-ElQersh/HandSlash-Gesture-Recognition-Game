# Gesture Trajectory Learning Engine

A research-grade framework for real-time hand gesture recognition using trajectory learning. Five model architectures are implemented, benchmarked, and integrated into a live inference pipeline. The fruit-slashing game (`demos/`) serves as a proof-of-concept demo.

---

## What This Is

Most gesture recognition demos call a prebuilt API and call it "AI". This is different.

The core technical problem: raw fingertip trajectories vary with position, scale, speed, and orientation. To classify gestures reliably across all users and conditions you need to:

1. Normalise the trajectory (translation, scale, PCA rotation)
2. Learn a sequence model on the normalised representation
3. Segment gestures in real time from a continuous motion stream

This framework implements all three, with five model backends to compare.

---

## Architecture

```
gesture_engine/
├── capture/
│   └── webcam_tracker.py        CameraStream (threaded) + HandTracker (MediaPipe)
├── processing/
│   ├── resampling.py            Variable-length → fixed 64 points (arc-length param)
│   ├── normalization.py         Translate, scale, optional PCA rotation
│   └── features.py              Velocity, curvature, direction histogram
├── data/
│   ├── collector.py             Interactive webcam recording tool
│   ├── dataset.py               PyTorch Dataset + train/val/test split
│   └── loader.py                External formats: $N XML, flat CSV, txt directories
├── models/
│   ├── base.py                  Abstract GestureModel interface
│   ├── dtw.py                   DTW 1-NN (dtaidistance)
│   ├── hmm.py                   Per-class GaussianHMM (hmmlearn)
│   ├── cnn.py                   1D CNN (PyTorch)
│   ├── lstm.py                  Bidirectional LSTM (PyTorch)
│   └── transformer.py           Transformer encoder with CLS token (PyTorch)
├── training/
│   ├── trainer.py               Training loop + checkpoint saving
│   └── evaluator.py             Accuracy, macro F1, confusion matrix PNG
├── realtime/
│   ├── segmenter.py             Velocity-threshold gesture start/end detection
│   └── inference.py             Full pipeline, <20ms latency target
├── benchmarks/
│   └── compare_models.py        Side-by-side accuracy + latency table
├── configs/
│   └── default.yaml             Gesture set, hyperparameters, paths
├── demos/
│   └── fruit_slash_game.py      Fruit-slicing game (classic or engine mode)
├── tests/
│   ├── test_normalization.py
│   ├── test_models.py
│   └── test_segmenter.py
└── main.py                      CLI entry point
```

---

## Setup

### 1. Create conda environment

```bash
conda create -n gesture_engine python=3.10 -y
conda activate gesture_engine
```

### 2. Install dependencies

The core constraint is `mediapipe==0.10.21` (last version with the `solutions` API) requiring `numpy<2`. Install in this exact order to avoid dependency conflicts:

```bash
pip install "mediapipe==0.10.21" "numpy<2"
pip install --no-deps "opencv-contrib-python==4.11.0.86"
pip install "numpy<2"
pip install torch scikit-learn hmmlearn dtaidistance PyYAML matplotlib pytest
```

---

## Quickstart

### Step 1 — Collect gesture data

```bash
python main.py collect
```

Controls:
- `1`–`9` — select gesture class
- `SPACE` — start / stop recording a sample
- `D` — delete last sample for current class
- `Q` / `ESC` — quit

Aim for **30–50 samples per class** for reliable training. Samples are saved to `data/raw/<class>/`.

### Step 2 — Train models

```bash
python main.py train --model all        # train all 5
python main.py train --model lstm       # single model
```

### Step 3 — Benchmark

```bash
python main.py benchmark
```

Outputs an accuracy / F1 / latency table and saves confusion matrix PNGs to `benchmarks/results/`.

### Step 4 — Run live inference

```bash
python main.py infer --model transformer
```

### Step 5 — Play the game

```bash
python main.py demo
```

If a trained model is found, the game runs in **Engine mode**: perform a `slash_left`, `slash_right`, or `slash_down` gesture to slice all fruits; perform a `circle` to activate a shield. Without a trained model it falls back to classic fingertip collision.

---

## Trajectory Normalization Pipeline

Raw pixel coordinates are invariant to gesture meaning — the same slash performed at different distances, positions, or speeds produces completely different raw values.

The normalization pipeline addresses this:

```
raw (x,y) sequence
  → resample to 64 evenly-spaced points       (arc-length parameterisation)
  → translate centroid to origin              (position invariance)
  → scale to unit bounding box               (scale invariance)
  → (optional) PCA rotation alignment        (orientation invariance)
  → compute vx, vy via finite differences
  → final tensor: (64, 4) [x, y, vx, vy]
```

PCA alignment is off by default — neural models learn orientation from data. Enable it for DTW and HMM via `configs/default.yaml`.

---

## Models

All five implement the same `GestureModel` interface: `fit(X, y)`, `predict(X)`, `predict_proba(X)`, `save(path)`, `load(path)`.

| Model       | Input           | Approach                                  | Library      |
|-------------|-----------------|-------------------------------------------|--------------|
| DTW         | (N, 64, 2)      | 1-NN with dynamic time warping distance   | dtaidistance |
| HMM         | (N, 64, 4)      | Per-class GaussianHMM, argmax log-P       | hmmlearn     |
| CNN         | (N, 4, 64)      | Conv1d → GlobalAvgPool → Linear           | PyTorch      |
| LSTM        | (N, 64, 4)      | Bidirectional LSTM(128) × 2 layers        | PyTorch      |
| Transformer | (N, 64, 4)      | Sinusoidal PE + 2 encoder layers, CLS token | PyTorch    |

---

## Loading External Datasets

```bash
# $N Multistroke XML
python main.py load-external --format dollar_n --path path/to/gestures.xml

# Flat CSV (label, x0, y0, x1, y1, ...)
python main.py load-external --format csv --path path/to/gestures.csv

# Directory of per-class .txt files
python main.py load-external --format txt --path path/to/gesture_dir/
```

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite covers:
- Resampling: output shape, even spacing, endpoint preservation
- Normalization: centring, bounding, scale/translation invariance
- All models: predict shape, dtype, proba sums, save/load round-trip
- Segmenter: state transitions, gesture completion, edge cases

---

## Configuration

All parameters are in `configs/default.yaml`:

```yaml
gestures: [slash_left, slash_right, slash_down, circle, zigzag, triangle]

processing:
  n_points: 64
  use_pca: false

segmenter:
  start_threshold: 15    # px/frame
  end_threshold: 8
  n_idle_frames: 8
  min_gesture_len: 10

models:
  transformer:
    d_model: 64
    nhead: 4
    num_encoder_layers: 2
    epochs: 60
    lr: 0.001
```

---

## Pinned Dependencies

| Package                 | Version   | Reason                                             |
|-------------------------|-----------|----------------------------------------------------|
| mediapipe               | 0.10.21   | `solutions` API removed in ≥0.10.30                |
| opencv-contrib-python   | 4.11.0.86 | Last version compatible with numpy<2               |
| numpy                   | < 2       | Required by mediapipe 0.10.21                      |

Install order matters — see Setup above.

---

## Gesture Set

Default gestures (configurable):

| Name         | Description                       |
|--------------|-----------------------------------|
| slash_left   | Right-to-left horizontal swipe    |
| slash_right  | Left-to-right horizontal swipe    |
| slash_down   | Top-to-bottom vertical swipe      |
| circle       | Full clockwise loop               |
| zigzag       | Alternating left-right motion     |
| triangle     | Three-corner closed path          |
