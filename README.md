# Hand-Tracking
Hand tracking and air draw using mediapipe, python 3.12 and oven cv


# ✋ Air Draw — Hand Gesture Drawing App

Draw in the air. No mouse. No touchscreen. Just your hand and a webcam.

## What is this?

Air Draw is a real-time hand tracking application that turns your webcam into a canvas. Point your index finger at the camera and draw — literally in thin air. Switch colors, change brush thickness, undo strokes, erase, and clear the canvas, all without touching your keyboard or mouse.

It uses **MediaPipe** to detect hand landmarks and **OpenCV** to render everything live on screen. The toolbar is gesture-controlled too — hover your finger over a button for ~0.8 seconds and it clicks itself.

---

## Features

- **Real-time hand tracking** via MediaPipe (index finger tip as cursor)
- **Air drawing** — leaves a stroke trail as you move your finger
- **Gesture switching** — one finger to draw, open palm to pause
- **Color palette** — Neon Green, Pink, Cyan, Orange
- **Brush thickness** — Thin, Medium, Thick
- **Undo / Redo** strokes
- **Eraser tool** — erase specific parts without clearing everything
- **Clear canvas** — trash everything and start fresh
- **Dwell-click toolbar** — no clicks needed, just hover

---

## Demo

> *(Add a screen recording GIF here once captured)*

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Runtime |
| OpenCV | Frame capture, rendering, UI drawing |
| MediaPipe | Hand landmark detection |
| NumPy | Canvas array operations |

---

## Setup & Installation

### Prerequisites

You need **Python 3.12** specifically. MediaPipe does not support Python 3.13 or 3.14 yet — pre-built wheels simply don't exist for those versions. I learned this the hard way.

### 1. Clone the repo

```bash
git clone https://github.com/your-username/air-draw.git
cd air-draw
```

### 2. Create a virtual environment with Python 3.12

```bash
# Windows
py -3.12 -m venv venv
venv\Scripts\activate

# Mac / Linux
python3.12 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install opencv-python mediapipe numpy
```

### 4. Run it

```bash
python air_draw.py
```

Press **`q`** to quit.

---

## How to Use

| Gesture | Action |
|---|---|
| ☝️ Index finger only | Draw mode — traces your fingertip |
| ✋ All 5 fingers open | Pause — move freely without drawing |
| Hover over toolbar ~0.8s | Activates button (dwell-click) |

The toolbar lives on the right side of the screen. From top to bottom: colors → brush size → undo → redo → eraser → clear.

---

## Honest Notes on How This Was Built

I'll be straight — the core code was generated using **Claude in agent mode** (Cursor's agentic workflow, sometimes called "antigravity" coding). I gave it a detailed feature spec and it wrote the Python file.

But here's the thing: the code working on paper and the code *actually running* are two completely different problems.

What I ended up learning hands-on:

- **Virtual environments** — I'd never created one before this project. Turns out managing Python versions and isolating dependencies is a whole skill in itself, not just something you skip past in tutorials.
- **Python version compatibility** — I was running Python 3.14. MediaPipe doesn't support it. Neither does 3.13. I had to go find, install, and configure Python 3.12 separately, then point a venv at it specifically. That took real debugging.
- **Dependency resolution** — understanding *why* a package fails, not just that it fails, is different from copy-pasting an install command.
- **Reading error messages** — "no matching distribution found" hits different when it's *your* project and you have to fix it.

The agent wrote the logic. I made it actually run. That gap — between generated code and a working environment — taught me more about Python packaging in two hours than months of tutorials did.

---

## Why I Built This

I came across a demo of a hand gesture drawing app and thought it was genuinely cool. I wanted to understand how it worked and have something tangible in my portfolio that goes beyond finance models and spreadsheets. Computer vision felt like a good stretch.

Also — it's just fun to draw in the air.

---

## Future Ideas

- Export canvas as PNG
- Multi-hand support
- Web version using MediaPipe.js (so it runs in a browser without any install)
- Shape recognition — draw a rough circle and it snaps to a perfect one
---

*Built by Snehal · SVNIT · April 2026*
