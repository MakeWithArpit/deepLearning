# DL-024: Sign Language Gesture Recognition from Short Clips

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/TensorFlow-2.17-orange.svg)](https://tensorflow.org/)
[![Inference](https://img.shields.io/badge/MediaPipe-Tasks_API-teal.svg)](https://developers.google.com/mediapipe)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containers_Ready-2496ED.svg)](https://www.docker.com/)

An accessibility deep learning research prototype designed to classify short video clips of discrete dynamic sign language gestures using spatial MediaPipe hand landmark extraction and sequence modeling with Long Short-Term Memory (LSTM) neural networks.

---

## Academic Affiliation & Attribution

* **Course Code / Problem ID:** DL-024 (Deep Learning / Accessibility)
* **Institution:** Department of Computer Science & Engineering, Invertis University, Bareilly
* **Degree Program:** Bachelor of Technology (B.Tech)
* **Project Mentor:** Mr. Raushan Raj

---

## Target Gesture Vocabulary

The system classifies 5 dynamic hand gestures represented across temporal sequences of 20 frames:

| ID | Class Label | Motion Characteristics | Real-World Meaning |
|---|---|---|---|
| **01** | `Hello` | Lateral open-palm oscillation | Friendly greeting wave |
| **02** | `Thank_You` | Forward push / translation from body | Expression of gratitude |
| **03** | `Yes` | Vertical nodding fist motion | Affirmation / Agreement |
| **04** | `No` | Index finger & thumb pinch oscillation | Negation / Disagreement |
| **05** | `Help` | Two-hand upward support lifting motion | Assistance request |

---

## System Architecture & Workflow

```
[ Short Video Clip (3–5s) ]
            │
            ▼
[ Temporal Sliding Window (frame_sampler.py) ]
  • Consecutive 30 FPS sampling
  • Size: 20 frames, Stride: 2
            │
            ▼
[ Spatial Feature Extraction (feature_extractor.py) ]
  • MediaPipe HandLandmarker (21 3D points per hand)
  • Wrist Normalization: (P[1:21] - Wrist)
  • Scale Invariance: Normalized by Palm Length
  • Feature Vector: 60 coords × 2 hands = 120-dim per frame
            │
            ▼
[ Sequence Classifier (model.py) ]
  • Input: (Batch, 20 Frames, 120 Features)
  • Layer 1: LSTM (64 units, return_sequences=True, dropout=0.2)
  • Layer 2: LSTM (32 units, return_sequences=False, dropout=0.2)
  • Dense (32, ReLU) → Dense (5, Softmax)
            │
            ▼
[ FastAPI Server & Laboratory Instrumentation UI ]
```

---

## Evaluation & Metrics (Trained on Real Gesture Sequences)

Evaluated on unseen test splits of real recorded gestures:

```
              precision    recall  f1-score   support

       Hello     1.0000    1.0000    1.0000        42
   Thank_You     1.0000    1.0000    1.0000        42
         Yes     1.0000    1.0000    1.0000        42
          No     1.0000    1.0000    1.0000        42
        Help     1.0000    1.0000    1.0000        42

    accuracy                         1.0000       210
   macro avg     1.0000    1.0000    1.0000       210
weighted avg     1.0000    1.0000    1.0000       210
```

Visual confusion matrix and training curves are stored under the [`report/`](report/) directory.

---

## Prerequisites

### For Docker Method
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### For Local Python Method
- Python 3.10 or higher
- pip

---

## Repository Structure

```text
deepLearning/
├── backend/
│   ├── Dockerfile             # Container image for FastAPI inference service
│   ├── main.py                # REST API endpoints (/health, /predict, /classes)
│   └── requirements.txt       # Production backend Python dependencies
├── frontend/
│   ├── Dockerfile             # Nginx Alpine container image
│   ├── index.html             # Academic Instrumentation UI (Stitch benchmark style)
│   ├── style.css              # Technical light canvas stylesheet (WCAG AAA)
│   └── app.js                 # Viewport controller and API client
├── ml/
│   ├── data/
│   │   └── real_gestures.pkl  # Calibrated real user gesture dataset
│   ├── models/
│   │   └── gesture_lstm_model.h5 # Trained 2-layer LSTM sequence weights
│   └── src/
│       ├── collect_data.py    # Real-time webcam sequence collector
│       ├── feature_extractor.py # 120-dim normalized MediaPipe extractor
│       ├── frame_sampler.py   # 20-frame sliding window clip sampler
│       ├── generate_dataset.py# Synthetic sequence generator (baseline)
│       ├── hand_landmarker.task # Pre-trained MediaPipe landmark weights
│       ├── model.py           # Keras LSTM architecture definition
│       └── train.py           # Training, augmentation & evaluation suite
├── report/
│   ├── confusion_matrix.png   # Class-wise evaluation matrix
│   ├── evaluation_metrics.txt # Raw classification report
│   └── training_curves.png    # Epoch loss & accuracy plots
├── .gitignore                 # Excludes cache, virtualenvs & artifacts
├── docker-compose.yml         # Multi-container orchestration (ports 3000, 8005)
└── README.md
```

---

## Quickstart — New User Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/MakeWithArpit/deepLearning.git
cd deepLearning
```

---

### Option A: Run with Docker (Recommended)

> Requires Docker Desktop to be installed and running.

```bash
docker compose up --build
```

First build takes ~10–15 minutes (downloads TensorFlow + MediaPipe).
Subsequent runs start in under 30 seconds.

| Service | URL |
|---|---|
| Frontend UI | http://localhost:3000 |
| Backend API | http://localhost:8005 |
| API Docs (Swagger) | http://localhost:8005/docs |
| Health Check | http://localhost:8005/health |

To stop:
```bash
docker compose down
```

---

### Option B: Run without Docker (Local Python)

> Requires Python 3.10+

**Step 1 — Install backend dependencies:**
```bash
pip install -r backend/requirements.txt
```

> First install takes ~5–10 minutes (TensorFlow + MediaPipe are large packages).

**Step 2 — Start the backend server:**

On Windows:
```bash
set PYTHONPATH=.;ml/src
uvicorn backend.main:app --host 0.0.0.0 --port 8005
```

On macOS / Linux:
```bash
PYTHONPATH=.:ml/src uvicorn backend.main:app --host 0.0.0.0 --port 8005
```

**Step 3 — Open the frontend:**

Open `frontend/index.html` directly in your browser, **or** serve it with Python:
```bash
cd frontend
python -m http.server 3000
```
Then open [http://localhost:3000](http://localhost:3000).

> Backend must be running on port 8005 before using the frontend.

---

## Project Scope & Boundary Disclaimers (DL-024)

As defined in the course curriculum specification:
* **In Scope:** Classification of discrete dynamic gestures extracted from short clips (3–5 seconds) utilizing spatial hand landmarks and sequence modeling.
* **Out of Scope:** Continuous unsegmented signing, grammatical inflection modeling, non-manual markers (facial affect), and real-time sentence construction.
