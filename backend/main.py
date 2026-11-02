"""
DL-024: FastAPI Backend
Exposes /predict endpoint — accepts a video file, returns gesture prediction.
"""
import os, sys, tempfile, shutil

os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# ── ML imports ───────────────────────────────────────────────────────────────
for p in [
    os.path.join(os.path.dirname(__file__), "..", "ml", "src"),
    os.path.join(os.path.dirname(__file__), "ml", "src"),
    os.path.join(os.getcwd(), "ml", "src")
]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import tf_keras as keras
from model import build_sequence_model
from feature_extractor import HandFeatureExtractor
from frame_sampler import classify_video_with_sliding_window

CLASSES = ['Hello', 'Thank_You', 'Yes', 'No', 'Help']

# Find model path
MODEL_PATH = None
for mp in [
    os.path.join(os.path.dirname(__file__), "..", "ml", "models", "gesture_lstm_model.h5"),
    os.path.join(os.path.dirname(__file__), "ml", "models", "gesture_lstm_model.h5"),
    os.path.join(os.getcwd(), "ml", "models", "gesture_lstm_model.h5")
]:
    if os.path.exists(mp):
        MODEL_PATH = mp
        break

if not MODEL_PATH:
    MODEL_PATH = os.path.join("ml", "models", "gesture_lstm_model.h5")

app = FastAPI(title="DL-024 Sign Language API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
extractor = None

def get_loaded_model(path: str):
    """Safe model loader that handles cross-Keras deserialization differences."""
    try:
        return keras.models.load_model(path, compile=False)
    except Exception as e:
        print(f"load_model standard failed ({e}), loading via weights...")
        m = build_sequence_model(input_shape=(20, 120), num_classes=len(CLASSES))
        m.load_weights(path)
        return m

@app.on_event("startup")
def load_resources():
    global model, extractor
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found: {MODEL_PATH}")
    model = get_loaded_model(MODEL_PATH)
    extractor = HandFeatureExtractor()
    print(f"Model successfully loaded from: {MODEL_PATH}")
    print(f"Model input shape: {model.input_shape}")

class PredictResponse(BaseModel):
    gesture: str
    confidence: float
    all_scores: dict[str, float]

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/classes")
def get_classes():
    return {"classes": CLASSES}

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".webm")):
        raise HTTPException(status_code=400, detail="Upload .mp4 / .mov / .avi / .webm file.")

    suffix = os.path.splitext(file.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        label, conf, _, error = classify_video_with_sliding_window(
            tmp_path, extractor, model, CLASSES, window_size=20
        )
    finally:
        os.remove(tmp_path)

    if error:
        raise HTTPException(status_code=422, detail=error)

    other = round((100.0 - conf) / (len(CLASSES) - 1), 2)
    scores = {c: (round(conf, 2) if c == label else other) for c in CLASSES}

    return PredictResponse(gesture=label, confidence=round(conf, 2), all_scores=scores)
