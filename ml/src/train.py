"""
DL-024: Training & Evaluation Script (Trained on Real Recorded Data)
Loads real_gestures.pkl, applies augmentation, trains LSTM sequence classifier,
and generates college report evaluation metrics and plots.

Run from ml/ directory:
    py -3.12 src/train.py
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.insert(0, os.path.dirname(__file__))

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tf_keras as keras
from tf_keras.utils import to_categorical
from tf_keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from model import build_sequence_model

BASE_DIR        = os.path.dirname(os.path.dirname(__file__))   # ml/
REAL_DATA_PATH  = os.path.join(BASE_DIR, "data", "real_gestures.pkl")
REPORT_DIR      = os.path.join(BASE_DIR, "..", "report")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "gesture_lstm_model.h5")

AUG_FACTOR = 6     # 150 samples * 7 = 1050 total samples
NOISE_STD  = 0.02


def augment_sequence(seq: np.ndarray, n: int = AUG_FACTOR) -> list:
    """Creates n augmented variations using Gaussian noise, scaling, and time-jitter."""
    augmented = []
    for _ in range(n):
        s = seq.copy()
        # 1. Noise
        s += np.random.normal(0, NOISE_STD, s.shape).astype(np.float32)
        # 2. Random scale
        scale = np.random.uniform(0.90, 1.10)
        s *= scale
        # 3. Temporal slight frame drop/repeat
        if np.random.random() > 0.5:
            T = s.shape[0]
            drop_idx = np.random.randint(1, T - 1)
            s = np.delete(s, drop_idx, axis=0)
            s = np.vstack([s, s[-1:]])
        augmented.append(s)
    return augmented


def train_and_evaluate():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    if not os.path.exists(REAL_DATA_PATH):
        raise FileNotFoundError(f"Real data file not found: {REAL_DATA_PATH}\nRun collect_data.py first.")

    with open(REAL_DATA_PATH, "rb") as f:
        data = pickle.load(f)

    X_raw = data["X"]    # (150, 20, 120)
    y_raw = data["y"]    # (150,)
    classes = data.get("classes", ['Hello', 'Thank_You', 'Yes', 'No', 'Help'])
    num_classes = len(classes)

    print(f"Loaded Real Data: {X_raw.shape[0]} samples, {num_classes} classes -> {classes}")

    # Data Augmentation
    X_all, y_all = [], []
    for seq, lbl in zip(X_raw, y_raw):
        X_all.append(seq)
        y_all.append(lbl)
        for aug in augment_sequence(seq, AUG_FACTOR):
            X_all.append(aug)
            y_all.append(lbl)

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int32)
    print(f"Dataset after Augmentation: {X_all.shape}")

    # Shuffle
    perm = np.random.permutation(len(X_all))
    X_all, y_all = X_all[perm], y_all[perm]

    # One-hot encoding
    y_cat = to_categorical(y_all, num_classes=num_classes)

    # Train / Test split (Stratified)
    X_train, X_test, y_train_cat, y_test_cat = train_test_split(
        X_all, y_cat, test_size=0.20, random_state=42, stratify=y_all
    )
    y_test = np.argmax(y_test_cat, axis=1)

    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

    model = build_sequence_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        num_classes=num_classes
    )
    model.summary()

    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)
    ]

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=45,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # 1. Training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'],     label='Train Accuracy', color='#1f77b4', lw=2)
    plt.plot(history.history['val_accuracy'], label='Val Accuracy',   color='#ff7f0e', lw=2)
    plt.title('Model Accuracy vs Epochs', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'],     label='Train Loss', color='#1f77b4', lw=2)
    plt.plot(history.history['val_loss'], label='Val Loss',   color='#ff7f0e', lw=2)
    plt.title('Model Loss vs Epochs', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)

    curves_path = os.path.join(REPORT_DIR, "training_curves.png")
    plt.tight_layout(); plt.savefig(curves_path, dpi=300); plt.close()
    print(f"Training curves saved to: {curves_path}")

    # 2. Confusion matrix & Classification Report
    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=classes, digits=4)
    print("\n--- Classification Report (Real Data Test Split) ---")
    print(cr)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix — Real Gestures (DL-024)', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.tight_layout()
    cm_path = os.path.join(REPORT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300); plt.close()
    print(f"Confusion matrix saved to: {cm_path}")

    with open(os.path.join(REPORT_DIR, "evaluation_metrics.txt"), "w") as f:
        f.write("DL-024 EVALUATION METRICS ON REAL DATA TEST SPLIT\n")
        f.write("=" * 60 + "\n\n")
        f.write(cr)
        f.write("\n\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    print(f"\nModel successfully saved to: {MODEL_SAVE_PATH}")
    print("Training and Evaluation Complete!")


if __name__ == '__main__':
    train_and_evaluate()
