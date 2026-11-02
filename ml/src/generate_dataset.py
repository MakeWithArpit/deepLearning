"""
DL-024: Dataset Generator — Shape-Only Features (120-dim)
60 wrist-normalised shape features × 2 hands.
No velocity: LSTM over 20 frames learns motion from shape deltas implicitly.
Gestures distinguished by:
  Hello      → lateral oscillation of ALL fingertip x-positions over time
  Thank_You  → palm gradually translates toward camera (pts shift forward)
  Yes        → fist shape + knuckle y oscillation
  No         → index/middle tips oscillating toward thumb tip (pinch cycle)
  Help       → two-hand configuration with upward drift
"""
import os, pickle
import numpy as np

GESTURE_CLASSES = ['Hello', 'Thank_You', 'Yes', 'No', 'Help']
FEATURE_DIM = 120  # 60 shape per hand × 2 hands

# --------------- Hand shape builders (wrist-normalised, scale=1) ---------------

def _open_palm():
    """Normalised open-palm keypoints (20 kpts × 3), wrist at origin, scale 1."""
    pts = np.zeros((20, 3), dtype=np.float32)
    # Thumb (lm 1-4)
    pts[0] = [0.25, -0.22, 0.0]
    pts[1] = [0.45, -0.48, 0.02]
    pts[2] = [0.55, -0.72, 0.02]
    pts[3] = [0.60, -0.96, 0.02]
    # Fingers: x-base, finger_length
    bases = [(0.20, 2.2), (0.00, 2.4), (-0.20, 2.2), (-0.38, 1.8)]
    for i, (bx, fl) in enumerate(bases):
        b = 4 + i * 4
        pts[b]   = [bx, -1.00, 0.0]
        pts[b+1] = [bx, -1.00 - fl*0.40, 0.0]
        pts[b+2] = [bx, -1.00 - fl*0.72, 0.0]
        pts[b+3] = [bx, -1.00 - fl*1.00, 0.0]
    return pts

def _fist():
    """Normalised closed-fist keypoints."""
    pts = np.zeros((20, 3), dtype=np.float32)
    pts[0] = [0.20, -0.20, 0.0]
    pts[1] = [0.35, -0.40, 0.1]
    pts[2] = [0.30, -0.55, 0.2]
    pts[3] = [0.15, -0.60, 0.2]
    for i, bx in enumerate([0.20, 0.00, -0.20, -0.38]):
        b = 4 + i * 4
        pts[b]   = [bx, -0.9,  0.0]
        pts[b+1] = [bx, -0.70, -0.20]
        pts[b+2] = [bx, -0.50, -0.30]
        pts[b+3] = [bx, -0.40, -0.20]
    return pts

def _build_frame(gesture, t, signer_noise=0.02):
    """Returns (120,) feature vector for one frame at time t ∈ [0,1]."""
    h1 = np.zeros((20, 3), dtype=np.float32)
    h2 = np.zeros((20, 3), dtype=np.float32)

    if gesture == 'Hello':
        h1 = _open_palm().copy()
        # Lateral wave: all x-coords oscillate
        dx = 0.28 * np.sin(2 * np.pi * 2.0 * t)
        h1[:, 0] += dx

    elif gesture == 'Thank_You':
        h1 = _open_palm().copy()
        # Fingers pressed together for a respectful bow shape
        for i in range(4):
            b = 4 + i * 4
            h1[b:b+4, 0] *= 0.55
        # Wrist-frame: the hand shape STAYS SAME but the arm moves forward.
        # In wrist-normalised coords the shape is stable — the discriminating
        # signal is the *gradually opening* of fingers mid-motion.
        open_factor = t  # 0 → closed, 1 → fully open
        for i in range(4):
            b = 4 + i * 4
            h1[b:b+4, 1] *= (0.6 + 0.4 * open_factor)

    elif gesture == 'Yes':
        h1 = _fist().copy()
        # Fist nods vertically
        dy = 0.22 * np.sin(2 * np.pi * 2.5 * t)
        h1[:, 1] += dy

    elif gesture == 'No':
        h1 = _open_palm().copy()
        # Index + middle tips oscillate toward thumb tip (pinch cycle)
        pinch = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
        # tip indices (0-indexed in the 20-kpt array): thumb=3, index=7, middle=11
        thumb_tip = h1[3].copy()
        h1[7] = thumb_tip + np.array([0.20 * pinch, -0.10 * pinch, 0.0])
        h1[11] = thumb_tip + np.array([0.28 * pinch, -0.10 * pinch, 0.0])

    elif gesture == 'Help':
        # Hand 1: thumbs-up fist
        h1 = _fist().copy()
        h1[3] = [0.20, -1.40, 0.0]   # thumb extended up
        # Hand 2: flat support palm (offset left)
        h2 = _open_palm().copy()
        h2[:, 0] -= 0.55             # shift left
        h2[:, 1] += 0.15             # slightly lower
        # Both drift upward together
        drift = -0.30 * t
        h1[:, 1] += drift
        h2[:, 1] += drift

    # Add per-signer noise
    h1 += np.random.normal(0, signer_noise, h1.shape).astype(np.float32)
    h2 += np.random.normal(0, signer_noise, h2.shape).astype(np.float32)

    # Flatten to (60,) each, concatenate → (120,)
    feat = np.concatenate([h1.flatten(), h2.flatten()])
    return feat.astype(np.float32)

# --------------- Dataset generation ---------------

def generate_full_dataset(samples_per_class=200, num_frames=20,
                          output_dir="D:/Anti/deepLearning/Sign_Language_DL024/data"):
    X_train, y_train = [], []
    X_test,  y_test  = [], []

    # 4 signers with varying hand sizes / positions, Signer D = unseen test
    signers = [
        ("A", 0.018, 1.00),
        ("B", 0.025, 0.90),
        ("C", 0.022, 1.10),
        ("D", 0.030, 1.05),   # Unseen test signer
    ]

    for cls_idx, cls in enumerate(GESTURE_CLASSES):
        print(f"Generating: {cls}")
        for s_name, noise, speed in signers:
            n = samples_per_class // 4
            for _ in range(n):
                t_vals = np.linspace(0, 1.0 * speed, num_frames)
                seq = np.array([_build_frame(cls, t, noise) for t in t_vals],
                               dtype=np.float32)
                if s_name == "D":
                    X_test.append(seq)
                    y_test.append(cls_idx)
                else:
                    X_train.append(seq)
                    y_train.append(cls_idx)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    X_test  = np.array(X_test,  dtype=np.float32)
    y_test  = np.array(y_test,  dtype=np.int32)

    path = os.path.join(output_dir, "dataset.pkl")
    with open(path, "wb") as f:
        pickle.dump({"X_train": X_train, "y_train": y_train,
                     "X_test": X_test,   "y_test": y_test,
                     "classes": GESTURE_CLASSES}, f)
    print(f"Saved -> {path}")
    print(f"Train {X_train.shape}, Test {X_test.shape}")
    return path

if __name__ == "__main__":
    generate_full_dataset()
