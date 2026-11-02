"""
DL-024: Pretrained Feature Extractor (MediaPipe Hands)
Extracts wrist-normalized, scale-invariant 3D hand shape features only.
- Shape: 20 keypoints relative to wrist / palm_scale = 60 features per hand
- No velocity (velocity distorts when sampling rate changes between live camera and uploaded video)
- 2 hands: 120 features total per frame
- LSTM over T=20 frames learns motion implicitly from shape deltas
"""
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

FEATURE_DIM = 120  # 60 shape per hand × 2 hands

class HandFeatureExtractor:
    def __init__(self, model_path=MODEL_PATH):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def reset_tracking(self):
        pass  # no state to reset

    def _extract_shape(self, landmarks):
        """
        Returns 60-dim wrist-normalised, scale-invariant shape vector.
        Subtracts wrist (pt 0), divides by palm length (wrist→middle_mcp).
        Invariant to hand position, size, and camera distance.
        """
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        wrist = pts[0]
        palm_scale = np.linalg.norm(pts[9, :2] - wrist[:2])
        if palm_scale < 1e-4:
            palm_scale = 0.1
        shape = ((pts[1:] - wrist) / palm_scale).flatten()  # 20×3 = 60
        return np.clip(shape, -5.0, 5.0)

    def extract_from_frame(self, frame_bgr, draw=True):
        """
        Returns:
          features: (120,) float32
          annotated_frame
          num_hands_detected: int
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect(mp_image)

        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        annotated = frame_bgr.copy() if draw else frame_bgr
        num_detected = 0

        if result.hand_landmarks:
            num_detected = len(result.hand_landmarks)
            for hand_idx, landmarks in enumerate(result.hand_landmarks[:2]):
                shape = self._extract_shape(landmarks)
                features[hand_idx * 60: (hand_idx + 1) * 60] = shape

                if draw:
                    coords_px = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
                    for cx, cy in coords_px:
                        cv2.circle(annotated, (cx, cy), 4, (0, 255, 120), -1)
                    for s, e in HAND_CONNECTIONS:
                        cv2.line(annotated, coords_px[s], coords_px[e], (0, 220, 255), 2)

        return features, annotated, num_detected

    def extract_landmarks_from_frame(self, frame_bgr):
        feats, _, _ = self.extract_from_frame(frame_bgr, draw=False)
        return feats

