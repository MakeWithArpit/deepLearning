"""
DL-024: Real Gesture Data Collection Tool
==========================================
Records labeled gesture feature sequences from webcam.
Run BEFORE training to collect real-world training data.

Usage (from ml/ directory):
    py -3.12 src/collect_data.py --camera 1   # USB mobile camera
    py -3.12 src/collect_data.py --camera 0   # Laptop camera
"""
import os, sys, time, pickle, cv2, numpy as np, argparse

sys.path.insert(0, os.path.dirname(__file__))
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from feature_extractor import HandFeatureExtractor

GESTURE_CLASSES   = ['Hello', 'Thank_You', 'Yes', 'No', 'Help']
SAMPLES_PER_CLASS = 30
FRAMES_PER_SAMPLE = 20
COUNTDOWN_SECS    = 3

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))   # ml/
DATA_DIR  = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "real_gestures.pkl")

GREEN  = (0, 220, 80);  ORANGE = (0, 165, 255)
RED    = (60, 60, 240);  WHITE  = (255, 255, 255)
GRAY   = (80, 80, 80);   YELLOW = (0, 220, 220)


def put_text(img, text, y, color=WHITE, scale=0.9, thickness=2):
    cv2.putText(img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_overlay(img, gesture, sample_num, total, phase, recorded=0):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    put_text(img, f"Gesture: {gesture}  [{sample_num}/{total}]", 35, GREEN, 1.0, 2)
    if phase == 'wait':
        put_text(img, "Perform gesture NOW. Press SPACE when ready.", 70, YELLOW, 0.75)
        put_text(img, "Press Q to quit & save.", 100, GRAY, 0.6)
    elif phase == 'countdown':
        put_text(img, f"Get ready... starting in {recorded}s", 70, ORANGE, 0.85)
    elif phase == 'record':
        put_text(img, "RECORDING -- keep gesture visible!", 70, RED, 0.85, 2)
        bar_x1, bar_y1, bar_x2, bar_y2 = 15, 88, w - 15, 108
        cv2.rectangle(img, (bar_x1, bar_y1), (bar_x2, bar_y2), GRAY, -1)
        filled = int((recorded / FRAMES_PER_SAMPLE) * (bar_x2 - bar_x1))
        cv2.rectangle(img, (bar_x1, bar_y1), (bar_x1 + filled, bar_y2), GREEN, -1)
    elif phase == 'ok':
        put_text(img, "Got it! Well done.", 70, GREEN, 0.9, 2)
    elif phase == 'no_hand':
        put_text(img, "No hand detected -- try again!", 70, RED, 0.85)


def collect_data(camera_idx: int = 1):
    ext = HandFeatureExtractor()
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Camera {camera_idx} not available. Try --camera 0")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    all_X, all_y = [], []
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, 'rb') as f:
            saved = pickle.load(f)
        all_X = list(saved['X']);  all_y = list(saved['y'])
        print(f"Loaded {len(all_X)} existing samples.")

    try:
        for class_idx, gesture in enumerate(GESTURE_CLASSES):
            existing = sum(1 for v in all_y if v == class_idx)
            need = SAMPLES_PER_CLASS - existing
            if need <= 0:
                print(f"[SKIP] {gesture}: already {existing} samples.")
                continue

            print(f"\n{'='*50}\n  {gesture} [{class_idx+1}/{len(GESTURE_CLASSES)}] — need {need} more\n{'='*50}")
            sample_count = existing

            while True:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                draw_overlay(frame, gesture, sample_count + 1, SAMPLES_PER_CLASS, 'wait')
                cv2.imshow("DL-024 Data Collector", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '): break
                if key == ord('q'): raise KeyboardInterrupt

            while sample_count < SAMPLES_PER_CLASS:
                for cd in range(COUNTDOWN_SECS, 0, -1):
                    end = time.time() + 1.0
                    while time.time() < end:
                        ret, frame = cap.read()
                        if not ret: break
                        frame = cv2.flip(frame, 1)
                        draw_overlay(frame, gesture, sample_count + 1, SAMPLES_PER_CLASS, 'countdown', cd)
                        cv2.imshow("DL-024 Data Collector", frame); cv2.waitKey(1)

                feats, attempts = [], 0
                while len(feats) < FRAMES_PER_SAMPLE and attempts < FRAMES_PER_SAMPLE * 5:
                    ret, frame = cap.read()
                    if not ret: break
                    attempts += 1
                    frame = cv2.flip(frame, 1)
                    feat, _, num_det = ext.extract_from_frame(frame, draw=False)
                    if num_det > 0:
                        feats.append(feat)
                    draw_overlay(frame, gesture, sample_count + 1, SAMPLES_PER_CLASS, 'record', len(feats))
                    cv2.imshow("DL-024 Data Collector", frame); cv2.waitKey(1)

                if len(feats) >= FRAMES_PER_SAMPLE:
                    all_X.append(np.array(feats[:FRAMES_PER_SAMPLE], dtype=np.float32))
                    all_y.append(class_idx)
                    sample_count += 1
                    print(f"  Sample {sample_count}/{SAMPLES_PER_CLASS} OK")
                    flash_end = time.time() + 0.7
                    while time.time() < flash_end:
                        ret, frame = cap.read()
                        if not ret: break
                        frame = cv2.flip(frame, 1)
                        draw_overlay(frame, gesture, sample_count, SAMPLES_PER_CLASS, 'ok')
                        cv2.imshow("DL-024 Data Collector", frame); cv2.waitKey(20)
                else:
                    print("  No hand detected — try again.")
                    flash_end = time.time() + 1.2
                    while time.time() < flash_end:
                        ret, frame = cap.read()
                        if not ret: break
                        frame = cv2.flip(frame, 1)
                        draw_overlay(frame, gesture, sample_count + 1, SAMPLES_PER_CLASS, 'no_hand')
                        cv2.imshow("DL-024 Data Collector", frame); cv2.waitKey(20)

    except KeyboardInterrupt:
        print("\n[STOPPED] Saving partial data...")
    finally:
        cap.release(); cv2.destroyAllWindows(); _save(all_X, all_y)


def _save(X_list, y_list):
    if not X_list:
        print("Nothing to save."); return
    os.makedirs(DATA_DIR, exist_ok=True)
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    with open(DATA_PATH, 'wb') as f:
        pickle.dump({'X': X, 'y': y, 'classes': GESTURE_CLASSES}, f)
    print(f"\n[SAVED] {len(X_list)} samples -> {DATA_PATH}  shape: {X.shape}")
    for i, g in enumerate(GESTURE_CLASSES):
        cnt = int((y == i).sum())
        print(f"  {g:12s}: {cnt}/{SAMPLES_PER_CLASS}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=1)
    args = parser.parse_args()
    print(f"DL-024 Data Collector | Camera: {args.camera}")
    collect_data(camera_idx=args.camera)
