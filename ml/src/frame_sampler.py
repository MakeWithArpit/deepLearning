"""
DL-024: Temporal Video Processor with Sliding-Window Gesture Analysis
Standardizes video frame rate, performs consecutive 30 FPS feature extraction,
and runs a temporal sliding window to find the peak gesture activation.
"""
import cv2
import numpy as np

def classify_video_with_sliding_window(video_path, extractor, model, classes, window_size=20):
    """
    Processes video clip using sequential 30 FPS feature extraction and
    sliding-window peak confidence pooling.
    
    Returns:
        best_label: str (e.g. 'Hello')
        best_conf: float (e.g. 96.5)
        best_annotated_frames: list of 20 annotated frames from winning window
        error_msg: str or None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0.0, None, "Could not open video file."

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    # Read all frames with FPS normalization (target ~25-30 FPS)
    step = 2 if fps >= 45 else 1
    raw_frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            # Resize if ultra 4K to prevent MediaPipe slowdown
            h, w = frame.shape[:2]
            if w > 1280 or h > 1280:
                scale = 640 / max(w, h)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            raw_frames.append(frame)
        frame_idx += 1

    cap.release()

    if not raw_frames:
        return None, 0.0, None, "Video file contains no readable frames."

    # Sequential frame-by-frame feature extraction (consecutive frames = stable shape features)
    extractor.reset_tracking()
    all_features = []
    annotated_frames = []
    hand_detected_flags = []

    for frame in raw_frames:
        feat, ann, num_detected = extractor.extract_from_frame(frame, draw=True)
        all_features.append(feat)
        annotated_frames.append(ann)
        hand_detected_flags.append(num_detected > 0)

    total_valid_hands = sum(hand_detected_flags)
    if total_valid_hands < 5:
        # Check if rotating 90 degrees helps (mobile portrait orientation)
        extractor.reset_tracking()
        rot_features = []
        rot_annotated = []
        rot_flags = []
        for frame in raw_frames:
            rot_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            feat, ann, num_detected = extractor.extract_from_frame(rot_frame, draw=True)
            rot_features.append(feat)
            rot_annotated.append(ann)
            rot_flags.append(num_detected > 0)

        if sum(rot_flags) > total_valid_hands:
            all_features = rot_features
            annotated_frames = rot_annotated
            hand_detected_flags = rot_flags
            total_valid_hands = sum(rot_flags)

    if total_valid_hands < 5:
        return None, 0.0, None, "No human hand detected in video. Please ensure your hand is visible and clearly lit."

    N = len(all_features)

    # If video is shorter than 20 frames, pad with last frame
    if N < window_size:
        pad_count = window_size - N
        all_features += [all_features[-1]] * pad_count
        annotated_frames += [annotated_frames[-1]] * pad_count
        hand_detected_flags += [hand_detected_flags[-1]] * pad_count
        N = window_size

    # Sliding Window Temporal Search
    best_conf = -1.0
    best_pred_idx = 0
    best_window_start = 0

    # Slide window across frames with stride = 2
    stride = 2
    windows = []
    window_starts = []

    for start in range(0, N - window_size + 1, stride):
        # Only evaluate windows where hand was present for at least 8 frames
        valid_in_win = sum(hand_detected_flags[start : start + window_size])
        if valid_in_win >= 8:
            win_tensor = np.array(all_features[start : start + window_size], dtype=np.float32)
            windows.append(win_tensor)
            window_starts.append(start)

    if not windows:
        # Fallback to the center window
        center_start = max(0, (N - window_size) // 2)
        win_tensor = np.array(all_features[center_start : center_start + window_size], dtype=np.float32)
        windows = [win_tensor]
        window_starts = [center_start]

    # Batch prediction on all sliding windows
    batch_input = np.array(windows, dtype=np.float32)
    predictions = model.predict(batch_input, verbose=0)

    for i, probs in enumerate(predictions):
        pred_c = int(np.argmax(probs))
        conf = float(probs[pred_c]) * 100.0

        if conf > best_conf:
            best_conf = conf
            best_pred_idx = pred_c
            best_window_start = window_starts[i]

    best_label = classes[best_pred_idx]
    winning_frames = annotated_frames[best_window_start : best_window_start + window_size]

    return best_label, best_conf, winning_frames, None
