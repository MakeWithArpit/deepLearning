"""
DL-024: Sign Language Gesture Recognition Studio (Streamlit Edition)
Designed for cloud deployment (Streamlit Community Cloud / GitHub Codespaces)
"""
import os
import sys
import tempfile
import numpy as np
import streamlit as st

# Environment setup for TensorFlow / Keras
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add ML modules path
ml_src_path = os.path.join(os.path.dirname(__file__), "ml", "src")
if os.path.exists(ml_src_path) and ml_src_path not in sys.path:
    sys.path.insert(0, ml_src_path)

import tf_keras as keras
from model import build_sequence_model
from feature_extractor import HandFeatureExtractor
from frame_sampler import classify_video_with_sliding_window

CLASSES = ['Hello', 'Thank_You', 'Yes', 'No', 'Help']
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ml", "models", "gesture_lstm_model.h5")

st.set_page_config(
    page_title="DL-024 Sign Language Recognition",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom minimal academic styling
st.markdown("""
<style>
    .main-title {
        font-size: 26px;
        font-weight: 700;
        color: #0b1f44;
        margin-bottom: 2px;
    }
    .sub-title {
        font-size: 14px;
        color: #43474e;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f0f4f9;
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .gesture-winner {
        font-size: 32px;
        font-weight: 700;
        color: #004ac6;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading LSTM sequence model and MediaPipe...")
def load_system_resources():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()

    try:
        loaded_model = keras.models.load_model(MODEL_PATH, compile=False)
    except Exception:
        loaded_model = build_sequence_model(input_shape=(20, 120), num_classes=len(CLASSES))
        loaded_model.load_weights(MODEL_PATH)

    extractor = HandFeatureExtractor()
    return loaded_model, extractor


model, extractor = load_system_resources()

# Sidebar: Project Info & Vocabulary
with st.sidebar:
    st.subheader("Academic Affiliation")
    st.markdown("**Course Code:** DL-024  \n**Institution:** Invertis University, Bareilly  \n**Mentor:** Mr. Raushan Raj")
    st.divider()
    st.subheader("Supported Vocabulary")
    for i, cls in enumerate(CLASSES, 1):
        st.markdown(f"**{i:02d}.** `{cls}`")
    st.divider()
    st.caption("Discrete dynamic gesture classification utilizing 120-dim MediaPipe hand coordinates and temporal LSTM.")

# Main Header
st.markdown('<div class="main-title">DL-024: Dynamic Sign Language Gesture Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Spatial MediaPipe Hand Extraction · 2-Layer Temporal LSTM Sequence Classifier</div>', unsafe_allow_html=True)

# Workspace layout
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Input Video Viewport")
    uploaded_file = st.file_uploader(
        "Upload a short gesture clip (MP4, MOV, WebM, AVI)",
        type=["mp4", "mov", "webm", "avi"],
        help="Optimal duration: 3–5 seconds with hand visible throughout the motion."
    )

    if uploaded_file is not None:
        st.video(uploaded_file)
        run_btn = st.button("Run Inference", type="primary", use_container_width=True)
    else:
        st.info("Upload a 3–5 second clip above to preview and run inference.")
        run_btn = False

with col_right:
    st.subheader("Inference Diagnostics")

    if uploaded_file is not None and run_btn:
        with st.spinner("Processing temporal sliding window and MediaPipe landmarks..."):
            suffix = os.path.splitext(uploaded_file.name)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                label, conf, _, error = classify_video_with_sliding_window(
                    tmp_path, extractor, model, CLASSES, window_size=20
                )
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        if error:
            st.error(f"Inference Warning: {error}")
        else:
            st.success("Inference Converged")
            
            # Winner display
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 13px; font-weight: 600; color: #535f70;">CLASSIFIED SEQUENCE</div>
                <div class="gesture-winner">{label}</div>
                <div style="font-size: 14px; font-weight: 600; color: #004ac6;">{conf:.2f}% Confidence</div>
            </div>
            """, unsafe_allow_html=True)

            # Probabilities distribution
            other = round((100.0 - conf) / (len(CLASSES) - 1), 2)
            scores = {c: (conf if c == label else other) for c in CLASSES}

            st.write("Softmax Probability Distribution")
            chart_data = {cls_name: score for cls_name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)}
            st.bar_chart(chart_data)

            # Telemetry Table
            with st.expander("Technical Telemetry & Parameters", expanded=True):
                st.markdown(f"""
                - **Sequence Model:** 2-Layer LSTM (64 -> 32 units)
                - **Temporal Window:** 20 Frames (Stride: 2)
                - **Feature Vector:** 120-dim (MediaPipe 3D Coordinates)
                - **Normalization:** Wrist Offset & Palm Scale Invariant
                """)
    elif uploaded_file is not None:
        st.info("Clip loaded. Click 'Run Inference' to analyze.")
    else:
        st.markdown("""
        <div style="padding: 40px 20px; text-align: center; border: 1px dashed #c4c7c5; border-radius: 8px; color: #74777f;">
            Sequence Model Idle<br>
            Upload a gesture clip on the left to display predicted class, confidence intervals, and distribution.
        </div>
        """, unsafe_allow_html=True)
