"""
DL-024: Sign Language Gesture Recognition Studio
Renders the exact Stitch research prototype frontend via Streamlit Component
"""
import os
import sys
import base64
import tempfile
import streamlit as st
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="Sign Language Gesture Recognition · DL-024",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Strip all default Streamlit chrome, decorations, headers and footers
st.markdown("""
<style>
/* Remove Streamlit decoration bar, header, toolbar, and footer */
[data-testid="stDecoration"],
[data-testid="stHeader"],
header,
footer,
#MainMenu,
.stDeployButton,
.stAppDeployButton,
div[data-testid="stToolbar"] {
    display: none !important;
    height: 0px !important;
    min-height: 0px !important;
    padding: 0 !important;
    margin: 0 !important;
}

html, body, .stApp, .stAppViewContainer, .stMain, .block-container, [data-testid="stAppViewBlockContainer"] {
    background-color: #f9f9ff !important;
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
}

iframe {
    width: 100% !important;
    border: none !important;
    display: block !important;
    margin: 0 !important;
    padding: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# Environment setup for TensorFlow / Keras
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ml_src_path = os.path.join(os.path.dirname(__file__), "ml", "src")
if os.path.exists(ml_src_path) and ml_src_path not in sys.path:
    sys.path.insert(0, ml_src_path)

import tf_keras as keras
from model import build_sequence_model
from feature_extractor import HandFeatureExtractor
from frame_sampler import classify_video_with_sliding_window

CLASSES = ['Hello', 'Thank_You', 'Yes', 'No', 'Help']
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ml", "models", "gesture_lstm_model.h5")


@st.cache_resource(show_spinner=False)
def load_system_resources():
    if not os.path.exists(MODEL_PATH):
        return None, None
    try:
        loaded_model = keras.models.load_model(MODEL_PATH, compile=False)
    except Exception:
        loaded_model = build_sequence_model(input_shape=(20, 120), num_classes=len(CLASSES))
        loaded_model.load_weights(MODEL_PATH)
    extractor = HandFeatureExtractor()
    return loaded_model, extractor


model, extractor = load_system_resources()

# Declare component pointing to the exact frontend/ directory
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
stitch_component = components.declare_component("stitch_studio", path=frontend_dir)

# Initialize session state for prediction results
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# Render the exact frontend/index.html interface
component_event = stitch_component(
    prediction=st.session_state.prediction_result,
    key="studio_root"
)

# Process inference requests from frontend JS
if component_event and isinstance(component_event, dict):
    action = component_event.get("action")
    if action == "predict":
        b64_data = component_event.get("base64")
        filename = component_event.get("filename", "clip.mp4")
        suffix = os.path.splitext(filename)[-1] or ".mp4"

        if b64_data and model is not None and extractor is not None:
            raw_bytes = base64.b64decode(b64_data)
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(raw_bytes)
                tmp_path = tmp.name

            try:
                label, conf, _, error = classify_video_with_sliding_window(
                    tmp_path, extractor, model, CLASSES, window_size=20
                )
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            if error:
                st.session_state.prediction_result = {"error": error}
            else:
                other = round((100.0 - conf) / (len(CLASSES) - 1), 2)
                scores = {c: (round(conf, 2) if c == label else other) for c in CLASSES}
                st.session_state.prediction_result = {
                    "gesture": label,
                    "confidence": round(conf, 2),
                    "all_scores": scores
                }
            st.rerun()

    elif action == "reset":
        st.session_state.prediction_result = None
        st.rerun()
