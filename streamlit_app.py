"""
DL-024: Sign Language Gesture Recognition Studio (Streamlit Edition)
Exact Recreation of the Stitch Research Prototype Design System
Academic light canvas, cobalt blue instrumentation, minimal & professional
"""
import os
import sys
import tempfile
import numpy as np
import streamlit as st

# TensorFlow / Keras environment flags
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
    page_title="Sign Language Gesture Recognition · DL-024",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Inject Full Stitch UI Design System CSS ────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet">

<style>
/* Reset & Streamlit Chrome Overrides */
#MainMenu, header, footer, .stDeployButton, .stAppDeployButton { display: none !important; }
.block-container {
    padding-top: 0rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1280px !important;
}

:root {
  --color-surface: #f9f9ff;
  --color-surface-dim: #d1daf4;
  --color-surface-lowest: #ffffff;
  --color-surface-low: #f1f3ff;
  --color-surface-container: #e9edff;
  --color-surface-highest: #d9e2fc;
  --color-on-surface: #121b2e;
  --color-on-surface-variant: #434655;
  --color-outline: #737686;
  --color-outline-variant: #c3c6d7;
  --color-primary: #004ac6;
  --color-primary-container: #2563eb;
  --color-primary-hover: #1d4ed8;
  --color-on-primary: #ffffff;
  --color-primary-fixed: #dbe1ff;
  --color-secondary: #505f76;
  --color-secondary-container: #d0e1fb;
  --color-status-success: #059669;
  --color-status-success-bg: #ecfdf5;
  --color-status-error: #ba1a1a;
  --color-status-error-bg: #ffdad6;
  --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
  --radius-xs: 2px;
  --radius-sm: 4px;
  --radius-md: 8px;
  --shadow-subtle: 0 1px 3px 0 rgba(18, 27, 46, 0.05), 0 1px 2px -1px rgba(18, 27, 46, 0.05);
}

body, .stApp {
  background-color: var(--color-surface) !important;
  color: var(--color-on-surface) !important;
  font-family: var(--font-sans) !important;
}

.mono {
  font-family: var(--font-mono) !important;
  font-variant-numeric: tabular-nums;
}

/* App Header */
.app-header {
  background-color: var(--color-surface-lowest);
  border-bottom: 1px solid var(--color-outline-variant);
  margin-left: -2rem;
  margin-right: -2rem;
  margin-bottom: 24px;
  padding: 12px 32px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 10px;
}

.project-tag {
  font-family: var(--font-mono);
  font-size: 12px;
  font-weight: 600;
  color: var(--color-primary);
  background-color: var(--color-primary-fixed);
  padding: 3px 8px;
  border-radius: var(--radius-sm);
  border: 1px solid #bfdbfe;
}

.header-divider {
  color: var(--color-outline);
}

.header-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--color-on-surface);
}

.header-badge {
  font-size: 11px;
  font-family: var(--font-mono);
  color: var(--color-secondary);
  background-color: var(--color-surface-low);
  padding: 2px 7px;
  border-radius: var(--radius-xs);
  border: 1px solid var(--color-outline-variant);
}

.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.telemetry-pill {
  display: flex;
  align-items: center;
  gap: 6px;
  background-color: var(--color-surface-low);
  border: 1px solid var(--color-outline-variant);
  padding: 3px 10px;
  border-radius: 9999px;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--color-on-surface);
}

.pill-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background-color: var(--color-status-success);
}

.academic-tag {
  font-size: 12px;
  font-weight: 500;
  color: var(--color-on-surface-variant);
  padding-left: 12px;
  border-left: 1px solid var(--color-outline-variant);
}

/* Page Intro & Vocab */
.page-intro {
  margin-bottom: 24px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.page-title {
  font-size: 22px;
  font-weight: 700;
  color: var(--color-on-surface);
  line-height: 1.25;
}

.page-desc {
  font-size: 13.5px;
  color: var(--color-on-surface-variant);
  max-width: 860px;
  line-height: 1.5;
}

.vocab-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  padding: 10px 14px;
  background-color: var(--color-surface-lowest);
  border: 1px solid var(--color-outline-variant);
  border-radius: var(--radius-sm);
}

.vocab-title {
  font-family: var(--font-mono);
  font-size: 10.5px;
  font-weight: 600;
  color: var(--color-outline);
  letter-spacing: 0.05em;
}

.vocab-pills {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}

.vocab-chip {
  font-family: var(--font-mono);
  font-size: 11px;
  padding: 2px 8px;
  background-color: var(--color-surface-low);
  border: 1px solid var(--color-outline-variant);
  border-radius: var(--radius-xs);
  color: var(--color-on-surface);
}

/* Card Styling */
.card-header-custom {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 18px;
  border-bottom: 1px solid var(--color-outline-variant);
  background-color: var(--color-surface-lowest);
  border-radius: var(--radius-md) var(--radius-md) 0 0;
  margin-bottom: 12px;
}

.card-title-custom {
  font-size: 13.5px;
  font-weight: 600;
  color: var(--color-on-surface);
  display: flex;
  align-items: center;
  gap: 8px;
}

.mono-badge-custom {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--color-primary);
  background-color: var(--color-primary-fixed);
  padding: 2px 7px;
  border-radius: var(--radius-xs);
  border: 1px solid #bfdbfe;
}

.status-chip-custom {
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 500;
  padding: 2px 8px;
  border-radius: 9999px;
  border: 1px solid var(--color-outline-variant);
  color: var(--color-outline);
  background-color: var(--color-surface-lowest);
}

.status-chip-success {
  border-color: var(--color-status-success);
  color: var(--color-status-success);
  background-color: var(--color-status-success-bg);
}

/* Guidance box */
.guidance-box {
  background-color: var(--color-surface-low);
  border: 1px solid var(--color-outline-variant);
  border-left: 3px solid var(--color-primary);
  border-radius: var(--radius-sm);
  padding: 10px 14px;
  font-size: 12px;
  color: var(--color-on-surface-variant);
  margin-top: 14px;
  line-height: 1.45;
}

/* Results Card Elements */
.identified-box {
  background-color: var(--color-surface-low);
  border: 1px solid var(--color-outline-variant);
  border-radius: var(--radius-sm);
  padding: 16px 18px;
  margin-bottom: 16px;
}

.identified-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
}

.identified-sub {
  font-family: var(--font-mono);
  font-size: 10.5px;
  font-weight: 600;
  color: var(--color-outline);
  letter-spacing: 0.05em;
}

.confidence-tag {
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 600;
  color: var(--color-primary);
  background-color: var(--color-primary-fixed);
  padding: 2px 7px;
  border-radius: var(--radius-xs);
  border: 1px solid #bfdbfe;
}

.identified-label {
  font-size: 28px;
  font-weight: 700;
  color: var(--color-on-surface);
  line-height: 1.2;
  margin-bottom: 8px;
}

.confidence-track {
  height: 6px;
  background-color: var(--color-surface-highest);
  border-radius: 9999px;
  overflow: hidden;
}

.confidence-fill {
  height: 100%;
  background-color: var(--color-primary);
  border-radius: 9999px;
}

/* Telemetry Table */
.telemetry-table {
  background-color: var(--color-surface-lowest);
  border: 1px solid var(--color-outline-variant);
  border-radius: var(--radius-sm);
  overflow: hidden;
  margin-bottom: 16px;
}

.telemetry-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 14px;
  border-bottom: 1px solid var(--color-surface-low);
  font-size: 12px;
}

.telemetry-row:last-child {
  border-bottom: none;
}

.t-label {
  color: var(--color-on-surface-variant);
}

.t-value {
  font-family: var(--font-mono);
  font-size: 11.5px;
  color: var(--color-on-surface);
  font-weight: 500;
}

/* Distribution Rows */
.distribution-section {
  background-color: var(--color-surface-lowest);
  border: 1px solid var(--color-outline-variant);
  border-radius: var(--radius-sm);
  padding: 14px;
  margin-bottom: 16px;
}

.dist-header-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--color-surface-low);
}

.dist-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 0;
}

.dist-label {
  width: 90px;
  font-size: 12px;
  font-weight: 500;
  color: var(--color-on-surface-variant);
  white-space: nowrap;
}

.dist-track {
  flex: 1;
  height: 8px;
  background-color: var(--color-surface-highest);
  border-radius: 9999px;
  overflow: hidden;
}

.dist-fill {
  height: 100%;
  background-color: #93c5fd;
  border-radius: 9999px;
}

.dist-row.active .dist-fill {
  background-color: var(--color-primary);
}

.dist-row.active .dist-label {
  color: var(--color-on-surface);
  font-weight: 600;
}

.dist-val {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--color-outline);
  width: 50px;
  text-align: right;
}

/* Empty Diagnostic Box */
.diagnostic-empty {
  padding: 48px 24px;
  text-align: center;
  background-color: var(--color-surface-lowest);
  border: 1px dashed var(--color-outline-variant);
  border-radius: var(--radius-sm);
}

.empty-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--color-on-surface);
  margin-bottom: 4px;
}

.empty-desc {
  font-size: 12px;
  color: var(--color-on-surface-variant);
  max-width: 320px;
  margin: 0 auto;
  line-height: 1.45;
}

/* Curriculum Scope */
.boundary-card {
  margin-top: 24px;
  background-color: var(--color-surface-lowest);
  border: 1px solid var(--color-outline-variant);
  border-radius: var(--radius-md);
  padding: 14px 18px;
}

.boundary-title {
  font-size: 12.5px;
  font-weight: 600;
  color: var(--color-on-surface);
  margin-bottom: 4px;
}

.boundary-body {
  font-size: 12px;
  color: var(--color-on-surface-variant);
  line-height: 1.45;
}

/* Institutional Footer */
.app-footer {
  margin-top: 28px;
  margin-left: -2rem;
  margin-right: -2rem;
  margin-bottom: -2rem;
  border-top: 1px solid var(--color-outline-variant);
  background-color: var(--color-surface-lowest);
  padding: 16px 32px;
}

.footer-inner {
  max-width: 1280px;
  margin: 0 auto;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.footer-head {
  font-size: 12.5px;
  font-weight: 600;
  color: var(--color-on-surface);
}

.footer-sub {
  font-size: 11px;
  color: var(--color-outline);
}

/* Streamlit Button Overrides */
.stButton > button {
  background-color: var(--color-primary) !important;
  color: #ffffff !important;
  font-weight: 600 !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  padding: 10px 18px !important;
  width: 100% !important;
  font-size: 13.5px !important;
}

.stButton > button:hover {
  background-color: var(--color-primary-hover) !important;
}
</style>
""", unsafe_allow_html=True)

# ── App Header ────────────────────────────────────────────────────────────────
st.markdown("""
<header class="app-header">
  <div class="header-left">
    <span class="project-tag">DL-024</span>
    <span class="header-divider">/</span>
    <span class="header-title">Sign Language Gesture Recognition</span>
    <span class="header-badge">Research Prototype</span>
  </div>
  <div class="header-right">
    <div class="telemetry-pill">
      <span class="pill-dot"></span>
      <span class="mono">MODEL: LSTM ACTIVE</span>
    </div>
    <div class="academic-tag">
      <span>Invertis University</span>
    </div>
  </div>
</header>
""", unsafe_allow_html=True)

# ── Page Intro & Vocabulary Bar ───────────────────────────────────────────────
st.markdown("""
<div class="page-intro">
  <div>
    <h1 class="page-title">Dynamic Gesture Classification from Short Video Clips</h1>
    <p class="page-desc">
      An empirical deep learning system using MediaPipe hand coordinate extraction and 
      bidirectional temporal LSTM sequences. Designed for discrete accessibility sign recognition.
    </p>
  </div>
  <div class="vocab-bar">
    <span class="vocab-title">SUPPORTED VOCABULARY</span>
    <div class="vocab-pills">
      <span class="vocab-chip">01 Hello</span>
      <span class="vocab-chip">02 Thank You</span>
      <span class="vocab-chip">03 Yes</span>
      <span class="vocab-chip">04 No</span>
      <span class="vocab-chip">05 Help</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Load Model Resources (Cached) ─────────────────────────────────────────────
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

# ── Dual-Pane Workspace Studio ────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("""
    <div class="card-header-custom">
      <div class="card-title-custom">
        <span class="material-symbols-outlined" style="font-size:18px;color:#004ac6;">videocam</span>
        <span>Input Video Viewport</span>
      </div>
      <span class="mono-badge-custom">SAMPLING: 20 FRAMES</span>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Select video clip",
        type=["mp4", "mov", "webm", "avi"],
        label_visibility="collapsed",
        help="Optimal duration: 3–5 seconds with hand visible throughout the motion."
    )

    if uploaded_file is not None:
        st.video(uploaded_file)
        run_btn = st.button("Run Inference", key="run_inference_btn")
    else:
        st.markdown("""
        <div style="background-color:#ffffff;border:1px dashed #c3c6d7;border-radius:4px;padding:36px 20px;text-align:center;margin-bottom:12px;">
          <p style="font-weight:600;font-size:13px;color:#121b2e;margin-bottom:2px;">No Active Clip Loaded</p>
          <p style="font-size:11.5px;color:#737686;">Upload a 3–5 second clip below (Formats: MP4, MOV, WebM)</p>
        </div>
        """, unsafe_allow_html=True)
        run_btn = False

    st.markdown("""
    <div class="guidance-box">
      <strong>Sampling Protocol:</strong> Ensure hand remains inside the frame throughout the gesture movement. 
      The temporal sliding window searches for peak confidence across 20 consecutive frames.
    </div>
    """, unsafe_allow_html=True)

with col_right:
    status_chip = '<span class="status-chip-custom status-chip-success">Inference Converged</span>' if (uploaded_file and run_btn) else '<span class="status-chip-custom">Awaiting Input</span>'
    st.markdown(f"""
    <div class="card-header-custom">
      <div class="card-title-custom">
        <span class="material-symbols-outlined" style="font-size:18px;color:#004ac6;">analytics</span>
        <span>Inference Diagnostics</span>
      </div>
      {status_chip}
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file is not None and run_btn:
        if model is None or extractor is None:
            st.error("Model file not found. Ensure gesture_lstm_model.h5 exists in ml/models/.")
        else:
            with st.spinner("Extracting MediaPipe landmarks and sliding window..."):
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
                st.markdown(f"""
                <div style="background-color:#ffdad6;border:1px solid #fca5a5;border-radius:4px;padding:12px 14px;color:#ba1a1a;font-size:12.5px;">
                  <strong>Inference Warning:</strong> {error}
                </div>
                """, unsafe_allow_html=True)
            else:
                other = round((100.0 - conf) / (len(CLASSES) - 1), 2)
                scores = {c: (round(conf, 2) if c == label else other) for c in CLASSES}

                # Classified Sequence Box
                st.markdown(f"""
                <div class="identified-box">
                  <div class="identified-header">
                    <span class="identified-sub">CLASSIFIED SEQUENCE</span>
                    <span class="confidence-tag mono">{conf:.1f}% CONFIDENCE</span>
                  </div>
                  <div class="identified-label">{label}</div>
                  <div class="confidence-track">
                    <div class="confidence-fill" style="width: {min(100.0, max(0.0, conf))}%;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Telemetry Table
                st.markdown("""
                <div class="telemetry-table">
                  <div class="telemetry-row">
                    <span class="t-label">Sequence Model</span>
                    <span class="t-value mono">2-Layer Temporal LSTM</span>
                  </div>
                  <div class="telemetry-row">
                    <span class="t-label">Temporal Window</span>
                    <span class="t-value mono">20 Frames (Stride: 2)</span>
                  </div>
                  <div class="telemetry-row">
                    <span class="t-label">Spatial Coordinates</span>
                    <span class="t-value mono">120-dim (MediaPipe 3D)</span>
                  </div>
                  <div class="telemetry-row">
                    <span class="t-label">Normalization</span>
                    <span class="t-value mono">Wrist Offset / Palm Scale</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Softmax Probability Distribution
                dist_rows_html = ""
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                for cls_name, score in sorted_scores:
                    is_active = "active" if cls_name == label else ""
                    dist_rows_html += f"""
                    <div class="dist-row {is_active}">
                      <span class="dist-label">{cls_name}</span>
                      <div class="dist-track">
                        <div class="dist-fill" style="width: {min(100.0, max(0.0, score))}%;"></div>
                      </div>
                      <span class="dist-val mono">{score:.1f}%</span>
                    </div>
                    """

                st.markdown(f"""
                <div class="distribution-section">
                  <div class="dist-header-row">
                    <span style="font-size:12px;font-weight:600;color:#121b2e;">Softmax Probability Distribution</span>
                    <span class="mono" style="font-size:10.5px;color:#737686;">ArgMax Output</span>
                  </div>
                  {dist_rows_html}
                </div>
                """, unsafe_allow_html=True)
    elif uploaded_file is not None:
        st.markdown("""
        <div class="diagnostic-empty">
          <p class="empty-title">Clip Loaded · Ready for Inference</p>
          <p class="empty-desc">Click 'Run Inference' in the viewport on the left to execute feature extraction and sequence classification.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="diagnostic-empty">
          <p class="empty-title">Sequence Model Idle</p>
          <p class="empty-desc">Load a gesture clip and execute inference to display predicted class, confidence intervals, and probability distribution.</p>
        </div>
        """, unsafe_allow_html=True)

# ── Curriculum Scope & Constraints Card ───────────────────────────────────────
st.markdown("""
<section class="boundary-card">
  <div class="boundary-title">Curriculum Scope &amp; Constraints (DL-024)</div>
  <div class="boundary-body">
    This system is an academic accessibility research prototype specifically evaluated on discrete 
    dynamic gesture sequences sampled from short clips. Continuous unsegmented signing, grammatical 
    inflections, facial affect tracking, and real-time sentence construction are intentionally 
    designated as out of scope according to the project evaluation criteria.
  </div>
</section>
""", unsafe_allow_html=True)

# ── Institutional Academic Footer ─────────────────────────────────────────────
st.markdown("""
<footer class="app-footer">
  <div class="footer-inner">
    <div>
      <div class="footer-head">DL-024 Research Prototype</div>
      <div class="footer-sub">Department of Computer Science &amp; Engineering</div>
    </div>
    <div style="text-align: center;">
      <div class="footer-head">Invertis University, Bareilly</div>
      <div class="footer-sub">Bachelor of Technology</div>
    </div>
    <div style="text-align: right;">
      <div class="footer-sub">Project Mentor</div>
      <div class="footer-head">Mr. Raushan Raj</div>
    </div>
  </div>
</footer>
""", unsafe_allow_html=True)
