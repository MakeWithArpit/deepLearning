/**
 * DL-024: Sign Language Gesture Recognition Studio Controller
 * Matches Stitch Academic Instrumentation Design Reference
 */

// Use 127.0.0.1 on local machines to avoid Windows IPv6 localhost timeout with Docker
const API_BASE = (!window.location.hostname || window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")
  ? "http://127.0.0.1:8005"
  : `http://${window.location.hostname}:8005`;

// DOM References
const dropzoneArea = document.getElementById("dropzone-area");
const fileInput = document.getElementById("file-input");
const browseBtn = document.getElementById("browse-btn");

const viewportBox = document.getElementById("viewport-box");
const viewportPlaceholder = document.getElementById("viewport-placeholder");
const mainVideo = document.getElementById("main-video");
const viewportHud = document.getElementById("viewport-hud");
const hudFilename = document.getElementById("hud-filename");

const progressContainer = document.getElementById("progress-container");
const progressBarFill = document.getElementById("progress-bar-fill");
const progressPhase = document.getElementById("progress-phase");
const progressPct = document.getElementById("progress-pct");

const replaceFileBtn = document.getElementById("replace-file-btn");
const analyzeBtn = document.getElementById("analyze-btn");

const diagnosticChip = document.getElementById("diagnostic-chip");
const diagnosticEmpty = document.getElementById("diagnostic-empty");
const errorCard = document.getElementById("error-card");
const errorTitle = document.getElementById("error-title");
const errorMessage = document.getElementById("error-message");

const resultsWrapper = document.getElementById("results-wrapper");
const resultGestureTitle = document.getElementById("result-gesture-title");
const resultConfTag = document.getElementById("result-conf-tag");
const resultConfFill = document.getElementById("result-conf-fill");
const distributionList = document.getElementById("distribution-list");

const systemStatus = document.getElementById("system-status");
const systemDot = document.getElementById("system-dot");

let selectedFile = null;

// Initial Health Check
async function pingBackend() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      const data = await res.json();
      if (data.model_loaded) {
        systemStatus.textContent = "MODEL: LSTM READY";
        systemDot.style.backgroundColor = "var(--color-status-success)";
      }
    }
  } catch (err) {
    systemStatus.textContent = "BACKEND DISCONNECTED";
    systemDot.style.backgroundColor = "var(--color-status-error)";
  }
}
pingBackend();

// File Selector Handlers
browseBtn.addEventListener("click", () => fileInput.click());
dropzoneArea.addEventListener("click", (e) => {
  if (e.target !== browseBtn) {
    fileInput.click();
  }
});

fileInput.addEventListener("change", (e) => {
  if (e.target.files && e.target.files[0]) {
    loadVideoFile(e.target.files[0]);
  }
});

// Drag & Drop
dropzoneArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzoneArea.classList.add("dragover");
});

dropzoneArea.addEventListener("dragleave", () => {
  dropzoneArea.classList.remove("dragover");
});

dropzoneArea.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzoneArea.classList.remove("dragover");
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    loadVideoFile(e.dataTransfer.files[0]);
  }
});

replaceFileBtn.addEventListener("click", () => {
  resetStudio();
});

analyzeBtn.addEventListener("click", () => {
  if (selectedFile) {
    executeInference(selectedFile);
  }
});

function loadVideoFile(file) {
  if (!file) return;

  if (file.size > 50 * 1024 * 1024) {
    displayError("File Limit Exceeded", "Uploaded clip exceeds the 50 MB threshold.");
    return;
  }

  selectedFile = file;

  // Update UI Viewport for Preview
  viewportPlaceholder.style.display = "none";
  mainVideo.style.display = "block";
  viewportHud.style.display = "flex";

  hudFilename.textContent = file.name;
  mainVideo.src = URL.createObjectURL(file);
  mainVideo.play().catch(() => {});

  // Enable Controls (User has full control to run when ready)
  replaceFileBtn.disabled = false;
  analyzeBtn.disabled = false;

  diagnosticEmpty.style.display = "flex";
  resultsWrapper.style.display = "none";
  errorCard.style.display = "none";
  progressContainer.style.display = "none";

  diagnosticChip.textContent = "Clip Ready · Click Run";
  diagnosticChip.style.color = "var(--color-primary)";
  diagnosticChip.style.borderColor = "var(--color-primary)";
}

function resetStudio() {
  selectedFile = null;
  fileInput.value = "";

  mainVideo.pause();
  mainVideo.src = "";
  mainVideo.style.display = "none";
  viewportHud.style.display = "none";
  viewportPlaceholder.style.display = "flex";

  replaceFileBtn.disabled = true;
  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = '<span class="material-symbols-outlined btn-icon">play_arrow</span> Run Inference';

  progressContainer.style.display = "none";
  resultsWrapper.style.display = "none";
  errorCard.style.display = "none";
  diagnosticEmpty.style.display = "flex";

  diagnosticChip.textContent = "Awaiting Input";
  diagnosticChip.style.color = "var(--color-outline)";
  diagnosticChip.style.borderColor = "var(--color-outline-variant)";
}

async function executeInference(file) {
  // UI Loading State
  diagnosticEmpty.style.display = "none";
  errorCard.style.display = "none";
  resultsWrapper.style.display = "none";
  progressContainer.style.display = "flex";

  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = '<span class="material-symbols-outlined btn-icon">hourglass_empty</span> Processing...';
  diagnosticChip.textContent = "Processing Pipeline";
  diagnosticChip.style.color = "var(--color-primary)";

  // Stage 1
  progressBarFill.style.width = "35%";
  progressPhase.textContent = "OpenCV: Extracting consecutive temporal frames...";
  progressPct.textContent = "35%";

  const formData = new FormData();
  formData.append("file", file);

  try {
    // Stage 2
    setTimeout(() => {
      progressBarFill.style.width = "75%";
      progressPhase.textContent = "MediaPipe & LSTM: Evaluating temporal sliding window...";
      progressPct.textContent = "75%";
    }, 400);

    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Inference pipeline failed to process clip.");
    }

    // Complete
    progressBarFill.style.width = "100%";
    progressPhase.textContent = "Pipeline complete.";
    progressPct.textContent = "100%";

    setTimeout(() => {
      progressContainer.style.display = "none";
      renderSuccess(data);
    }, 300);

  } catch (err) {
    progressContainer.style.display = "none";
    displayError("Inference Exception", err.message || "Failed to communicate with prediction server.");
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = '<span class="material-symbols-outlined btn-icon">replay</span> Re-run Inference';
  }
}

function renderSuccess(data) {
  diagnosticEmpty.style.display = "none";
  errorCard.style.display = "none";
  resultsWrapper.style.display = "flex";

  diagnosticChip.textContent = "Inference Converged";
  diagnosticChip.style.color = "var(--color-status-success)";
  diagnosticChip.style.borderColor = "var(--color-status-success)";

  resultGestureTitle.textContent = data.gesture;
  resultConfTag.textContent = `${data.confidence.toFixed(1)}% CONFIDENCE`;
  resultConfFill.style.width = `${Math.min(100, Math.max(0, data.confidence))}%`;

  // Softmax Distribution rendering
  distributionList.innerHTML = "";
  if (data.all_scores) {
    const sorted = Object.entries(data.all_scores).sort((a, b) => b[1] - a[1]);
    sorted.forEach(([clsName, score]) => {
      const isWinner = clsName === data.gesture;
      const row = document.createElement("div");
      row.className = `dist-row ${isWinner ? 'active' : ''}`;
      row.innerHTML = `
        <span class="dist-label">${clsName}</span>
        <div class="dist-track">
          <div class="dist-fill" style="width: ${Math.min(100, Math.max(0, score))}%;"></div>
        </div>
        <span class="dist-val mono">${score.toFixed(1)}%</span>
      `;
      distributionList.appendChild(row);
    });
  }
}

function displayError(title, msg) {
  progressContainer.style.display = "none";
  resultsWrapper.style.display = "none";
  diagnosticEmpty.style.display = "none";
  errorCard.style.display = "flex";

  errorTitle.textContent = title;
  errorMessage.textContent = msg;

  diagnosticChip.textContent = "Pipeline Error";
  diagnosticChip.style.color = "var(--color-status-error)";
  diagnosticChip.style.borderColor = "var(--color-status-error)";
}
