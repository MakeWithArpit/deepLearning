FROM python:3.12-slim

WORKDIR /app

# System deps needed by OpenCV + MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ML source (feature extractor + frame sampler + mediapipe task file)
COPY ml/src/feature_extractor.py  ./ml/src/
COPY ml/src/frame_sampler.py      ./ml/src/
COPY ml/src/model.py              ./ml/src/
COPY ml/src/hand_landmarker.task  ./ml/src/

# Copy trained model
COPY ml/models/gesture_lstm_model.h5 ./ml/models/

# Copy backend app
COPY backend/main.py .

ENV PYTHONPATH="/app:/app/ml/src"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

