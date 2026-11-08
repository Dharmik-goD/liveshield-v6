FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    OBJECT_DEVICE=auto \
    ENABLE_SCENE_CAPTIONER=0

WORKDIR /app

# System libraries required by OpenCV and numerical inference packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install the CUDA-enabled PyTorch build separately.
RUN python -m pip install --upgrade pip && \
    python -m pip install \
    torch==2.14.1 \
    torchvision==0.29.1 \
    --index-url https://download.pytorch.org/whl/cu130

# Install application runtime dependencies.
COPY requirements-runtime.txt .
RUN python -m pip install -r requirements-runtime.txt

# Copy application code and required model assets.
COPY app/ ./app/
COPY web/v6/ ./web/v6/
COPY models/ ./models/
COPY yolov8n.pt ./yolov8n.pt

EXPOSE 8000

CMD ["uvicorn", "app.v6.main:app", "--host", "0.0.0.0", "--port", "8000"]
