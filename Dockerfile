# =============================================
# Cancer Fusion AI - Dockerfile
# Cloud Run deployment target — no volume mounts available, so the model
# checkpoint and metadata CSV must be baked into the image (see COPY steps
# below). Cloud Run injects $PORT (default 8080) and expects the container
# to bind to it; a hardcoded port makes the revision fail to become ready.
# =============================================

# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# curl is required for the HEALTHCHECK below — the previous Dockerfile ran
# `curl` in HEALTHCHECK without ever installing it, so it always failed.
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install torch/torchvision from the CPU-only wheel index first — this alone
# is what shrinks the image (the default PyPI wheels pull in CUDA/cuDNN
# libraries that are dead weight on a CPU-only Cloud Run instance).
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install the rest. torch/torchvision are already
# satisfied by the CPU wheels above (unpinned in requirements.txt), so this
# won't re-resolve/overwrite them with the default CUDA-enabled build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY app.py .

# Bake in the model checkpoint and the metadata CSV. app.py loads both at
# module import time (metadata CSV to fit the age scaler / one-hot columns,
# checkpoint into the model) — with no volume mount on Cloud Run, anything
# not COPYed here doesn't exist and the container crashes on startup before
# ever serving /health.
COPY models/best_model.pt ./models/best_model.pt
COPY data/HAM10000_metadata.csv ./data/HAM10000_metadata.csv

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Cloud Run ignores EXPOSE for routing (it reads $PORT), but it's still
# useful documentation and keeps `docker run -p` mapping obvious locally.
EXPOSE 8080

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Health check — respects $PORT for local `docker run -e PORT=...` testing.
# Cloud Run does not use Docker HEALTHCHECK for its own liveness probing
# (that's configured on the Cloud Run service itself against /health), but
# this keeps `docker ps` / local verification meaningful.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

# Single shared vCPU, one worker: the model + checkpoint load once at
# import (app.py:79-82), so each extra worker process duplicates that full
# load in its own memory rather than sharing it. Must be shell form (not the
# exec JSON array form) so ${PORT:-8080} actually expands — Cloud Run sets
# $PORT at container start, not build time.
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
