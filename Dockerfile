FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip python3.11-dev \
    ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download MERaLiON-SER-v1 model at build time
# Pass HF_TOKEN as build arg: docker build --build-arg HF_TOKEN=hf_xxx .
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
COPY download_models.py .
COPY src/ src/
RUN if [ -n "$HF_TOKEN" ]; then python download_models.py; fi

# Copy application code
COPY slim_pipeline.py slim_main.py api.py ./
COPY .env.example .env

# Create output dirs
RUN mkdir -p /app/outputs/visualizations /app/cache

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app /models
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default: run API server
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
