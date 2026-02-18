FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python and install pip
RUN ln -sf /usr/bin/python3.11 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (--ignore-installed to handle system-pkg conflicts like blinker)
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Copy application code
COPY . .

# Create output directories
RUN mkdir -p outputs team_reports

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GROQ_API_KEY=""
ENV HF_TOKEN=""

# Expose port for API (optional)
EXPOSE 8000

# Default command (can be overridden)
CMD ["python", "main.py", "--help"]
