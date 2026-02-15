FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.13 \
    python3.13-pip \
    python3.13-dev \
    python3.13-venv \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.13 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create output directories
RUN mkdir -p outputs team_reports

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GROQ_API_KEY=""

# Expose port for API (optional)
EXPOSE 8000

# Default command (can be overridden)
CMD ["python", "main.py", "--help"]
