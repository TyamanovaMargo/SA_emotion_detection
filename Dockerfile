FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

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
