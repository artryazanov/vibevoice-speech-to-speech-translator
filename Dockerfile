# Use PyTorch base image with CUDA support
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    WORKDIR=/app

# Set working directory
WORKDIR $WORKDIR

# Install system dependencies
# ffmpeg: required for audio processing (pydub)
# libsndfile1: required for soundfile
# git: required for some pip installations or submodule checks
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Upgrade pip first
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories for output and temp files
RUN mkdir -p output temp_files

# Define the entrypoint
ENTRYPOINT ["python", "main.py"]

# Default command arguments (can be overridden)
CMD ["--help"]
