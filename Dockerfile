# Use NVIDIA CUDA base image compatible with PyTorch
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    COQUI_TOS_AGREED=1 \
    TTS_HOME=/app/models \
    WORKDIR=/app

# Set working directory
WORKDIR $WORKDIR

# Install system dependencies including Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    rubberband-cli \
    nodejs \
    && rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3.10 /usr/bin/python

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install Python dependencies
# Pyworld 0.2.10 (TTS dependency) fails to compile against numpy 2.x, so we build it explicitly first without build isolation
# It also possesses a setup.py bug making it crash on modern pip/setuptools. We use ubuntu's native pip 22 and setuptools 59.
RUN pip install "numpy<2.0.0" Cython wheel && \
    pip install --no-build-isolation pyworld==0.2.10 && \
    python -m pip install --upgrade pip && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories for output, temp files, and models
RUN mkdir -p output temp_files models

# Expose models directory as volume for persistence
VOLUME ["/app/models"]

# Define the entrypoint
ENTRYPOINT ["python", "main.py"]

# Default command arguments (can be overridden)
CMD ["--help"]
