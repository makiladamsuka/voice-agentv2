# Multi-platform support for x86 and ARM (Raspberry Pi)
FROM python:3.10-slim-bullseye

# Install system dependencies required for face-recognition, mediapipe, and OpenCV
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p known_faces assets/events assets/maps assets/fallback

# Expose port for image server (if needed)
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command to run the greeting agent
CMD ["python", "greeting_agent.py", "dev"]
