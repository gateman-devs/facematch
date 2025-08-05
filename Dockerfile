# Face Recognition API Dockerfile
# Multi-stage build for optimized production image

FROM python:3.10-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies and dlib pre-requisites
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    curl \
    wget \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install dlib first (heaviest dependency)
RUN pip install --no-cache-dir dlib

# Copy requirements and install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    SIMILARITY_THRESHOLD=0.6 \
    MAX_IMAGE_SIZE=10485760 \
    REQUEST_TIMEOUT=30 \
    MAX_IMAGE_DIMENSION=1024

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    wget \
    libjpeg62-turbo \
    libpng16-16 \
    libtiff5 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r facematch && useradd -r -g facematch facematch

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/facematch/.local

# Make sure scripts in .local are usable
ENV PATH=/home/facematch/.local/bin:$PATH

# Copy application code
COPY infrastructure/ infrastructure/
COPY requirements.txt .
COPY download_models.sh .

# Create directories
RUN mkdir -p models faces logs static && chown -R facematch:facematch /app

# Make scripts executable
RUN chmod +x download_models.sh

# Switch to non-root user
USER facematch

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Start the server directly
CMD ["python", "-m", "infrastructure.facematch.server", "--host", "0.0.0.0", "--port", "8000"]
