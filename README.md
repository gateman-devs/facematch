# Gateman Liveness Check API

A production-ready face recognition and liveness detection API optimized for ECS deployment.

## Features

- **Face Recognition**: Compare two face images for similarity
- **Liveness Detection**: Anti-spoofing detection for static images
- **Video Challenges**: Head movement validation with video uploads
- **Face Comparison**: URL and base64 image support
- **Session Management**: Redis-based session handling
- **API-First**: Pure REST API without frontend dependencies

## Quick Start

### 1. Build Docker Image

```bash
docker build -t facematch-api .
```

### 2. Test Locally

```bash
docker run -p 8000:8000 facematch-api
```

### 3. Deploy to ECS

The container is optimized for ECS deployment with:
- Models baked into the image
- Health checks at `/health`
- Environment variable configuration
- Non-root user for security

### Environment Variables

```bash
# Required
PORT=8000
PYTHONPATH=/app

# Optional (with defaults)
SIMILARITY_THRESHOLD=0.6
MAX_IMAGE_SIZE=10485760
REQUEST_TIMEOUT=30
MAX_IMAGE_DIMENSION=1024
```

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation
- `POST /match` - Face matching
- `POST /image-liveness-check` - Static image liveness
- `POST /face-comparison` - Face comparison
- `POST /create-simple-challenge` - Create liveness challenge
- `POST /submit-simple-challenge` - Submit video challenge
- `GET /liveness-result/{session_id}` - Get challenge results

## Architecture

- **FastAPI**: Modern Python web framework
- **OpenCV**: Computer vision and image processing
- **MediaPipe**: Face detection and tracking
- **Redis**: Session management (optional)
- **Docker**: Containerized deployment

## Models

The following models are automatically downloaded during build:
- MTCNN (face detection)
- ArcFace (face recognition)
- CRMNET (liveness detection)

## Health Check

The container includes a health check endpoint at `/health` that verifies:
- All ML models are loaded
- API is responding
- System resources are available

## Security

- Non-root user execution
- Input validation and sanitization
- File size limits
- CORS configuration
- Error handling without information leakage 