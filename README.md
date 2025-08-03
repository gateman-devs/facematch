# Face Recognition API

A production-grade face recognition system built with TensorFlow, MTCNN, ArcFace, and CRMNET. This system provides comprehensive face analysis including detection, liveness verification, and similarity matching.

## 🚀 Features

- **Face Detection**: MTCNN-based single face detection with quality validation
- **Liveness Detection**: CRMNET-based anti-spoofing to detect real vs fake faces
- **Face Recognition**: ArcFace-based face embedding extraction and similarity matching
- **Async Processing**: Concurrent image loading and processing for optimal performance
- **Production Ready**: Docker containerization, health checks, and comprehensive error handling
- **Flexible Input**: Support for both URL and base64 image inputs
- **Quality Control**: Automatic image quality validation and preprocessing

## 📋 Requirements

- Python 3.10+
- Docker (optional, for containerized deployment)
- CUDA-compatible GPU (optional, for accelerated inference)

## 🛠️ Installation

### Option 1: Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd facematch
   ```

2. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

3. **Access the API**
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Option 2: Local Installation

1. **Clone and setup**
   ```bash
   git clone <repository-url>
   cd facematch
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download models**
   ```bash
   ./download_models.sh
   ```

4. **Start the server**
   ```bash
   python main.py
   ```

## 📁 Project Structure

```
facematch/
├── infrastructure/facematch/    # Core application modules
│   ├── face_detection.py       # MTCNN face detection
│   ├── liveness.py             # CRMNET liveness detection
│   ├── facematch.py            # ArcFace face recognition
│   ├── image_utils.py          # Image loading utilities
│   ├── server.py               # FastAPI server
│   └── __init__.py
├── models/                     # Model files (downloaded by script)
│   ├── mtcnn.pb
│   ├── arcface_resnet50.onnx
│   └── crmnet.onnx
├── download_models.sh          # Model download script
├── startup.sh                  # Container startup script
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose configuration
├── requirements.txt            # Python dependencies
├── main.py                     # Application entry point
└── README.md                   # This file
```

## 🔧 Configuration

Environment variables for customization:

| Variable | Default | Description |
|----------|---------|-------------|
| `SIMILARITY_THRESHOLD` | `0.6` | Face similarity threshold for matching |
| `MAX_IMAGE_SIZE` | `10485760` | Maximum image size in bytes (10MB) |
| `REQUEST_TIMEOUT` | `30` | Request timeout in seconds |
| `MAX_IMAGE_DIMENSION` | `1024` | Maximum image width/height |
| `PORT` | `8000` | Server port |

## 📖 API Documentation

### Endpoints

#### `GET /health`
Health check endpoint returning system status and model availability.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1673123456.789,
  "models": {
    "mtcnn": true,
    "crmnet": true,
    "arcface": true
  },
  "version": "1.0.0"
}
```

#### `POST /match`
Compare two face images for similarity with comprehensive analysis.

**Request Body:**
```json
{
  "image1": "https://example.com/face1.jpg",
  "image2": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA...",
  "threshold": 0.6
}
```

**Parameters:**
- `image1`: First image (URL or base64 data)
- `image2`: Second image (URL or base64 data)  
- `threshold`: Similarity threshold (0.0-1.0, optional, default: 0.6)

**Response:**
```json
{
  "success": true,
  "match": true,
  "similarity_score": 0.87,
  "confidence": 0.92,
  "threshold": 0.6,
  "liveness_results": {
    "image1": {
      "success": true,
      "is_live": true,
      "liveness_score": 0.95,
      "confidence": 0.93
    },
    "image2": {
      "success": true,
      "is_live": true,
      "liveness_score": 0.89,
      "confidence": 0.91
    }
  },
  "face_detection_results": {
    "image1": {
      "face_count": 1,
      "confidence": 0.99,
      "detection_time": 0.156
    },
    "image2": {
      "face_count": 1,
      "confidence": 0.97,
      "detection_time": 0.142
    }
  },
  "recognition_results": {
    "distance_metrics": {
      "cosine_similarity": 0.87,
      "cosine_distance": 0.13,
      "euclidean_distance": 0.52,
      "manhattan_distance": 2.34
    },
    "embedding_quality": {
      "face1_quality": {
        "embedding_norm": 1.0,
        "embedding_mean": 0.02,
        "embedding_std": 0.35
      },
      "face2_quality": {
        "embedding_norm": 1.0,
        "embedding_mean": 0.01,
        "embedding_std": 0.33
      }
    }
  },
  "processing_time": 2.45,
  "image_loading_time": 0.78
}
```

### Error Responses

**400 Bad Request:**
```json
{
  "success": false,
  "error": "Image1 quality issue: Image too small: 100x80. Minimum size: 224x224"
}
```

**503 Service Unavailable:**
```json
{
  "success": false,
  "error": "Face recognition model not available"
}
```

## 🧪 Usage Examples

### Python Client Example

```python
import requests
import base64

# Load images
def load_image_as_base64(file_path):
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

# Compare faces
response = requests.post('http://localhost:8000/match', json={
    'image1': 'https://example.com/person1.jpg',
    'image2': f'data:image/jpeg;base64,{load_image_as_base64("person2.jpg")}',
    'threshold': 0.7
})

result = response.json()
if result['success']:
    if result['match']:
        print(f"MATCH! Similarity: {result['similarity_score']:.3f}")
    else:
        print(f"NO MATCH. Similarity: {result['similarity_score']:.3f}")
else:
    print(f"Error: {result['error']}")
```

### cURL Example

```bash
# Test with URLs
curl -X POST http://localhost:8000/match \
  -H "Content-Type: application/json" \
  -d '{
    "image1": "https://example.com/face1.jpg",
    "image2": "https://example.com/face2.jpg",
    "threshold": 0.6
  }'

# Health check
curl http://localhost:8000/health
```

## 🔍 Model Information

### MTCNN (Face Detection)
- **Purpose**: Single face detection and facial landmark extraction
- **Input**: RGB images
- **Output**: Face bounding boxes, confidence scores, facial keypoints
- **Requirements**: Minimum 224x224 image resolution

### ArcFace (Face Recognition)  
- **Purpose**: Face embedding extraction for similarity comparison
- **Architecture**: ResNet50-based
- **Input**: 112x112 aligned face regions
- **Output**: 512-dimensional face embeddings
- **Similarity Metric**: Cosine similarity

### CRMNET (Liveness Detection)
- **Purpose**: Anti-spoofing to detect real vs fake faces
- **Input**: Face regions 
- **Output**: Liveness probability scores
- **Threshold**: 0.5 (configurable)

## 🚦 Quality Controls

The system implements several quality controls:

1. **Single Face Requirement**: Only processes images with exactly one face
2. **Image Quality Validation**: Checks resolution, brightness, and contrast
3. **File Size Limits**: Configurable maximum file size (default: 10MB)
4. **Timeout Protection**: Request timeouts to prevent hanging
5. **Error Recovery**: Comprehensive error handling and logging

## 🐳 Docker Usage

### Build and Run
```bash
# Build the image
docker build -t facematch-api .

# Run the container
docker run -p 8000:8000 facematch-api

# Or use Docker Compose
docker-compose up --build
```

### Docker Compose Services
- **facematch**: Main API service
- **Volume Mounts**: Models and logs directories
- **Health Checks**: Automatic container health monitoring
- **Networks**: Isolated network for services

## 🛡️ Security Considerations

- **Input Validation**: Comprehensive validation of image inputs
- **File Size Limits**: Protection against large file uploads
- **Rate Limiting**: Implement rate limiting in production
- **CORS**: Configure CORS policies appropriately
- **Authentication**: Add authentication for production use

## 📊 Performance

### Typical Processing Times
- **Face Detection**: 100-200ms per image
- **Liveness Detection**: 50-100ms per face
- **Face Recognition**: 30-50ms per face pair
- **Total Pipeline**: 500-1000ms for two images

### Optimization Tips
- Use GPU acceleration when available
- Implement connection pooling for URL loading
- Consider batch processing for multiple comparisons
- Monitor memory usage with large images

## 🔧 Troubleshooting

### Common Issues

1. **Models not downloading**
   ```bash
   # Manual download
   ./download_models.sh --force --verbose
   ```

2. **Permission errors with Docker**
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER models/
   ```

3. **Memory issues with large images**
   - Reduce `MAX_IMAGE_DIMENSION` environment variable
   - Implement image preprocessing before upload

4. **Performance issues**
   - Enable GPU support in Docker
   - Increase container memory allocation
   - Use SSD storage for model files

## 📝 Development

### Running Tests
```bash
# Run the download script in test mode
./download_models.sh --help

# Test the API endpoints
python -m pytest tests/ # (if test suite exists)
```

### Adding New Models
1. Update model URLs in `download_models.sh`
2. Add model validation in respective modules
3. Update Docker container startup logic
4. Test with sample images

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MTCNN Paper](https://arxiv.org/abs/1604.02878)
- [ArcFace Paper](https://arxiv.org/abs/1801.07698)
- [Docker Documentation](https://docs.docker.com/)

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation at `/docs`
3. Check container logs: `docker-compose logs facematch`
4. Open an issue on the repository 