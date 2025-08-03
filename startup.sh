#!/bin/bash
set -e

echo "Starting Face Recognition API..."

# Check if models exist, download if missing
if [ ! -f "models/arcface_resnet50.onnx" ] || [ ! -f "models/mtcnn.pb" ] || [ ! -f "models/crmnet.onnx" ]; then
    echo "Models not found, downloading models..."
    if ./download_models.sh; then
        echo "Models downloaded successfully"
    else
        echo "WARNING: Model download failed. Some features may not work."
        echo "You may need to manually place model files in the models/ directory."
    fi
else
    echo "All models found"
fi

# Start the server
echo "Starting server on port ${PORT:-8000}..."
exec python -m infrastructure.facematch.server --host 0.0.0.0 --port ${PORT:-8000} 