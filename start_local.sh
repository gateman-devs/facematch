#!/bin/bash

# Face Recognition API - Development Start Script
# This script builds the Docker image and downloads models only if missing
# Use this for development environments where you need to rebuild and test changes

set -e  # Exit on any error

echo "🔨 Starting Face Recognition API (Development Mode)..."
echo "=================================================="

# Function to print colored output
print_step() {
    echo -e "\n🔧 $1"
}

print_success() {
    echo -e "\n✅ $1"
}

print_warning() {
    echo -e "\n⚠️  $1"
}

# Step 1: Stop existing containers
print_step "Stopping existing containers..."
docker compose down --remove-orphans || true

# Step 2: Check if models exist locally
print_step "Checking for existing models..."
if [ ! -f "models/arcface_resnet50.onnx" ] || [ ! -f "models/mtcnn.pb" ] || [ ! -f "models/crmnet.onnx" ]; then
    print_warning "Models not found locally. They will be downloaded at runtime."
    print_step "Downloading models..."
    if ./download_models.sh; then
        print_success "Models downloaded successfully"
    else
        print_warning "Model download failed. Models will be downloaded at runtime."
    fi
else
    print_success "All models found locally"
fi

# Step 3: Build new image
print_step "Building Docker image..."
docker compose build --pull

# Step 4: Start services
print_step "Starting services..."
docker compose up -d

# Step 5: Show status
print_step "Checking service status..."
sleep 3
docker compose ps

# Step 6: Show logs
print_step "Recent logs:"
docker compose logs --tail=20

print_success "Face Recognition API started successfully (Development Mode)!"
echo ""
echo "📡 API Endpoints:"
echo "   - Main API: http://localhost:8000"
echo "   - Health Check: http://localhost:8000/health"
echo "   - API Docs: http://localhost:8000/docs"
echo "   - Frontend: http://localhost:8000/static/index.html"
echo ""
echo "📊 To view live logs: docker compose logs -f"
echo "🛑 To stop services: docker compose down"
echo "🚀 To start without rebuilding (Production): ./start.sh"
echo ""
echo "==================================================" 