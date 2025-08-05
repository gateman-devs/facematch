#!/bin/bash

# Face Recognition API - Fresh Start Script
# This script rebuilds the Docker image from scratch and starts the services

set -e  # Exit on any error

echo "🚀 Starting Face Recognition API with fresh Docker build..."
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

# Step 1: Stop and remove existing containers
print_step "Stopping existing containers..."
docker compose down --remove-orphans || true

# Step 2: Remove only the main application image (keep Redis and base images)
print_step "Removing main application image only..."
# Get the application image name and remove it specifically
APP_IMAGE=$(docker compose config | grep -A5 "facematch:" | grep "image:" | awk '{print $2}' 2>/dev/null || echo "")
if [ -z "$APP_IMAGE" ]; then
    # If no image specified in compose, it will be built with default naming
    APP_IMAGE="facematch-facematch"
fi

# Remove the specific application image if it exists
docker image rm "$APP_IMAGE" 2>/dev/null || true
docker image rm "facematch_facematch" 2>/dev/null || true

# Remove only dangling images (not all unused images)
docker image prune -f || true

# Step 3: Build new image without cache
print_step "Building fresh Docker image (no cache)..."
docker compose build --no-cache --pull

# Step 4: Start services
print_step "Starting services with docker compose up..."
docker compose up -d

# Step 5: Show status
print_step "Checking service status..."
sleep 3
docker compose ps

# Step 6: Show logs
print_step "Recent logs:"
docker compose logs --tail=20

print_success "Face Recognition API started successfully!"
echo ""
echo "📡 API Endpoints:"
echo "   - Main API: http://localhost:8000"
echo "   - Health Check: http://localhost:8000/health"
echo "   - API Docs: http://localhost:8000/docs"
echo "   - Frontend: http://localhost:8000/static/index.html"
echo ""
echo "📊 To view live logs: docker compose logs -f"
echo "🛑 To stop services: docker compose down"
echo ""
echo "==================================================" 