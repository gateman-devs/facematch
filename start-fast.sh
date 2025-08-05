#!/bin/bash

# Face Recognition API - Fast Start Script
# This script uses Docker layer caching for much faster builds

set -e  # Exit on any error

echo "⚡ Starting Face Recognition API with fast caching..."
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

# Step 1: Stop existing containers (but keep images)
print_step "Stopping existing containers..."
docker compose down --remove-orphans || true

# Step 2: Build with layer caching (much faster!)
print_step "Building with layer caching..."
docker compose build --pull

# Step 3: Start services
print_step "Starting services..."
docker compose up -d

# Step 4: Show status
print_step "Checking service status..."
sleep 3
docker compose ps

# Step 5: Show logs
print_step "Recent logs:"
docker compose logs --tail=20

print_success "Face Recognition API started successfully (with caching)!"
echo ""
echo "📡 API Endpoints:"
echo "   - Main API: http://localhost:8000"
echo "   - Health Check: http://localhost:8000/health"
echo "   - API Docs: http://localhost:8000/docs"
echo "   - Frontend: http://localhost:8000/static/index.html"
echo "   - Challenge: http://localhost:8000/static/challenge.html"
echo ""
echo "⚡ This script uses layer caching for faster builds"
echo "🐌 Use start.sh for clean rebuilds (slower)"
echo "📊 To view live logs: docker compose logs -f"
echo "🛑 To stop services: docker compose down"
echo ""
echo "==================================================" 