#!/bin/bash

# Face Recognition API - Production Start Script
# This script starts the application using docker compose without building
# Use this for production environments where the image is already built

set -e  # Exit on any error

echo "🚀 Starting Face Recognition API (Production Mode)..."
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

# Step 2: Start services (will pull image if not exists)
print_step "Starting services with docker compose up..."
docker compose up -d

# Step 3: Show status
print_step "Checking service status..."
sleep 3
docker compose ps

# Step 4: Show logs
print_step "Recent logs:"
docker compose logs --tail=20

print_success "Face Recognition API started successfully (Production Mode)!"
echo ""
echo "📡 API Endpoints:"
echo "   - Main API: http://localhost:8000"
echo "   - Health Check: http://localhost:8000/health"
echo "   - API Docs: http://localhost:8000/docs"
echo "   - Frontend: http://localhost:8000/static/index.html"
echo ""
echo "📊 To view live logs: docker compose logs -f"
echo "🛑 To stop services: docker compose down"
echo "🔨 To rebuild and start (Development): ./start_local.sh"
echo ""
echo "==================================================" 