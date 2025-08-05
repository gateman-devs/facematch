#!/bin/bash

# Face Recognition API - Stop Script
# This script stops all services and optionally cleans up

set -e  # Exit on any error

echo "🛑 Stopping Face Recognition API..."
echo "=================================================="

# Function to print colored output
print_step() {
    echo -e "\n🔧 $1"
}

print_success() {
    echo -e "\n✅ $1"
}

# Parse command line arguments
CLEANUP=false
if [[ "$1" == "--cleanup" || "$1" == "-c" ]]; then
    CLEANUP=true
fi

# Step 1: Stop services
print_step "Stopping Docker Compose services..."
docker compose down

# Step 2: Optional cleanup
if [ "$CLEANUP" = true ]; then
    print_step "Cleaning up application image and volumes..."
    
    # Remove only the main application image (keep Redis and base images)
    APP_IMAGE=$(docker compose config | grep -A5 "facematch:" | grep "image:" | awk '{print $2}' 2>/dev/null || echo "")
    if [ -z "$APP_IMAGE" ]; then
        APP_IMAGE="facematch-facematch"
    fi
    
    # Remove the specific application image if it exists
    docker image rm "$APP_IMAGE" 2>/dev/null || true
    docker image rm "facematch_facematch" 2>/dev/null || true
    
    # Remove volumes and dangling images
    docker compose down --volumes --remove-orphans
    docker image prune -f
    print_success "Services stopped and application image cleaned up!"
else
    print_success "Services stopped!"
    echo ""
    echo "💡 To also remove application image and volumes, run: ./stop.sh --cleanup"
fi

echo ""
echo "🔄 To restart with fresh build: ./start.sh"
echo "==================================================" 