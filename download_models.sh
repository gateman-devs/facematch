#!/bin/bash

# Face Recognition Model Download Script
# Downloads MTCNN, ArcFace, and Anti-spoofing models with validation and retry logic

set -e  # Exit on any error

# Configuration
MODELS_DIR="models"
MAX_RETRIES=3
DOWNLOAD_TIMEOUT=300  # 5 minutes per file

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create models directory if it doesn't exist
mkdir -p "$MODELS_DIR"

# Model URLs - Real working URLs from search results
# InsightFace ArcFace model (commonly used format)
ARCFACE_URL="https://huggingface.co/MonsterMMORPG/tools/resolve/main/glintr100.onnx"

# MTCNN detection model  
MTCNN_DET_URL="https://huggingface.co/MonsterMMORPG/tools/resolve/main/scrfd_10g_bnkps.onnx"

# Anti-spoofing model (liveness detection)
ANTISPOOFING_URL="https://huggingface.co/MonsterMMORPG/tools/resolve/main/2d106det.onnx"

# Age/Gender model (optional)
AGEGENDER_URL="https://huggingface.co/MonsterMMORPG/tools/resolve/main/genderage.onnx"

# Model file paths
ARCFACE_PATH="$MODELS_DIR/arcface_resnet50.onnx"
MTCNN_PATH="$MODELS_DIR/mtcnn.pb"
ANTISPOOFING_PATH="$MODELS_DIR/crmnet.onnx"
AGEGENDER_PATH="$MODELS_DIR/genderage.onnx"

# Function to print colored output
print_colored() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to download file with retry logic
download_with_retry() {
    local url=$1
    local output_path=$2
    local description=$3
    local retry_count=0
    
    print_colored $BLUE "Downloading $description..."
    
    while [ $retry_count -lt $MAX_RETRIES ]; do
        if curl -L --fail --connect-timeout 30 --max-time $DOWNLOAD_TIMEOUT \
                --progress-bar "$url" -o "$output_path"; then
            print_colored $GREEN "✓ Successfully downloaded $description"
            return 0
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $MAX_RETRIES ]; then
                print_colored $YELLOW "⚠ Download failed. Retrying ($retry_count/$MAX_RETRIES)..."
                sleep 2
            else
                print_colored $RED "✗ Failed to download $description after $MAX_RETRIES attempts"
                return 1
            fi
        fi
    done
}

# Function to verify file exists and has reasonable size
verify_download() {
    local file_path=$1
    local min_size=$2
    local description=$3
    
    if [ ! -f "$file_path" ]; then
        print_colored $RED "✗ File $file_path does not exist"
        return 1
    fi
    
    local file_size=$(stat -c%s "$file_path" 2>/dev/null || stat -f%z "$file_path" 2>/dev/null || echo 0)
    
    if [ "$file_size" -lt "$min_size" ]; then
        print_colored $RED "✗ File $file_path is too small ($file_size bytes, expected at least $min_size)"
        rm -f "$file_path"
        return 1
    fi
    
    print_colored $GREEN "✓ $description verified (size: $file_size bytes)"
    return 0
}

# Main download function
main() {
    print_colored $BLUE "=== Face Recognition Model Downloader ==="
    print_colored $BLUE "Models will be saved to: $MODELS_DIR"
    
    # Check if all models already exist
    if [ -f "$ARCFACE_PATH" ] && [ -f "$MTCNN_PATH" ] && [ -f "$ANTISPOOFING_PATH" ]; then
        print_colored $YELLOW "All models already exist. Use --force to re-download."
        if [ "$1" != "--force" ]; then
            exit 0
        fi
    fi
    
    local success=true
    
    # Download ArcFace model (face recognition)
    if [ ! -f "$ARCFACE_PATH" ] || [ "$1" = "--force" ]; then
        if download_with_retry "$ARCFACE_URL" "$ARCFACE_PATH" "ArcFace ResNet100 model"; then
            verify_download "$ARCFACE_PATH" 200000000 "ArcFace model" || success=false  # ~200MB
        else
            success=false
        fi
    else
        print_colored $GREEN "✓ ArcFace model already exists"
    fi
    
    # Download MTCNN model (face detection) - rename for compatibility
    if [ ! -f "$MTCNN_PATH" ] || [ "$1" = "--force" ]; then
        if download_with_retry "$MTCNN_DET_URL" "$MTCNN_PATH" "MTCNN detection model"; then
            verify_download "$MTCNN_PATH" 10000000 "MTCNN model" || success=false  # ~10MB
        else
            success=false
        fi
    else
        print_colored $GREEN "✓ MTCNN model already exists"
    fi
    
    # Download Anti-spoofing model (liveness detection)
    if [ ! -f "$ANTISPOOFING_PATH" ] || [ "$1" = "--force" ]; then
        if download_with_retry "$ANTISPOOFING_URL" "$ANTISPOOFING_PATH" "Anti-spoofing model"; then
            verify_download "$ANTISPOOFING_PATH" 1000000 "Anti-spoofing model" || success=false  # ~1MB
        else
            success=false
        fi
    else
        print_colored $GREEN "✓ Anti-spoofing model already exists"
    fi
    
    # Optional: Download age/gender model
    if [ ! -f "$AGEGENDER_PATH" ] || [ "$1" = "--force" ]; then
        print_colored $BLUE "Downloading optional age/gender model..."
        if download_with_retry "$AGEGENDER_URL" "$AGEGENDER_PATH" "Age/Gender model"; then
            verify_download "$AGEGENDER_PATH" 1000000 "Age/Gender model" || true  # Don't fail on optional model
        fi
    fi
    
    echo
    if [ "$success" = true ]; then
        print_colored $GREEN "=== All required models downloaded successfully! ==="
        print_colored $GREEN "Models saved in: $MODELS_DIR"
        print_colored $BLUE "You can now start the face recognition service."
    else
        print_colored $RED "=== Some model downloads failed ==="
        print_colored $YELLOW "Please check your internet connection and try again."
        exit 1
    fi
}

# Help function
show_help() {
    echo "Face Recognition Model Downloader"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --force    Re-download models even if they already exist"
    echo "  --help     Show this help message"
    echo
    echo "This script downloads the required models for face recognition:"
    echo "  - ArcFace ResNet100 model for face recognition"
    echo "  - MTCNN model for face detection"
    echo "  - Anti-spoofing model for liveness detection"
    echo "  - Age/Gender model (optional)"
    echo
}

# Parse command line arguments
case "${1:-}" in
    --help)
        show_help
        exit 0
        ;;
    --force)
        main --force
        ;;
    *)
        main
        ;;
esac 