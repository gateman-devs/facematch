#!/bin/bash

# Performance Check Script for Video Liveness System
# This script analyzes the performance logs and determines if the system is performing well enough

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Video Liveness Performance Checker${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if logs directory exists
if [ ! -d "logs" ]; then
    echo -e "${YELLOW}Warning: logs directory not found. Creating it...${NC}"
    mkdir -p logs
fi

# Check if performance log exists
if [ ! -f "logs/facematch.log" ]; then
    echo -e "${YELLOW}Warning: Application log file not found at logs/facematch.log${NC}"
    echo -e "${YELLOW}This means no video liveness requests have been processed yet.${NC}"
    echo -e "${YELLOW}Run some video liveness tests first to generate performance data.${NC}"
    exit 1
fi

# Check log file size
log_size=$(stat -c%s "logs/facematch.log" 2>/dev/null || stat -f%z "logs/facematch.log" 2>/dev/null || echo "0")
if [ "$log_size" -eq 0 ]; then
    echo -e "${YELLOW}Warning: Performance log file is empty.${NC}"
    echo -e "${YELLOW}No video liveness requests have been processed yet.${NC}"
    exit 1
fi

echo -e "${GREEN}Found performance log file. Analyzing...${NC}"

# Run the performance analysis
python3 analyze_performance.py --log-file logs/facematch.log

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Performance Check Complete${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if we should save detailed results
if [ "$1" = "--save" ] || [ "$1" = "-s" ]; then
    timestamp=$(date +"%Y%m%d_%H%M%S")
    output_file="logs/performance_analysis_${timestamp}.json"
    python3 analyze_performance.py --log-file logs/facematch.log --output "$output_file"
    echo -e "${GREEN}Detailed analysis saved to: ${output_file}${NC}"
fi

echo ""
echo -e "${BLUE}Usage:${NC}"
echo -e "  $0              - Run performance analysis"
echo -e "  $0 --save       - Run analysis and save detailed results to JSON"
echo -e "  $0 -s           - Same as --save"
