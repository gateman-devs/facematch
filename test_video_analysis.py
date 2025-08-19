#!/usr/bin/env python3
"""
Test script to analyze the provided video and detect head movements.
Expected pattern: left twice, right twice, up twice, down twice, then left, right, up, down
"""

import logging
import sys
import os
from pathlib import Path

# Add the infrastructure directory to the path
sys.path.insert(0, str(Path(__file__).parent / "infrastructure"))

from facematch.simple_mediapipe_detector import SimpleMediaPipeDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_video_movements():
    """Analyze the provided video and detect head movements."""
    
    print("=== Video Movement Analysis ===\n")
    
    # Create detector with debug mode for detailed analysis
    detector = SimpleMediaPipeDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        movement_threshold=0.015,  # Lower threshold for more sensitive detection
        max_history=50,            # Keep more history for analysis
        debug_mode=True
    )
    
    video_path = "test_video.mov"
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    print(f"📹 Analyzing video: {video_path}")
    print("Expected pattern: left, left, right, right, up, up, down, down, left, right, up, down\n")
    
    # Process video with optimized method for better performance
    result = detector.process_video_optimized(video_path, target_fps=15)
    
    if not result.success:
        print(f"❌ Video processing failed: {result.error}")
        return
    
    print(f"✅ Video processing completed:")
    print(f"   - Processing time: {result.processing_time:.3f}s")
    print(f"   - Frames processed: {result.frames_processed}")
    print(f"   - Movements detected: {len(result.movements)}")
    print()
    
    # Get detected movement sequence
    detected_sequence = detector.get_movement_sequence()
    
    print("📊 Detected Movements:")
    if detected_sequence:
        for i, movement in enumerate(detected_sequence, 1):
            print(f"   {i:2d}. {movement}")
    else:
        print("   No movements detected!")
    
    print()
    
    # Get detailed movement information
    detailed_movements = detector.get_detailed_movements()
    
    print("📈 Detailed Movement Analysis:")
    if detailed_movements:
        for i, movement in enumerate(detailed_movements, 1):
            print(f"   {i:2d}. {movement['direction']:5} - "
                  f"Confidence: {movement['confidence']:.3f}, "
                  f"Magnitude: {movement['magnitude']:.4f}, "
                  f"Time: {movement['timestamp']:.2f}s")
    else:
        print("   No detailed movement data available!")
    
    print()
    
    # Expected pattern analysis
    expected_pattern = ["left", "left", "right", "right", "up", "up", "down", "down", "left", "right", "up", "down"]
    
    print("🎯 Pattern Analysis:")
    print(f"   Expected: {expected_pattern}")
    print(f"   Detected:  {detected_sequence}")
    
    # Calculate accuracy
    if detected_sequence:
        # Use improved validation
        validation = detector.validate_sequence_improved(expected_pattern, tolerance=0.5)
        
        print(f"\n📊 Accuracy Results:")
        print(f"   - Final Accuracy: {validation['accuracy']:.3f}")
        print(f"   - Sequence Similarity: {validation['sequence_similarity']:.3f}")
        print(f"   - Timing Accuracy: {validation['timing_accuracy']:.3f}")
        
        # Check if pattern matches
        if validation['accuracy'] > 0.7:
            print("   ✅ Pattern detection: GOOD")
        elif validation['accuracy'] > 0.4:
            print("   ⚠️  Pattern detection: PARTIAL")
        else:
            print("   ❌ Pattern detection: POOR")
    else:
        print("\n❌ No movements detected - cannot calculate accuracy")
    
    print()
    
    # Log movement summary
    detector.log_movement_summary()
    
    # Clean up
    detector.release()
    
    print("=== Analysis Complete ===")

if __name__ == "__main__":
    analyze_video_movements()
