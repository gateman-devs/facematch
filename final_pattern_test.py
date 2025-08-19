#!/usr/bin/env python3
"""
Final test to analyze the complete movement pattern and improve accuracy.
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

def final_pattern_test():
    """Final test to analyze the complete movement pattern."""
    
    print("=== Final Pattern Analysis ===\n")
    
    # Create detector with optimized settings
    detector = SimpleMediaPipeDetector(
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        movement_threshold=0.008,  # Lower threshold for better sensitivity
        max_history=100,
        debug_mode=True
    )
    
    video_path = "test_video.mov"
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    print(f"📹 Final analysis of: {video_path}")
    print("Expected pattern: left, left, right, right, up, up, down, down, left, right, up, down\n")
    
    # Process video with optimized method
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
    
    # Expected pattern analysis
    expected_pattern = ["left", "left", "right", "right", "up", "up", "down", "down", "left", "right", "up", "down"]
    
    print("🎯 Pattern Analysis:")
    print(f"   Expected: {expected_pattern}")
    print(f"   Detected:  {detected_sequence}")
    
    # Calculate accuracy with different tolerances
    print(f"\n📊 Accuracy Analysis:")
    
    tolerances = [0.3, 0.5, 0.7, 1.0]
    for tolerance in tolerances:
        validation = detector.validate_sequence_improved(expected_pattern, tolerance=tolerance)
        print(f"   Tolerance {tolerance}: Accuracy={validation['accuracy']:.3f}, "
              f"Similarity={validation['sequence_similarity']:.3f}, "
              f"Timing={validation['timing_accuracy']:.3f}")
    
    # Try to find the best subsequence match
    print(f"\n🔍 Best Subsequence Analysis:")
    
    # Find the longest common subsequence manually
    def find_lcs(expected, detected):
        m, n = len(expected), len(detected)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if expected[i-1] == detected[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        # Reconstruct the LCS
        lcs = []
        i, j = m, n
        while i > 0 and j > 0:
            if expected[i-1] == detected[j-1]:
                lcs.append(expected[i-1])
                i -= 1
                j -= 1
            elif dp[i-1][j] > dp[i][j-1]:
                i -= 1
            else:
                j -= 1
        
        return list(reversed(lcs))
    
    if detected_sequence:
        lcs = find_lcs(expected_pattern, detected_sequence)
        print(f"   Longest Common Subsequence: {lcs}")
        print(f"   LCS Length: {len(lcs)} / {len(expected_pattern)} = {len(lcs)/len(expected_pattern):.3f}")
        
        # Find which movements are missing
        missing = [m for m in expected_pattern if m not in lcs]
        extra = [m for m in detected_sequence if m not in expected_pattern]
        
        print(f"   Missing movements: {missing}")
        print(f"   Extra movements: {extra}")
    
    # Analyze timing
    print(f"\n⏱️  Timing Analysis:")
    if result.movements:
        timestamps = [m.timestamp for m in result.movements]
        intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        
        print(f"   Total duration: {timestamps[-1] - timestamps[0]:.2f}s")
        print(f"   Average interval: {sum(intervals)/len(intervals):.2f}s")
        print(f"   Min interval: {min(intervals):.2f}s")
        print(f"   Max interval: {max(intervals):.2f}s")
    
    # Summary
    print(f"\n📋 Summary:")
    if detected_sequence:
        # Count each direction
        direction_counts = {}
        for direction in detected_sequence:
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        
        print(f"   Detected directions:")
        for direction, count in direction_counts.items():
            print(f"     {direction}: {count}")
        
        # Check if we have the basic pattern
        has_left = direction_counts.get('left', 0) >= 2
        has_right = direction_counts.get('right', 0) >= 2
        has_up = direction_counts.get('up', 0) >= 2
        has_down = direction_counts.get('down', 0) >= 2
        
        print(f"   Pattern completeness:")
        print(f"     Left (≥2): {'✅' if has_left else '❌'} ({direction_counts.get('left', 0)})")
        print(f"     Right (≥2): {'✅' if has_right else '❌'} ({direction_counts.get('right', 0)})")
        print(f"     Up (≥2): {'✅' if has_up else '❌'} ({direction_counts.get('up', 0)})")
        print(f"     Down (≥2): {'✅' if has_down else '❌'} ({direction_counts.get('down', 0)})")
        
        if has_left and has_right and has_up and has_down:
            print(f"   🎉 All directions detected!")
        else:
            print(f"   ⚠️  Some directions missing")
    
    detector.release()
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    final_pattern_test()
