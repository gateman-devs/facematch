#!/usr/bin/env python3
"""
Test script to demonstrate the new movement detection algorithm.
Shows how the system now detects all movements and finds matching sequences.
"""

import sys
import os
sys.path.append('infrastructure/facematch')

from simple_liveness import SimpleLivenessDetector

def test_movement_detection():
    """Test the new movement detection algorithm."""
    
    # Initialize the detector
    detector = SimpleLivenessDetector()
    
    # Create sample nose positions that simulate a video with movements
    # This simulates a video where the user makes various movements
    sample_positions = []
    
    # Simulate a video with movements: up, up, right, down (but not in that exact order)
    # Frame 0-30: up movement
    for i in range(30):
        sample_positions.append({
            'timestamp': i * 0.033,  # 30fps
            'x': 0.5,  # center
            'y': 0.5 - (i * 0.01)  # moving up
        })
    
    # Frame 31-60: right movement  
    for i in range(30):
        sample_positions.append({
            'timestamp': (i + 30) * 0.033,
            'x': 0.5 + (i * 0.01),  # moving right
            'y': 0.2  # stay at top
        })
    
    # Frame 61-90: down movement
    for i in range(30):
        sample_positions.append({
            'timestamp': (i + 60) * 0.033,
            'x': 0.8,  # stay at right
            'y': 0.2 + (i * 0.01)  # moving down
        })
    
    # Frame 91-120: up movement (second up)
    for i in range(30):
        sample_positions.append({
            'timestamp': (i + 90) * 0.033,
            'x': 0.8,  # stay at right
            'y': 0.5 - (i * 0.01)  # moving up again
        })
    
    # Frame 121-150: some random movements
    for i in range(30):
        sample_positions.append({
            'timestamp': (i + 120) * 0.033,
            'x': 0.5 + (i * 0.005),  # slight right
            'y': 0.2 + (i * 0.005)  # slight down
        })
    
    expected_sequence = ['up', 'up', 'right', 'down']
    
    print("Testing new movement detection algorithm...")
    print(f"Expected sequence: {expected_sequence}")
    print(f"Total frames: {len(sample_positions)}")
    print()
    
    # Test the new algorithm
    result = detector._validate_movement_sequence(sample_positions, expected_sequence)
    
    print("=== RESULTS ===")
    print(f"Passed: {result['passed']}")
    print(f"Accuracy: {result.get('accuracy', 0):.3f}")
    print(f"All detected movements: {result.get('all_movements', [])}")
    print(f"Matched sequence: {result.get('detected_sequence', [])}")
    print(f"Total movements detected: {result.get('total_movements_detected', 0)}")
    
    if result.get('best_match_start_index') is not None:
        print(f"Best match found at positions {result['best_match_start_index']}-{result['best_match_end_index']}")
    
    if result.get('partial_match'):
        partial = result['partial_match']
        print(f"Best partial match: {partial['matched']}/{len(expected_sequence)} movements")
        print(f"Partial match sequence: {partial['matched_movements']}")
    
    print()
    print("=== ALGORITHM EXPLANATION ===")
    print("1. The system analyzes the entire video in sliding windows")
    print("2. It detects ALL significant movements (not just first 4 time segments)")
    print("3. It then looks for ANY sequence of 4 consecutive movements that match")
    print("4. If found, the test passes - much more flexible than the old approach!")
    print()
    print("This approach is much more user-friendly because:")
    print("- Users don't need to follow exact timing")
    print("- Extra movements are ignored")
    print("- Any 4 consecutive movements that match will pass")
    print("- Much more forgiving of natural head movement patterns")

if __name__ == "__main__":
    test_movement_detection()
