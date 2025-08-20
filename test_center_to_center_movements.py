#!/usr/bin/env python3
"""
Test script for center-to-center movement detection.
This script tests the new movement detection system that removes time-based constraints.
"""

import sys
import os
import numpy as np
import time

# Add the infrastructure directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'infrastructure'))

from facematch.simple_liveness import SimpleLivenessDetector

def create_mock_nose_positions():
    """Create mock nose positions that simulate center-to-center movements."""
    positions = []
    timestamp = 0.0
    
    # Center position
    center_x, center_y = 0.5, 0.5
    
    # Movement 1: Center -> Right -> Center
    # Start at center
    for i in range(5):
        positions.append({
            'x': center_x + np.random.normal(0, 0.01),  # Small noise
            'y': center_y + np.random.normal(0, 0.01),
            'timestamp': timestamp
        })
        timestamp += 0.1
    
    # Move right
    for i in range(10):
        x = center_x + (i + 1) * 0.05  # Move right
        y = center_y + np.random.normal(0, 0.02)
        positions.append({
            'x': x,
            'y': y,
            'timestamp': timestamp
        })
        timestamp += 0.1
    
    # Return to center
    for i in range(5):
        x = center_x + (5 - i) * 0.05  # Move back to center
        y = center_y + np.random.normal(0, 0.01)
        positions.append({
            'x': x,
            'y': y,
            'timestamp': timestamp
        })
        timestamp += 0.1
    
    # Movement 2: Center -> Left -> Center
    # Start at center
    for i in range(5):
        positions.append({
            'x': center_x + np.random.normal(0, 0.01),
            'y': center_y + np.random.normal(0, 0.01),
            'timestamp': timestamp
        })
        timestamp += 0.1
    
    # Move left
    for i in range(10):
        x = center_x - (i + 1) * 0.05  # Move left
        y = center_y + np.random.normal(0, 0.02)
        positions.append({
            'x': x,
            'y': y,
            'timestamp': timestamp
        })
        timestamp += 0.1
    
    # Return to center
    for i in range(5):
        x = center_x - (5 - i) * 0.05  # Move back to center
        y = center_y + np.random.normal(0, 0.01)
        positions.append({
            'x': x,
            'y': y,
            'timestamp': timestamp
        })
        timestamp += 0.1
    
    # Movement 3: Center -> Up -> Center
    # Start at center
    for i in range(5):
        positions.append({
            'x': center_x + np.random.normal(0, 0.01),
            'y': center_y + np.random.normal(0, 0.01),
            'timestamp': timestamp
        })
        timestamp += 0.1
    
    # Move up
    for i in range(10):
        x = center_x + np.random.normal(0, 0.02)
        y = center_y - (i + 1) * 0.05  # Move up
        positions.append({
            'x': x,
            'y': y,
            'timestamp': timestamp
        })
        timestamp += 0.1
    
    # Return to center
    for i in range(5):
        x = center_x + np.random.normal(0, 0.01)
        y = center_y - (5 - i) * 0.05  # Move back to center
        positions.append({
            'x': x,
            'y': y,
            'timestamp': timestamp
        })
        timestamp += 0.1
    
    # Movement 4: Center -> Down -> Center
    # Start at center
    for i in range(5):
        positions.append({
            'x': center_x + np.random.normal(0, 0.01),
            'y': center_y + np.random.normal(0, 0.01),
            'timestamp': timestamp
        })
        timestamp += 0.1
    
    # Move down
    for i in range(10):
        x = center_x + np.random.normal(0, 0.02)
        y = center_y + (i + 1) * 0.05  # Move down
        positions.append({
            'x': x,
            'y': y,
            'timestamp': timestamp
        })
        timestamp += 0.1
    
    # Return to center
    for i in range(5):
        x = center_x + np.random.normal(0, 0.01)
        y = center_y + (5 - i) * 0.05  # Move back to center
        positions.append({
            'x': x,
            'y': y,
            'timestamp': timestamp
        })
        timestamp += 0.1
    
    return positions

def test_center_to_center_detection():
    """Test the center-to-center movement detection."""
    print("Testing center-to-center movement detection...")
    
    # Create detector
    detector = SimpleLivenessDetector()
    
    # Create mock nose positions
    nose_positions = create_mock_nose_positions()
    print(f"Created {len(nose_positions)} mock nose positions")
    
    # Test center-to-center movement detection
    movements = detector._detect_center_to_center_movements(nose_positions)
    
    print(f"Detected {len(movements)} center-to-center movements:")
    for i, movement in enumerate(movements):
        print(f"  Movement {i+1}: {movement['direction']} (confidence: {movement['confidence']:.3f})")
    
    # Test sequence validation
    expected_sequence = ['right', 'left', 'up', 'down']
    validation_result = detector._validate_movement_sequence(nose_positions, expected_sequence)
    
    print(f"\nSequence validation result:")
    print(f"  Passed: {validation_result['passed']}")
    print(f"  Accuracy: {validation_result['accuracy']:.3f}")
    print(f"  Detected sequence: {validation_result['detected_sequence']}")
    print(f"  Expected sequence: {expected_sequence}")
    
    if validation_result.get('error'):
        print(f"  Error: {validation_result['error']}")
    
    return validation_result['passed']

def test_challenge_generation():
    """Test the new challenge generation without time constraints."""
    print("\nTesting challenge generation...")
    
    detector = SimpleLivenessDetector()
    challenge = detector.generate_challenge()
    
    print(f"Generated challenge:")
    print(f"  Type: {challenge['type']}")
    print(f"  Instruction: {challenge['instruction']}")
    print(f"  Duration: {challenge['duration']}")
    print(f"  Movement sequence: {challenge['movement_sequence']}")
    print(f"  Movement type: {challenge['movement_type']}")
    print(f"  Direction duration: {challenge['direction_duration']}")
    
    # Verify no time constraints
    assert challenge['direction_duration'] is None, "Direction duration should be None for center-to-center movements"
    assert challenge['movement_type'] == 'center_to_center', "Movement type should be center_to_center"
    
    print("✓ Challenge generation test passed")

if __name__ == "__main__":
    print("Center-to-Center Movement Detection Test")
    print("=" * 50)
    
    try:
        # Test challenge generation
        test_challenge_generation()
        
        # Test movement detection
        success = test_center_to_center_detection()
        
        if success:
            print("\n✓ All tests passed! Center-to-center movement detection is working correctly.")
        else:
            print("\n✗ Some tests failed. Please check the implementation.")
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
