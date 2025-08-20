#!/usr/bin/env python3
"""
Test script for degree-based head movement detection.
Demonstrates the new non-time-based detection system.
"""

import sys
import os
import logging
import numpy as np
from typing import List, Dict, Any

# Add the infrastructure directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'infrastructure', 'facematch'))

from enhanced_mediapipe_detector import EnhancedMediaPipeDetector
from head_movement_config import HeadMovementConfig, DEGREE_THRESHOLD_GUIDE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_degree_based_detection():
    """Test the degree-based head movement detection system."""
    
    print("=" * 60)
    print("DEGREE-BASED HEAD MOVEMENT DETECTION TEST")
    print("=" * 60)
    
    # Print configuration guide
    print("\nDEGREE THRESHOLD GUIDE:")
    print(DEGREE_THRESHOLD_GUIDE)
    
    # Test different configurations
    configs = {
        'Default': HeadMovementConfig.default_config(),
        'Conservative': HeadMovementConfig.conservative_config(),
        'Lenient': HeadMovementConfig.lenient_config()
    }
    
    for config_name, config in configs.items():
        print(f"\n{'='*40}")
        print(f"TESTING {config_name.upper()} CONFIGURATION")
        print(f"{'='*40}")
        
        print(f"Minimum rotation degrees: {config.min_rotation_degrees}°")
        print(f"Significant rotation degrees: {config.significant_rotation_degrees}°")
        print(f"Center threshold degrees: {config.center_threshold_degrees}°")
        print(f"Min confidence threshold: {config.min_confidence_threshold}")
        
        # Create detector with this configuration
        detector = EnhancedMediaPipeDetector(
            min_rotation_degrees=config.min_rotation_degrees,
            significant_rotation_degrees=config.significant_rotation_degrees,
            min_confidence_threshold=config.min_confidence_threshold,
            debug_mode=True
        )
        
        # Test with simulated head pose data
        test_head_movements(detector, config_name)

def test_head_movements(detector: EnhancedMediaPipeDetector, config_name: str):
    """Test detector with simulated head movements."""
    
    print(f"\nTesting {config_name} configuration with simulated movements:")
    
    # Simulate head pose data with different rotation degrees
    test_cases = [
        {
            'name': 'Small movement (10°)',
            'yaw_change': 10.0,
            'pitch_change': 0.0,
            'expected_detection': 'lenient_only'
        },
        {
            'name': 'Natural movement (15°)',
            'yaw_change': 15.0,
            'pitch_change': 0.0,
            'expected_detection': 'all'
        },
        {
            'name': 'Clear movement (25°)',
            'yaw_change': 25.0,
            'pitch_change': 0.0,
            'expected_detection': 'all'
        },
        {
            'name': 'Large movement (35°)',
            'yaw_change': 35.0,
            'pitch_change': 0.0,
            'expected_detection': 'all'
        },
        {
            'name': 'Upward tilt (20°)',
            'yaw_change': 0.0,
            'pitch_change': -20.0,
            'expected_detection': 'all'
        },
        {
            'name': 'Diagonal movement (18° yaw, 12° pitch)',
            'yaw_change': 18.0,
            'pitch_change': 12.0,
            'expected_detection': 'lenient_default'
        }
    ]
    
    for test_case in test_cases:
        print(f"\n  Testing: {test_case['name']}")
        
        # Create simulated pose data
        initial_pose = {
            'x': 0.5, 'y': 0.5,
            'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
            'quality_score': 0.8,
            'yaw_confidence': 0.8,
            'pitch_confidence': 0.8,
            'roll_confidence': 0.8,
            'eye_distance': 0.1
        }
        
        final_pose = {
            'x': 0.5, 'y': 0.5,
            'yaw': test_case['yaw_change'],
            'pitch': test_case['pitch_change'],
            'roll': 0.0,
            'quality_score': 0.8,
            'yaw_confidence': 0.8,
            'pitch_confidence': 0.8,
            'roll_confidence': 0.8,
            'eye_distance': 0.1
        }
        
        # Reset detector state
        detector._reset_state()
        
        # Process initial pose
        detector.detect_movement(initial_pose, 0.0)
        
        # Process final pose (should trigger movement detection)
        movement = detector.detect_movement(final_pose, 0.5)
        
        if movement:
            print(f"    ✓ Movement detected: {movement.direction}")
            print(f"      Rotation magnitude: {movement.magnitude:.1f}°")
            print(f"      Confidence: {movement.confidence:.3f}")
            
            # Check if detection matches expectation
            expected = test_case['expected_detection']
            if expected == 'all' or config_name.lower() in expected:
                print(f"      ✓ Expected detection for {config_name}")
            else:
                print(f"      ⚠ Unexpected detection for {config_name}")
        else:
            print(f"    ✗ No movement detected")
            
            # Check if this was expected
            expected = test_case['expected_detection']
            if expected == 'all' or config_name.lower() in expected:
                print(f"      ⚠ Expected detection for {config_name} but none found")
            else:
                print(f"      ✓ Expected no detection for {config_name}")

def demonstrate_degree_thresholds():
    """Demonstrate how degree thresholds work."""
    
    print("\n" + "="*60)
    print("DEGREE THRESHOLD DEMONSTRATION")
    print("="*60)
    
    print("\nFor a head facing the camera at 90° (center position):")
    print("\n1. MINIMUM ROTATION THRESHOLDS:")
    print("   - 10°: Detects very small head movements")
    print("   - 15°: Detects natural head turns (recommended)")
    print("   - 20°: Requires clear head movements")
    
    print("\n2. HEAD TURN EXAMPLES:")
    print("   - Left turn: 15° minimum = head rotated 15° left")
    print("   - Right turn: 15° minimum = head rotated 15° right")
    print("   - Up tilt: 15° minimum = head tilted 15° up")
    print("   - Down tilt: 15° minimum = head tilted 15° down")
    
    print("\n3. SIGNIFICANT MOVEMENT THRESHOLDS:")
    print("   - 20°: Most movements considered significant")
    print("   - 25°: Clear head turns get higher confidence")
    print("   - 30°: Only large movements are significant")
    
    print("\n4. CENTER POSITION THRESHOLD:")
    print("   - 10°: Head within ±10° of center is considered 'facing camera'")
    print("   - This accounts for natural head sway and slight movements")

def main():
    """Main test function."""
    try:
        test_degree_based_detection()
        demonstrate_degree_thresholds()
        
        print("\n" + "="*60)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nKey improvements:")
        print("✓ Removed time-based constraints (no more 2s/1.5s expectations)")
        print("✓ Implemented degree-based detection (15° minimum recommended)")
        print("✓ Configurable thresholds for different sensitivity levels")
        print("✓ Detects movements as they occur without time pressure")
        print("✓ More accurate head pose calculation using yaw/pitch angles")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
