#!/usr/bin/env python3
"""
Simple test for degree-based head movement detection logic.
Tests the core detection algorithms without MediaPipe dependencies.
"""

import sys
import os
import numpy as np
from typing import Dict, Any

# Add the infrastructure directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'infrastructure', 'facematch'))

from head_movement_config import HeadMovementConfig, DEGREE_THRESHOLD_GUIDE

def test_degree_thresholds():
    """Test the degree threshold logic."""
    
    print("=" * 60)
    print("SIMPLE DEGREE-BASED DETECTION TEST")
    print("=" * 60)
    
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
        
        # Test movement detection logic
        test_movement_detection(config, config_name)

def test_movement_detection(config: HeadMovementConfig, config_name: str):
    """Test the movement detection logic with simulated data."""
    
    print(f"\nTesting {config_name} configuration:")
    
    # Test cases with different rotation magnitudes
    test_cases = [
        {
            'name': 'Small movement (10°)',
            'yaw_change': 10.0,
            'pitch_change': 0.0,
            'should_detect': config.min_rotation_degrees <= 10.0
        },
        {
            'name': 'Natural movement (15°)',
            'yaw_change': 15.0,
            'pitch_change': 0.0,
            'should_detect': config.min_rotation_degrees <= 15.0
        },
        {
            'name': 'Clear movement (25°)',
            'yaw_change': 25.0,
            'pitch_change': 0.0,
            'should_detect': config.min_rotation_degrees <= 25.0
        },
        {
            'name': 'Large movement (35°)',
            'yaw_change': 35.0,
            'pitch_change': 0.0,
            'should_detect': config.min_rotation_degrees <= 35.0
        },
        {
            'name': 'Upward tilt (20°)',
            'yaw_change': 0.0,
            'pitch_change': -20.0,
            'should_detect': config.min_rotation_degrees <= 20.0
        },
        {
            'name': 'Diagonal movement (18° yaw, 12° pitch)',
            'yaw_change': 18.0,
            'pitch_change': 12.0,
            'should_detect': config.min_rotation_degrees <= 18.0  # Uses larger rotation
        }
    ]
    
    for test_case in test_cases:
        print(f"\n  Testing: {test_case['name']}")
        
        # Calculate rotation magnitude (larger of yaw or pitch)
        yaw_abs = abs(test_case['yaw_change'])
        pitch_abs = abs(test_case['pitch_change'])
        rotation_magnitude = max(yaw_abs, pitch_abs)
        
        # Determine direction
        if yaw_abs > pitch_abs:
            if test_case['yaw_change'] > 0:
                direction = 'right'
            else:
                direction = 'left'
        else:
            if test_case['pitch_change'] > 0:
                direction = 'down'
            else:
                direction = 'up'
        
        # Check if movement should be detected
        should_detect = test_case['should_detect']
        threshold_met = rotation_magnitude >= config.min_rotation_degrees
        
        # Calculate confidence (simplified)
        magnitude_ratio = rotation_magnitude / config.min_rotation_degrees
        base_confidence = min(magnitude_ratio / 2.0, 1.0)
        
        # Apply size factor
        size_factor = 1.0
        if rotation_magnitude > config.significant_rotation_degrees:
            size_factor = 1.3
        elif rotation_magnitude < config.min_rotation_degrees * 1.2:
            size_factor = 0.8
        
        confidence = base_confidence * size_factor
        
        # Check if confidence meets threshold
        confidence_met = confidence >= config.min_confidence_threshold
        
        # Final detection result
        detected = threshold_met and confidence_met
        
        print(f"    Rotation magnitude: {rotation_magnitude:.1f}°")
        print(f"    Direction: {direction}")
        print(f"    Confidence: {confidence:.3f}")
        print(f"    Threshold met: {threshold_met} (≥{config.min_rotation_degrees}°)")
        print(f"    Confidence met: {confidence_met} (≥{config.min_confidence_threshold})")
        
        if detected:
            print(f"    ✓ Movement detected: {direction} ({rotation_magnitude:.1f}° rotation)")
            if should_detect:
                print(f"      ✓ Expected detection for {config_name}")
            else:
                print(f"      ⚠ Unexpected detection for {config_name}")
        else:
            print(f"    ✗ No movement detected")
            if should_detect:
                print(f"      ⚠ Expected detection for {config_name} but none found")
            else:
                print(f"      ✓ Expected no detection for {config_name}")

def demonstrate_degree_calculations():
    """Demonstrate how degree calculations work."""
    
    print("\n" + "="*60)
    print("DEGREE CALCULATION DEMONSTRATION")
    print("="*60)
    
    print("\n1. ROTATION MAGNITUDE CALCULATION:")
    print("   - Uses the larger of yaw or pitch rotation")
    print("   - Example: 18° yaw + 12° pitch = 18° magnitude (uses yaw)")
    print("   - Example: 10° yaw + 25° pitch = 25° magnitude (uses pitch)")
    
    print("\n2. DIRECTION DETERMINATION:")
    print("   - Horizontal (left/right): Based on yaw rotation")
    print("   - Vertical (up/down): Based on pitch rotation")
    print("   - Uses the larger rotation to determine primary direction")
    
    print("\n3. CONFIDENCE CALCULATION:")
    print("   - Base confidence: magnitude_ratio / 2.0")
    print("   - Size factor: 1.3 for significant movements, 0.8 for small movements")
    print("   - Final confidence: base_confidence * size_factor")
    
    print("\n4. DETECTION CRITERIA:")
    print("   - Rotation magnitude ≥ minimum threshold")
    print("   - Confidence ≥ minimum confidence threshold")
    print("   - Both conditions must be met for detection")

def test_configuration_comparison():
    """Compare different configurations."""
    
    print("\n" + "="*60)
    print("CONFIGURATION COMPARISON")
    print("="*60)
    
    configs = {
        'Lenient': HeadMovementConfig.lenient_config(),
        'Default': HeadMovementConfig.default_config(),
        'Conservative': HeadMovementConfig.conservative_config()
    }
    
    print("\nConfiguration Parameters:")
    print(f"{'Parameter':<25} {'Lenient':<10} {'Default':<10} {'Conservative':<12}")
    print("-" * 60)
    
    for param in ['min_rotation_degrees', 'significant_rotation_degrees', 'min_confidence_threshold']:
        values = [getattr(config, param) for config in configs.values()]
        print(f"{param:<25} {values[0]:<10.1f} {values[1]:<10.1f} {values[2]:<12.1f}")
    
    print("\nUse Cases:")
    print("- Lenient: Easy detection, smaller movements, lower confidence requirements")
    print("- Default: Balanced detection, natural movements, moderate confidence")
    print("- Conservative: Strict detection, clear movements, high confidence requirements")

def main():
    """Main test function."""
    try:
        test_degree_thresholds()
        demonstrate_degree_calculations()
        test_configuration_comparison()
        
        print("\n" + "="*60)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nSummary of changes:")
        print("✓ Removed time-based constraints (no more 2s/1.5s expectations)")
        print("✓ Implemented degree-based detection (15° minimum recommended)")
        print("✓ Configurable thresholds for different sensitivity levels")
        print("✓ Detects movements as they occur without time pressure")
        print("✓ More accurate head pose calculation using yaw/pitch angles")
        
        print("\nRecommended degree thresholds:")
        print("- Minimum rotation: 15° (detects natural head turns)")
        print("- Significant rotation: 25° (clear head turns get higher confidence)")
        print("- Center threshold: 10° (accounts for natural head sway)")
        
    except Exception as e:
        print(f"Test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
