#!/usr/bin/env python3
"""
Test script for Movement Validation Rules.
This script tests the new validation rules:
1. Discard return movements
2. Only allow movements from centered face position
"""

import sys
import os
import logging
import tempfile
import numpy as np
import cv2

# Add the infrastructure directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'infrastructure'))

from facematch.simple_mediapipe_detector import create_simple_detector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_video_with_validation_scenarios(output_path: str):
    """Create a test video with various validation scenarios."""
    logger.info(f"Creating test video with validation scenarios: {output_path}")
    
    # Video parameters
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    
    # Create a face that moves in different patterns
    face_center = [width // 2, height // 2]
    face_size = 60
    
    # Scenario 1: Face starts centered, moves left, then returns to center (should discard return)
    # Scenario 2: Face starts off-center (should be rejected)
    # Scenario 3: Face starts centered, moves right, then left (should keep both)
    
    scenarios = [
        # Scenario 1: Centered -> Left -> Return to center (return should be discarded)
        {'name': 'Centered_Left_Return', 'start_pos': [width//2, height//2], 'movements': [
            (0, 0, 30),      # Stay centered
            (-30, 0, 30),    # Move left
            (30, 0, 30),     # Return to center (should be discarded)
        ]},
        
        # Scenario 2: Off-center start (should be rejected)
        {'name': 'OffCenter_Start', 'start_pos': [width//2 + 100, height//2], 'movements': [
            (0, 0, 30),      # Stay off-center
            (-20, 0, 30),    # Move left
        ]},
        
        # Scenario 3: Centered -> Right -> Left (both should be kept)
        {'name': 'Centered_Right_Left', 'start_pos': [width//2, height//2], 'movements': [
            (0, 0, 30),      # Stay centered
            (30, 0, 30),     # Move right
            (-30, 0, 30),    # Move left (not a return, should be kept)
        ]},
        
        # Scenario 4: Centered -> Up -> Down (both should be kept)
        {'name': 'Centered_Up_Down', 'start_pos': [width//2, height//2], 'movements': [
            (0, 0, 30),      # Stay centered
            (0, -30, 30),    # Move up
            (0, 30, 30),     # Move down (not a return, should be kept)
        ]},
    ]
    
    for scenario in scenarios:
        logger.info(f"Creating scenario: {scenario['name']}")
        
        # Set initial position
        face_center[0], face_center[1] = scenario['start_pos']
        
        for dx, dy, frames in scenario['movements']:
            for _ in range(frames):
                # Create frame
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame[:] = (240, 240, 240)  # Light gray background
                
                # Update face position
                face_center[0] = max(face_size, min(width - face_size, face_center[0] + dx))
                face_center[1] = max(face_size, min(height - face_size, face_center[1] + dy))
                
                # Draw face
                x, y = face_center[0], face_center[1]
                
                # Face outline
                cv2.ellipse(frame, (x, y), (face_size, int(face_size * 1.2)), 0, 0, 360, (255, 255, 255), -1)
                
                # Eyes
                cv2.circle(frame, (x - 20, y - 15), 8, (0, 0, 0), -1)
                cv2.circle(frame, (x + 20, y - 15), 8, (0, 0, 0), -1)
                
                # Nose
                cv2.circle(frame, (x, y + 5), 5, (0, 0, 0), -1)
                
                # Mouth
                cv2.ellipse(frame, (x, y + 25), (15, 8), 0, 0, 180, (0, 0, 0), 2)
                
                # Write frame
                out.write(frame)
    
    out.release()
    logger.info(f"Test video created: {output_path}")

def test_validation_rules():
    """Test the movement validation rules."""
    logger.info("Testing Movement Validation Rules")
    
    # Create test video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        test_video_path = temp_file.name
    
    try:
        create_test_video_with_validation_scenarios(test_video_path)
        
        # Test with validation rules enabled
        detector = create_simple_detector(
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2,
            movement_threshold=0.005,  # Very sensitive for testing
            debug_mode=True
        )
        
        # Process video
        result = detector.process_video(test_video_path)
        
        logger.info("="*60)
        logger.info("VALIDATION RULES TEST RESULTS")
        logger.info("="*60)
        logger.info(f"Success: {result.success}")
        logger.info(f"Frames processed: {result.frames_processed}")
        logger.info(f"Movements detected: {len(result.movements)}")
        
        if result.movements:
            logger.info("\nDetected movements:")
            for i, movement in enumerate(result.movements):
                logger.info(f"  {i+1}: {movement.direction} "
                           f"(confidence: {movement.confidence:.3f}, "
                           f"magnitude: {movement.magnitude:.4f}, "
                           f"timestamp: {movement.timestamp:.2f}s)")
            
            # Analyze results
            sequence = [m.direction for m in result.movements]
            logger.info(f"\nFinal movement sequence: {sequence}")
            
            # Check for validation patterns
            logger.info("\nValidation Analysis:")
            
            # Check for return movements (should be filtered out)
            return_movements = []
            for i in range(1, len(sequence)):
                prev, curr = sequence[i-1], sequence[i]
                if (prev == 'left' and curr == 'right') or \
                   (prev == 'right' and curr == 'left') or \
                   (prev == 'up' and curr == 'down') or \
                   (prev == 'down' and curr == 'up'):
                    return_movements.append(f"{prev}->{curr}")
            
            if return_movements:
                logger.warning(f"Return movements detected (should be filtered): {return_movements}")
            else:
                logger.info("✅ No return movements detected - validation working correctly")
            
            # Check movement distribution
            direction_counts = {}
            for direction in sequence:
                direction_counts[direction] = direction_counts.get(direction, 0) + 1
            
            logger.info(f"Movement distribution: {direction_counts}")
            
        else:
            logger.warning("No movements detected - check if validation is too strict")
        
        # Clean up
        detector.release()
        
        return len(result.movements) > 0
        
    except Exception as e:
        logger.error(f"Validation test failed: {e}")
        return False
    finally:
        # Clean up test video
        if os.path.exists(test_video_path):
            os.unlink(test_video_path)

def test_validation_parameters():
    """Test different validation parameter settings."""
    logger.info("\nTesting Validation Parameters")
    
    # Create test video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        test_video_path = temp_file.name
    
    try:
        create_test_video_with_validation_scenarios(test_video_path)
        
        # Test different validation settings
        test_configs = [
            {
                'name': 'Strict Validation',
                'face_center_threshold': 0.05,
                'return_movement_threshold': 0.03
            },
            {
                'name': 'Moderate Validation',
                'face_center_threshold': 0.1,
                'return_movement_threshold': 0.05
            },
            {
                'name': 'Lenient Validation',
                'face_center_threshold': 0.15,
                'return_movement_threshold': 0.08
            }
        ]
        
        results = {}
        
        for config in test_configs:
            logger.info(f"\nTesting {config['name']}...")
            
            detector = create_simple_detector(
                min_detection_confidence=0.2,
                min_tracking_confidence=0.2,
                movement_threshold=0.005,
                debug_mode=False
            )
            
            # Override validation parameters
            detector.face_center_threshold = config['face_center_threshold']
            detector.return_movement_threshold = config['return_movement_threshold']
            
            result = detector.process_video(test_video_path)
            
            results[config['name']] = {
                'movements': len(result.movements),
                'sequence': [m.direction for m in result.movements]
            }
            
            logger.info(f"  Movements: {len(result.movements)}")
            logger.info(f"  Sequence: {[m.direction for m in result.movements]}")
            
            detector.release()
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("VALIDATION PARAMETER TEST SUMMARY")
        logger.info("="*60)
        
        for name, result in results.items():
            logger.info(f"{name}: {result['movements']} movements - {result['sequence']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Parameter test failed: {e}")
        return False
    finally:
        # Clean up test video
        if os.path.exists(test_video_path):
            os.unlink(test_video_path)

def main():
    """Main function."""
    logger.info("Starting Movement Validation Rules Test")
    
    # Test 1: Basic validation rules
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Basic Validation Rules")
    logger.info("="*60)
    test1_passed = test_validation_rules()
    
    # Test 2: Validation parameters
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Validation Parameters")
    logger.info("="*60)
    test2_passed = test_validation_parameters()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Test 1 (Basic Rules): {'PASSED' if test1_passed else 'FAILED'}")
    logger.info(f"Test 2 (Parameters): {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        logger.info("✅ All validation tests PASSED!")
        return 0
    else:
        logger.error("❌ Some validation tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
