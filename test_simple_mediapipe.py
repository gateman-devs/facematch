#!/usr/bin/env python3
"""
Test script for Simple MediaPipe Head Movement Detection.
This script tests the simplified MediaPipe implementation.
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

def create_realistic_test_video(output_path: str, duration: int = 5, fps: int = 30):
    """Create a more realistic test video with face-like features."""
    logger.info(f"Creating realistic test video: {output_path}")
    
    # Video parameters
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create a more realistic face-like object
    face_center = [width // 2, height // 2]
    face_size = 80
    
    # Movement pattern (more realistic head movements)
    movement_pattern = [
        (0, -15),   # Up
        (20, 0),    # Right
        (0, 15),    # Down
        (-20, 0),   # Left
        (0, -15),   # Up
    ]
    
    frame_count = 0
    pattern_index = 0
    
    for _ in range(duration * fps):
        # Create frame with realistic face
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (240, 240, 240)  # Light gray background
        
        # Calculate face position
        movement = movement_pattern[pattern_index % len(movement_pattern)]
        face_center[0] = max(face_size, min(width - face_size, face_center[0] + movement[0]))
        face_center[1] = max(face_size, min(height - face_size, face_center[1] + movement[1]))
        
        # Draw realistic face features
        x, y = face_center[0], face_center[1]
        
        # Face outline (oval)
        cv2.ellipse(frame, (x, y), (face_size, int(face_size * 1.2)), 0, 0, 360, (255, 255, 255), -1)
        
        # Eyes
        cv2.circle(frame, (x - 20, y - 15), 8, (0, 0, 0), -1)  # Left eye
        cv2.circle(frame, (x + 20, y - 15), 8, (0, 0, 0), -1)  # Right eye
        
        # Nose
        cv2.circle(frame, (x, y + 5), 5, (0, 0, 0), -1)  # Nose tip
        
        # Mouth
        cv2.ellipse(frame, (x, y + 25), (15, 8), 0, 0, 180, (0, 0, 0), 2)  # Mouth
        
        # Ears
        cv2.circle(frame, (x - face_size + 10, y), 12, (255, 255, 255), -1)  # Left ear
        cv2.circle(frame, (x + face_size - 10, y), 12, (255, 255, 255), -1)  # Right ear
        
        # Write frame
        out.write(frame)
        
        # Change movement pattern every second
        if frame_count % fps == 0:
            pattern_index += 1
        
        frame_count += 1
    
    out.release()
    logger.info(f"Realistic test video created: {output_path}")

def test_simple_mediapipe_detector():
    """Test the simple MediaPipe detector."""
    logger.info("Testing Simple MediaPipe detector...")
    
    # Create test video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        test_video_path = temp_file.name
    
    try:
        create_realistic_test_video(test_video_path)
        
        # Create detector with custom parameters
        detector = create_simple_detector(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            movement_threshold=0.01,  # Lower threshold for testing
            max_history=20
        )
        
        # Test with expected sequence
        expected_sequence = ['up', 'right', 'down', 'left', 'up']
        
        # Process video
        result = detector.process_video(
            video_path=test_video_path,
            expected_sequence=expected_sequence
        )
        
        logger.info(f"Simple MediaPipe detection result:")
        logger.info(f"  Success: {result.success}")
        logger.info(f"  Frames processed: {result.frames_processed}")
        logger.info(f"  Processing time: {result.processing_time:.3f}s")
        logger.info(f"  Movements detected: {len(result.movements)}")
        
        if result.movements:
            logger.info("  Detected movements:")
            for i, movement in enumerate(result.movements):
                logger.info(f"    {i+1}: {movement.direction} "
                           f"(confidence: {movement.confidence:.3f}, "
                           f"magnitude: {movement.magnitude:.4f})")
            
            # Test sequence validation
            validation = detector.validate_sequence(expected_sequence)
            logger.info(f"  Sequence validation:")
            logger.info(f"    Accuracy: {validation['accuracy']:.3f}")
            logger.info(f"    Expected: {validation['expected_sequence']}")
            logger.info(f"    Detected: {validation['detected_sequence']}")
            logger.info(f"    Correct movements: {validation['correct_movements']}/{validation['total_expected']}")
        
        # Clean up
        detector.release()
        
        return result.success
        
    except Exception as e:
        logger.error(f"Simple MediaPipe detector test failed: {e}")
        return False
    finally:
        # Clean up test video
        if os.path.exists(test_video_path):
            os.unlink(test_video_path)

def test_detector_parameters():
    """Test different detector parameters."""
    logger.info("Testing detector parameters...")
    
    # Create test video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        test_video_path = temp_file.name
    
    try:
        create_realistic_test_video(test_video_path)
        
        # Test different parameter combinations
        test_configs = [
            {
                'name': 'High Sensitivity',
                'params': {
                    'min_detection_confidence': 0.2,
                    'min_tracking_confidence': 0.2,
                    'movement_threshold': 0.005
                }
            },
            {
                'name': 'Standard',
                'params': {
                    'min_detection_confidence': 0.5,
                    'min_tracking_confidence': 0.5,
                    'movement_threshold': 0.02
                }
            },
            {
                'name': 'Low Sensitivity',
                'params': {
                    'min_detection_confidence': 0.7,
                    'min_tracking_confidence': 0.7,
                    'movement_threshold': 0.05
                }
            }
        ]
        
        results = {}
        
        for config in test_configs:
            logger.info(f"Testing {config['name']} configuration...")
            
            detector = create_simple_detector(**config['params'])
            
            result = detector.process_video(test_video_path)
            
            results[config['name']] = {
                'success': result.success,
                'movements': len(result.movements),
                'processing_time': result.processing_time,
                'frames_processed': result.frames_processed
            }
            
            detector.release()
        
        # Report results
        logger.info("Parameter test results:")
        for name, result in results.items():
            logger.info(f"  {name}:")
            logger.info(f"    Success: {result['success']}")
            logger.info(f"    Movements: {result['movements']}")
            logger.info(f"    Processing time: {result['processing_time']:.3f}s")
            logger.info(f"    Frames processed: {result['frames_processed']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Parameter test failed: {e}")
        return False
    finally:
        # Clean up test video
        if os.path.exists(test_video_path):
            os.unlink(test_video_path)

def main():
    """Run all tests."""
    logger.info("Starting Simple MediaPipe detector tests...")
    
    # Test 1: Basic functionality
    logger.info("\n" + "="*50)
    logger.info("TEST 1: Basic Functionality")
    logger.info("="*50)
    test1_passed = test_simple_mediapipe_detector()
    
    # Test 2: Parameter variations
    logger.info("\n" + "="*50)
    logger.info("TEST 2: Parameter Variations")
    logger.info("="*50)
    test2_passed = test_detector_parameters()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    logger.info(f"Test 1 (Basic Functionality): {'PASSED' if test1_passed else 'FAILED'}")
    logger.info(f"Test 2 (Parameter Variations): {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        logger.info("All tests PASSED! Simple MediaPipe detector is working correctly.")
        return 0
    else:
        logger.error("Some tests FAILED! Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
