#!/usr/bin/env python3
"""
Test script for MediaPipe head movement detection integration.
This script tests the new MediaPipe-based head movement detection.
"""

import sys
import os
import logging
import tempfile
import numpy as np
import cv2

# Add the infrastructure directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'infrastructure'))

from facematch.mediapipe_head_detector import create_mediapipe_detector, MediaPipeConfig
from facematch.optimized_video_processor import create_optimized_processor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_video(output_path: str, duration: int = 5, fps: int = 30):
    """Create a test video with simulated head movements."""
    logger.info(f"Creating test video: {output_path}")
    
    # Video parameters
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create a simple face-like object that moves
    face_center = [width // 2, height // 2]
    movement_pattern = [
        (0, -20),   # Up
        (20, 0),    # Right
        (0, 20),    # Down
        (-20, 0),   # Left
        (0, -20),   # Up
    ]
    
    frame_count = 0
    pattern_index = 0
    
    for _ in range(duration * fps):
        # Create frame with moving face
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (128, 128, 128)  # Gray background
        
        # Calculate face position
        movement = movement_pattern[pattern_index % len(movement_pattern)]
        face_center[0] = max(50, min(width - 50, face_center[0] + movement[0]))
        face_center[1] = max(50, min(height - 50, face_center[1] + movement[1]))
        
        # Draw face (simple circle)
        cv2.circle(frame, (face_center[0], face_center[1]), 30, (255, 255, 255), -1)
        cv2.circle(frame, (face_center[0] - 10, face_center[1] - 10), 5, (0, 0, 0), -1)  # Left eye
        cv2.circle(frame, (face_center[0] + 10, face_center[1] - 10), 5, (0, 0, 0), -1)  # Right eye
        cv2.circle(frame, (face_center[0], face_center[1] + 10), 3, (0, 0, 0), -1)      # Nose
        
        # Write frame
        out.write(frame)
        
        # Change movement pattern every second
        if frame_count % fps == 0:
            pattern_index += 1
        
        frame_count += 1
    
    out.release()
    logger.info(f"Test video created: {output_path}")

def test_mediapipe_detector():
    """Test the MediaPipe head detector directly."""
    logger.info("Testing MediaPipe head detector...")
    
    # Create test video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        test_video_path = temp_file.name
    
    try:
        create_test_video(test_video_path)
        
        # Test MediaPipe detector
        config = MediaPipeConfig(
            pose_confidence_threshold=0.3,
            face_mesh_confidence_threshold=0.3,
            min_movement_threshold=5.0,
            enable_optimization=False  # Disable optimization for testing
        )
        
        detector = create_mediapipe_detector(config)
        
        # Test with expected sequence
        expected_sequence = ['up', 'right', 'down', 'left', 'up']
        
        result = detector.detect_head_movements(
            video_path=test_video_path,
            expected_sequence=expected_sequence
        )
        
        logger.info(f"MediaPipe detection result:")
        logger.info(f"  Success: {result.success}")
        logger.info(f"  Frames processed: {result.frames_processed}")
        logger.info(f"  Processing time: {result.processing_time:.3f}s")
        logger.info(f"  Movements detected: {len(result.movements)}")
        logger.info(f"  Quality metrics: {result.quality_metrics}")
        
        if result.movements:
            logger.info("  Detected movements:")
            for i, movement in enumerate(result.movements):
                logger.info(f"    {i+1}: {movement.direction} (confidence: {movement.confidence:.3f})")
        
        # Clean up
        detector.release()
        
        return result.success
        
    except Exception as e:
        logger.error(f"MediaPipe detector test failed: {e}")
        return False
    finally:
        # Clean up test video
        if os.path.exists(test_video_path):
            os.unlink(test_video_path)

def test_optimized_processor():
    """Test the optimized video processor with MediaPipe."""
    logger.info("Testing optimized video processor with MediaPipe...")
    
    # Create test video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        test_video_path = temp_file.name
    
    try:
        create_test_video(test_video_path)
        
        # Test optimized processor with MediaPipe
        processor = create_optimized_processor(use_mediapipe=True)
        
        expected_sequence = ['up', 'right', 'down', 'left', 'up']
        
        result = processor.process_video_for_liveness(
            video_path=test_video_path,
            expected_sequence=expected_sequence
        )
        
        logger.info(f"Optimized processor result:")
        logger.info(f"  Success: {result.success}")
        logger.info(f"  Frames processed: {result.frames_processed}")
        logger.info(f"  Processing time: {result.processing_time:.3f}s")
        logger.info(f"  Movements detected: {len(result.movements)}")
        logger.info(f"  Optimization applied: {result.optimization_applied}")
        
        if result.movements:
            logger.info("  Detected movements:")
            for i, movement in enumerate(result.movements):
                logger.info(f"    {i+1}: {movement['direction']} (confidence: {movement['confidence']:.3f})")
        
        return result.success
        
    except Exception as e:
        logger.error(f"Optimized processor test failed: {e}")
        return False
    finally:
        # Clean up test video
        if os.path.exists(test_video_path):
            os.unlink(test_video_path)

def main():
    """Run all tests."""
    logger.info("Starting MediaPipe integration tests...")
    
    # Test 1: Direct MediaPipe detector
    logger.info("\n" + "="*50)
    logger.info("TEST 1: Direct MediaPipe Detector")
    logger.info("="*50)
    test1_passed = test_mediapipe_detector()
    
    # Test 2: Optimized processor with MediaPipe
    logger.info("\n" + "="*50)
    logger.info("TEST 2: Optimized Processor with MediaPipe")
    logger.info("="*50)
    test2_passed = test_optimized_processor()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    logger.info(f"Test 1 (Direct MediaPipe): {'PASSED' if test1_passed else 'FAILED'}")
    logger.info(f"Test 2 (Optimized Processor): {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        logger.info("All tests PASSED! MediaPipe integration is working correctly.")
        return 0
    else:
        logger.error("Some tests FAILED! Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
