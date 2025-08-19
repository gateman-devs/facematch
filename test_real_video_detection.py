#!/usr/bin/env python3
"""
Test script for Real Video Head Movement Detection.
This script tests the enhanced MediaPipe detector with real video files.
"""

import sys
import os
import logging
import tempfile
import shutil

# Add the infrastructure directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'infrastructure'))

from facematch.simple_mediapipe_detector import create_simple_detector

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_video_file(video_path: str, expected_sequence: list = None):
    """Test a real video file with the enhanced detector."""
    logger.info(f"Testing video file: {video_path}")
    
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return False
    
    try:
        # Create detector with debug mode and sensitive settings
        detector = create_simple_detector(
            min_detection_confidence=0.2,  # Very low for testing
            min_tracking_confidence=0.2,
            movement_threshold=0.005,  # Very sensitive
            debug_mode=True
        )
        
        # Process video
        result = detector.process_video(
            video_path=video_path,
            expected_sequence=expected_sequence
        )
        
        logger.info("="*60)
        logger.info("VIDEO PROCESSING RESULTS")
        logger.info("="*60)
        logger.info(f"Success: {result.success}")
        logger.info(f"Frames processed: {result.frames_processed}")
        logger.info(f"Processing time: {result.processing_time:.3f}s")
        logger.info(f"Movements detected: {len(result.movements)}")
        
        if result.movements:
            logger.info("\nDetected movements:")
            for i, movement in enumerate(result.movements):
                logger.info(f"  {i+1}: {movement.direction} "
                           f"(confidence: {movement.confidence:.3f}, "
                           f"magnitude: {movement.magnitude:.4f}, "
                           f"timestamp: {movement.timestamp:.2f}s)")
            
            # Test sequence validation if expected sequence provided
            if expected_sequence:
                validation = detector.validate_sequence(expected_sequence)
                logger.info(f"\nSequence validation:")
                logger.info(f"  Accuracy: {validation['accuracy']:.3f}")
                logger.info(f"  Expected: {validation['expected_sequence']}")
                logger.info(f"  Detected: {validation['detected_sequence']}")
                logger.info(f"  Correct movements: {validation['correct_movements']}/{validation['total_expected']}")
        else:
            logger.warning("No movements detected!")
            logger.info("Possible causes:")
            logger.info("  1. Movement threshold too high")
            logger.info("  2. Face not clearly visible")
            logger.info("  3. Video quality issues")
            logger.info("  4. No actual head movements in video")
        
        # Clean up
        detector.release()
        
        return result.success and len(result.movements) > 0
        
    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        return False

def test_with_sample_video():
    """Test with a sample video if available."""
    # Look for common video files in the current directory
    video_extensions = ['.mp4', '.webm', '.avi', '.mov', '.mkv']
    sample_videos = []
    
    for ext in video_extensions:
        for file in os.listdir('.'):
            if file.endswith(ext):
                sample_videos.append(file)
    
    if not sample_videos:
        logger.info("No sample video files found in current directory.")
        logger.info("Please provide a video file path as an argument.")
        return False
    
    logger.info(f"Found sample videos: {sample_videos}")
    
    # Test with the first video found
    video_path = sample_videos[0]
    expected_sequence = ['up', 'right', 'down', 'left']  # Common test sequence
    
    return test_video_file(video_path, expected_sequence)

def main():
    """Main function."""
    logger.info("Starting Real Video Head Movement Detection Test")
    
    if len(sys.argv) > 1:
        # Test with provided video file
        video_path = sys.argv[1]
        expected_sequence = sys.argv[2:] if len(sys.argv) > 2 else None
        
        success = test_video_file(video_path, expected_sequence)
    else:
        # Test with sample video
        success = test_with_sample_video()
    
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    if success:
        logger.info("✅ Test PASSED - Movements detected successfully!")
        return 0
    else:
        logger.error("❌ Test FAILED - No movements detected or processing failed")
        logger.info("\nTroubleshooting tips:")
        logger.info("1. Ensure the video contains clear head movements")
        logger.info("2. Check that the face is clearly visible")
        logger.info("3. Try with a higher quality video")
        logger.info("4. Verify the video format is supported")
        return 1

if __name__ == "__main__":
    sys.exit(main())
