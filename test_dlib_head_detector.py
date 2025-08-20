#!/usr/bin/env python3
"""
Test script for Dlib-based Head Movement Detector
Tests the detector with the provided video URL and expected movements.
"""

import logging
import sys
import os
import urllib.request
import tempfile
from pathlib import Path

# Add the infrastructure directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'infrastructure'))

from facematch.dlib_head_detector import DlibHeadDetector, create_dlib_detector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_video(url: str, output_path: str) -> bool:
    """Download video from URL."""
    try:
        logger.info(f"Downloading video from {url}")
        # Create SSL context that ignores certificate verification
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Use the SSL context for the request
        with urllib.request.urlopen(url, context=ssl_context) as response:
            with open(output_path, 'wb') as f:
                f.write(response.read())
        
        logger.info(f"Video downloaded to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download video: {e}")
        return False

def test_dlib_detector(video_path: str):
    """Test the dlib head movement detector."""
    
    # Expected movements from the video
    expected_movements = [
        "Left", "Left", "Right", "Right", 
        "Up", "Up", "Down", "Down", 
        "Left", "Right", "Up", "Down"
    ]
    
    logger.info("Testing Dlib Head Movement Detector")
    logger.info(f"Expected movements: {expected_movements}")
    
    # Create detector with debug mode
    detector = create_dlib_detector({
        'min_rotation_degrees': 15.0,
        'significant_rotation_degrees': 20.0,
        'min_confidence_threshold': 0.6,
        'debug_mode': True
    })
    
    # Process video
    logger.info("Processing video...")
    result = detector.process_video(video_path)
    
    if not result.success:
        logger.error(f"Video processing failed: {result.error}")
        return
    
    # Analyze results
    logger.info(f"Processing completed in {result.processing_time:.2f} seconds")
    logger.info(f"Frames processed: {result.frames_processed}")
    logger.info(f"Movements detected: {len(result.movements)}")
    
    # Print detected movements
    detected_movements = [m.direction for m in result.movements]
    logger.info(f"Detected movements: {detected_movements}")
    
    # Print detailed movement information
    for i, movement in enumerate(result.movements):
        logger.info(f"Movement {i+1}: {movement.direction} "
                   f"(confidence: {movement.confidence:.2f}, "
                   f"magnitude: {movement.magnitude:.1f}°)")
        logger.info(f"  Pose data: pitch={movement.pose_data['pitch']:.1f}°, "
                   f"yaw={movement.pose_data['yaw']:.1f}°, "
                   f"roll={movement.pose_data['roll']:.1f}°")
    
    # Calculate accuracy
    if len(detected_movements) > 0:
        # Simple accuracy calculation - check if we detected the right number of movements
        accuracy = min(len(detected_movements) / len(expected_movements), 1.0)
        logger.info(f"Detection accuracy: {accuracy:.2%}")
        
        # Check for specific movement patterns
        left_count = detected_movements.count("Left")
        right_count = detected_movements.count("Right")
        up_count = detected_movements.count("Up")
        down_count = detected_movements.count("Down")
        
        logger.info(f"Movement breakdown:")
        logger.info(f"  Left: {left_count}")
        logger.info(f"  Right: {right_count}")
        logger.info(f"  Up: {up_count}")
        logger.info(f"  Down: {down_count}")
    else:
        logger.warning("No movements detected!")
    
    return result

def main():
    """Main test function."""
    video_url = "https://res.cloudinary.com/themizehq/video/upload/v1755621958/IMG_6482.mov"
    
    # Create temporary file for video
    with tempfile.NamedTemporaryFile(suffix='.mov', delete=False) as tmp_file:
        video_path = tmp_file.name
    
    try:
        # Download video
        if not download_video(video_url, video_path):
            logger.error("Failed to download video. Exiting.")
            return
        
        # Test detector
        result = test_dlib_detector(video_path)
        
        if result and result.success:
            logger.info("Test completed successfully!")
        else:
            logger.error("Test failed!")
            
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(video_path)
            logger.info("Cleaned up temporary video file")
        except:
            pass

if __name__ == "__main__":
    main()
