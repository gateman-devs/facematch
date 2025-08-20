#!/usr/bin/env python3
"""
Test script for dlib integration with the liveness detection system.
Tests the unified liveness detector and optimized video processor with dlib.
"""

import logging
import sys
import os
import urllib.request
import tempfile
import ssl

# Add the infrastructure directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'infrastructure'))

from facematch.unified_liveness_detector import get_unified_liveness_detector, LivenessMode
from facematch.optimized_video_processor import create_optimized_processor

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

def test_unified_liveness_detector(video_path: str):
    """Test the unified liveness detector with dlib."""
    logger.info("Testing Unified Liveness Detector with dlib")
    
    # Expected movements from the video
    expected_movements = ["Left", "Left", "Right", "Right", "Up", "Up", "Down", "Down", "Left", "Right", "Up", "Down"]
    
    try:
        # Get unified liveness detector
        detector = get_unified_liveness_detector()
        
        # Check available modes
        detector_status = detector.get_detector_status()
        logger.info(f"Available detector modes: {detector_status}")
        
        # Test video movement detection
        if LivenessMode.VIDEO_MOVEMENT in detector._detectors:
            logger.info("Testing video movement detection with dlib")
            
            result = detector.detect_liveness(
                video_path, 
                mode=LivenessMode.VIDEO_MOVEMENT,
                expected_sequence=expected_movements
            )
            
            logger.info(f"Detection result: success={result.success}, passed={result.passed}")
            logger.info(f"Confidence: {result.confidence:.2f}, Liveness score: {result.liveness_score:.2f}")
            logger.info(f"Processing time: {result.processing_time:.2f}s")
            
            if hasattr(result, 'detected_sequence') and result.detected_sequence:
                logger.info(f"Detected sequence: {result.detected_sequence}")
                logger.info(f"Expected sequence: {expected_movements}")
                if hasattr(result, 'sequence_accuracy') and result.sequence_accuracy is not None:
                    logger.info(f"Sequence accuracy: {result.sequence_accuracy:.2%}")
            
            if result.movement_details:
                logger.info(f"Movement details: {result.movement_details}")
            
            return result
        else:
            logger.error("Video movement mode not available in unified detector")
            return None
            
    except Exception as e:
        logger.error(f"Unified liveness detector test failed: {e}")
        return None

def test_optimized_video_processor(video_path: str):
    """Test the optimized video processor with dlib."""
    logger.info("Testing Optimized Video Processor with dlib")
    
    # Expected movements from the video
    expected_movements = ["Left", "Left", "Right", "Right", "Up", "Up", "Down", "Down", "Left", "Right", "Up", "Down"]
    
    try:
        # Create optimized processor with dlib
        processor = create_optimized_processor(use_dlib=True)
        
        # Process video
        result = processor.process_video_for_liveness(video_path, expected_movements)
        
        logger.info(f"Processing result: success={result.success}")
        logger.info(f"Processing time: {result.processing_time:.2f}s")
        logger.info(f"Frames processed: {result.frames_processed}")
        logger.info(f"Movements detected: {len(result.movements)}")
        
        if result.optimization_applied:
            logger.info(f"Optimizations applied: {result.optimization_applied}")
        
        if result.movements:
            movements = [m['direction'] for m in result.movements]
            logger.info(f"Detected movements: {movements}")
            logger.info(f"Expected movements: {expected_movements}")
        
        if result.validation_result:
            logger.info(f"Validation result: {result.validation_result}")
        
        if result.performance_metrics:
            logger.info(f"Performance metrics: {result.performance_metrics}")
        
        return result
        
    except Exception as e:
        logger.error(f"Optimized video processor test failed: {e}")
        return None

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
        
        logger.info("=" * 60)
        logger.info("TESTING DLIB INTEGRATION WITH LIVENESS DETECTION SYSTEM")
        logger.info("=" * 60)
        
        # Test 1: Unified Liveness Detector
        logger.info("\n" + "=" * 40)
        logger.info("TEST 1: Unified Liveness Detector")
        logger.info("=" * 40)
        
        unified_result = test_unified_liveness_detector(video_path)
        
        # Test 2: Optimized Video Processor
        logger.info("\n" + "=" * 40)
        logger.info("TEST 2: Optimized Video Processor")
        logger.info("=" * 40)
        
        processor_result = test_optimized_video_processor(video_path)
        
        # Summary
        logger.info("\n" + "=" * 40)
        logger.info("INTEGRATION TEST SUMMARY")
        logger.info("=" * 40)
        
        unified_success = unified_result is not None and unified_result.success
        processor_success = processor_result is not None and processor_result.success
        
        logger.info(f"Unified Liveness Detector: {'PASS' if unified_success else 'FAIL'}")
        logger.info(f"Optimized Video Processor: {'PASS' if processor_success else 'FAIL'}")
        logger.info(f"Overall Integration: {'PASS' if unified_success and processor_success else 'FAIL'}")
        
        if unified_success and processor_success:
            logger.info("\n✅ Dlib integration completed successfully!")
            logger.info("The system can now use dlib for head movement detection.")
        else:
            logger.error("\n❌ Integration test failed!")
            logger.error("Please check the errors above and fix any issues.")
            
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
