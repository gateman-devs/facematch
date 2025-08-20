#!/usr/bin/env python3
"""
Test script for dlib integration with the liveness detection system (local test).
Tests the system components without requiring video download.
"""

import logging
import sys
import os

# Add the infrastructure directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'infrastructure'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_dlib_detector_import():
    """Test if dlib detector can be imported and initialized."""
    logger.info("Testing dlib detector import and initialization")
    
    try:
        from facematch.dlib_head_detector import DlibHeadDetector, create_dlib_detector
        
        # Test creation
        detector = create_dlib_detector({
            'min_rotation_degrees': 15.0,
            'significant_rotation_degrees': 25.0,
            'debug_mode': True
        })
        
        logger.info("✅ Dlib detector created successfully")
        logger.info(f"Detector type: {type(detector)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to import/create dlib detector: {e}")
        return False

def test_unified_liveness_detector_initialization():
    """Test if unified liveness detector can initialize with dlib."""
    logger.info("Testing unified liveness detector initialization")
    
    try:
        from facematch.unified_liveness_detector import get_unified_liveness_detector, LivenessMode
        
        # Get detector
        detector = get_unified_liveness_detector()
        
        # Check available modes
        detector_status = detector.get_detector_status()
        logger.info(f"Available detector modes: {detector_status}")
        
        # Check if video movement mode is available
        if LivenessMode.VIDEO_MOVEMENT in detector._detectors:
            logger.info("✅ Video movement mode available with dlib detector")
            
            # Get the specific detector
            video_detector = detector._detectors[LivenessMode.VIDEO_MOVEMENT]
            logger.info(f"Video detector type: {type(video_detector)}")
            logger.info(f"Video detector available: {video_detector.is_available()}")
            
            return True
        else:
            logger.error("❌ Video movement mode not available")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize unified liveness detector: {e}")
        return False

def test_optimized_video_processor_initialization():
    """Test if optimized video processor can initialize with dlib."""
    logger.info("Testing optimized video processor initialization")
    
    try:
        from facematch.optimized_video_processor import create_optimized_processor
        
        # Create processor with dlib
        processor = create_optimized_processor(use_dlib=True)
        
        logger.info("✅ Optimized video processor created successfully")
        logger.info(f"Processor type: {type(processor)}")
        logger.info(f"Using dlib: {processor.use_dlib}")
        logger.info(f"Dlib detector available: {processor.dlib_detector is not None}")
        
        if processor.dlib_detector:
            logger.info(f"Dlib detector type: {type(processor.dlib_detector)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create optimized video processor: {e}")
        return False

def test_server_integration():
    """Test if the server components can be imported."""
    logger.info("Testing server integration compatibility")
    
    try:
        # Import server module to check for compatibility
        from facematch import server
        logger.info("✅ Server module imported successfully")
        
        # Check if unified detector can be retrieved
        from facematch.server import get_unified_liveness_detector
        detector = get_unified_liveness_detector()
        
        if detector:
            logger.info("✅ Unified detector available in server context")
            detector_status = detector.get_detector_status()
            logger.info(f"Server detector status: {detector_status}")
            return True
        else:
            logger.error("❌ Unified detector not available in server context")
            return False
            
    except Exception as e:
        logger.error(f"❌ Server integration test failed: {e}")
        return False

def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("TESTING DLIB INTEGRATION (LOCAL TESTS)")
    logger.info("=" * 60)
    
    tests = [
        ("Dlib Detector Import", test_dlib_detector_import),
        ("Unified Liveness Detector", test_unified_liveness_detector_initialization),
        ("Optimized Video Processor", test_optimized_video_processor_initialization),
        ("Server Integration", test_server_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*40}")
        logger.info(f"TEST: {test_name}")
        logger.info(f"{'='*40}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test '{test_name}' failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 40)
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info("=" * 40)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n✅ All integration tests passed!")
        logger.info("Dlib integration is ready for use.")
    else:
        logger.error(f"\n❌ {total - passed} test(s) failed!")
        logger.error("Please fix the issues before using dlib integration.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
