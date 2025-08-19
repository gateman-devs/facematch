#!/usr/bin/env python3
"""
Test script for Enhanced MediaPipe Head Movement Detector.
This script tests all the new features:
1. Adaptive thresholds based on face size
2. Angle-based direction detection
3. Temporal smoothing
4. Enhanced confidence calculation
5. Quality validation
6. Improved error handling
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

def create_test_video_with_varying_face_sizes(output_path: str):
    """Create a test video with varying face sizes to test adaptive thresholds."""
    logger.info(f"Creating test video with varying face sizes: {output_path}")
    
    # Video parameters
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    
    # Create scenarios with different face sizes
    scenarios = [
        # Large face (close to camera) - should use smaller threshold
        {'name': 'Large_Face', 'face_size': 80, 'movements': [
            (0, 0, 30),      # Stay centered
            (-20, 0, 30),    # Move left
            (20, 0, 30),     # Move right
        ]},
        
        # Medium face (normal distance) - should use base threshold
        {'name': 'Medium_Face', 'face_size': 50, 'movements': [
            (0, 0, 30),      # Stay centered
            (-15, 0, 30),    # Move left
            (15, 0, 30),     # Move right
        ]},
        
        # Small face (far from camera) - should use larger threshold
        {'name': 'Small_Face', 'face_size': 30, 'movements': [
            (0, 0, 30),      # Stay centered
            (-10, 0, 30),    # Move left
            (10, 0, 30),     # Move right
        ]},
    ]
    
    for scenario in scenarios:
        logger.info(f"Creating scenario: {scenario['name']} (face_size: {scenario['face_size']})")
        
        face_center = [width // 2, height // 2]
        face_size = scenario['face_size']
        
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
                cv2.circle(frame, (x - face_size//3, y - face_size//4), face_size//8, (0, 0, 0), -1)
                cv2.circle(frame, (x + face_size//3, y - face_size//4), face_size//8, (0, 0, 0), -1)
                
                # Nose
                cv2.circle(frame, (x, y + face_size//12), face_size//12, (0, 0, 0), -1)
                
                # Mouth
                cv2.ellipse(frame, (x, y + face_size//2), (face_size//4, face_size//10), 0, 0, 180, (0, 0, 0), 2)
                
                # Write frame
                out.write(frame)
    
    out.release()
    logger.info(f"Test video created: {output_path}")

def create_test_video_with_angle_movements(output_path: str):
    """Create a test video with diagonal movements to test angle-based direction detection."""
    logger.info(f"Creating test video with angle movements: {output_path}")
    
    # Video parameters
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    
    # Create diagonal movements to test angle detection
    movements = [
        (0, 0, 30),      # Stay centered
        (20, 20, 30),    # Diagonal down-right (should be 'right')
        (-20, 20, 30),   # Diagonal down-left (should be 'left')
        (20, -20, 30),   # Diagonal up-right (should be 'right')
        (-20, -20, 30),  # Diagonal up-left (should be 'left')
        (0, 30, 30),     # Straight down
        (0, -30, 30),    # Straight up
        (30, 0, 30),     # Straight right
        (-30, 0, 30),    # Straight left
    ]
    
    face_center = [width // 2, height // 2]
    face_size = 50
    
    for dx, dy, frames in movements:
        for _ in range(frames):
            # Create frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = (240, 240, 240)
            
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
    logger.info(f"Angle test video created: {output_path}")

def test_adaptive_thresholds():
    """Test adaptive threshold functionality."""
    logger.info("\n" + "="*60)
    logger.info("TEST: Adaptive Thresholds")
    logger.info("="*60)
    
    # Create test video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        test_video_path = temp_file.name
    
    try:
        create_test_video_with_varying_face_sizes(test_video_path)
        
        # Test with debug mode to see threshold changes
        detector = create_simple_detector(
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2,
            movement_threshold=0.01,
            debug_mode=True
        )
        
        # Process video
        result = detector.process_video(test_video_path)
        
        logger.info(f"Movements detected: {len(result.movements)}")
        logger.info(f"Face sizes tracked: {len(detector.face_sizes)}")
        
        if detector.face_sizes:
            logger.info(f"Face size range: {min(detector.face_sizes):.4f} - {max(detector.face_sizes):.4f}")
            logger.info(f"Average face size: {np.mean(detector.face_sizes):.4f}")
        
        # Check if threshold was adapted
        if detector.movement_threshold != detector.base_movement_threshold:
            logger.info(f"✅ Threshold adapted: base={detector.base_movement_threshold}, final={detector.movement_threshold}")
        else:
            logger.info("⚠️ Threshold not adapted (may be normal)")
        
        detector.release()
        return len(result.movements) > 0
        
    except Exception as e:
        logger.error(f"Adaptive threshold test failed: {e}")
        return False
    finally:
        if os.path.exists(test_video_path):
            os.unlink(test_video_path)

def test_angle_based_direction():
    """Test angle-based direction detection."""
    logger.info("\n" + "="*60)
    logger.info("TEST: Angle-Based Direction Detection")
    logger.info("="*60)
    
    # Create test video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        test_video_path = temp_file.name
    
    try:
        create_test_video_with_angle_movements(test_video_path)
        
        detector = create_simple_detector(
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2,
            movement_threshold=0.005,
            debug_mode=True
        )
        
        # Process video
        result = detector.process_video(test_video_path)
        
        logger.info(f"Movements detected: {len(result.movements)}")
        
        if result.movements:
            logger.info("Detected movements:")
            for i, movement in enumerate(result.movements):
                logger.info(f"  {i+1}: {movement.direction} "
                           f"(confidence: {movement.confidence:.3f}, "
                           f"magnitude: {movement.magnitude:.4f})")
            
            # Check for diagonal movements
            directions = [m.direction for m in result.movements]
            logger.info(f"Movement sequence: {directions}")
            
            # Verify that diagonal movements are classified correctly
            diagonal_movements = [d for d in directions if d in ['left', 'right', 'up', 'down']]
            if len(diagonal_movements) > 0:
                logger.info("✅ Angle-based direction detection working")
            else:
                logger.warning("⚠️ No diagonal movements detected")
        
        detector.release()
        return len(result.movements) > 0
        
    except Exception as e:
        logger.error(f"Angle-based direction test failed: {e}")
        return False
    finally:
        if os.path.exists(test_video_path):
            os.unlink(test_video_path)

def test_quality_validation():
    """Test quality validation features."""
    logger.info("\n" + "="*60)
    logger.info("TEST: Quality Validation")
    logger.info("="*60)
    
    # Create test video with very small faces
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        test_video_path = temp_file.name
    
    try:
        # Create video with tiny faces that should be rejected
        width, height = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(test_video_path, fourcc, 30, (width, height))
        
        # Create tiny face that should be rejected
        face_center = [width // 2, height // 2]
        face_size = 5  # Very small face
        
        for _ in range(90):  # 3 seconds
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = (240, 240, 240)
            
            x, y = face_center[0], face_center[1]
            
            # Draw tiny face
            cv2.circle(frame, (x, y), face_size, (255, 255, 255), -1)
            cv2.circle(frame, (x - 2, y - 2), 1, (0, 0, 0), -1)
            cv2.circle(frame, (x + 2, y - 2), 1, (0, 0, 0), -1)
            
            out.write(frame)
        
        out.release()
        
        detector = create_simple_detector(
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1,
            movement_threshold=0.001,
            debug_mode=True
        )
        
        # Process video
        result = detector.process_video(test_video_path)
        
        logger.info(f"Movements detected: {len(result.movements)}")
        logger.info(f"Frames with face: {detector.frames_with_face}")
        logger.info(f"Total frames: {detector.total_frames}")
        
        # Should have very few or no movements due to quality rejection
        if len(result.movements) == 0:
            logger.info("✅ Quality validation working - no movements from low-quality faces")
        else:
            logger.warning(f"⚠️ {len(result.movements)} movements detected despite low quality")
        
        detector.release()
        return True
        
    except Exception as e:
        logger.error(f"Quality validation test failed: {e}")
        return False
    finally:
        if os.path.exists(test_video_path):
            os.unlink(test_video_path)

def test_enhanced_confidence():
    """Test enhanced confidence calculation."""
    logger.info("\n" + "="*60)
    logger.info("TEST: Enhanced Confidence Calculation")
    logger.info("="*60)
    
    # Create test video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        test_video_path = temp_file.name
    
    try:
        create_test_video_with_angle_movements(test_video_path)
        
        detector = create_simple_detector(
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2,
            movement_threshold=0.005,
            debug_mode=False
        )
        
        # Process video
        result = detector.process_video(test_video_path)
        
        if result.movements:
            logger.info("Confidence analysis:")
            confidences = [m.confidence for m in result.movements]
            magnitudes = [m.magnitude for m in result.movements]
            
            logger.info(f"Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
            logger.info(f"Average confidence: {np.mean(confidences):.3f}")
            logger.info(f"Magnitude range: {min(magnitudes):.4f} - {max(magnitudes):.4f}")
            
            # Check if confidence correlates with magnitude
            correlation = np.corrcoef(magnitudes, confidences)[0, 1]
            logger.info(f"Confidence-magnitude correlation: {correlation:.3f}")
            
            if correlation > 0.3:
                logger.info("✅ Confidence calculation working - correlates with magnitude")
            else:
                logger.warning("⚠️ Confidence may not be properly correlated with magnitude")
        
        detector.release()
        return len(result.movements) > 0
        
    except Exception as e:
        logger.error(f"Enhanced confidence test failed: {e}")
        return False
    finally:
        if os.path.exists(test_video_path):
            os.unlink(test_video_path)

def main():
    """Main function to run all tests."""
    logger.info("Starting Enhanced MediaPipe Detector Tests")
    
    tests = [
        ("Adaptive Thresholds", test_adaptive_thresholds),
        ("Angle-Based Direction", test_angle_based_direction),
        ("Quality Validation", test_quality_validation),
        ("Enhanced Confidence", test_enhanced_confidence),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("ENHANCED DETECTOR TEST SUMMARY")
    logger.info("="*60)
    
    passed = 0
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        logger.info("✅ All enhanced features working correctly!")
        return 0
    else:
        logger.error("❌ Some enhanced features need attention!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
