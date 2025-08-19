#!/usr/bin/env python3
"""
Test script to verify threshold adjustments for real video data.
"""

import sys
import os
import logging

# Add the infrastructure directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'infrastructure'))

from facematch.simple_mediapipe_detector import create_simple_detector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_thresholds():
    """Test different threshold settings to find optimal values."""
    
    # Test thresholds based on the real video analysis
    test_configs = [
        {
            'name': 'Very Low (0.001)',
            'movement_threshold': 0.001,
            'min_detection_confidence': 0.4,
            'min_tracking_confidence': 0.4
        },
        {
            'name': 'Low (0.002)',
            'movement_threshold': 0.002,
            'min_detection_confidence': 0.4,
            'min_tracking_confidence': 0.4
        },
        {
            'name': 'Adjusted (0.003)',
            'movement_threshold': 0.003,
            'min_detection_confidence': 0.4,
            'min_tracking_confidence': 0.4
        },
        {
            'name': 'Medium (0.005)',
            'movement_threshold': 0.005,
            'min_detection_confidence': 0.4,
            'min_tracking_confidence': 0.4
        },
        {
            'name': 'High (0.010)',
            'movement_threshold': 0.010,
            'min_detection_confidence': 0.4,
            'min_tracking_confidence': 0.4
        }
    ]
    
    logger.info("Testing different threshold configurations...")
    logger.info("Based on real video analysis: avg_magnitude=0.0028, suggested_threshold=0.0008")
    
    # Look for test video files
    video_extensions = ['.mp4', '.webm', '.avi', '.mov', '.mkv']
    test_videos = []
    
    for ext in video_extensions:
        for file in os.listdir('.'):
            if file.endswith(ext):
                test_videos.append(file)
    
    if not test_videos:
        logger.info("No test video files found. Please provide a video file path as argument.")
        if len(sys.argv) > 1:
            test_videos = [sys.argv[1]]
        else:
            return
    
    video_path = test_videos[0]
    logger.info(f"Using test video: {video_path}")
    
    results = {}
    
    for config in test_configs:
        logger.info(f"\nTesting {config['name']} configuration...")
        
        try:
            detector = create_simple_detector(
                movement_threshold=config['movement_threshold'],
                min_detection_confidence=config['min_detection_confidence'],
                min_tracking_confidence=config['min_tracking_confidence'],
                debug_mode=False  # Disable debug for cleaner output
            )
            
            result = detector.process_video(video_path)
            
            results[config['name']] = {
                'movements_detected': len(result.movements),
                'frames_processed': result.frames_processed,
                'processing_time': result.processing_time,
                'success': result.success
            }
            
            if result.movements:
                # Get movement sequence
                sequence = [m.direction for m in result.movements]
                results[config['name']]['sequence'] = sequence
                
                # Calculate confidence stats
                confidences = [m.confidence for m in result.movements]
                avg_confidence = sum(confidences) / len(confidences)
                results[config['name']]['avg_confidence'] = avg_confidence
                
                logger.info(f"  Movements: {len(result.movements)}")
                logger.info(f"  Sequence: {sequence}")
                logger.info(f"  Avg Confidence: {avg_confidence:.3f}")
            else:
                logger.info(f"  Movements: 0")
            
            detector.release()
            
        except Exception as e:
            logger.error(f"  Error testing {config['name']}: {e}")
            results[config['name']] = {'error': str(e)}
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("THRESHOLD TESTING SUMMARY")
    logger.info("="*60)
    
    for name, result in results.items():
        if 'error' in result:
            logger.info(f"{name}: ERROR - {result['error']}")
        else:
            movements = result.get('movements_detected', 0)
            confidence = result.get('avg_confidence', 0)
            sequence = result.get('sequence', [])
            
            logger.info(f"{name}: {movements} movements, avg_confidence={confidence:.3f}")
            if sequence:
                logger.info(f"  Sequence: {sequence}")
    
    # Recommendations
    logger.info("\nRECOMMENDATIONS:")
    
    # Find best configuration
    best_config = None
    best_score = 0
    
    for name, result in results.items():
        if 'error' not in result and 'movements_detected' in result:
            movements = result['movements_detected']
            confidence = result.get('avg_confidence', 0)
            
            # Score based on movements detected and confidence
            score = movements * confidence
            
            if score > best_score and movements > 0:
                best_score = score
                best_config = name
    
    if best_config:
        logger.info(f"Best configuration: {best_config}")
        logger.info(f"  - Detected {results[best_config]['movements_detected']} movements")
        logger.info(f"  - Average confidence: {results[best_config]['avg_confidence']:.3f}")
    else:
        logger.info("No movements detected with any configuration")
        logger.info("Consider using threshold < 0.002 for very subtle movements")

if __name__ == "__main__":
    test_thresholds()
