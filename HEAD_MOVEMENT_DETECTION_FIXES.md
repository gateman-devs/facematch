# Head Movement Detection Fixes and Improvements

This document outlines the fixes and improvements made to address the head movement detection issues identified in the logs.

## Issues Identified

From the logs, the main issue was:
```
INFO:infrastructure.facematch.simple_mediapipe_detector:Video processing completed: 0 movements in 2.452s
```

The MediaPipe detector was processing videos but detecting **0 movements**, causing liveness checks to fail.

## Root Causes

1. **Movement Threshold Too High**: Default threshold (0.02) was too high for real video movements
2. **Insufficient Debugging**: No detailed information about why movements weren't detected
3. **Detection Confidence Too High**: MediaPipe confidence thresholds were too restrictive
4. **Lack of Movement Tracking**: No detailed logging of detected movements

## Fixes Implemented

### 1. Enhanced Debugging and Logging

#### Added Debug Mode
```python
def __init__(self, 
             min_detection_confidence: float = 0.5,
             min_tracking_confidence: float = 0.5,
             movement_threshold: float = 0.02,
             max_history: int = 10,
             debug_mode: bool = False):  # New parameter
```

#### Detailed Movement Logging
```python
if self.debug_mode:
    logger.debug(f"Movement: dx={dx:.4f}, dy={dy:.4f}, magnitude={magnitude:.4f}, threshold={self.movement_threshold:.4f}")
    logger.debug(f"Movement detected: {direction} (magnitude={magnitude:.4f})")
```

#### Statistics Tracking
```python
# Statistics for adaptive threshold
self.total_frames = 0
self.frames_with_face = 0
self.movement_magnitudes = []
```

### 2. Adaptive Threshold System

#### Movement Magnitude Tracking
```python
# Store magnitude for adaptive threshold
self.movement_magnitudes.append(magnitude)
```

#### Suggested Threshold Calculation
```python
if len(movements) == 0 and self.movement_magnitudes:
    suggested_threshold = np.percentile(self.movement_magnitudes, 25)
    logger.info(f"No movements detected. Suggested threshold: {suggested_threshold:.4f} "
               f"(current: {self.movement_threshold:.4f})")
```

### 3. More Sensitive Default Settings

#### Updated VideoLivenessAdapter Configuration
```python
self.simple_mediapipe_detector = create_simple_detector(
    min_detection_confidence=0.3,    # Lowered from 0.5
    min_tracking_confidence=0.3,     # Lowered from 0.5
    movement_threshold=0.01,         # Lowered from 0.02
    debug_mode=True                  # Enable debug logging
)
```

### 4. Enhanced Movement Tracking

#### Detailed Movement Information
```python
def get_detailed_movements(self) -> List[Dict[str, Any]]:
    """Get detailed movement information for debugging."""
    return [
        {
            'direction': movement.direction,
            'confidence': movement.confidence,
            'magnitude': movement.magnitude,
            'timestamp': movement.timestamp,
            'pose_data': movement.pose_data
        }
        for movement in self.movement_history
    ]
```

#### Movement Summary Logging
```python
def log_movement_summary(self):
    """Log a summary of all detected movements."""
    # Groups movements by direction
    # Shows movement sequence
    # Provides confidence and magnitude statistics
```

### 5. Improved Error Messages

#### Better No-Movement Feedback
```python
return {
    'success': True,
    'passed': False,
    'error': 'No movements detected - possible causes: movement too subtle, face not clearly visible, or video quality issues',
    'debug_info': {
        'frames_processed': result.frames_processed,
        'processing_time': result.processing_time,
        'suggested_threshold': 'Try lowering movement_threshold to 0.005'
    }
}
```

### 6. Enhanced API Response

#### Movement Details in Response
```python
'movement_details': {
    'total_movements': len(movements),
    'movements': [...],  # Detailed movement data
    'detected_sequence': detected_sequence,
    'movement_summary': {
        'up_count': detected_sequence.count('up'),
        'down_count': detected_sequence.count('down'),
        'left_count': detected_sequence.count('left'),
        'right_count': detected_sequence.count('right')
    }
}
```

## Testing and Validation

### Test Scripts Created

1. **`test_real_video_detection.py`**: Tests with real video files
2. **Enhanced `test_simple_mediapipe.py`**: More comprehensive testing

### Debug Mode Usage

```python
# Enable debug logging
detector = create_simple_detector(
    debug_mode=True,
    movement_threshold=0.005  # Very sensitive for testing
)
```

## Configuration Recommendations

### For Production Use
```python
detector = create_simple_detector(
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
    movement_threshold=0.01,
    debug_mode=False  # Disable in production
)
```

### For Testing/Debugging
```python
detector = create_simple_detector(
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2,
    movement_threshold=0.005,
    debug_mode=True
)
```

## Expected Improvements

### 1. Better Detection Rate
- Lower thresholds should catch more subtle movements
- Improved face detection with lower confidence requirements

### 2. Enhanced Debugging
- Detailed logs show exactly what's happening during processing
- Movement statistics help identify issues

### 3. Adaptive Thresholds
- System suggests optimal thresholds based on video content
- Automatic adjustment recommendations

### 4. Comprehensive Movement Tracking
- Full movement history with timestamps
- Direction-based movement counting
- Confidence and magnitude statistics

## Monitoring and Troubleshooting

### Key Log Messages to Watch

1. **Face Detection Rate**: `face_rate={face_detection_rate:.3f}`
2. **Average Movement Magnitude**: `avg_magnitude={avg_magnitude:.4f}`
3. **Suggested Threshold**: When no movements detected
4. **Movement Summary**: Detailed breakdown of detected movements

### Common Issues and Solutions

#### Issue: No movements detected
**Check**: Face detection rate and average magnitude
**Solution**: Lower movement threshold or detection confidence

#### Issue: Too many false positives
**Check**: Movement magnitude statistics
**Solution**: Increase movement threshold

#### Issue: Poor face detection
**Check**: Face detection rate
**Solution**: Lower detection confidence or improve video quality

## Next Steps

1. **Monitor Production Logs**: Watch for improved detection rates
2. **Fine-tune Thresholds**: Adjust based on real-world performance
3. **Add Performance Metrics**: Track detection accuracy over time
4. **Consider Adaptive Thresholds**: Automatically adjust based on video characteristics

## Files Modified

1. `infrastructure/facematch/simple_mediapipe_detector.py` - Enhanced with debugging and adaptive thresholds
2. `infrastructure/facematch/unified_liveness_detector.py` - Updated with sensitive settings and better error handling
3. `test_real_video_detection.py` - New test script for real video files
4. `HEAD_MOVEMENT_DETECTION_FIXES.md` - This documentation

The enhanced system should now provide much better movement detection and comprehensive debugging information to help identify and resolve any remaining issues.
