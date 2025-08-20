# Non-Time-Based Head Movement Detection

## Overview

The video head movement detector has been modified to remove time-based constraints and implement degree-based detection. This addresses the user's request to not expect movements within specific time windows (2 seconds, 1.5 seconds, etc.) and instead detect movements as they occur.

## Key Changes

### 1. Removed Time-Based Constraints

**Before:**
- Expected movements within 2 seconds per direction
- Required 1.5 seconds to return to center
- Fixed time windows for analysis

**After:**
- No time constraints on movement detection
- Movements detected as they occur
- Flexible timing - users can move at their own pace

### 2. Implemented Degree-Based Detection

**Before:**
- Pixel-based movement detection
- Movement thresholds in pixels
- Less accurate for different face sizes and distances

**After:**
- Degree-based head rotation detection
- Uses actual yaw/pitch angles from MediaPipe
- More accurate and consistent across different scenarios

## Degree Thresholds

### Recommended Settings

For a head facing the camera at 90° (center position):

1. **Minimum Rotation: 15°**
   - Left turn: -15° to -90° (15° minimum for detection)
   - Right turn: +15° to +90° (15° minimum for detection)
   - Up tilt: -15° to -90° (15° minimum for detection)
   - Down tilt: +15° to +90° (15° minimum for detection)

2. **Significant Rotation: 25°**
   - Movements above 25° get higher confidence scores
   - Clear head turns are more reliably detected

3. **Center Threshold: 10°**
   - Head within ±10° of center is considered "facing camera"
   - Accounts for natural head sway

### Configuration Options

The system provides three preset configurations:

1. **Default Configuration**
   - Min rotation: 15°
   - Significant rotation: 25°
   - Center threshold: 10°
   - Balanced for most use cases

2. **Conservative Configuration**
   - Min rotation: 20°
   - Significant rotation: 30°
   - Center threshold: 15°
   - Requires clear, deliberate movements

3. **Lenient Configuration**
   - Min rotation: 10°
   - Significant rotation: 20°
   - Center threshold: 8°
   - Detects smaller movements

## Technical Implementation

### Modified Files

1. **`enhanced_mediapipe_detector.py`**
   - Replaced pixel-based detection with degree-based detection
   - Added yaw/pitch angle calculation
   - Removed time-based movement cooldowns
   - Updated movement detection logic

2. **`simple_liveness.py`**
   - Removed time-segmented analysis
   - Implemented movement-based analysis
   - No longer expects specific time windows

3. **`head_movement_config.py`** (New)
   - Centralized configuration management
   - Predefined configurations for different sensitivity levels
   - Documentation of degree thresholds

### Key Functions

1. **`_calculate_head_pose_angles()`**
   - Calculates yaw, pitch, and roll angles from face landmarks
   - Converts pixel positions to degree measurements

2. **`_detect_direction_from_rotation()`**
   - Determines movement direction from rotation angles
   - Uses the larger rotation (yaw or pitch) as primary direction

3. **`detect_movement()`**
   - Checks if rotation exceeds minimum threshold
   - No longer uses time-based constraints
   - Returns movement with degree magnitude

## Benefits

### 1. Improved User Experience
- No time pressure on users
- Natural movement detection
- Works with different movement speeds

### 2. Better Accuracy
- Degree-based detection is more precise
- Consistent across different face sizes
- Less affected by camera distance

### 3. Configurable Sensitivity
- Easy to adjust for different use cases
- Conservative settings for strict validation
- Lenient settings for easier detection

### 4. More Reliable Detection
- Based on actual head rotation angles
- Not affected by pixel-level noise
- Better handling of edge cases

## Usage Examples

### Basic Usage
```python
from enhanced_mediapipe_detector import EnhancedMediaPipeDetector

# Use default configuration (15° minimum rotation)
detector = EnhancedMediaPipeDetector()

# Process video
result = detector.process_video("video.mp4")
```

### Custom Configuration
```python
from head_movement_config import HeadMovementConfig

# Create custom configuration
config = HeadMovementConfig(
    min_rotation_degrees=20.0,  # Require 20° minimum
    significant_rotation_degrees=30.0,  # 30° for high confidence
    min_confidence_threshold=0.8  # Higher confidence requirement
)

# Use with detector
detector = EnhancedMediaPipeDetector(
    min_rotation_degrees=config.min_rotation_degrees,
    significant_rotation_degrees=config.significant_rotation_degrees,
    min_confidence_threshold=config.min_confidence_threshold
)
```

### Predefined Configurations
```python
from head_movement_config import CONSERVATIVE_CONFIG, LENIENT_CONFIG

# Use conservative settings for strict validation
detector = EnhancedMediaPipeDetector(
    min_rotation_degrees=CONSERVATIVE_CONFIG.min_rotation_degrees,
    significant_rotation_degrees=CONSERVATIVE_CONFIG.significant_rotation_degrees
)

# Use lenient settings for easier detection
detector = EnhancedMediaPipeDetector(
    min_rotation_degrees=LENIENT_CONFIG.min_rotation_degrees,
    significant_rotation_degrees=LENIENT_CONFIG.significant_rotation_degrees
)
```

## Testing

Run the test script to verify the implementation:

```bash
python test_degree_based_detection.py
```

This will test different configurations and demonstrate how the degree thresholds work.

## Migration Notes

### For Existing Users
- The system is backward compatible
- Default settings provide balanced detection
- Can be adjusted based on specific requirements

### For Developers
- New configuration system makes it easy to tune parameters
- Degree-based detection provides more predictable results
- Time constraints can be completely disabled

## Future Enhancements

1. **Adaptive Thresholds**
   - Automatically adjust based on video quality
   - Learn from user behavior patterns

2. **Advanced Pose Estimation**
   - Integration with more sophisticated 3D pose estimators
   - Better handling of extreme head angles

3. **Real-time Processing**
   - Optimize for live video streams
   - Reduce processing latency

## Conclusion

The non-time-based degree detection system provides a more natural and accurate way to detect head movements. Users are no longer constrained by time limits and can move at their own pace, while the system provides more reliable detection based on actual head rotation angles rather than pixel movements.
