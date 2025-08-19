# Enhanced MediaPipe Head Movement Detector Features

## Overview

The `SimpleMediaPipeDetector` has been significantly enhanced with advanced features for more accurate and robust head movement detection. This document outlines all the improvements and their implementation details.

## 🎯 **Core Enhancements**

### 1. **Landmark Indices Definition**
```python
self.landmarks = {
    'nose_tip': 1,
    'chin': 175,
    'left_eye': 33,
    'right_eye': 263,
    'forehead': 10
}
```
- **Purpose**: Centralized landmark mapping for consistency
- **Benefits**: Easy maintenance, clear documentation, reduced errors

### 2. **Adaptive Movement Thresholds**
```python
def _calculate_adaptive_threshold(self, eye_distance: float) -> float:
    if eye_distance < 0.05:  # Face far from camera
        return self.base_movement_threshold * 1.5
    elif eye_distance > 0.15:  # Face close to camera
        return self.base_movement_threshold * 0.7
    else:  # Normal distance
        return self.base_movement_threshold
```
- **Purpose**: Adjusts sensitivity based on face size/distance
- **Logic**: 
  - Large faces (close camera) → Lower threshold (more sensitive)
  - Small faces (far camera) → Higher threshold (less sensitive)
- **Benefits**: Consistent detection across different camera distances

### 3. **Angle-Based Direction Detection**
```python
def _detect_direction_angle(self, dx: float, dy: float) -> str:
    angle = np.arctan2(dy, dx) * 180 / np.pi
    # Normalize to 0-360 range
    if angle < 0:
        angle += 360
    
    # Direction ranges
    self.direction_angles = {
        'right': (-45, 45),
        'down': (45, 135),
        'left': (135, 225),
        'up': (225, 315)
    }
```
- **Purpose**: More accurate direction classification for diagonal movements
- **Benefits**: 
  - Handles diagonal movements correctly
  - Prevents direction flipping with hysteresis
  - More intuitive angle ranges

### 4. **Temporal Smoothing**
```python
def _apply_temporal_smoothing(self, current_pose: Dict[str, float]) -> Dict[str, float]:
    smoothing_weights = [0.1, 0.3, 0.6]  # Weights for last 3 poses
    # Apply weighted moving average to x, y, yaw_ratio, pitch_ratio
```
- **Purpose**: Reduces pose jitter and noise
- **Implementation**: Weighted moving average with recent poses getting higher weight
- **Benefits**: Smoother movement detection, reduced false positives

### 5. **Enhanced Pose Calculation**
```python
# Normalized pose ratios
yaw_ratio = (nose_tip.x - x_center) / (eye_distance + 1e-6)
pitch_ratio = (nose_tip.y - y_center) / (face_height + 1e-6)

# Quality score based on face size
quality_score = min(eye_distance * 10, 1.0)
```
- **Purpose**: Better pose representation and quality assessment
- **Features**:
  - Normalized ratios for consistent measurements
  - Quality score based on face size
  - Enhanced pose data structure

### 6. **Improved Confidence Calculation**
```python
def _calculate_movement_confidence(self, direction: str, magnitude: float, pose_data: Dict[str, float]) -> float:
    # Base confidence from magnitude (rewards larger movements)
    magnitude_ratio = magnitude / self.movement_threshold
    base_confidence = min(magnitude_ratio / 2.0, 1.0)
    
    # Quality boost
    quality_boost = pose_data.get('quality_score', 0.5) * 0.3
    
    # Direction consistency (rewards natural patterns)
    # Movement size factor (rewards significant movements)
```
- **Purpose**: More accurate confidence scoring
- **Improvements**:
  - Rewards larger movements instead of penalizing them
  - Considers pose quality
  - Rewards natural movement patterns
  - Penalizes repetitive noise

### 7. **Quality Validation**
```python
# Quality thresholds
self.min_eye_distance = 0.01
self.min_face_height = 0.01
self.min_quality_score = 0.3

# Validation in get_head_pose
if eye_distance < self.min_eye_distance or face_height < self.min_face_height:
    return None  # Reject low-quality poses
```
- **Purpose**: Ensures only high-quality poses are processed
- **Benefits**: Reduces false detections from poor face detection

## 🔧 **Configuration Parameters**

### **Adaptive Threshold Settings**
```python
self.base_movement_threshold = movement_threshold  # Base threshold
self.movement_threshold = movement_threshold       # Current adaptive threshold
```

### **Direction Detection Settings**
```python
self.direction_angles = {
    'right': (-45, 45),
    'down': (45, 135),
    'left': (135, 225),
    'up': (225, 315)
}
self.direction_hysteresis = 10  # Degrees to prevent flipping
```

### **Temporal Smoothing Settings**
```python
self.smoothing_weights = [0.1, 0.3, 0.6]  # Weights for last 3 poses
self.min_poses_for_smoothing = 3
```

### **Quality Validation Settings**
```python
self.min_eye_distance = 0.01
self.min_face_height = 0.01
self.min_quality_score = 0.3
```

## 📊 **Enhanced Logging and Statistics**

### **New Statistics Tracked**
- `face_sizes`: List of eye distances for adaptive threshold analysis
- `quality_score`: Pose quality metrics
- `yaw_ratio`, `pitch_ratio`: Normalized pose measurements

### **Enhanced Logging**
```python
logger.info(f"Movement: dx={dx:.4f}, dy={dy:.4f}, magnitude={magnitude:.4f}, "
           f"threshold={self.movement_threshold:.4f}, eye_distance={eye_distance:.4f}")

logger.info(f"Movement detected: {direction} (magnitude={magnitude:.4f}, "
           f"angle={np.arctan2(dy, dx) * 180 / np.pi:.1f}°)")
```

## 🧪 **Testing and Validation**

### **Test Scenarios**
1. **Adaptive Thresholds**: Varying face sizes to verify threshold adaptation
2. **Angle-Based Direction**: Diagonal movements to test direction classification
3. **Quality Validation**: Low-quality faces to ensure proper rejection
4. **Enhanced Confidence**: Movement patterns to verify confidence calculation

### **Test Scripts**
- `test_enhanced_detector.py`: Comprehensive test suite for all features
- `test_movement_validation.py`: Tests for movement validation rules
- `test_threshold_adjustment.py`: Threshold sensitivity testing

## 🚀 **Performance Improvements**

### **Computational Efficiency**
- Temporal smoothing reduces processing of noisy frames
- Quality validation prevents unnecessary calculations
- Adaptive thresholds optimize detection sensitivity

### **Accuracy Improvements**
- Angle-based direction detection handles edge cases
- Enhanced confidence calculation reduces false positives
- Quality validation ensures reliable pose data

## 🔄 **Backward Compatibility**

### **Maintained Interface**
- All existing method signatures preserved
- `MovementResult` and `DetectionResult` classes unchanged
- Existing configuration parameters still supported

### **Enhanced Capabilities**
- New features are additive and don't break existing functionality
- Debug mode provides detailed information about new features
- Graceful fallback to base functionality if enhanced features fail

## 📈 **Usage Examples**

### **Basic Usage (Unchanged)**
```python
detector = create_simple_detector(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    movement_threshold=0.02,
    debug_mode=True
)
result = detector.process_video(video_path)
```

### **Enhanced Configuration**
```python
detector = create_simple_detector(
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4,
    movement_threshold=0.003,  # Will be adapted based on face size
    debug_mode=True
)

# Access enhanced statistics
print(f"Face sizes tracked: {len(detector.face_sizes)}")
print(f"Average face size: {np.mean(detector.face_sizes):.4f}")
print(f"Adaptive threshold: {detector.movement_threshold:.4f}")
```

## 🎯 **Expected Behavior**

### **Adaptive Thresholds**
- Large faces (close camera): More sensitive detection
- Small faces (far camera): Less sensitive detection
- Automatic adjustment during processing

### **Direction Detection**
- Diagonal movements classified as primary direction
- Hysteresis prevents rapid direction changes
- Angle-based classification more accurate than x/y comparison

### **Quality Validation**
- Low-quality poses automatically rejected
- Quality score influences confidence calculation
- Face size requirements prevent false detections

### **Temporal Smoothing**
- Smoother movement detection
- Reduced jitter in pose tracking
- Better handling of noisy video

## 🔧 **Troubleshooting**

### **Common Issues**
1. **No movements detected**: Check quality validation thresholds
2. **Too many false positives**: Adjust adaptive threshold parameters
3. **Direction classification errors**: Verify angle ranges and hysteresis settings

### **Debug Information**
Enable debug mode to see:
- Adaptive threshold changes
- Quality validation results
- Angle calculations
- Temporal smoothing effects

## 📚 **Future Enhancements**

### **Potential Improvements**
1. **Machine Learning Integration**: Learn optimal thresholds from training data
2. **Multi-Face Support**: Handle multiple faces in frame
3. **Advanced Filtering**: Kalman filtering for pose prediction
4. **Real-time Optimization**: GPU acceleration for faster processing

### **Extensibility**
The enhanced architecture supports easy addition of new features:
- Modular design with clear separation of concerns
- Configurable parameters for easy tuning
- Comprehensive logging for debugging and optimization
