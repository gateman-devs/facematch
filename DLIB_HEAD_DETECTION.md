# Dlib-based Head Movement Detection

This document describes the new dlib-based head movement detection system that replaces MediaPipe with dlib facial landmarks and OpenCV for more accurate and reliable head pose estimation.

## Overview

The dlib-based head movement detector uses:
- **dlib**: For robust 68-point facial landmark detection
- **OpenCV**: For 3D head pose estimation using solvePnP
- **NumPy**: For mathematical calculations and array operations

## Key Features

### 1. Accurate 3D Head Pose Estimation
- Uses dlib's 68-point facial landmark model
- Implements OpenCV's solvePnP for 3D pose calculation
- Provides yaw, pitch, and roll angles in degrees

### 2. Robust Movement Detection
- Degree-based movement thresholds (configurable)
- Prevents consecutive same-direction movements
- Quality scoring based on face size and landmark visibility

### 3. Automatic Resource Management
- Downloads shape predictor file automatically if missing
- Handles video processing with proper cleanup
- Configurable parameters for different use cases

## Implementation

### Core Components

#### 1. DlibHeadDetector Class
Located in `infrastructure/facematch/dlib_head_detector.py`

```python
from facematch.dlib_head_detector import DlibHeadDetector, create_dlib_detector

# Create detector with custom configuration
detector = create_dlib_detector({
    'min_rotation_degrees': 15.0,
    'significant_rotation_degrees': 25.0,
    'min_confidence_threshold': 0.7,
    'debug_mode': True
})
```

#### 2. Movement Detection
```python
# Process video file
result = detector.process_video('video.mp4')

# Process single frame
frame_result = detector.process_frame(frame)

# Get detected movements
movements = detector.get_movements()
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_rotation_degrees` | 15.0 | Minimum degrees for valid movement |
| `significant_rotation_degrees` | 25.0 | Degrees for significant movement |
| `min_confidence_threshold` | 0.7 | Minimum confidence to keep movement |
| `max_history` | 10 | Maximum poses to keep in history |
| `debug_mode` | False | Enable debug logging |

## Usage Examples

### 1. Basic Usage
```python
import cv2
import dlib
import numpy as np

# Initialize detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Process video
cap = cv2.VideoCapture('video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])
        
        # Calculate head pose
        pose = get_head_pose(frame, landmarks)
        if pose:
            pitch, yaw = pose
            print(f"Direction: {determine_direction(pitch, yaw)}")

cap.release()
```

### 2. Advanced Usage with DlibHeadDetector
```python
from facematch.dlib_head_detector import create_dlib_detector

# Create detector
detector = create_dlib_detector({
    'min_rotation_degrees': 20.0,
    'significant_rotation_degrees': 30.0,
    'debug_mode': True
})

# Process video
result = detector.process_video('test_video.mp4')

if result.success:
    print(f"Detected {len(result.movements)} movements")
    for movement in result.movements:
        print(f"{movement.direction}: {movement.magnitude:.1f}° (confidence: {movement.confidence:.2f})")
```

## Testing

### Test Scripts

1. **Basic Test**: `example_dlib_head_detection.py`
   - Simple implementation similar to original code
   - Downloads and processes the test video
   - Shows basic movement detection

2. **Advanced Test**: `test_dlib_head_detector.py`
   - Full-featured test with detailed analysis
   - Compares against expected movements
   - Provides accuracy metrics

### Running Tests
```bash
# Basic test
python example_dlib_head_detection.py

# Advanced test
python test_dlib_head_detector.py
```

## Performance Comparison

### Advantages over MediaPipe

1. **Accuracy**: More precise 3D pose estimation using solvePnP
2. **Reliability**: dlib's facial landmarks are more stable
3. **Customization**: Easier to tune parameters for specific use cases
4. **Dependencies**: Fewer external dependencies (no MediaPipe)

### Performance Metrics

| Metric | MediaPipe | Dlib + OpenCV |
|--------|-----------|---------------|
| Face Detection | Good | Excellent |
| Landmark Stability | Variable | High |
| 3D Pose Accuracy | Good | Excellent |
| Processing Speed | Fast | Moderate |
| Memory Usage | High | Low |

## Integration

### Replacing MediaPipe in Existing Code

1. **Update imports**:
   ```python
   # Old
   from facematch.enhanced_mediapipe_detector import EnhancedMediaPipeDetector
   
   # New
   from facematch.dlib_head_detector import DlibHeadDetector, create_dlib_detector
   ```

2. **Update detector creation**:
   ```python
   # Old
   detector = EnhancedMediaPipeDetector()
   
   # New
   detector = create_dlib_detector()
   ```

3. **Update method calls**:
   ```python
   # Old
   result = detector.process_video(video_path)
   
   # New (same interface)
   result = detector.process_video(video_path)
   ```

## Troubleshooting

### Common Issues

1. **Shape Predictor Not Found**
   ```
   Error: Could not download shape predictor
   ```
   **Solution**: The script automatically downloads the file. Check internet connection.

2. **No Face Detected**
   ```
   Warning: No movements detected!
   ```
   **Solution**: Adjust `min_quality_score` or check video quality.

3. **Too Many False Positives**
   ```
   Solution: Increase `min_rotation_degrees` and `significant_rotation_degrees`
   ```

4. **Too Few Detections**
   ```
   Solution: Decrease `min_rotation_degrees` and `significant_rotation_degrees`
   ```

### Debug Mode

Enable debug mode for detailed logging:
```python
detector = create_dlib_detector({'debug_mode': True})
```

## Dependencies

### Required Packages
- `dlib>=19.24.0`: Facial landmark detection
- `opencv-python>=4.8.0`: Computer vision operations
- `numpy>=1.24.0`: Numerical computations

### Optional Packages
- `scikit-learn>=1.3.0`: For advanced analysis
- `Pillow>=10.0.0`: Image processing utilities

## Future Enhancements

1. **Multi-face Support**: Detect movements for multiple faces
2. **Real-time Processing**: Optimize for live video streams
3. **Advanced Filtering**: Implement Kalman filtering for smoother results
4. **GPU Acceleration**: Add CUDA support for faster processing
5. **Custom Models**: Support for custom landmark models

## Migration Guide

### From MediaPipe to Dlib

1. **Install dependencies**:
   ```bash
   pip install dlib opencv-python numpy
   ```

2. **Update code**:
   ```python
   # Replace MediaPipe imports
   from facematch.dlib_head_detector import create_dlib_detector
   
   # Create detector
   detector = create_dlib_detector()
   
   # Use same interface
   result = detector.process_video(video_path)
   ```

3. **Test thoroughly**:
   - Run test scripts
   - Adjust parameters as needed
   - Monitor performance

## Support

For issues or questions regarding the dlib-based head movement detection:

1. Check the troubleshooting section
2. Review the test scripts for examples
3. Enable debug mode for detailed logging
4. Consult the dlib and OpenCV documentation
