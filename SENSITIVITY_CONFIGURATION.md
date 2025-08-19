# Head Movement Detection Sensitivity Configuration

This guide provides different configuration presets for the MediaPipe head movement detector to balance between sensitivity and accuracy.

## Configuration Presets

### 1. **Conservative (Low Sensitivity)**
Best for: Production environments where false positives are costly
```python
detector = create_simple_detector(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    movement_threshold=0.025,  # Higher threshold
    debug_mode=False
)
```
**Characteristics:**
- Fewer false positives
- May miss subtle movements
- Higher confidence requirements
- Suitable for high-quality videos

### 2. **Balanced (Current Default)**
Best for: General use with good video quality
```python
detector = create_simple_detector(
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4,
    movement_threshold=0.015,  # Balanced threshold
    debug_mode=True
)
```
**Characteristics:**
- Good balance of sensitivity and accuracy
- Handles most real-world scenarios
- Moderate confidence requirements
- Includes movement cooldown and filtering

### 3. **Sensitive (High Sensitivity)**
Best for: Testing or low-quality videos
```python
detector = create_simple_detector(
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
    movement_threshold=0.01,  # Lower threshold
    debug_mode=True
)
```
**Characteristics:**
- Catches subtle movements
- May have more false positives
- Lower confidence requirements
- Good for debugging

### 4. **Very Sensitive (Debug Mode)**
Best for: Development and troubleshooting
```python
detector = create_simple_detector(
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2,
    movement_threshold=0.005,  # Very low threshold
    debug_mode=True
)
```
**Characteristics:**
- Maximum sensitivity
- Many false positives expected
- Detailed debugging information
- Only for development/testing

## Current Implementation Features

### Movement Filtering
The current implementation includes several filtering mechanisms:

1. **Movement Cooldown**: 0.3 seconds minimum between movements
2. **Significant Movement Check**: Movements must be 1.2x threshold or part of a pattern
3. **Confidence Penalties**: 
   - Repetitive movements penalized
   - Small movements penalized
   - Significant movements get minimum confidence boost

### Confidence Calculation
```python
# Base confidence from magnitude (more selective)
base_confidence = min(magnitude / (self.movement_threshold * 3), 1.0)

# Penalize repetitive movements
if direction_consistency > 0.6:
    base_confidence *= 0.5

# Penalize very small movements
if magnitude < self.movement_threshold * 1.5:
    base_confidence *= 0.8

# Ensure minimum confidence for significant movements
if magnitude > self.movement_threshold * 2:
    base_confidence = max(base_confidence, 0.6)
```

## Recommended Settings by Use Case

### Production Environment
```python
# Conservative settings
min_detection_confidence=0.5
min_tracking_confidence=0.5
movement_threshold=0.025
debug_mode=False
```

### Development/Testing
```python
# Balanced settings with debug
min_detection_confidence=0.4
min_tracking_confidence=0.4
movement_threshold=0.015
debug_mode=True
```

### Troubleshooting
```python
# Very sensitive for debugging
min_detection_confidence=0.2
min_tracking_confidence=0.2
movement_threshold=0.005
debug_mode=True
```

## Adjusting Sensitivity

### If Detection is Too Sensitive (Too Many False Positives)

1. **Increase Movement Threshold**:
   ```python
   movement_threshold=0.020  # Increase from 0.015
   ```

2. **Increase Confidence Requirements**:
   ```python
   min_detection_confidence=0.5  # Increase from 0.4
   min_tracking_confidence=0.5   # Increase from 0.4
   ```

3. **Increase Movement Cooldown**:
   ```python
   self.movement_cooldown = 0.5  # Increase from 0.3 seconds
   ```

### If Detection is Not Sensitive Enough (Missing Movements)

1. **Decrease Movement Threshold**:
   ```python
   movement_threshold=0.010  # Decrease from 0.015
   ```

2. **Decrease Confidence Requirements**:
   ```python
   min_detection_confidence=0.3  # Decrease from 0.4
   min_tracking_confidence=0.3   # Decrease from 0.4
   ```

3. **Decrease Movement Cooldown**:
   ```python
   self.movement_cooldown = 0.2  # Decrease from 0.3 seconds
   ```

## Monitoring and Tuning

### Key Metrics to Watch

1. **Face Detection Rate**: Should be > 0.7 for good videos
2. **Movement Count**: Should be reasonable (5-20 for a 5-second video)
3. **Average Confidence**: Should be > 0.6 for reliable movements
4. **False Positive Rate**: Monitor for unexpected movements

### Log Analysis

Look for these patterns in logs:

```
# Good detection
Movement Summary - Total: 8
  up: 2 movements
  right: 2 movements
  down: 2 movements
  left: 2 movements
Confidence stats: avg=0.750, min=0.600, max=0.900

# Too sensitive
Movement Summary - Total: 45
  up: 12 movements
  right: 11 movements
  down: 11 movements
  left: 11 movements
Confidence stats: avg=0.450, min=0.200, max=0.700

# Not sensitive enough
Movement Summary - Total: 1
  up: 1 movements
Confidence stats: avg=0.800, min=0.800, max=0.800
```

## Quick Configuration Changes

### For Immediate Use

To make detection less sensitive right now, update the VideoLivenessAdapter:

```python
# In unified_liveness_detector.py
self.simple_mediapipe_detector = create_simple_detector(
    min_detection_confidence=0.5,    # Increased from 0.4
    min_tracking_confidence=0.5,     # Increased from 0.4
    movement_threshold=0.020,        # Increased from 0.015
    debug_mode=True
)
```

### For Testing Different Levels

Create a test script with different configurations:

```python
configs = {
    'conservative': {'movement_threshold': 0.025, 'min_detection_confidence': 0.5},
    'balanced': {'movement_threshold': 0.015, 'min_detection_confidence': 0.4},
    'sensitive': {'movement_threshold': 0.010, 'min_detection_confidence': 0.3},
}

for name, config in configs.items():
    detector = create_simple_detector(**config, debug_mode=True)
    # Test with your video
    result = detector.process_video(video_path)
    print(f"{name}: {len(result.movements)} movements")
```

The current balanced configuration should provide a good middle ground, but you can adjust based on your specific needs and video quality.
