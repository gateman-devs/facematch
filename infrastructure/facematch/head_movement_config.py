"""
Head Movement Detection Configuration
Configurable parameters for degree-based head movement detection.
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class HeadMovementConfig:
    """
    Configuration for head movement detection using degree-based thresholds.
    
    Key Parameters:
    - min_rotation_degrees: Minimum degrees of head rotation to count as a valid movement
    - significant_rotation_degrees: Degrees for significant movement classification
    - center_threshold_degrees: Degrees within which head is considered "center"
    """
    
    # Degree-based movement thresholds
    min_rotation_degrees: float = 15.0
    """Minimum degrees of head rotation to count as a valid movement.
    Recommended: 15-20 degrees for natural head movements."""
    
    significant_rotation_degrees: float = 25.0
    """Degrees for significant movement classification.
    Movements above this threshold get higher confidence scores.
    Recommended: 25-30 degrees for clear head turns."""
    
    center_threshold_degrees: float = 10.0
    """Degrees within which head is considered "center" (facing camera).
    Recommended: 10-15 degrees to account for natural head sway."""
    
    # Confidence thresholds
    min_confidence_threshold: float = 0.7
    """Minimum confidence to keep a detected movement."""
    
    high_confidence_threshold: float = 0.8
    """Confidence threshold for high-quality movements."""
    
    # Quality parameters
    min_quality_score: float = 0.3
    """Minimum quality score for pose detection."""
    
    # Movement filtering
    movement_cooldown: float = 0.1
    """Minimum time between movements (seconds). Reduced for non-time-based detection."""
    
    max_consecutive_same_direction: int = 1
    """Maximum consecutive movements in the same direction."""
    
    # Return movement detection
    return_movement_threshold: float = 0.05
    """Distance threshold for detecting return movements."""
    
    face_center_threshold: float = 0.1
    """Distance threshold for center position."""
    
    # Performance parameters
    max_history: int = 10
    """Maximum number of poses to keep in history."""
    
    # Debug mode
    debug_mode: bool = False
    """Enable debug logging."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'min_rotation_degrees': self.min_rotation_degrees,
            'significant_rotation_degrees': self.significant_rotation_degrees,
            'center_threshold_degrees': self.center_threshold_degrees,
            'min_confidence_threshold': self.min_confidence_threshold,
            'high_confidence_threshold': self.high_confidence_threshold,
            'min_quality_score': self.min_quality_score,
            'movement_cooldown': self.movement_cooldown,
            'max_consecutive_same_direction': self.max_consecutive_same_direction,
            'return_movement_threshold': self.return_movement_threshold,
            'face_center_threshold': self.face_center_threshold,
            'max_history': self.max_history,
            'debug_mode': self.debug_mode
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HeadMovementConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def conservative_config(cls) -> 'HeadMovementConfig':
        """Conservative configuration for strict movement detection."""
        return cls(
            min_rotation_degrees=20.0,
            significant_rotation_degrees=30.0,
            center_threshold_degrees=15.0,
            min_confidence_threshold=0.8,
            high_confidence_threshold=0.9,
            min_quality_score=0.5,
            movement_cooldown=0.2,
            debug_mode=False
        )
    
    @classmethod
    def lenient_config(cls) -> 'HeadMovementConfig':
        """Lenient configuration for easier movement detection."""
        return cls(
            min_rotation_degrees=10.0,
            significant_rotation_degrees=20.0,
            center_threshold_degrees=8.0,
            min_confidence_threshold=0.6,
            high_confidence_threshold=0.7,
            min_quality_score=0.2,
            movement_cooldown=0.05,
            debug_mode=False
        )
    
    @classmethod
    def default_config(cls) -> 'HeadMovementConfig':
        """Default balanced configuration."""
        return cls()

# Predefined configurations
CONSERVATIVE_CONFIG = HeadMovementConfig.conservative_config()
LENIENT_CONFIG = HeadMovementConfig.lenient_config()
DEFAULT_CONFIG = HeadMovementConfig.default_config()

# Configuration documentation
DEGREE_THRESHOLD_GUIDE = """
Head Movement Degree Thresholds Guide:

1. MINIMUM ROTATION (15° default):
   - 10°: Very sensitive, may detect small head movements
   - 15°: Balanced, detects natural head turns
   - 20°: Conservative, requires clear head movements

2. SIGNIFICANT ROTATION (25° default):
   - 20°: Lenient, most movements considered significant
   - 25°: Balanced, clear head turns get higher confidence
   - 30°: Conservative, only large movements are significant

3. CENTER THRESHOLD (10° default):
   - 8°: Very strict center detection
   - 10°: Balanced, accounts for natural head sway
   - 15°: Lenient, wider center zone

For 90° center position (head facing camera):
- Left turn: -15° to -90° (15° minimum for detection)
- Right turn: +15° to +90° (15° minimum for detection)
- Up tilt: -15° to -90° (15° minimum for detection)
- Down tilt: +15° to +90° (15° minimum for detection)

Recommended settings:
- Natural movements: 15° minimum, 25° significant
- Clear movements: 20° minimum, 30° significant
- Sensitive detection: 10° minimum, 20° significant
"""
