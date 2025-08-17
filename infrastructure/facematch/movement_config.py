"""
Enhanced Movement Threshold Configuration System
Provides configurable parameters for movement detection with adaptive thresholds.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class MovementThresholdConfig:
    """Configuration for movement detection thresholds."""
    
    # Base movement thresholds (in pixels)
    min_movement_threshold: float = 10.0
    significant_movement_threshold: float = 12.0
    
    # Confidence thresholds
    min_confidence_threshold: float = 0.2
    high_confidence_threshold: float = 0.8
    
    # Adaptive threshold parameters
    enable_adaptive_thresholds: bool = True
    face_size_scaling_factor: float = 0.02  # Scale thresholds based on face size
    video_quality_scaling_factor: float = 0.15  # Scale thresholds based on video quality
    
    # Movement magnitude scaling
    horizontal_magnitude_scale: float = 80.0  # For confidence calculation
    vertical_magnitude_scale: float = 80.0    # For confidence calculation
    
    # Pixel-to-movement ratio parameters
    base_frame_width: int = 640
    base_frame_height: int = 480
    movement_ratio_threshold: float = 0.015  # Minimum movement as ratio of frame size
    
    # Temporal parameters
    window_size: int = 12  # Frames to analyze for movement detection
    step_size_ratio: float = 0.33  # Step size as ratio of window size
    consistency_boost_factor: float = 1.2  # Boost for consistent movements
    
    # Quality-based adjustments
    low_quality_threshold_multiplier: float = 1.5  # Increase thresholds for low quality
    high_quality_threshold_multiplier: float = 0.8  # Decrease thresholds for high quality
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.min_movement_threshold <= 0:
            raise ValueError("min_movement_threshold must be positive")
        if self.significant_movement_threshold < self.min_movement_threshold:
            raise ValueError("significant_movement_threshold must be >= min_movement_threshold")
        if not (0 < self.min_confidence_threshold < 1):
            raise ValueError("min_confidence_threshold must be between 0 and 1")
        if not (0 < self.high_confidence_threshold <= 1):
            raise ValueError("high_confidence_threshold must be between 0 and 1")
        if self.min_confidence_threshold >= self.high_confidence_threshold:
            raise ValueError("min_confidence_threshold must be < high_confidence_threshold")

@dataclass
class AdaptiveThresholdCalculator:
    """Calculator for adaptive movement thresholds based on face size and video quality."""
    
    config: MovementThresholdConfig
    
    def calculate_adaptive_thresholds(
        self, 
        face_size: Tuple[int, int], 
        video_quality_metrics: Dict[str, float],
        frame_dimensions: Tuple[int, int] = (640, 480)
    ) -> Dict[str, float]:
        """
        Calculate adaptive thresholds based on face size and video quality.
        
        Args:
            face_size: (width, height) of detected face in pixels
            video_quality_metrics: Dictionary containing quality metrics
            frame_dimensions: (width, height) of video frame
            
        Returns:
            Dictionary with calculated adaptive thresholds
        """
        if not self.config.enable_adaptive_thresholds:
            return self._get_base_thresholds()
        
        # Calculate face size factor
        face_area = face_size[0] * face_size[1]
        frame_area = frame_dimensions[0] * frame_dimensions[1]
        face_size_ratio = face_area / frame_area
        
        # Face size scaling: larger faces allow for smaller pixel thresholds
        # because the same head movement covers more pixels
        face_scale_factor = max(0.5, min(2.0, 1.0 / (face_size_ratio * 10 + 0.1)))
        
        # Video quality scaling
        quality_scale_factor = self._calculate_quality_scale_factor(video_quality_metrics)
        
        # Frame size scaling for pixel-to-movement ratio
        frame_scale_x = frame_dimensions[0] / self.config.base_frame_width
        frame_scale_y = frame_dimensions[1] / self.config.base_frame_height
        
        # Calculate adaptive thresholds
        adaptive_thresholds = {
            'min_movement_threshold': self.config.min_movement_threshold * face_scale_factor * quality_scale_factor,
            'significant_movement_threshold': self.config.significant_movement_threshold * face_scale_factor * quality_scale_factor,
            'horizontal_magnitude_scale': self.config.horizontal_magnitude_scale * frame_scale_x,
            'vertical_magnitude_scale': self.config.vertical_magnitude_scale * frame_scale_y,
            'movement_ratio_threshold': self.config.movement_ratio_threshold * quality_scale_factor,
            'face_scale_factor': face_scale_factor,
            'quality_scale_factor': quality_scale_factor,
            'frame_scale_x': frame_scale_x,
            'frame_scale_y': frame_scale_y
        }
        
        logger.debug(f"Adaptive thresholds calculated: face_size={face_size}, "
                    f"face_scale_factor={face_scale_factor:.3f}, "
                    f"quality_scale_factor={quality_scale_factor:.3f}, "
                    f"min_movement_threshold={adaptive_thresholds['min_movement_threshold']:.1f}")
        
        return adaptive_thresholds
    
    def _calculate_quality_scale_factor(self, quality_metrics: Dict[str, float]) -> float:
        """Calculate scaling factor based on video quality metrics."""
        # Extract quality indicators
        brightness = quality_metrics.get('brightness', 128)  # 0-255 scale
        contrast = quality_metrics.get('contrast', 50)       # Standard deviation
        sharpness = quality_metrics.get('sharpness', 100)    # Laplacian variance
        
        # Normalize quality metrics to 0-1 scale
        brightness_score = self._normalize_brightness_score(brightness)
        contrast_score = self._normalize_contrast_score(contrast)
        sharpness_score = self._normalize_sharpness_score(sharpness)
        
        # Calculate overall quality score
        overall_quality = (brightness_score + contrast_score + sharpness_score) / 3.0
        
        # Map quality score to threshold scaling factor
        if overall_quality >= 0.8:
            # High quality - can use lower thresholds
            scale_factor = self.config.high_quality_threshold_multiplier
        elif overall_quality <= 0.3:
            # Low quality - need higher thresholds
            scale_factor = self.config.low_quality_threshold_multiplier
        else:
            # Medium quality - interpolate between high and low
            # Linear interpolation between low and high quality multipliers
            t = (overall_quality - 0.3) / (0.8 - 0.3)  # Normalize to 0-1
            scale_factor = (self.config.low_quality_threshold_multiplier * (1 - t) + 
                          self.config.high_quality_threshold_multiplier * t)
        
        return scale_factor
    
    def _normalize_brightness_score(self, brightness: float) -> float:
        """Normalize brightness to quality score (0-1)."""
        # Optimal brightness range is around 80-180
        if 80 <= brightness <= 180:
            return 1.0
        elif brightness < 80:
            return max(0.0, brightness / 80)
        else:  # brightness > 180
            return max(0.0, (255 - brightness) / (255 - 180))
    
    def _normalize_contrast_score(self, contrast: float) -> float:
        """Normalize contrast to quality score (0-1)."""
        # Good contrast is typically above 30
        return min(1.0, max(0.0, contrast / 60))
    
    def _normalize_sharpness_score(self, sharpness: float) -> float:
        """Normalize sharpness to quality score (0-1)."""
        # Good sharpness is typically above 100
        return min(1.0, max(0.0, sharpness / 200))
    
    def _get_base_thresholds(self) -> Dict[str, float]:
        """Get base thresholds without adaptive scaling."""
        return {
            'min_movement_threshold': self.config.min_movement_threshold,
            'significant_movement_threshold': self.config.significant_movement_threshold,
            'horizontal_magnitude_scale': self.config.horizontal_magnitude_scale,
            'vertical_magnitude_scale': self.config.vertical_magnitude_scale,
            'movement_ratio_threshold': self.config.movement_ratio_threshold,
            'face_scale_factor': 1.0,
            'quality_scale_factor': 1.0,
            'frame_scale_x': 1.0,
            'frame_scale_y': 1.0
        }

@dataclass
class MovementValidationConfig:
    """Configuration for movement validation parameters."""
    
    # Minimum movement validation
    enable_minimum_movement_validation: bool = True
    minimum_movement_pixels: float = 8.0  # Absolute minimum movement in pixels
    minimum_movement_ratio: float = 0.01   # Minimum movement as ratio of frame size
    
    # Pixel-to-movement ratio validation
    enable_pixel_movement_ratio_validation: bool = True
    expected_movement_efficiency: float = 0.8  # Expected efficiency of head movement
    
    # Temporal consistency validation
    enable_temporal_consistency: bool = True
    consistency_window_size: int = 5  # Frames to check for consistency
    consistency_threshold: float = 0.6  # Minimum consistency ratio
    
    def validate_movement_magnitude(
        self, 
        movement_pixels: float, 
        frame_dimensions: Tuple[int, int],
        adaptive_thresholds: Dict[str, float]
    ) -> Tuple[bool, str]:
        """
        Validate if movement magnitude meets minimum requirements.
        
        Args:
            movement_pixels: Movement magnitude in pixels
            frame_dimensions: (width, height) of video frame
            adaptive_thresholds: Calculated adaptive thresholds
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if not self.enable_minimum_movement_validation:
            return True, "Validation disabled"
        
        # Check absolute minimum
        if movement_pixels < self.minimum_movement_pixels:
            return False, f"Movement too small: {movement_pixels:.1f} < {self.minimum_movement_pixels}"
        
        # Check adaptive threshold
        min_threshold = adaptive_thresholds.get('min_movement_threshold', self.minimum_movement_pixels)
        if movement_pixels < min_threshold:
            return False, f"Movement below adaptive threshold: {movement_pixels:.1f} < {min_threshold:.1f}"
        
        # Check ratio-based threshold
        if self.enable_pixel_movement_ratio_validation:
            frame_diagonal = np.sqrt(frame_dimensions[0]**2 + frame_dimensions[1]**2)
            movement_ratio = movement_pixels / frame_diagonal
            min_ratio = adaptive_thresholds.get('movement_ratio_threshold', self.minimum_movement_ratio)
            
            if movement_ratio < min_ratio:
                return False, f"Movement ratio too small: {movement_ratio:.4f} < {min_ratio:.4f}"
        
        return True, "Movement magnitude valid"

def create_default_movement_config() -> MovementThresholdConfig:
    """Create default movement threshold configuration."""
    return MovementThresholdConfig()

def create_adaptive_calculator(config: Optional[MovementThresholdConfig] = None) -> AdaptiveThresholdCalculator:
    """Create adaptive threshold calculator with optional custom config."""
    if config is None:
        config = create_default_movement_config()
    return AdaptiveThresholdCalculator(config)

def create_movement_validation_config() -> MovementValidationConfig:
    """Create default movement validation configuration."""
    return MovementValidationConfig()