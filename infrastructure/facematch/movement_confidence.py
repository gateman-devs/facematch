"""
Comprehensive Movement Confidence Scoring System
Provides advanced confidence calculation, temporal smoothing, and filtering for movement detection.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import time

logger = logging.getLogger(__name__)

@dataclass
class MovementData:
    """Data structure for movement information."""
    direction: str
    confidence: float
    timestamp: float
    magnitude: float
    start_position: Tuple[float, float]
    end_position: Tuple[float, float]
    frame_indices: Tuple[int, int]
    dx_pixels: float = 0.0
    dy_pixels: float = 0.0
    dx_norm: float = 0.0
    dy_norm: float = 0.0
    raw_confidence: float = 0.0
    temporal_confidence: float = 0.0
    consistency_score: float = 0.0

@dataclass
class ConfidenceConfig:
    """Configuration for movement confidence scoring."""
    
    # Base confidence calculation parameters
    magnitude_weight: float = 0.4
    consistency_weight: float = 0.3
    temporal_weight: float = 0.2
    quality_weight: float = 0.1
    
    # Temporal smoothing parameters
    enable_temporal_smoothing: bool = True
    temporal_window_size: int = 5
    temporal_decay_factor: float = 0.8
    
    # Consistency analysis parameters
    consistency_window_size: int = 7
    direction_consistency_threshold: float = 0.6
    magnitude_consistency_threshold: float = 0.7
    
    # Confidence filtering parameters
    min_confidence_threshold: float = 0.2
    high_confidence_threshold: float = 0.8
    enable_confidence_filtering: bool = True
    
    # Noise reduction parameters
    noise_reduction_enabled: bool = True
    outlier_detection_threshold: float = 2.0  # Standard deviations
    minimum_movement_samples: int = 3

class MovementConfidenceScorer:
    """Advanced movement confidence scoring system with temporal smoothing and filtering."""
    
    def __init__(self, config: Optional[ConfidenceConfig] = None):
        """Initialize the confidence scorer with configuration."""
        self.config = config or ConfidenceConfig()
        
        # Temporal smoothing buffers
        self.movement_history: deque = deque(maxlen=self.config.temporal_window_size)
        self.confidence_history: deque = deque(maxlen=self.config.temporal_window_size)
        
        # Statistics for outlier detection
        self.magnitude_stats = {'mean': 0.0, 'std': 1.0, 'count': 0}
        self.confidence_stats = {'mean': 0.0, 'std': 1.0, 'count': 0}
        
        logger.info("MovementConfidenceScorer initialized with temporal smoothing and filtering")
    
    def calculate_comprehensive_confidence(
        self, 
        movement_data: Dict[str, Any],
        frame_context: Optional[Dict[str, Any]] = None,
        historical_movements: Optional[List[Dict[str, Any]]] = None
    ) -> MovementData:
        """
        Calculate comprehensive confidence score for a movement.
        
        Args:
            movement_data: Basic movement information
            frame_context: Additional frame quality and context information
            historical_movements: Previous movements for temporal analysis
            
        Returns:
            Enhanced MovementData with comprehensive confidence scores
        """
        try:
            # Extract basic movement information
            direction = movement_data.get('direction', 'none')
            magnitude = movement_data.get('magnitude', 0.0)
            dx_pixels = movement_data.get('dx_pixels', 0.0)
            dy_pixels = movement_data.get('dy_pixels', 0.0)
            timestamp = movement_data.get('timestamp', time.time())
            
            # Calculate raw confidence based on magnitude
            raw_confidence = self._calculate_magnitude_confidence(
                magnitude, dx_pixels, dy_pixels, direction
            )
            
            # Calculate consistency score
            consistency_score = self._calculate_consistency_score(
                movement_data, historical_movements or []
            )
            
            # Calculate temporal confidence with smoothing
            temporal_confidence = self._calculate_temporal_confidence(
                raw_confidence, direction
            )
            
            # Calculate quality-based confidence adjustment
            quality_adjustment = self._calculate_quality_adjustment(frame_context)
            
            # Combine all confidence components
            final_confidence = self._combine_confidence_scores(
                raw_confidence, consistency_score, temporal_confidence, quality_adjustment
            )
            
            # Create enhanced movement data
            enhanced_movement = MovementData(
                direction=direction,
                confidence=final_confidence,
                timestamp=timestamp,
                magnitude=magnitude,
                start_position=movement_data.get('start_position', (0.0, 0.0)),
                end_position=movement_data.get('end_position', (0.0, 0.0)),
                frame_indices=movement_data.get('frame_indices', (0, 0)),
                dx_pixels=dx_pixels,
                dy_pixels=dy_pixels,
                dx_norm=movement_data.get('dx_norm', 0.0),
                dy_norm=movement_data.get('dy_norm', 0.0),
                raw_confidence=raw_confidence,
                temporal_confidence=temporal_confidence,
                consistency_score=consistency_score
            )
            
            # Update temporal buffers
            self._update_temporal_buffers(enhanced_movement)
            
            # Update statistics for outlier detection
            self._update_statistics(enhanced_movement)
            
            return enhanced_movement
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive confidence: {e}")
            # Return basic movement data with minimal confidence
            return MovementData(
                direction=movement_data.get('direction', 'none'),
                confidence=0.1,
                timestamp=movement_data.get('timestamp', time.time()),
                magnitude=movement_data.get('magnitude', 0.0),
                start_position=movement_data.get('start_position', (0.0, 0.0)),
                end_position=movement_data.get('end_position', (0.0, 0.0)),
                frame_indices=movement_data.get('frame_indices', (0, 0))
            )
    
    def _calculate_magnitude_confidence(
        self, 
        magnitude: float, 
        dx_pixels: float, 
        dy_pixels: float, 
        direction: str
    ) -> float:
        """Calculate confidence based on movement magnitude and direction consistency."""
        if magnitude <= 0:
            return 0.0
        
        # Base confidence from magnitude (normalized to reasonable range)
        # Typical head movements are 10-100 pixels
        magnitude_confidence = min(magnitude / 50.0, 1.0)
        
        # Direction-specific confidence adjustments
        abs_dx = abs(dx_pixels)
        abs_dy = abs(dy_pixels)
        
        if direction in ['left', 'right']:
            # For horizontal movements, horizontal component should dominate
            if abs_dx > abs_dy:
                direction_consistency = abs_dx / (abs_dx + abs_dy + 1e-6)
            else:
                direction_consistency = 0.5  # Mixed movement
        elif direction in ['up', 'down']:
            # For vertical movements, vertical component should dominate
            if abs_dy > abs_dx:
                direction_consistency = abs_dy / (abs_dx + abs_dy + 1e-6)
            else:
                direction_consistency = 0.5  # Mixed movement
        else:
            direction_consistency = 0.0
        
        # Combine magnitude and direction consistency
        raw_confidence = (magnitude_confidence * 0.7 + direction_consistency * 0.3)
        
        return min(max(raw_confidence, 0.0), 1.0)
    
    def _calculate_consistency_score(
        self, 
        current_movement: Dict[str, Any], 
        historical_movements: List[Dict[str, Any]]
    ) -> float:
        """Calculate consistency score based on historical movement patterns."""
        if not historical_movements or len(historical_movements) < 2:
            return 0.5  # Neutral score for insufficient history
        
        try:
            # Get recent movements for consistency analysis
            recent_movements = historical_movements[-self.config.consistency_window_size:]
            
            current_direction = current_movement.get('direction', 'none')
            current_magnitude = current_movement.get('magnitude', 0.0)
            
            # Direction consistency: how often recent movements were in similar directions
            similar_directions = sum(
                1 for mov in recent_movements 
                if mov.get('direction') == current_direction
            )
            direction_consistency = similar_directions / len(recent_movements)
            
            # Magnitude consistency: how similar current magnitude is to recent magnitudes
            recent_magnitudes = [mov.get('magnitude', 0.0) for mov in recent_movements]
            if recent_magnitudes:
                mean_magnitude = np.mean(recent_magnitudes)
                std_magnitude = np.std(recent_magnitudes) + 1e-6
                
                # Calculate z-score for current magnitude
                z_score = abs(current_magnitude - mean_magnitude) / std_magnitude
                magnitude_consistency = max(0.0, 1.0 - z_score / 3.0)  # 3-sigma rule
            else:
                magnitude_consistency = 0.5
            
            # Combine consistency scores
            overall_consistency = (
                direction_consistency * 0.6 + 
                magnitude_consistency * 0.4
            )
            
            return min(max(overall_consistency, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating consistency score: {e}")
            return 0.5
    
    def _calculate_temporal_confidence(self, raw_confidence: float, direction: str) -> float:
        """Calculate temporal confidence using smoothing and historical data."""
        if not self.config.enable_temporal_smoothing:
            return raw_confidence
        
        try:
            # Add current confidence to history
            self.confidence_history.append(raw_confidence)
            
            if len(self.confidence_history) < 2:
                return raw_confidence
            
            # Apply exponential smoothing
            smoothed_confidence = raw_confidence
            weight = 1.0
            
            for i, historical_confidence in enumerate(reversed(list(self.confidence_history)[:-1])):
                weight *= self.config.temporal_decay_factor
                smoothed_confidence += historical_confidence * weight
            
            # Normalize by total weight
            total_weight = sum(
                self.config.temporal_decay_factor ** i 
                for i in range(len(self.confidence_history))
            )
            smoothed_confidence /= total_weight
            
            # Boost confidence if direction is consistent with recent movements
            if len(self.movement_history) >= 2:
                recent_directions = [mov.direction for mov in list(self.movement_history)[-3:]]
                direction_consistency = sum(1 for d in recent_directions if d == direction) / len(recent_directions)
                
                if direction_consistency > self.config.direction_consistency_threshold:
                    smoothed_confidence *= 1.1  # 10% boost for consistent direction
            
            return min(max(smoothed_confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating temporal confidence: {e}")
            return raw_confidence
    
    def _calculate_quality_adjustment(self, frame_context: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence adjustment based on frame quality metrics."""
        if not frame_context:
            return 1.0  # No adjustment if no context provided
        
        try:
            # Extract quality metrics
            brightness = frame_context.get('brightness', 128.0)
            contrast = frame_context.get('contrast', 50.0)
            sharpness = frame_context.get('sharpness', 100.0)
            
            # Normalize quality scores (0-1 range)
            brightness_score = self._normalize_brightness_quality(brightness)
            contrast_score = self._normalize_contrast_quality(contrast)
            sharpness_score = self._normalize_sharpness_quality(sharpness)
            
            # Calculate overall quality adjustment
            quality_score = (brightness_score + contrast_score + sharpness_score) / 3.0
            
            # Map quality score to adjustment factor (0.7 to 1.3 range)
            adjustment_factor = 0.7 + (quality_score * 0.6)
            
            return min(max(adjustment_factor, 0.5), 1.5)
            
        except Exception as e:
            logger.warning(f"Error calculating quality adjustment: {e}")
            return 1.0
    
    def _normalize_brightness_quality(self, brightness: float) -> float:
        """Normalize brightness to quality score (0-1)."""
        # Optimal brightness range is 80-180
        if 80 <= brightness <= 180:
            return 1.0
        elif brightness < 80:
            return max(0.0, brightness / 80.0)
        else:  # brightness > 180
            return max(0.0, (255 - brightness) / 75.0)
    
    def _normalize_contrast_quality(self, contrast: float) -> float:
        """Normalize contrast to quality score (0-1)."""
        # Good contrast is typically above 30
        return min(1.0, max(0.0, contrast / 60.0))
    
    def _normalize_sharpness_quality(self, sharpness: float) -> float:
        """Normalize sharpness to quality score (0-1)."""
        # Good sharpness is typically above 100
        return min(1.0, max(0.0, sharpness / 200.0))
    
    def _combine_confidence_scores(
        self, 
        raw_confidence: float, 
        consistency_score: float, 
        temporal_confidence: float, 
        quality_adjustment: float
    ) -> float:
        """Combine all confidence components into final score."""
        # Weighted combination of confidence components
        combined_confidence = (
            raw_confidence * self.config.magnitude_weight +
            consistency_score * self.config.consistency_weight +
            temporal_confidence * self.config.temporal_weight +
            raw_confidence * self.config.quality_weight  # Quality affects base confidence
        )
        
        # Apply quality adjustment
        final_confidence = combined_confidence * quality_adjustment
        
        return min(max(final_confidence, 0.0), 1.0)
    
    def _update_temporal_buffers(self, movement: MovementData) -> None:
        """Update temporal buffers with new movement data."""
        self.movement_history.append(movement)
    
    def _update_statistics(self, movement: MovementData) -> None:
        """Update running statistics for outlier detection."""
        try:
            # Update magnitude statistics
            self.magnitude_stats['count'] += 1
            n = self.magnitude_stats['count']
            
            # Online mean and variance calculation (Welford's algorithm)
            delta = movement.magnitude - self.magnitude_stats['mean']
            self.magnitude_stats['mean'] += delta / n
            
            if n > 1:
                delta2 = movement.magnitude - self.magnitude_stats['mean']
                variance = ((n - 2) * self.magnitude_stats['std'] ** 2 + delta * delta2) / (n - 1)
                self.magnitude_stats['std'] = np.sqrt(max(variance, 1e-6))
            
            # Update confidence statistics
            delta_conf = movement.confidence - self.confidence_stats['mean']
            self.confidence_stats['mean'] += delta_conf / n
            
            if n > 1:
                delta2_conf = movement.confidence - self.confidence_stats['mean']
                conf_variance = ((n - 2) * self.confidence_stats['std'] ** 2 + delta_conf * delta2_conf) / (n - 1)
                self.confidence_stats['std'] = np.sqrt(max(conf_variance, 1e-6))
            
            self.confidence_stats['count'] = n
            
        except Exception as e:
            logger.warning(f"Error updating statistics: {e}")
    
    def filter_movements_by_confidence(
        self, 
        movements: List[MovementData]
    ) -> List[MovementData]:
        """Filter movements based on confidence scores and outlier detection."""
        if not self.config.enable_confidence_filtering:
            return movements
        
        try:
            # Basic confidence filtering
            confidence_filtered = [
                mov for mov in movements 
                if mov.confidence >= self.config.min_confidence_threshold
            ]
            
            if not self.config.noise_reduction_enabled or len(confidence_filtered) < self.config.minimum_movement_samples:
                return confidence_filtered
            
            # Outlier detection based on magnitude and confidence
            filtered_movements = []
            
            for movement in confidence_filtered:
                is_outlier = self._is_movement_outlier(movement)
                
                if not is_outlier:
                    filtered_movements.append(movement)
                else:
                    logger.debug(f"Filtered outlier movement: {movement.direction} "
                               f"(confidence: {movement.confidence:.3f}, magnitude: {movement.magnitude:.1f})")
            
            return filtered_movements
            
        except Exception as e:
            logger.error(f"Error filtering movements by confidence: {e}")
            return movements
    
    def _is_movement_outlier(self, movement: MovementData) -> bool:
        """Determine if a movement is an outlier based on statistical analysis."""
        try:
            if self.magnitude_stats['count'] < self.config.minimum_movement_samples:
                return False  # Not enough data for outlier detection
            
            # Check magnitude outlier
            magnitude_z_score = abs(
                movement.magnitude - self.magnitude_stats['mean']
            ) / (self.magnitude_stats['std'] + 1e-6)
            
            # Check confidence outlier (unusually low confidence)
            confidence_z_score = abs(
                movement.confidence - self.confidence_stats['mean']
            ) / (self.confidence_stats['std'] + 1e-6)
            
            # Consider it an outlier if magnitude is too extreme or confidence is unusually low
            is_magnitude_outlier = magnitude_z_score > self.config.outlier_detection_threshold
            is_confidence_outlier = (
                movement.confidence < self.confidence_stats['mean'] - 
                self.config.outlier_detection_threshold * self.confidence_stats['std']
            )
            
            return is_magnitude_outlier or is_confidence_outlier
            
        except Exception as e:
            logger.warning(f"Error in outlier detection: {e}")
            return False
    
    def get_confidence_statistics(self) -> Dict[str, Any]:
        """Get current confidence scoring statistics."""
        return {
            'movement_history_size': len(self.movement_history),
            'confidence_history_size': len(self.confidence_history),
            'magnitude_stats': self.magnitude_stats.copy(),
            'confidence_stats': self.confidence_stats.copy(),
            'config': {
                'temporal_smoothing_enabled': self.config.enable_temporal_smoothing,
                'confidence_filtering_enabled': self.config.enable_confidence_filtering,
                'noise_reduction_enabled': self.config.noise_reduction_enabled,
                'min_confidence_threshold': self.config.min_confidence_threshold,
                'high_confidence_threshold': self.config.high_confidence_threshold
            }
        }
    
    def reset_temporal_state(self) -> None:
        """Reset temporal buffers and statistics."""
        self.movement_history.clear()
        self.confidence_history.clear()
        self.magnitude_stats = {'mean': 0.0, 'std': 1.0, 'count': 0}
        self.confidence_stats = {'mean': 0.0, 'std': 1.0, 'count': 0}
        logger.info("Temporal state reset")

def create_default_confidence_config() -> ConfidenceConfig:
    """Create default confidence configuration."""
    return ConfidenceConfig()

def create_confidence_scorer(config: Optional[ConfidenceConfig] = None) -> MovementConfidenceScorer:
    """Create movement confidence scorer with optional custom config."""
    return MovementConfidenceScorer(config or create_default_confidence_config())