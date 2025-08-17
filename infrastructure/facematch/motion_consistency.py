"""
Motion Consistency Validation Module
Provides advanced analysis of movement patterns, edge density, and motion blur detection
to identify unnatural or artificial motion and video artifacts.
"""

import logging
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import time

logger = logging.getLogger(__name__)

@dataclass
class MotionConsistencyConfig:
    """Configuration for motion consistency validation."""
    
    # Movement pattern analysis parameters
    enable_pattern_analysis: bool = True
    pattern_window_size: int = 5
    natural_motion_threshold: float = 0.6
    artificial_motion_penalty: float = 0.3
    
    # Edge density analysis parameters
    enable_edge_analysis: bool = True
    edge_low_threshold: int = 50
    edge_high_threshold: int = 150
    optimal_edge_density_min: float = 0.02
    optimal_edge_density_max: float = 0.15
    edge_quality_weight: float = 0.3
    
    # Motion blur detection parameters
    enable_blur_detection: bool = True
    blur_kernel_size: int = 3
    blur_threshold: float = 100.0
    motion_blur_penalty: float = 0.4
    
    # Temporal consistency parameters
    temporal_consistency_window: int = 7
    velocity_smoothness_threshold: float = 0.7
    acceleration_limit: float = 2.0
    
    # Overall scoring weights
    pattern_weight: float = 0.4
    edge_weight: float = 0.3
    blur_weight: float = 0.3

@dataclass
class MotionAnalysisResult:
    """Result of motion consistency analysis."""
    overall_score: float
    pattern_score: float
    edge_score: float
    blur_score: float
    is_natural_motion: bool
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FrameMotionData:
    """Motion data for a single frame."""
    frame_index: int
    timestamp: float
    velocity: Tuple[float, float]
    acceleration: Tuple[float, float]
    edge_density: float
    blur_score: float
    landmarks: Optional[Any] = None

class MotionConsistencyValidator:
    """Advanced motion consistency validation for detecting artificial motion and video artifacts."""
    
    def __init__(self, config: Optional[MotionConsistencyConfig] = None):
        """Initialize the motion consistency validator."""
        self.config = config or MotionConsistencyConfig()
        
        # Temporal buffers for motion analysis
        self.motion_history: deque = deque(maxlen=self.config.temporal_consistency_window)
        self.velocity_history: deque = deque(maxlen=self.config.pattern_window_size)
        self.acceleration_history: deque = deque(maxlen=self.config.pattern_window_size)
        
        # Edge detection kernel
        self.sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        self.sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        logger.info("MotionConsistencyValidator initialized")
    
    def analyze_motion_consistency(
        self, 
        frames: List[np.ndarray], 
        landmarks_sequence: List[Any],
        timestamps: Optional[List[float]] = None
    ) -> MotionAnalysisResult:
        """
        Analyze motion consistency across a sequence of frames.
        
        Args:
            frames: List of video frames
            landmarks_sequence: List of face landmarks for each frame
            timestamps: Optional timestamps for each frame
            
        Returns:
            MotionAnalysisResult with comprehensive analysis
        """
        try:
            if len(frames) < 3:
                logger.warning("Insufficient frames for motion consistency analysis")
                return MotionAnalysisResult(
                    overall_score=0.5,
                    pattern_score=0.5,
                    edge_score=0.5,
                    blur_score=0.5,
                    is_natural_motion=False,
                    confidence=0.0,
                    details={'error': 'Insufficient frames'}
                )
            
            # Generate timestamps if not provided
            if timestamps is None:
                timestamps = [i / 30.0 for i in range(len(frames))]  # Assume 30 FPS
            
            # Extract motion data for each frame
            motion_data = self._extract_motion_data(frames, landmarks_sequence, timestamps)
            
            # Analyze movement patterns
            pattern_score = self._analyze_movement_patterns(motion_data) if self.config.enable_pattern_analysis else 0.5
            
            # Analyze edge density across frames
            edge_score = self._analyze_edge_density(frames) if self.config.enable_edge_analysis else 0.5
            
            # Detect motion blur artifacts
            blur_score = self._detect_motion_blur(frames) if self.config.enable_blur_detection else 0.5
            
            # Calculate overall consistency score
            overall_score = (
                pattern_score * self.config.pattern_weight +
                edge_score * self.config.edge_weight +
                blur_score * self.config.blur_weight
            )
            
            # Determine if motion appears natural
            is_natural = overall_score >= self.config.natural_motion_threshold
            
            # Calculate confidence based on score distribution
            confidence = self._calculate_confidence(pattern_score, edge_score, blur_score)
            
            return MotionAnalysisResult(
                overall_score=overall_score,
                pattern_score=pattern_score,
                edge_score=edge_score,
                blur_score=blur_score,
                is_natural_motion=is_natural,
                confidence=confidence,
                details={
                    'frame_count': len(frames),
                    'motion_data_points': len(motion_data),
                    'analysis_timestamp': time.time()
                }
            )
            
        except Exception as e:
            logger.error(f"Error in motion consistency analysis: {e}")
            return MotionAnalysisResult(
                overall_score=0.0,
                pattern_score=0.0,
                edge_score=0.0,
                blur_score=0.0,
                is_natural_motion=False,
                confidence=0.0,
                details={'error': str(e)}
            )
    
    def _extract_motion_data(
        self, 
        frames: List[np.ndarray], 
        landmarks_sequence: List[Any], 
        timestamps: List[float]
    ) -> List[FrameMotionData]:
        """Extract motion data from frame sequence."""
        motion_data = []
        
        for i in range(len(frames)):
            try:
                frame = frames[i]
                landmarks = landmarks_sequence[i] if i < len(landmarks_sequence) else None
                timestamp = timestamps[i] if i < len(timestamps) else i / 30.0
                
                # Calculate velocity and acceleration if we have previous frames
                velocity = (0.0, 0.0)
                acceleration = (0.0, 0.0)
                
                if i > 0 and landmarks and landmarks_sequence[i-1]:
                    velocity = self._calculate_velocity(
                        landmarks_sequence[i-1], landmarks, 
                        timestamps[i-1] if i-1 < len(timestamps) else (i-1)/30.0, 
                        timestamp
                    )
                
                if i > 1 and len(motion_data) > 0:
                    prev_velocity = motion_data[-1].velocity
                    dt = timestamp - motion_data[-1].timestamp
                    if dt > 0:
                        acceleration = (
                            (velocity[0] - prev_velocity[0]) / dt,
                            (velocity[1] - prev_velocity[1]) / dt
                        )
                
                # Calculate edge density for this frame
                edge_density = self._calculate_frame_edge_density(frame)
                
                # Calculate blur score for this frame
                blur_score = self._calculate_frame_blur_score(frame)
                
                motion_data.append(FrameMotionData(
                    frame_index=i,
                    timestamp=timestamp,
                    velocity=velocity,
                    acceleration=acceleration,
                    edge_density=edge_density,
                    blur_score=blur_score,
                    landmarks=landmarks
                ))
                
            except Exception as e:
                logger.warning(f"Error extracting motion data for frame {i}: {e}")
                continue
        
        return motion_data
    
    def _calculate_velocity(
        self, 
        landmarks_prev: Any, 
        landmarks_curr: Any, 
        timestamp_prev: float, 
        timestamp_curr: float
    ) -> Tuple[float, float]:
        """Calculate velocity between two landmark sets."""
        try:
            dt = timestamp_curr - timestamp_prev
            if dt <= 0:
                return (0.0, 0.0)
            
            # Use nose tip landmark (index 1) as reference point
            if hasattr(landmarks_prev, 'landmark') and hasattr(landmarks_curr, 'landmark'):
                if len(landmarks_prev.landmark) > 1 and len(landmarks_curr.landmark) > 1:
                    prev_nose = landmarks_prev.landmark[1]
                    curr_nose = landmarks_curr.landmark[1]
                    
                    dx = curr_nose.x - prev_nose.x
                    dy = curr_nose.y - prev_nose.y
                    
                    velocity_x = dx / dt
                    velocity_y = dy / dt
                    
                    return (velocity_x, velocity_y)
            
            return (0.0, 0.0)
            
        except Exception as e:
            logger.warning(f"Error calculating velocity: {e}")
            return (0.0, 0.0)
    
    def _analyze_movement_patterns(self, motion_data: List[FrameMotionData]) -> float:
        """
        Analyze movement patterns to detect unnatural or artificial motion.
        
        Natural human head movement characteristics:
        - Smooth velocity changes
        - Limited acceleration
        - Consistent directional changes
        - Natural pauses and variations
        """
        try:
            if len(motion_data) < 3:
                return 0.5
            
            # Extract velocity and acceleration sequences
            velocities = [data.velocity for data in motion_data]
            accelerations = [data.acceleration for data in motion_data[1:]]  # Skip first frame
            
            # 1. Analyze velocity smoothness
            velocity_smoothness = self._calculate_velocity_smoothness(velocities)
            
            # 2. Analyze acceleration limits (detect sudden jerky movements)
            acceleration_naturalness = self._analyze_acceleration_patterns(accelerations)
            
            # 3. Analyze directional consistency
            directional_consistency = self._analyze_directional_patterns(velocities)
            
            # 4. Detect robotic/artificial patterns
            artificial_pattern_penalty = self._detect_artificial_patterns(motion_data)
            
            # Combine pattern analysis scores
            pattern_score = (
                velocity_smoothness * 0.3 +
                acceleration_naturalness * 0.3 +
                directional_consistency * 0.2 +
                (1.0 - artificial_pattern_penalty) * 0.2
            )
            
            return min(max(pattern_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error analyzing movement patterns: {e}")
            return 0.5
    
    def _calculate_velocity_smoothness(self, velocities: List[Tuple[float, float]]) -> float:
        """Calculate smoothness of velocity changes."""
        try:
            if len(velocities) < 3:
                return 0.5
            
            # Calculate velocity magnitudes
            velocity_magnitudes = [np.sqrt(vx**2 + vy**2) for vx, vy in velocities]
            
            # Calculate smoothness using variance of velocity changes
            velocity_changes = []
            for i in range(1, len(velocity_magnitudes)):
                change = abs(velocity_magnitudes[i] - velocity_magnitudes[i-1])
                velocity_changes.append(change)
            
            if not velocity_changes:
                return 0.5
            
            # Lower variance indicates smoother motion
            variance = np.var(velocity_changes)
            smoothness = max(0.0, 1.0 - variance / 0.1)  # Normalize by expected variance
            
            return min(smoothness, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating velocity smoothness: {e}")
            return 0.5
    
    def _analyze_acceleration_patterns(self, accelerations: List[Tuple[float, float]]) -> float:
        """Analyze acceleration patterns for naturalness."""
        try:
            if not accelerations:
                return 0.5
            
            # Calculate acceleration magnitudes
            accel_magnitudes = [np.sqrt(ax**2 + ay**2) for ax, ay in accelerations]
            
            # Check for excessive accelerations (unnatural jerky movements)
            excessive_accels = sum(1 for accel in accel_magnitudes if accel > self.config.acceleration_limit)
            excessive_ratio = excessive_accels / len(accel_magnitudes)
            
            # Natural motion should have limited excessive accelerations
            naturalness = max(0.0, 1.0 - excessive_ratio * 2.0)
            
            return min(naturalness, 1.0)
            
        except Exception as e:
            logger.warning(f"Error analyzing acceleration patterns: {e}")
            return 0.5
    
    def _analyze_directional_patterns(self, velocities: List[Tuple[float, float]]) -> float:
        """Analyze directional consistency and natural variation."""
        try:
            if len(velocities) < 4:
                return 0.5
            
            # Calculate direction angles
            directions = []
            for vx, vy in velocities:
                if vx != 0 or vy != 0:
                    angle = np.arctan2(vy, vx)
                    directions.append(angle)
            
            if len(directions) < 3:
                return 0.5
            
            # Analyze direction changes
            direction_changes = []
            for i in range(1, len(directions)):
                change = abs(directions[i] - directions[i-1])
                # Handle angle wrapping
                if change > np.pi:
                    change = 2 * np.pi - change
                direction_changes.append(change)
            
            # Natural motion has moderate directional variation
            mean_change = np.mean(direction_changes)
            
            # Optimal range for natural head movement direction changes
            if 0.2 <= mean_change <= 1.5:  # Radians
                consistency = 1.0
            else:
                # Penalize too little or too much directional change
                consistency = max(0.0, 1.0 - abs(mean_change - 0.85) / 2.0)
            
            return min(consistency, 1.0)
            
        except Exception as e:
            logger.warning(f"Error analyzing directional patterns: {e}")
            return 0.5
    
    def _detect_artificial_patterns(self, motion_data: List[FrameMotionData]) -> float:
        """Detect artificial or robotic movement patterns."""
        try:
            if len(motion_data) < 5:
                return 0.0
            
            penalty = 0.0
            
            # 1. Detect perfectly linear movements (too perfect to be human)
            linear_penalty = self._detect_linear_patterns(motion_data)
            penalty += linear_penalty * 0.4
            
            # 2. Detect repetitive patterns (robotic behavior)
            repetitive_penalty = self._detect_repetitive_patterns(motion_data)
            penalty += repetitive_penalty * 0.3
            
            # 3. Detect sudden stops/starts (unnatural motion)
            sudden_change_penalty = self._detect_sudden_changes(motion_data)
            penalty += sudden_change_penalty * 0.3
            
            return min(penalty, 1.0)
            
        except Exception as e:
            logger.warning(f"Error detecting artificial patterns: {e}")
            return 0.0
    
    def _detect_linear_patterns(self, motion_data: List[FrameMotionData]) -> float:
        """Detect overly linear movement patterns."""
        try:
            velocities = [data.velocity for data in motion_data]
            
            # Check for sequences of very similar velocities (too consistent)
            similar_velocity_count = 0
            threshold = 0.01  # Very small threshold for "identical" velocities
            
            for i in range(1, len(velocities)):
                vx_diff = abs(velocities[i][0] - velocities[i-1][0])
                vy_diff = abs(velocities[i][1] - velocities[i-1][1])
                
                if vx_diff < threshold and vy_diff < threshold:
                    similar_velocity_count += 1
            
            # High ratio of similar velocities indicates artificial motion
            similarity_ratio = similar_velocity_count / max(len(velocities) - 1, 1)
            
            return min(similarity_ratio * 2.0, 1.0)  # Amplify penalty
            
        except Exception as e:
            logger.warning(f"Error detecting linear patterns: {e}")
            return 0.0
    
    def _detect_repetitive_patterns(self, motion_data: List[FrameMotionData]) -> float:
        """Detect repetitive movement patterns."""
        try:
            if len(motion_data) < 6:
                return 0.0
            
            velocities = [data.velocity for data in motion_data]
            
            # Look for repeating velocity patterns
            pattern_length = 3  # Look for patterns of length 3
            repetitions = 0
            
            for i in range(len(velocities) - 2 * pattern_length):
                pattern1 = velocities[i:i + pattern_length]
                pattern2 = velocities[i + pattern_length:i + 2 * pattern_length]
                
                # Check if patterns are similar
                similarity = self._calculate_pattern_similarity(pattern1, pattern2)
                if similarity > 0.8:  # High similarity threshold
                    repetitions += 1
            
            repetition_ratio = repetitions / max(len(velocities) - 2 * pattern_length, 1)
            
            return min(repetition_ratio * 3.0, 1.0)  # Amplify penalty
            
        except Exception as e:
            logger.warning(f"Error detecting repetitive patterns: {e}")
            return 0.0
    
    def _detect_sudden_changes(self, motion_data: List[FrameMotionData]) -> float:
        """Detect sudden stops and starts in motion."""
        try:
            velocities = [data.velocity for data in motion_data]
            velocity_magnitudes = [np.sqrt(vx**2 + vy**2) for vx, vy in velocities]
            
            sudden_changes = 0
            threshold = 0.1  # Threshold for sudden change
            
            for i in range(1, len(velocity_magnitudes)):
                change_ratio = abs(velocity_magnitudes[i] - velocity_magnitudes[i-1])
                if velocity_magnitudes[i-1] > 0:
                    change_ratio /= velocity_magnitudes[i-1]
                
                if change_ratio > threshold:
                    sudden_changes += 1
            
            sudden_change_ratio = sudden_changes / max(len(velocity_magnitudes) - 1, 1)
            
            return min(sudden_change_ratio * 2.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Error detecting sudden changes: {e}")
            return 0.0
    
    def _calculate_pattern_similarity(
        self, 
        pattern1: List[Tuple[float, float]], 
        pattern2: List[Tuple[float, float]]
    ) -> float:
        """Calculate similarity between two velocity patterns."""
        try:
            if len(pattern1) != len(pattern2):
                return 0.0
            
            similarities = []
            for (vx1, vy1), (vx2, vy2) in zip(pattern1, pattern2):
                # Calculate cosine similarity
                mag1 = np.sqrt(vx1**2 + vy1**2)
                mag2 = np.sqrt(vx2**2 + vy2**2)
                
                if mag1 > 0 and mag2 > 0:
                    dot_product = vx1 * vx2 + vy1 * vy2
                    similarity = dot_product / (mag1 * mag2)
                    similarities.append(max(0.0, similarity))
                else:
                    similarities.append(1.0 if mag1 == mag2 == 0 else 0.0)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating pattern similarity: {e}")
            return 0.0
    
    def _analyze_edge_density(self, frames: List[np.ndarray]) -> float:
        """Analyze edge density across frames for image quality assessment."""
        try:
            edge_scores = []
            
            for frame in frames:
                edge_density = self._calculate_frame_edge_density(frame)
                
                # Score based on optimal edge density range
                if self.config.optimal_edge_density_min <= edge_density <= self.config.optimal_edge_density_max:
                    score = 1.0
                else:
                    # Penalize values outside optimal range
                    if edge_density < self.config.optimal_edge_density_min:
                        distance = self.config.optimal_edge_density_min - edge_density
                    else:
                        distance = edge_density - self.config.optimal_edge_density_max
                    
                    score = max(0.0, 1.0 - distance * 10.0)
                
                edge_scores.append(score)
            
            # Return average edge quality score
            return np.mean(edge_scores) if edge_scores else 0.5
            
        except Exception as e:
            logger.error(f"Error analyzing edge density: {e}")
            return 0.5
    
    def _calculate_frame_edge_density(self, frame: np.ndarray) -> float:
        """Calculate edge density for a single frame."""
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Apply Canny edge detection
            edges = cv2.Canny(
                gray, 
                self.config.edge_low_threshold, 
                self.config.edge_high_threshold
            )
            
            # Calculate edge density
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.size
            edge_density = edge_pixels / total_pixels
            
            return edge_density
            
        except Exception as e:
            logger.warning(f"Error calculating frame edge density: {e}")
            return 0.0
    
    def _detect_motion_blur(self, frames: List[np.ndarray]) -> float:
        """Detect motion blur artifacts across frames."""
        try:
            blur_scores = []
            
            for frame in frames:
                blur_score = self._calculate_frame_blur_score(frame)
                
                # Higher blur score means less blur (better quality)
                # Normalize to 0-1 range where 1 is good quality
                normalized_score = min(blur_score / self.config.blur_threshold, 1.0)
                blur_scores.append(normalized_score)
            
            # Return average blur quality score
            return np.mean(blur_scores) if blur_scores else 0.5
            
        except Exception as e:
            logger.error(f"Error detecting motion blur: {e}")
            return 0.5
    
    def _calculate_frame_blur_score(self, frame: np.ndarray) -> float:
        """Calculate blur score for a single frame using Laplacian variance."""
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Calculate Laplacian variance (higher = less blur)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            blur_score = laplacian.var()
            
            return blur_score
            
        except Exception as e:
            logger.warning(f"Error calculating frame blur score: {e}")
            return 0.0
    
    def _calculate_confidence(self, pattern_score: float, edge_score: float, blur_score: float) -> float:
        """Calculate confidence based on score distribution and consistency."""
        try:
            scores = [pattern_score, edge_score, blur_score]
            
            # Calculate mean and standard deviation
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # High confidence when scores are consistent and high
            consistency_factor = max(0.0, 1.0 - std_score * 2.0)  # Lower std = higher consistency
            quality_factor = mean_score  # Higher mean = higher quality
            
            confidence = (consistency_factor + quality_factor) / 2.0
            
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 0.5
    
    def reset_temporal_state(self) -> None:
        """Reset temporal buffers for new analysis."""
        self.motion_history.clear()
        self.velocity_history.clear()
        self.acceleration_history.clear()
        logger.info("Motion consistency temporal state reset")

def create_default_motion_consistency_config() -> MotionConsistencyConfig:
    """Create default motion consistency configuration."""
    return MotionConsistencyConfig()

def create_motion_consistency_validator(
    config: Optional[MotionConsistencyConfig] = None
) -> MotionConsistencyValidator:
    """Create motion consistency validator with optional custom config."""
    return MotionConsistencyValidator(config or create_default_motion_consistency_config())