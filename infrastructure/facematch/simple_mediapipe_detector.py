"""
Simplified MediaPipe Head Movement Detector
A streamlined implementation for head movement detection using MediaPipe Face Mesh.
"""

import logging
import time
import numpy as np
import cv2
import mediapipe as mp
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class MovementResult:
    """Result of movement detection."""
    direction: str
    confidence: float
    magnitude: float
    timestamp: float
    pose_data: Dict[str, float]

@dataclass
class DetectionResult:
    """Result of video processing."""
    success: bool
    movements: List[MovementResult]
    processing_time: float
    frames_processed: int
    error: Optional[str] = None

class SimpleMediaPipeDetector:
    """
    Enhanced MediaPipe-based head movement detector using FaceMesh with adaptive thresholds,
    temporal smoothing, and improved direction detection.
    """
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 movement_threshold: float = 0.02,
                 max_history: int = 10,
                 debug_mode: bool = False):
        """
        Initialize the detector with enhanced features.
        
        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
            movement_threshold: Base movement magnitude threshold (will be adapted)
            max_history: Maximum number of poses to keep in history
            debug_mode: Enable debug logging for troubleshooting
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Landmark indices definition - Updated with more reliable indices
        self.landmarks = {
            'nose_tip': 1,      # Keep this - it's reliable
            'chin': 18,         # Use 18 instead of 175
            'left_eye': 33,     # Good
            'right_eye': 263,   # Good
            'forehead': 9       # Use 9 instead of 10
        }
        
        # Alternative: Use landmark groups for better reliability
        self.face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        # Movement detection parameters
        self.base_movement_threshold = movement_threshold
        self.movement_threshold = movement_threshold  # Will be adapted based on face size
        self.debug_mode = debug_mode
        self.previous_pose = None
        self.movement_history = deque(maxlen=max_history)
        self.pose_history = deque(maxlen=max_history)
        
        # Temporal smoothing parameters
        self.smoothing_weights = [0.1, 0.3, 0.6]  # Weights for last 3 poses
        self.min_poses_for_smoothing = 3
        
        # Direction detection parameters
        self.direction_angles = {
            'right': (-45, 45),
            'down': (45, 135),
            'left': (135, 225),  # 135 to -135 (225)
            'up': (225, 315)     # -135 to -45 (315)
        }
        self.direction_hysteresis = 10  # Degrees of hysteresis to prevent flipping
        self.last_direction = None
        
        # Statistics for adaptive threshold
        self.total_frames = 0
        self.frames_with_face = 0
        self.movement_magnitudes = []
        self.face_sizes = []  # Track face sizes for adaptive threshold
        
        # Movement cooldown to prevent rapid-fire detections
        self.last_movement_time = 0
        self.movement_cooldown = 0.2
        
        # Movement validation state
        self.initial_face_center = None
        self.face_center_threshold = 0.1
        self.return_movement_threshold = 0.05
        self.last_movement_direction = None
        self.center_position = None
        
        # Quality validation parameters
        self.min_eye_distance = 0.01
        self.min_face_height = 0.01
        self.min_quality_score = 0.3
        
        logger.info("Enhanced MediaPipe detector initialized")
    
    def calibrate_thresholds(self, video_path: str, calibration_frames: int = 60):
        """Calibrate movement thresholds based on natural head micro-movements"""
        cap = cv2.VideoCapture(video_path)
        movements = []
        
        frame_count = 0
        while frame_count < calibration_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame and collect movement data
            result = self._process_frame(frame, frame_count)
            if result and result.get('pose'):
                if self.previous_pose:
                    dx = result['pose']['x'] - self.previous_pose['x']
                    dy = result['pose']['y'] - self.previous_pose['y']
                    magnitude = np.sqrt(dx**2 + dy**2)
                    movements.append(magnitude)
                self.previous_pose = result['pose']
            
            frame_count += 1
        
        cap.release()
        
        if movements:
            # Set threshold at 95th percentile of natural movements
            self.base_movement_threshold = np.percentile(movements, 95) * 1.5
            logger.info(f"Calibrated threshold: {self.base_movement_threshold:.4f}")
        else:
            logger.warning("No movement data collected during calibration")
    
    def _validate_face_detection(self, landmarks, image_shape) -> bool:
        """Validate that face detection is reliable"""
        try:
            # Check minimum number of landmarks
            if len(landmarks) < 400:  # FaceMesh should have 468 landmarks
                return False
                
            # Check that key landmarks are within frame bounds
            key_points = [landmarks[i] for i in [1, 33, 263, 18, 9]]
            for point in key_points:
                if not (0 <= point.x <= 1 and 0 <= point.y <= 1):
                    return False
                    
            # Check face size is reasonable
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            eye_distance = abs(right_eye.x - left_eye.x)
            
            if eye_distance < 0.01 or eye_distance > 0.5:  # Too small or too large
                return False
                
            return True
            
        except (IndexError, AttributeError):
            return False
    
    def get_head_pose(self, landmarks, image_shape) -> Optional[Dict[str, float]]:
        """
        Calculate enhanced head pose from facial landmarks with quality validation.
        
        Args:
            landmarks: MediaPipe face mesh landmarks
            image_shape: Image dimensions (height, width)
            
        Returns:
            Head pose data or None if landmarks are invalid or quality is poor
        """
        try:
            # Extract key landmarks with error handling
            nose_tip = landmarks[self.landmarks['nose_tip']]
            chin = landmarks[self.landmarks['chin']]
            left_eye = landmarks[self.landmarks['left_eye']]
            right_eye = landmarks[self.landmarks['right_eye']]
            forehead = landmarks[self.landmarks['forehead']]
            
            # Calculate head center (between eyes)
            x_center = (left_eye.x + right_eye.x) / 2
            y_center = (left_eye.y + right_eye.y) / 2
            
            # Calculate face metrics
            eye_distance = np.sqrt((right_eye.x - left_eye.x)**2 + (right_eye.y - left_eye.y)**2)
            face_height = abs(chin.y - forehead.y)
            
            # Quality validation
            if eye_distance < self.min_eye_distance or face_height < self.min_face_height:
                if self.debug_mode:
                    logger.debug(f"Face too small: eye_distance={eye_distance:.4f}, face_height={face_height:.4f}")
                return None
            
            # Calculate normalized pose ratios
            yaw_ratio = (nose_tip.x - x_center) / (eye_distance + 1e-6)
            pitch_ratio = (nose_tip.y - y_center) / (face_height + 1e-6)
            
            # Calculate quality score based on face size
            quality_score = min(eye_distance * 10, 1.0)
            
            # Enhanced pose data
            pose_data = {
                'x': x_center,
                'y': y_center,
                'nose_y': nose_tip.y,
                'chin_y': chin.y,
                'eye_distance': eye_distance,
                'face_height': face_height,
                'yaw_ratio': yaw_ratio,
                'pitch_ratio': pitch_ratio,
                'quality_score': quality_score,
                'confidence': quality_score
            }
            
            # Apply temporal smoothing if we have enough history
            if len(self.pose_history) >= self.min_poses_for_smoothing:
                pose_data = self._apply_temporal_smoothing(pose_data)
            
            return pose_data
            
        except (IndexError, AttributeError) as e:
            logger.warning(f"Failed to extract head pose: {e}")
            return None
    
    def _apply_temporal_smoothing(self, current_pose: Dict[str, float]) -> Dict[str, float]:
        """
        Apply weighted moving average smoothing to pose coordinates.
        
        Args:
            current_pose: Current pose data
            
        Returns:
            Smoothed pose data
        """
        if len(self.pose_history) < self.min_poses_for_smoothing:
            return current_pose
        
        # Get recent poses for smoothing
        recent_poses = list(self.pose_history)[-self.min_poses_for_smoothing:]
        
        # Smooth key coordinates
        smoothed_pose = current_pose.copy()
        
        # Smooth x, y coordinates
        x_values = [p['x'] for p in recent_poses] + [current_pose['x']]
        y_values = [p['y'] for p in recent_poses] + [current_pose['y']]
        
        smoothed_pose['x'] = np.average(x_values, weights=self.smoothing_weights + [0.6])
        smoothed_pose['y'] = np.average(y_values, weights=self.smoothing_weights + [0.6])
        
        # Smooth pose ratios if available
        if 'yaw_ratio' in current_pose:
            yaw_values = [p.get('yaw_ratio', 0) for p in recent_poses] + [current_pose['yaw_ratio']]
            smoothed_pose['yaw_ratio'] = np.average(yaw_values, weights=self.smoothing_weights + [0.6])
        
        if 'pitch_ratio' in current_pose:
            pitch_values = [p.get('pitch_ratio', 0) for p in recent_poses] + [current_pose['pitch_ratio']]
            smoothed_pose['pitch_ratio'] = np.average(pitch_values, weights=self.smoothing_weights + [0.6])
        
        return smoothed_pose
    
    def _calculate_adaptive_threshold(self, eye_distance: float) -> float:
        """
        Calculate adaptive movement threshold based on face size.
        
        Args:
            eye_distance: Distance between eyes (face size indicator)
            
        Returns:
            Adaptive movement threshold
        """
        # Base threshold adjustment based on face size
        # Larger faces (closer to camera) need smaller thresholds
        # Smaller faces (farther from camera) need larger thresholds
        
        if eye_distance < 0.05:  # Face far from camera
            return self.base_movement_threshold * 1.5
        elif eye_distance > 0.15:  # Face close to camera
            return self.base_movement_threshold * 0.7
        else:  # Normal distance
            return self.base_movement_threshold
    
    def _detect_direction_angle(self, dx: float, dy: float) -> str:
        """
        Detect movement direction using angle-based approach.
        
        Args:
            dx: X movement component
            dy: Y movement component
            
        Returns:
            Movement direction
        """
        # Calculate movement angle
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        # Normalize angle to 0-360 range
        if angle < 0:
            angle += 360
        
        # Determine direction based on angle ranges
        direction = None
        for dir_name, (start_angle, end_angle) in self.direction_angles.items():
            if start_angle <= angle < end_angle:
                direction = dir_name
                break
        
        # Apply hysteresis to prevent direction flipping
        if self.last_direction and direction != self.last_direction:
            # Check if the angle change is small (within hysteresis range)
            last_angle = self._direction_to_angle(self.last_direction)
            angle_diff = abs(angle - last_angle)
            if angle_diff < self.direction_hysteresis:
                direction = self.last_direction  # Keep previous direction
        
        self.last_direction = direction
        return direction
    
    def _detect_direction_improved(self, dx: float, dy: float, magnitude: float) -> str:
        """Improved direction detection with better angle handling"""
        
        # Use dominant axis approach for clearer directions
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        
        # Require minimum movement ratio to avoid noise
        min_ratio = 0.3  # Secondary axis should be at least 30% of primary
        
        if abs_dx > abs_dy:
            # Horizontal movement dominant
            if abs_dy / abs_dx > min_ratio:
                # Mixed movement - use angle
                angle = np.arctan2(dy, dx) * 180 / np.pi
                return self._angle_to_direction(angle)
            else:
                # Pure horizontal
                return 'right' if dx > 0 else 'left'
        else:
            # Vertical movement dominant  
            if abs_dx / abs_dy > min_ratio:
                # Mixed movement - use angle
                angle = np.arctan2(dy, dx) * 180 / np.pi
                return self._angle_to_direction(angle)
            else:
                # Pure vertical
                return 'down' if dy > 0 else 'up'

    def _angle_to_direction(self, angle: float) -> str:
        """Convert angle to direction with proper quadrant handling"""
        # Normalize angle to 0-360
        angle = angle % 360
        
        if 315 <= angle or angle < 45:
            return 'right'
        elif 45 <= angle < 135:
            return 'down'
        elif 135 <= angle < 225:
            return 'left'
        else:  # 225 <= angle < 315
            return 'up'
    
    def _direction_to_angle(self, direction: str) -> float:
        """
        Convert direction to center angle.
        
        Args:
            direction: Movement direction
            
        Returns:
            Center angle for the direction
        """
        direction_centers = {
            'right': 0,
            'down': 90,
            'left': 180,
            'up': 270
        }
        return direction_centers.get(direction, 0)
    
    def detect_movement(self, current_pose: Dict[str, float], timestamp: float) -> Optional[MovementResult]:
        """
        Detect movement between current and previous pose with enhanced logic.
        
        Args:
            current_pose: Current head pose data
            timestamp: Current timestamp
            
        Returns:
            Movement result or None if no significant movement
        """
        # Quality validation
        if current_pose.get('quality_score', 0) < self.min_quality_score:
            if self.debug_mode:
                logger.debug(f"Pose quality too low: {current_pose.get('quality_score', 0):.3f}")
            return None
        
        if self.previous_pose is None:
            self.previous_pose = current_pose
            self.pose_history.append(current_pose)
            if self.debug_mode:
                logger.debug(f"First pose detected: x={current_pose['x']:.4f}, y={current_pose['y']:.4f}")
            return None
        
        # Calculate movement
        dx = current_pose['x'] - self.previous_pose['x']
        dy = current_pose['y'] - self.previous_pose['y']
        
        # Calculate movement magnitude
        magnitude = np.sqrt(dx**2 + dy**2)
        
        # Update adaptive threshold based on face size
        eye_distance = current_pose.get('eye_distance', 0.1)
        self.movement_threshold = self._calculate_adaptive_threshold(eye_distance)
        
        # Store statistics
        self.movement_magnitudes.append(magnitude)
        self.face_sizes.append(eye_distance)
        
        if self.debug_mode:
            logger.debug(f"Movement: dx={dx:.4f}, dy={dy:.4f}, magnitude={magnitude:.4f}, "
                        f"threshold={self.movement_threshold:.4f}, eye_distance={eye_distance:.4f}")
        
        # Check if movement exceeds threshold
        if magnitude < self.movement_threshold:
            self.previous_pose = current_pose
            self.pose_history.append(current_pose)
            return None
        
        # Additional filtering: check for minimum significant movement
        min_significant_movement = self.movement_threshold * 1.1
        if magnitude < min_significant_movement:
            # Only count as movement if it's part of a pattern
            if len(self.movement_history) > 0:
                last_movement = self.movement_history[-1]
                time_diff = timestamp - last_movement.timestamp
                # If this is a continuation of recent movement, allow it
                if time_diff < 0.8:
                    pass  # Allow the movement
                else:
                    # Too small and isolated, skip
                    self.previous_pose = current_pose
                    self.pose_history.append(current_pose)
                    return None
            else:
                # First movement should be more significant but not too restrictive
                if magnitude < min_significant_movement * 1.2:
                    self.previous_pose = current_pose
                    self.pose_history.append(current_pose)
                    return None
        
        # Check cooldown period
        if timestamp - self.last_movement_time < self.movement_cooldown:
            if self.debug_mode:
                logger.debug(f"Movement skipped due to cooldown (time since last: {timestamp - self.last_movement_time:.3f}s)")
            self.previous_pose = current_pose
            self.pose_history.append(current_pose)
            return None
        
        # Determine movement direction using improved approach
        direction = self._detect_direction_improved(dx, dy, magnitude)
        
        if self.debug_mode:
            logger.debug(f"Movement detected: {direction} (magnitude={magnitude:.4f}, angle={np.arctan2(dy, dx) * 180 / np.pi:.1f}°)")
        
        # Validate movement based on rules
        validation_result = self._validate_movement(direction, current_pose, timestamp)
        if not validation_result['valid']:
            if self.debug_mode:
                logger.debug(f"Movement rejected: {validation_result['reason']}")
            self.previous_pose = current_pose
            self.pose_history.append(current_pose)
            return None
        
        # Calculate confidence based on movement characteristics
        confidence = self._calculate_movement_confidence(direction, magnitude, current_pose)
        
        # Create movement result
        movement = MovementResult(
            direction=direction,
            confidence=confidence,
            magnitude=magnitude,
            timestamp=timestamp,
            pose_data=current_pose
        )
        
        # Update state
        self.previous_pose = current_pose
        self.pose_history.append(current_pose)
        self.movement_history.append(movement)
        self.last_movement_time = timestamp
        
        return movement
    
    def _calculate_movement_confidence(self, direction: str, magnitude: float, pose_data: Dict[str, float]) -> float:
        """
        Calculate enhanced confidence score for detected movement.
        
        Args:
            direction: Movement direction
            magnitude: Movement magnitude
            pose_data: Current pose data including quality metrics
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Base confidence from magnitude (reward larger movements)
        magnitude_ratio = magnitude / self.movement_threshold
        base_confidence = min(magnitude_ratio / 2.0, 1.0)
        
        # Quality boost based on pose quality
        quality_score = pose_data.get('quality_score', 0.5)
        quality_boost = quality_score * 0.3  # Up to 30% boost for high quality
        
        # Direction consistency check
        direction_consistency = 1.0
        if len(self.movement_history) > 0:
            recent_directions = [m.direction for m in list(self.movement_history)[-3:]]
            consistency_ratio = sum(1 for d in recent_directions if d == direction) / len(recent_directions)
            
            # Reward consistent movements (natural head movements)
            if consistency_ratio < 0.5:  # Not repetitive
                direction_consistency = 1.2  # Boost for natural movement patterns
            elif consistency_ratio > 0.8:  # Too repetitive
                direction_consistency = 0.7  # Penalty for likely noise
        
        # Movement size reward/penalty
        size_factor = 1.0
        if magnitude > self.movement_threshold * 2.0:  # Large movement
            size_factor = 1.3  # Reward significant movements
        elif magnitude < self.movement_threshold * 1.2:  # Small movement
            size_factor = 0.8  # Penalty for very small movements
        
        # Calculate final confidence
        confidence = base_confidence * direction_consistency * size_factor + quality_boost
        
        # Ensure minimum confidence for significant movements
        if magnitude > self.movement_threshold * 1.5:
            confidence = max(confidence, 0.6)
        
        # Ensure maximum confidence cap
        return min(max(confidence, 0.0), 1.0)
    
    def process_video(self, video_path: str, expected_sequence: Optional[List[str]] = None) -> DetectionResult:
        """
        Process video for head movement detection.
        
        Args:
            video_path: Path to video file
            expected_sequence: Expected movement sequence for validation
            
        Returns:
            Detection result with movements and metrics
        """
        start_time = time.time()
        movements = []
        frames_processed = 0
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return DetectionResult(
                    success=False,
                    movements=[],
                    processing_time=time.time() - start_time,
                    frames_processed=0,
                    error="Failed to open video file"
                )
            
            logger.info(f"Processing video: {video_path}")
            
            # Reset state
            self._reset_state()
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.total_frames += 1
                
                # Process frame
                frame_result = self._process_frame(frame, frame_count)
                if frame_result:
                    frames_processed += 1
                    self.frames_with_face += 1
                    if frame_result.get('movement'):
                        movements.append(frame_result['movement'])
                
                frame_count += 1
                
                # Limit processing for performance
                if frame_count > 300:  # Max 300 frames
                    break
            
            cap.release()
            
            processing_time = time.time() - start_time
            
            # Calculate enhanced statistics
            face_detection_rate = self.frames_with_face / max(self.total_frames, 1)
            avg_magnitude = np.mean(self.movement_magnitudes) if self.movement_magnitudes else 0
            avg_face_size = np.mean(self.face_sizes) if self.face_sizes else 0
            
            logger.info(f"Video processing completed: {len(movements)} movements in {processing_time:.3f}s")
            logger.info(f"Statistics: frames={self.total_frames}, faces_detected={self.frames_with_face}, "
                       f"face_rate={face_detection_rate:.3f}, avg_magnitude={avg_magnitude:.4f}, "
                       f"avg_face_size={avg_face_size:.4f}")
            
            # Log movement summary
            self.log_movement_summary()
            
            # Enhanced adaptive threshold adjustment
            if len(movements) == 0 and self.movement_magnitudes:
                suggested_threshold = np.percentile(self.movement_magnitudes, 25)
                logger.info(f"No movements detected. Suggested threshold: {suggested_threshold:.4f} "
                           f"(current: {self.movement_threshold:.4f})")
            
            return DetectionResult(
                success=True,
                movements=movements,
                processing_time=processing_time,
                frames_processed=frames_processed
            )
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return DetectionResult(
                success=False,
                movements=[],
                processing_time=time.time() - start_time,
                frames_processed=frames_processed,
                error=str(e)
            )
    
    def process_video_optimized(self, video_path: str, target_fps: int = 10) -> DetectionResult:
        """Optimized video processing with frame skipping"""
        start_time = time.time()
        movements = []
        frames_processed = 0
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return DetectionResult(
                    success=False,
                    movements=[],
                    processing_time=time.time() - start_time,
                    frames_processed=0,
                    error="Failed to open video file"
                )
            
            # Get video properties
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = max(1, int(original_fps / target_fps))
            
            logger.info(f"Processing video optimized: {video_path} (original_fps={original_fps:.1f}, target_fps={target_fps}, frame_skip={frame_skip})")
            
            # Reset state
            self._reset_state()
            
            frame_count = 0
            processed_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.total_frames += 1
                
                # Skip frames to maintain target FPS
                if frame_count % frame_skip == 0:
                    # Resize frame for faster processing
                    height, width = frame.shape[:2]
                    if width > 640:
                        scale = 640 / width
                        new_width = 640
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Process frame
                    frame_result = self._process_frame(frame, processed_count)
                    if frame_result:
                        frames_processed += 1
                        self.frames_with_face += 1
                        if frame_result.get('movement'):
                            movements.append(frame_result['movement'])
                    
                    processed_count += 1
                
                frame_count += 1
                
                # Limit processing for performance
                if frame_count > 300:  # Max 300 frames
                    break
            
            cap.release()
            
            processing_time = time.time() - start_time
            
            # Calculate enhanced statistics
            face_detection_rate = self.frames_with_face / max(self.total_frames, 1)
            avg_magnitude = np.mean(self.movement_magnitudes) if self.movement_magnitudes else 0
            avg_face_size = np.mean(self.face_sizes) if self.face_sizes else 0
            
            logger.info(f"Optimized video processing completed: {len(movements)} movements in {processing_time:.3f}s")
            logger.info(f"Statistics: frames={self.total_frames}, faces_detected={self.frames_with_face}, "
                       f"face_rate={face_detection_rate:.3f}, avg_magnitude={avg_magnitude:.4f}, "
                       f"avg_face_size={avg_face_size:.4f}")
            
            # Log movement summary
            self.log_movement_summary()
            
            return DetectionResult(
                success=True,
                movements=movements,
                processing_time=processing_time,
                frames_processed=frames_processed
            )
            
        except Exception as e:
            logger.error(f"Optimized video processing failed: {e}")
            return DetectionResult(
                success=False,
                movements=[],
                processing_time=time.time() - start_time,
                frames_processed=frames_processed,
                error=str(e)
            )
    
    def _process_frame(self, frame: np.ndarray, frame_index: int) -> Optional[Dict[str, Any]]:
        """
        Process a single frame.
        
        Args:
            frame: Video frame
            frame_index: Frame index
            
        Returns:
            Processing result or None
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return None
            
            # Get first face
            face_landmarks = results.multi_face_landmarks[0]
            
            # Validate face detection
            if not self._validate_face_detection(face_landmarks.landmark, frame.shape):
                if self.debug_mode:
                    logger.debug(f"Face detection unreliable at frame {frame_index}")
                return None

            # Extract head pose
            pose = self.get_head_pose(face_landmarks.landmark, frame.shape)
            if not pose:
                return None
            
            # Calculate timestamp
            timestamp = frame_index / 30.0  # Assume 30 FPS
            
            # Detect movement
            movement = self.detect_movement(pose, timestamp)
            
            return {
                'pose': pose,
                'movement': movement,
                'frame_index': frame_index,
                'timestamp': timestamp
            }
            
        except Exception as e:
            logger.warning(f"Frame processing failed: {e}")
            return None
    
    def _reset_state(self):
        """Reset detector state for new video."""
        self.previous_pose = None
        self.movement_history.clear()
        self.pose_history.clear()
        self.total_frames = 0
        self.frames_with_face = 0
        self.movement_magnitudes = []
        self.face_sizes = []
        self.last_movement_time = 0
        
        # Reset movement validation state
        self.initial_face_center = None
        self.last_movement_direction = None
        self.center_position = None
        
        # Reset direction tracking
        self.last_direction = None
        
        # Reset adaptive threshold
        self.movement_threshold = self.base_movement_threshold
        
        logger.info("Enhanced detector state reset")
    
    def release(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        logger.info("MediaPipe resources released")
    
    def get_movement_sequence(self) -> List[str]:
        """Get detected movement sequence."""
        return [movement.direction for movement in self.movement_history]
    
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
    
    def log_movement_summary(self):
        """Log a summary of all detected movements."""
        if not self.movement_history:
            logger.info("No movements detected during processing")
            return
        
        logger.info(f"Movement Summary - Total: {len(self.movement_history)}")
        
        # Log initial face position
        if self.initial_face_center:
            logger.info(f"Initial face center: ({self.initial_face_center[0]:.3f}, {self.initial_face_center[1]:.3f})")
        
        # Group movements by direction
        direction_counts = {}
        for movement in self.movement_history:
            direction = movement.direction
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        
        for direction, count in direction_counts.items():
            logger.info(f"  {direction}: {count} movements")
        
        # Log sequence
        sequence = [m.direction for m in self.movement_history]
        logger.info(f"Movement sequence: {sequence}")
        
        # Log statistics
        confidences = [m.confidence for m in self.movement_history]
        magnitudes = [m.magnitude for m in self.movement_history]
        
        logger.info(f"Confidence stats: avg={np.mean(confidences):.3f}, min={min(confidences):.3f}, max={max(confidences):.3f}")
        logger.info(f"Magnitude stats: avg={np.mean(magnitudes):.4f}, min={min(magnitudes):.4f}, max={max(magnitudes):.4f}")
        
        # Log enhanced validation and quality info
        if self.debug_mode:
            logger.info(f"Validation settings: face_center_threshold={self.face_center_threshold}, return_threshold={self.return_movement_threshold}")
            logger.info(f"Quality settings: min_eye_distance={self.min_eye_distance}, min_face_height={self.min_face_height}, min_quality_score={self.min_quality_score}")
            logger.info(f"Adaptive threshold: base={self.base_movement_threshold}, current={self.movement_threshold}")
    
    def _validate_movement(self, direction: str, current_pose: Dict[str, float], timestamp: float) -> Dict[str, Any]:
        """
        Validate movement based on rules:
        1. Discard return movements (movements back to center)
        2. Only allow movements from centered face position
        
        Args:
            direction: Movement direction
            current_pose: Current pose data
            timestamp: Current timestamp
            
        Returns:
            Validation result with 'valid' boolean and 'reason' string
        """
        # Initialize center position if not set
        if self.center_position is None:
            self.center_position = (0.5, 0.5)  # Normalized center of frame
        
        # Rule 1: Check if face is initially centered
        if self.initial_face_center is None:
            # First movement - check if face is centered
            face_x, face_y = current_pose['x'], current_pose['y']
            center_x, center_y = self.center_position
            
            # Calculate distance from center
            distance_from_center = ((face_x - center_x) ** 2 + (face_y - center_y) ** 2) ** 0.5
            
            if distance_from_center > self.face_center_threshold:
                return {
                    'valid': False,
                    'reason': f'Face not centered initially (distance: {distance_from_center:.3f})'
                }
            
            # Store initial face center
            self.initial_face_center = (face_x, face_y)
            if self.debug_mode:
                logger.debug(f"Initial face center set: ({face_x:.3f}, {face_y:.3f})")
        
        # Rule 2: Check for return movements
        if self.last_movement_direction is not None:
            # Check if this is a return movement
            is_return_movement = self._is_return_movement(direction, current_pose)
            
            if is_return_movement:
                return {
                    'valid': False,
                    'reason': f'Return movement detected: {self.last_movement_direction} -> {direction}'
                }
        
        # Movement is valid
        self.last_movement_direction = direction
        return {'valid': True, 'reason': 'Valid movement'}
    
    def _is_return_movement(self, direction: str, current_pose: Dict[str, float]) -> bool:
        """
        Check if current movement is a return movement to center.
        
        Args:
            direction: Current movement direction
            current_pose: Current pose data
            
        Returns:
            True if this is a return movement
        """
        if self.initial_face_center is None:
            return False
        
        # Get current face position
        current_x, current_y = current_pose['x'], current_pose['y']
        initial_x, initial_y = self.initial_face_center
        
        # Calculate distance from initial center
        distance_from_initial = ((current_x - initial_x) ** 2 + (current_y - initial_y) ** 2) ** 0.5
        
        # Check if movement is bringing face back toward center
        if distance_from_initial < self.return_movement_threshold:
            # Face is close to initial center - likely a return movement
            return True
        
        # Check for opposite direction movements
        opposite_pairs = [
            ('left', 'right'),
            ('right', 'left'),
            ('up', 'down'),
            ('down', 'up')
        ]
        
        if self.last_movement_direction:
            for first, second in opposite_pairs:
                if (self.last_movement_direction == first and direction == second) or \
                   (self.last_movement_direction == second and direction == first):
                    # This is an opposite direction movement - likely a return
                    return True
        
        return False
    
    def validate_sequence(self, expected_sequence: List[str]) -> Dict[str, Any]:
        """
        Validate detected movements against expected sequence.
        
        Args:
            expected_sequence: Expected movement sequence
            
        Returns:
            Validation result
        """
        detected_sequence = self.get_movement_sequence()
        
        if not expected_sequence or not detected_sequence:
            return {
                'accuracy': 0.0,
                'expected_sequence': expected_sequence,
                'detected_sequence': detected_sequence,
                'correct_movements': 0,
                'total_expected': len(expected_sequence) if expected_sequence else 0
            }
        
        # Calculate accuracy
        correct_movements = 0
        for i, expected in enumerate(expected_sequence):
            if i < len(detected_sequence) and detected_sequence[i] == expected:
                correct_movements += 1
        
        accuracy = correct_movements / len(expected_sequence)
        
        return {
            'accuracy': accuracy,
            'expected_sequence': expected_sequence,
            'detected_sequence': detected_sequence,
            'correct_movements': correct_movements,
            'total_expected': len(expected_sequence)
        }

    def validate_sequence_improved(self, expected_sequence: List[str], tolerance: float = 0.3) -> Dict[str, Any]:
        """Improved sequence validation with timing tolerance and partial matching"""
        detected_sequence = self.get_movement_sequence()
        
        if not expected_sequence:
            return {'accuracy': 0.0, 'message': 'No expected sequence provided'}
        
        if not detected_sequence:
            return {'accuracy': 0.0, 'message': 'No movements detected'}
        
        # Use dynamic programming for best subsequence match
        score = self._calculate_sequence_similarity(expected_sequence, detected_sequence)
        
        # Calculate timing-based accuracy
        timing_accuracy = self._validate_movement_timing(expected_sequence, tolerance)
        
        # Combined score
        final_accuracy = (score * 0.7) + (timing_accuracy * 0.3)
        
        return {
            'accuracy': final_accuracy,
            'sequence_similarity': score,
            'timing_accuracy': timing_accuracy,
            'expected_sequence': expected_sequence,
            'detected_sequence': detected_sequence,
            'matched_movements': self._find_best_match(expected_sequence, detected_sequence)
        }

    def _calculate_sequence_similarity(self, expected: List[str], detected: List[str]) -> float:
        """Calculate similarity between sequences using Longest Common Subsequence"""
        if not expected or not detected:
            return 0.0
        
        # Dynamic programming LCS
        m, n = len(expected), len(detected)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if expected[i-1] == detected[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        if lcs_length == 0:
            return 0.0
        return lcs_length / len(expected)
    
    def _validate_movement_timing(self, expected_sequence: List[str], tolerance: float) -> float:
        """Validate timing of movements relative to expected sequence"""
        if not self.movement_history or not expected_sequence:
            return 0.0
        
        # Calculate expected timing intervals
        total_duration = self.movement_history[-1].timestamp - self.movement_history[0].timestamp
        expected_interval = total_duration / len(expected_sequence)
        
        # Check if detected movements are reasonably spaced
        timing_errors = []
        for i in range(1, len(self.movement_history)):
            actual_interval = self.movement_history[i].timestamp - self.movement_history[i-1].timestamp
            timing_error = abs(actual_interval - expected_interval) / expected_interval
            timing_errors.append(timing_error)
        
        if not timing_errors:
            return 1.0
        
        # Calculate timing accuracy
        avg_timing_error = np.mean(timing_errors)
        timing_accuracy = max(0, 1 - avg_timing_error / tolerance)
        
        return timing_accuracy
    
    def _find_best_match(self, expected: List[str], detected: List[str]) -> List[Dict[str, Any]]:
        """Find the best matching movements between expected and detected sequences"""
        matches = []
        
        # Simple greedy matching
        detected_idx = 0
        for i, expected_move in enumerate(expected):
            if detected_idx < len(detected):
                if detected[detected_idx] == expected_move:
                    matches.append({
                        'expected_index': i,
                        'detected_index': detected_idx,
                        'movement': expected_move,
                        'matched': True
                    })
                    detected_idx += 1
                else:
                    matches.append({
                        'expected_index': i,
                        'detected_index': -1,
                        'movement': expected_move,
                        'matched': False
                    })
        
        return matches

def create_simple_detector(**kwargs) -> SimpleMediaPipeDetector:
    """Create a simple MediaPipe detector with optional parameters."""
    return SimpleMediaPipeDetector(**kwargs)
