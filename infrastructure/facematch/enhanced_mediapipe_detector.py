"""
Enhanced MediaPipe Head Movement Detector
Fixed version that properly discards return movements:
- Only detects the main movements (down, left, right, up)
- Discards return movements that bring face back to center
- Filters out consecutive same directions
- Only keeps movements with confidence >= 0.8
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

class EnhancedMediaPipeDetector:
    """
    Enhanced MediaPipe-based head movement detector that discards return movements.
    """

    def __init__(self,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 movement_threshold: float = 0.009,  # Optimized threshold for both videos
                 min_confidence_threshold: float = 0.8,
                 max_history: int = 10,
                 debug_mode: bool = False):
        """
        Initialize the enhanced detector.
        
        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
            movement_threshold: Base movement magnitude threshold
            min_confidence_threshold: Minimum confidence to keep a movement
            max_history: Maximum number of poses to keep in history
            debug_mode: Enable debug logging
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Landmark indices
        self.landmarks = {
            'nose_tip': 1,
            'chin': 18,
            'left_eye': 33,
            'right_eye': 263,
            'forehead': 9
        }
        
        # Movement detection parameters
        self.base_movement_threshold = movement_threshold
        self.movement_threshold = movement_threshold
        self.min_confidence_threshold = min_confidence_threshold
        self.debug_mode = debug_mode
        self.previous_pose = None
        self.movement_history = deque(maxlen=max_history)
        self.pose_history = deque(maxlen=max_history)
        
        # Track last direction to prevent consecutive repeats
        self.last_direction = None
        self.consecutive_count = 0
        self.max_consecutive_same_direction = 1  # Allow max 1 consecutive same direction
        
        # Timing parameters
        self.movement_cooldown = 0.3  # Increased cooldown to avoid return movements
        self.last_movement_time = 0
        
        # Quality parameters
        self.min_eye_distance = 0.01
        self.min_face_height = 0.01
        self.min_quality_score = 0.3
        
        # Statistics
        self.total_frames = 0
        self.frames_with_face = 0
        self.movement_magnitudes = []
        self.face_sizes = []
        
        # Center tracking - MODIFIED: Discard return movements
        self.center_position = None
        self.initial_face_center = None
        self.face_center_threshold = 0.1
        self.return_movement_threshold = 0.05  # Strict threshold for return movements
        
        logger.info("Enhanced MediaPipe detector initialized")
    
    def _reset_state(self):
        """Reset detector state for new video."""
        self.previous_pose = None
        self.movement_history.clear()
        self.pose_history.clear()
        self.last_direction = None
        self.consecutive_count = 0
        self.last_movement_time = 0
        self.center_position = None
        self.initial_face_center = None
        self.total_frames = 0
        self.frames_with_face = 0
        self.movement_magnitudes = []
        self.face_sizes = []
        logger.info("Enhanced detector state reset")
    
    def _extract_head_pose(self, frame: np.ndarray) -> Optional[Dict[str, float]]:
        """Extract head pose from frame using MediaPipe Face Mesh."""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return None
            
            # Get the first face
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract key landmarks
            nose_tip = face_landmarks.landmark[self.landmarks['nose_tip']]
            chin = face_landmarks.landmark[self.landmarks['chin']]
            left_eye = face_landmarks.landmark[self.landmarks['left_eye']]
            right_eye = face_landmarks.landmark[self.landmarks['right_eye']]
            forehead = face_landmarks.landmark[self.landmarks['forehead']]
            
            # Calculate head center (between eyes)
            x_center = (left_eye.x + right_eye.x) / 2
            y_center = (left_eye.y + right_eye.y) / 2
            
            # Calculate face metrics
            eye_distance = np.sqrt((right_eye.x - left_eye.x)**2 + (right_eye.y - left_eye.y)**2)
            face_height = abs(chin.y - forehead.y)
            
            # Quality validation
            if eye_distance < self.min_eye_distance or face_height < self.min_face_height:
                return None
            
            # Calculate quality score
            quality_score = min(eye_distance * 10, 1.0)
            
            pose_data = {
                'x': x_center,
                'y': y_center,
                'eye_distance': eye_distance,
                'face_height': face_height,
                'quality_score': quality_score
            }
            
            return pose_data
            
        except Exception as e:
            logger.warning(f"Failed to extract head pose: {e}")
            return None
    
    def _calculate_adaptive_threshold(self, eye_distance: float) -> float:
        """Calculate adaptive movement threshold based on face size."""
        if eye_distance < 0.05:
            return self.base_movement_threshold * 1.5
        elif eye_distance > 0.15:
            return self.base_movement_threshold * 0.7
        else:
            return self.base_movement_threshold
    
    def _detect_direction(self, dx: float, dy: float) -> str:
        """Detect movement direction using angle-based approach."""
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        if angle < 0:
            angle += 360
        
        # Direction ranges
        if -45 <= angle < 45:
            return 'right'
        elif 45 <= angle < 135:
            return 'down'
        elif 135 <= angle < 225:
            return 'left'
        else:
            return 'up'
    
    def _is_consecutive_direction(self, direction: str) -> bool:
        """Check if this direction is consecutive to the last one."""
        if self.last_direction == direction:
            self.consecutive_count += 1
            return self.consecutive_count > self.max_consecutive_same_direction
        else:
            self.consecutive_count = 1
            return False
    
    def _is_return_movement(self, direction: str, current_pose: Dict[str, float]) -> bool:
        """
        Check if this is a return movement that should be discarded.
        Return movements bring the face back toward center.
        """
        if self.initial_face_center is None:
            return False
        
        current_x, current_y = current_pose['x'], current_pose['y']
        initial_x, initial_y = self.initial_face_center
        
        # Calculate distance from initial center
        distance_from_initial = np.sqrt((current_x - initial_x)**2 + (current_y - initial_y)**2)
        
        # If face is close to initial center, it's a return movement
        if distance_from_initial < self.return_movement_threshold:
            return True
        
        # Check for opposite direction movements (return movements)
        opposite_pairs = [
            ('left', 'right'),
            ('right', 'left'),
            ('up', 'down'),
            ('down', 'up')
        ]
        
        if self.last_direction:
            for dir1, dir2 in opposite_pairs:
                if (self.last_direction == dir1 and direction == dir2) or \
                   (self.last_direction == dir2 and direction == dir1):
                    # This is a return movement - check if it's bringing face toward center
                    center_x, center_y = 0.5, 0.5
                    distance_from_center = np.sqrt((current_x - center_x)**2 + (current_y - center_y)**2)
                    if distance_from_center < self.face_center_threshold:
                        return True
        
        return False
    
    def _calculate_confidence(self, direction: str, magnitude: float, pose_data: Dict[str, float]) -> float:
        """Calculate confidence score for movement."""
        # Base confidence from magnitude
        magnitude_ratio = magnitude / self.movement_threshold
        base_confidence = min(magnitude_ratio / 2.0, 1.0)
        
        # Quality boost
        quality_score = pose_data.get('quality_score', 0.5)
        quality_boost = quality_score * 0.3
        
        # Size factor
        size_factor = 1.0
        if magnitude > self.movement_threshold * 2.0:
            size_factor = 1.3
        elif magnitude < self.movement_threshold * 1.2:
            size_factor = 0.8
        
        confidence = base_confidence * size_factor + quality_boost
        return min(max(confidence, 0.0), 1.0)
    
    def detect_movement(self, current_pose: Dict[str, float], timestamp: float) -> Optional[MovementResult]:
        """Detect movement with filtering that discards return movements."""
        # Quality validation
        if current_pose.get('quality_score', 0) < self.min_quality_score:
            return None
        
        if self.previous_pose is None:
            self.previous_pose = current_pose
            self.pose_history.append(current_pose)
            self.initial_face_center = (current_pose['x'], current_pose['y'])
            return None
        
        # Calculate movement
        dx = current_pose['x'] - self.previous_pose['x']
        dy = current_pose['y'] - self.previous_pose['y']
        magnitude = np.sqrt(dx**2 + dy**2)
        
        # Update adaptive threshold
        eye_distance = current_pose.get('eye_distance', 0.1)
        self.movement_threshold = self._calculate_adaptive_threshold(eye_distance)
        
        # Store statistics
        self.movement_magnitudes.append(magnitude)
        self.face_sizes.append(eye_distance)
        
        # Check if movement exceeds threshold
        if magnitude < self.movement_threshold:
            self.previous_pose = current_pose
            self.pose_history.append(current_pose)
            return None
        
        # Check cooldown period
        if timestamp - self.last_movement_time < self.movement_cooldown:
            self.previous_pose = current_pose
            self.pose_history.append(current_pose)
            return None
        
        # Determine direction
        direction = self._detect_direction(dx, dy)
        
        # Check for consecutive same direction
        if self._is_consecutive_direction(direction):
            if self.debug_mode:
                logger.debug(f"Consecutive direction rejected: {direction}")
            self.previous_pose = current_pose
            self.pose_history.append(current_pose)
            return None
        
        # MODIFIED: Discard return movements
        if self._is_return_movement(direction, current_pose):
            if self.debug_mode:
                logger.debug(f"Return movement discarded: {direction}")
            self.previous_pose = current_pose
            self.pose_history.append(current_pose)
            return None
        
        # Calculate confidence
        confidence = self._calculate_confidence(direction, magnitude, current_pose)
        
        # Filter by confidence threshold
        if confidence < self.min_confidence_threshold:
            if self.debug_mode:
                logger.debug(f"Low confidence rejected: {confidence:.3f} < {self.min_confidence_threshold}")
            self.previous_pose = current_pose
            self.pose_history.append(current_pose)
            return None
        
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
        self.last_direction = direction
        
        if self.debug_mode:
            logger.debug(f"Movement detected: {direction} (confidence: {confidence:.3f}, magnitude: {magnitude:.4f})")
        
        return movement
    
    def process_video(self, video_path: str) -> DetectionResult:
        """Process video for head movement detection."""
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
            
            logger.info(f"Processing video: {video_path}")
            self._reset_state()
            
            frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.total_frames += 1
                
                # Process frame
                pose_data = self._extract_head_pose(frame)
                if pose_data:
                    self.frames_with_face += 1
                    timestamp = frame_count / fps
                    
                    movement = self.detect_movement(pose_data, timestamp)
                    if movement:
                        movements.append(movement)
                        frames_processed += 1
                
                frame_count += 1
                
                # Limit processing for performance
                if frame_count > 600:
                    break
            
            cap.release()
            
            processing_time = time.time() - start_time
            
            logger.info(f"Video processing completed: {len(movements)} movements in {processing_time:.3f}s")
            
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
