"""
Enhanced MediaPipe Head Movement Detector - Non-Time-Based Version
Modified to remove time constraints and use degree-based detection:
- No time-based expectations (2s, 1.5s, etc.)
- Uses actual head rotation degrees (yaw/pitch) instead of pixel movement
- Detects movements as they occur without time pressure
- Configurable degree thresholds for head turns
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
    magnitude: float  # Now represents degrees of rotation
    timestamp: float
    pose_data: Dict[str, float]
    rotation_degrees: Dict[str, float]  # yaw, pitch, roll in degrees

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
    Enhanced MediaPipe-based head movement detector with degree-based detection.
    No time constraints - detects movements as they occur.
    """

    def __init__(self,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 min_rotation_degrees: float = 15.0,  # Minimum degrees for a valid head turn
                 significant_rotation_degrees: float = 25.0,  # Degrees for significant movement classification
                 min_confidence_threshold: float = 0.7,
                 max_history: int = 10,
                 debug_mode: bool = False):
        """
        Initialize the enhanced detector.
        
        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
            min_rotation_degrees: Minimum degrees of head rotation to count as movement
            significant_rotation_degrees: Degrees for significant movement classification
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
        
        # Landmark indices for head pose calculation
        self.landmarks = {
            'nose_tip': 1,
            'chin': 18,
            'left_eye': 33,
            'right_eye': 263,
            'forehead': 9,
            'left_ear': 234,
            'right_ear': 454
        }
        
        # Degree-based movement detection parameters
        self.min_rotation_degrees = min_rotation_degrees
        self.significant_rotation_degrees = significant_rotation_degrees
        self.min_confidence_threshold = min_confidence_threshold
        self.debug_mode = debug_mode
        self.previous_pose = None
        self.movement_history = deque(maxlen=max_history)
        self.pose_history = deque(maxlen=max_history)
        
        # Track last direction to prevent consecutive repeats
        self.last_direction = None
        self.consecutive_count = 0
        self.max_consecutive_same_direction = 1  # Allow max 1 consecutive same direction
        
        # Quality parameters
        self.min_quality_score = 0.3
        
        # Statistics tracking
        self.total_frames = 0
        self.frames_with_face = 0
        self.rotation_magnitudes = []
        self.face_sizes = []
        
        # Center position tracking (for return movement detection)
        self.initial_face_center = None
        self.face_center_threshold = 0.1  # Distance threshold for center position
        self.return_movement_threshold = 0.05  # Distance threshold for return movements
        
        # Movement cooldown (reduced since we're not time-constrained)
        self.movement_cooldown = 0.1  # 100ms minimum between movements
        self.last_movement_time = 0
        
        logger.info(f"Enhanced MediaPipe Detector initialized - Min rotation: {min_rotation_degrees}°, Significant: {significant_rotation_degrees}°")

    def _reset_state(self):
        """Reset detector state for new video."""
        self.previous_pose = None
        self.movement_history.clear()
        self.pose_history.clear()
        self.last_direction = None
        self.consecutive_count = 0
        self.total_frames = 0
        self.frames_with_face = 0
        self.rotation_magnitudes = []
        self.face_sizes = []
        self.initial_face_center = None
        self.last_movement_time = 0

    def _extract_head_pose(self, frame: np.ndarray) -> Optional[Dict[str, float]]:
        """Extract head pose data from frame using MediaPipe."""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return None
            
            # Get the first face
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract key landmarks
            landmarks = {}
            for name, idx in self.landmarks.items():
                landmark = face_landmarks.landmark[idx]
                landmarks[name] = (landmark.x, landmark.y)
            
            # Calculate head pose angles (yaw, pitch, roll)
            pose_angles = self._calculate_head_pose_angles(landmarks, frame.shape)
            if not pose_angles:
                return None
            
            # Calculate face center position
            nose_x, nose_y = landmarks['nose_tip']
            
            # Calculate eye distance for quality assessment
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            eye_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(landmarks)
            
            return {
                'x': nose_x,
                'y': nose_y,
                'eye_distance': eye_distance,
                'quality_score': quality_score,
                'yaw': pose_angles['yaw'],
                'pitch': pose_angles['pitch'],
                'roll': pose_angles['roll'],
                'yaw_confidence': pose_angles.get('yaw_confidence', 0.8),
                'pitch_confidence': pose_angles.get('pitch_confidence', 0.8),
                'roll_confidence': pose_angles.get('roll_confidence', 0.8)
            }
            
        except Exception as e:
            if self.debug_mode:
                logger.debug(f"Head pose extraction failed: {e}")
            return None

    def _calculate_head_pose_angles(self, landmarks: Dict[str, Tuple[float, float]], frame_shape: Tuple[int, int, int]) -> Optional[Dict[str, float]]:
        """Calculate head pose angles using face landmarks."""
        try:
            # Get key landmarks
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            nose = landmarks['nose_tip']
            left_ear = landmarks['left_ear']
            right_ear = landmarks['right_ear']
            
            # Calculate yaw (left-right rotation) from ear positions
            ear_center_x = (left_ear[0] + right_ear[0]) / 2
            nose_ear_offset = nose[0] - ear_center_x
            
            # Convert to degrees (assuming 0.1 offset = 15 degrees)
            yaw_degrees = nose_ear_offset * 150  # Scale factor for degree conversion
            
            # Calculate pitch (up-down rotation) from eye positions
            eye_center_y = (left_eye[1] + right_eye[1]) / 2
            nose_eye_offset = nose[1] - eye_center_y
            
            # Convert to degrees (assuming 0.1 offset = 15 degrees)
            pitch_degrees = nose_eye_offset * 150  # Scale factor for degree conversion
            
            # Calculate roll (tilt) from eye positions
            eye_dy = right_eye[1] - left_eye[1]
            eye_dx = right_eye[0] - left_eye[0]
            roll_degrees = np.arctan2(eye_dy, eye_dx) * 180 / np.pi
            
            # Clamp angles to reasonable ranges
            yaw_degrees = np.clip(yaw_degrees, -90, 90)
            pitch_degrees = np.clip(pitch_degrees, -90, 90)
            roll_degrees = np.clip(roll_degrees, -45, 45)
            
            return {
                'yaw': yaw_degrees,
                'pitch': pitch_degrees,
                'roll': roll_degrees,
                'yaw_confidence': 0.8,
                'pitch_confidence': 0.8,
                'roll_confidence': 0.8
            }
            
        except Exception as e:
            if self.debug_mode:
                logger.debug(f"Head pose angle calculation failed: {e}")
            return None

    def _calculate_quality_score(self, landmarks: Dict[str, Tuple[float, float]]) -> float:
        """Calculate quality score for landmark detection."""
        try:
            # Count number of landmarks detected
            landmark_count = len(landmarks)
            
            # Calculate average distance between landmarks (should be reasonable)
            distances = []
            landmark_list = list(landmarks.values())
            
            for i in range(len(landmark_list)):
                for j in range(i + 1, len(landmark_list)):
                    dist = np.sqrt(
                        (landmark_list[i][0] - landmark_list[j][0])**2 +
                        (landmark_list[i][1] - landmark_list[j][1])**2
                    )
                    distances.append(dist)
            
            if distances:
                avg_distance = np.mean(distances)
                std_distance = np.std(distances)
                
                # Quality based on consistency of landmark distances
                distance_consistency = max(0, 1 - std_distance / avg_distance) if avg_distance > 0 else 0
            else:
                distance_consistency = 0
            
            # Combine quality factors
            quality_score = (
                min(landmark_count / 10, 1.0) * 0.4 +  # Landmark count
                distance_consistency * 0.6  # Distance consistency
            )
            
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception as e:
            if self.debug_mode:
                logger.debug(f"Quality score calculation failed: {e}")
            return 0.5

    def _detect_direction_from_rotation(self, yaw_degrees: float, pitch_degrees: float) -> str:
        """Detect movement direction based on head rotation degrees."""
        # Determine primary direction based on which rotation is larger
        yaw_abs = abs(yaw_degrees)
        pitch_abs = abs(pitch_degrees)
        
        if yaw_abs > pitch_abs:
            # Horizontal movement (left/right)
            if yaw_degrees > 0:
                return 'right'
            else:
                return 'left'
        else:
            # Vertical movement (up/down)
            if pitch_degrees > 0:
                return 'down'
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

    def _calculate_confidence(self, direction: str, rotation_magnitude: float, pose_data: Dict[str, float]) -> float:
        """Calculate confidence score for movement based on rotation degrees."""
        # Base confidence from rotation magnitude
        magnitude_ratio = rotation_magnitude / self.min_rotation_degrees
        base_confidence = min(magnitude_ratio / 2.0, 1.0)
        
        # Quality boost
        quality_score = pose_data.get('quality_score', 0.5)
        quality_boost = quality_score * 0.3
        
        # Rotation confidence boost
        if direction in ['left', 'right']:
            rotation_confidence = pose_data.get('yaw_confidence', 0.8)
        else:  # up, down
            rotation_confidence = pose_data.get('pitch_confidence', 0.8)
        
        rotation_boost = rotation_confidence * 0.2
        
        # Size factor based on rotation magnitude
        size_factor = 1.0
        if rotation_magnitude > self.significant_rotation_degrees:
            size_factor = 1.3
        elif rotation_magnitude < self.min_rotation_degrees * 1.2:
            size_factor = 0.8
        
        confidence = base_confidence * size_factor + quality_boost + rotation_boost
        return min(max(confidence, 0.0), 1.0)

    def detect_movement(self, current_pose: Dict[str, float], timestamp: float) -> Optional[MovementResult]:
        """Detect movement using degree-based rotation detection."""
        # Quality validation
        if current_pose.get('quality_score', 0) < self.min_quality_score:
            return None
        
        if self.previous_pose is None:
            self.previous_pose = current_pose
            self.pose_history.append(current_pose)
            self.initial_face_center = (current_pose['x'], current_pose['y'])
            return None
        
        # Calculate rotation changes in degrees
        yaw_change = abs(current_pose['yaw'] - self.previous_pose['yaw'])
        pitch_change = abs(current_pose['pitch'] - self.previous_pose['pitch'])
        
        # Use the larger rotation as the primary movement
        if yaw_change > pitch_change:
            rotation_magnitude = yaw_change
            primary_rotation = 'yaw'
        else:
            rotation_magnitude = pitch_change
            primary_rotation = 'pitch'
        
        # Store statistics
        self.rotation_magnitudes.append(rotation_magnitude)
        self.face_sizes.append(current_pose.get('eye_distance', 0.1))
        
        # Check if rotation exceeds minimum threshold
        if rotation_magnitude < self.min_rotation_degrees:
            self.previous_pose = current_pose
            self.pose_history.append(current_pose)
            return None
        
        # Check cooldown period (reduced since we're not time-constrained)
        if timestamp - self.last_movement_time < self.movement_cooldown:
            self.previous_pose = current_pose
            self.pose_history.append(current_pose)
            return None
        
        # Determine direction from rotation
        direction = self._detect_direction_from_rotation(
            current_pose['yaw'] - self.previous_pose['yaw'],
            current_pose['pitch'] - self.previous_pose['pitch']
        )
        
        # Check for consecutive same direction
        if self._is_consecutive_direction(direction):
            if self.debug_mode:
                logger.debug(f"Consecutive direction rejected: {direction}")
            self.previous_pose = current_pose
            self.pose_history.append(current_pose)
            return None
        
        # Check for return movements
        if self._is_return_movement(direction, current_pose):
            if self.debug_mode:
                logger.debug(f"Return movement discarded: {direction}")
            self.previous_pose = current_pose
            self.pose_history.append(current_pose)
            return None
        
        # Calculate confidence
        confidence = self._calculate_confidence(direction, rotation_magnitude, current_pose)
        
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
            magnitude=rotation_magnitude,  # Now represents degrees
            timestamp=timestamp,
            pose_data=current_pose,
            rotation_degrees={
                'yaw': current_pose['yaw'],
                'pitch': current_pose['pitch'],
                'roll': current_pose['roll']
            }
        )
        
        # Update state
        self.previous_pose = current_pose
        self.pose_history.append(current_pose)
        self.movement_history.append(movement)
        self.last_movement_time = timestamp
        self.last_direction = direction
        
        if self.debug_mode:
            logger.debug(f"Movement detected: {direction} ({rotation_magnitude:.1f}° rotation, confidence: {confidence:.3f})")
        
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
