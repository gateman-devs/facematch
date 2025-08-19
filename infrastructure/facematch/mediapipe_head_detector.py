"""
MediaPipe-based Head Movement Detection Module
Provides advanced head movement detection using MediaPipe Pose and Face Mesh
for more accurate and robust movement analysis.
"""

import logging
import numpy as np
import cv2
import mediapipe as mp
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import time

logger = logging.getLogger(__name__)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

@dataclass
class MediaPipeConfig:
    """Configuration for MediaPipe head movement detection."""
    
    # MediaPipe settings
    pose_confidence_threshold: float = 0.5
    face_mesh_confidence_threshold: float = 0.5
    max_num_faces: int = 1
    refine_landmarks: bool = True
    
    # Movement detection parameters
    min_movement_threshold: float = 8.0  # pixels
    significant_movement_threshold: float = 12.0  # pixels
    movement_smoothing_window: int = 5
    velocity_smoothing_window: int = 3
    
    # Head pose estimation parameters
    enable_head_pose_estimation: bool = True
    head_pose_confidence_threshold: float = 0.7
    
    # Quality assessment parameters
    enable_quality_assessment: bool = True
    min_landmark_confidence: float = 0.3
    min_face_size: int = 50  # minimum face size in pixels
    
    # Performance parameters
    enable_optimization: bool = True
    skip_frames: int = 1  # process every nth frame
    max_frames_to_process: int = 300  # maximum frames to process

@dataclass
class HeadMovementData:
    """Data structure for head movement information."""
    direction: str
    confidence: float
    magnitude: float
    timestamp: float
    start_position: Tuple[float, float]
    end_position: Tuple[float, float]
    head_pose: Optional[Dict[str, float]] = None
    quality_score: float = 0.0
    velocity: Tuple[float, float] = (0.0, 0.0)
    acceleration: Tuple[float, float] = (0.0, 0.0)
    landmarks_used: List[int] = field(default_factory=list)

@dataclass
class MediaPipeResult:
    """Result of MediaPipe head movement detection."""
    success: bool
    movements: List[HeadMovementData]
    head_poses: List[Dict[str, float]]
    quality_metrics: Dict[str, float]
    processing_time: float
    frames_processed: int
    error: Optional[str] = None

class MediaPipeHeadDetector:
    """Advanced head movement detector using MediaPipe Pose and Face Mesh."""
    
    def __init__(self, config: Optional[MediaPipeConfig] = None):
        """Initialize MediaPipe head detector."""
        self.config = config or MediaPipeConfig()
        
        # Initialize MediaPipe solutions
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=self.config.pose_confidence_threshold,
            min_tracking_confidence=self.config.pose_confidence_threshold
        )
        
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.config.max_num_faces,
            refine_landmarks=self.config.refine_landmarks,
            min_detection_confidence=self.config.face_mesh_confidence_threshold,
            min_tracking_confidence=self.config.face_mesh_confidence_threshold
        )
        
        # Movement tracking state
        self.movement_history: deque = deque(maxlen=self.config.movement_smoothing_window)
        self.velocity_history: deque = deque(maxlen=self.config.velocity_smoothing_window)
        self.head_pose_history: deque = deque(maxlen=10)
        
        # Key landmarks for head movement detection
        self.pose_landmarks = {
            'nose': mp_pose.PoseLandmark.NOSE,
            'left_ear': mp_pose.PoseLandmark.LEFT_EAR,
            'right_ear': mp_pose.PoseLandmark.RIGHT_EAR,
            'left_eye': mp_pose.PoseLandmark.LEFT_EYE,
            'right_eye': mp_pose.PoseLandmark.RIGHT_EYE,
            'left_shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER
        }
        
        # Face mesh landmarks for detailed head pose
        self.face_landmarks = {
            'nose_tip': 1,
            'nose_bottom': 2,
            'nose_top': 168,
            'left_eye_center': 33,
            'right_eye_center': 263,
            'left_ear': 234,
            'right_ear': 454,
            'mouth_center': 13,
            'left_cheek': 50,
            'right_cheek': 280
        }
        
        logger.info("MediaPipe Head Detector initialized successfully")
    
    def detect_head_movements(
        self, 
        video_path: str,
        expected_sequence: Optional[List[str]] = None
    ) -> MediaPipeResult:
        """
        Detect head movements in video using MediaPipe.
        
        Args:
            video_path: Path to video file
            expected_sequence: Expected movement sequence for validation
            
        Returns:
            MediaPipeResult with detected movements and analysis
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting MediaPipe head movement detection: {video_path}")
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return MediaPipeResult(
                    success=False,
                    movements=[],
                    head_poses=[],
                    quality_metrics={},
                    processing_time=time.time() - start_time,
                    frames_processed=0,
                    error="Failed to open video file"
                )
            
            movements = []
            head_poses = []
            frame_count = 0
            processed_frames = 0
            
            # Reset tracking state
            self._reset_tracking_state()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames for optimization
                if self.config.enable_optimization and frame_count % (self.config.skip_frames + 1) != 0:
                    frame_count += 1
                    continue
                
                # Limit frames to process
                if processed_frames >= self.config.max_frames_to_process:
                    logger.info(f"Reached maximum frames limit: {self.config.max_frames_to_process}")
                    break
                
                # Process frame
                frame_result = self._process_frame(frame, frame_count)
                
                if frame_result['success']:
                    processed_frames += 1
                    
                    # Add movement if detected
                    if frame_result.get('movement'):
                        movements.append(frame_result['movement'])
                    
                    # Add head pose if available
                    if frame_result.get('head_pose'):
                        head_poses.append(frame_result['head_pose'])
                
                frame_count += 1
            
            cap.release()
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(movements, head_poses, processed_frames)
            
            # Validate sequence if provided
            if expected_sequence and movements:
                self._validate_movement_sequence(movements, expected_sequence)
            
            processing_time = time.time() - start_time
            
            return MediaPipeResult(
                success=True,
                movements=movements,
                head_poses=head_poses,
                quality_metrics=quality_metrics,
                processing_time=processing_time,
                frames_processed=processed_frames
            )
            
        except Exception as e:
            logger.error(f"MediaPipe head movement detection failed: {e}")
            return MediaPipeResult(
                success=False,
                movements=[],
                head_poses=[],
                quality_metrics={},
                processing_time=time.time() - start_time,
                frames_processed=0,
                error=str(e)
            )
    
    def _process_frame(self, frame: np.ndarray, frame_index: int) -> Dict[str, Any]:
        """Process a single frame for head movement detection."""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe Pose
            pose_results = self.pose.process(rgb_frame)
            
            # Process with MediaPipe Face Mesh
            face_results = self.face_mesh.process(rgb_frame)
            
            result = {
                'success': False,
                'frame_index': frame_index,
                'timestamp': frame_index / 30.0  # Assume 30 FPS
            }
            
            # Extract pose landmarks
            pose_landmarks = self._extract_pose_landmarks(pose_results, frame.shape)
            if pose_landmarks:
                result['pose_landmarks'] = pose_landmarks
                result['success'] = True
            
            # Extract face mesh landmarks
            face_landmarks = self._extract_face_landmarks(face_results, frame.shape)
            if face_landmarks:
                result['face_landmarks'] = face_landmarks
                result['success'] = True
            
            # Calculate head movement if we have landmarks
            if result['success']:
                movement = self._calculate_head_movement(
                    pose_landmarks, face_landmarks, result['timestamp']
                )
                if movement:
                    result['movement'] = movement
                
                # Calculate head pose if enabled
                if self.config.enable_head_pose_estimation and face_landmarks:
                    head_pose = self._calculate_head_pose(face_landmarks, frame.shape)
                    if head_pose:
                        result['head_pose'] = head_pose
            
            return result
            
        except Exception as e:
            logger.warning(f"Frame processing failed: {e}")
            return {
                'success': False,
                'frame_index': frame_index,
                'error': str(e)
            }
    
    def _extract_pose_landmarks(
        self, 
        pose_results: Any, 
        frame_shape: Tuple[int, int, int]
    ) -> Optional[Dict[str, Tuple[float, float]]]:
        """Extract relevant pose landmarks for head movement detection."""
        if not pose_results.pose_landmarks:
            return None
        
        landmarks = {}
        height, width = frame_shape[:2]
        
        for name, landmark_id in self.pose_landmarks.items():
            landmark = pose_results.pose_landmarks.landmark[landmark_id]
            
            # Convert normalized coordinates to pixel coordinates
            x = landmark.x * width
            y = landmark.y * height
            
            # Only include landmarks with good visibility
            if landmark.visibility > self.config.min_landmark_confidence:
                landmarks[name] = (x, y)
        
        return landmarks if len(landmarks) >= 3 else None  # Need at least 3 landmarks
    
    def _extract_face_landmarks(
        self, 
        face_results: Any, 
        frame_shape: Tuple[int, int, int]
    ) -> Optional[Dict[str, Tuple[float, float]]]:
        """Extract relevant face mesh landmarks for detailed head pose."""
        if not face_results.multi_face_landmarks:
            return None
        
        # Use the first face
        face_landmarks = face_results.multi_face_landmarks[0]
        landmarks = {}
        height, width = frame_shape[:2]
        
        for name, landmark_id in self.face_landmarks.items():
            landmark = face_landmarks.landmark[landmark_id]
            
            # Convert normalized coordinates to pixel coordinates
            x = landmark.x * width
            y = landmark.y * height
            
            landmarks[name] = (x, y)
        
        return landmarks
    
    def _calculate_head_movement(
        self, 
        pose_landmarks: Optional[Dict[str, Tuple[float, float]]],
        face_landmarks: Optional[Dict[str, Tuple[float, float]]],
        timestamp: float
    ) -> Optional[HeadMovementData]:
        """Calculate head movement between current and previous landmarks."""
        try:
            # Use face landmarks if available, otherwise use pose landmarks
            current_landmarks = face_landmarks if face_landmarks else pose_landmarks
            if not current_landmarks:
                return None
            
            # Calculate head center (use nose as reference)
            if 'nose_tip' in current_landmarks:
                current_center = current_landmarks['nose_tip']
            elif 'nose' in current_landmarks:
                current_center = current_landmarks['nose']
            else:
                # Fallback to average of available landmarks
                x_coords = [landmark[0] for landmark in current_landmarks.values()]
                y_coords = [landmark[1] for landmark in current_landmarks.values()]
                current_center = (np.mean(x_coords), np.mean(y_coords))
            
            # Check if we have previous landmarks for comparison
            if not self.movement_history:
                # First frame, store current position
                self.movement_history.append({
                    'center': current_center,
                    'timestamp': timestamp,
                    'landmarks': current_landmarks
                })
                return None
            
            # Get previous position
            prev_data = self.movement_history[-1]
            prev_center = prev_data['center']
            
            # Calculate movement vector
            dx = current_center[0] - prev_center[0]
            dy = current_center[1] - prev_center[1]
            magnitude = np.sqrt(dx**2 + dy**2)
            
            # Check if movement is significant
            if magnitude < self.config.min_movement_threshold:
                # Update current position without creating movement
                self.movement_history.append({
                    'center': current_center,
                    'timestamp': timestamp,
                    'landmarks': current_landmarks
                })
                return None
            
            # Determine movement direction
            direction = self._determine_movement_direction(dx, dy, magnitude)
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_movement_confidence(
                current_landmarks, prev_data['landmarks'], magnitude
            )
            
            # Calculate velocity and acceleration
            velocity = self._calculate_velocity(dx, dy, timestamp - prev_data['timestamp'])
            acceleration = self._calculate_acceleration(velocity, timestamp)
            
            # Create movement data
            movement = HeadMovementData(
                direction=direction,
                confidence=confidence,
                magnitude=magnitude,
                timestamp=timestamp,
                start_position=prev_center,
                end_position=current_center,
                velocity=velocity,
                acceleration=acceleration,
                landmarks_used=list(current_landmarks.keys()),
                quality_score=self._calculate_quality_score(current_landmarks)
            )
            
            # Update movement history
            self.movement_history.append({
                'center': current_center,
                'timestamp': timestamp,
                'landmarks': current_landmarks
            })
            
            return movement
            
        except Exception as e:
            logger.warning(f"Head movement calculation failed: {e}")
            return None
    
    def _determine_movement_direction(self, dx: float, dy: float, magnitude: float) -> str:
        """Determine the primary direction of head movement."""
        # Normalize movement vector
        if magnitude == 0:
            return 'none'
        
        dx_norm = dx / magnitude
        dy_norm = dy / magnitude
        
        # Determine primary direction with threshold
        direction_threshold = 0.7  # Must be 70% in one direction
        
        if abs(dx_norm) > direction_threshold:
            return 'right' if dx_norm > 0 else 'left'
        elif abs(dy_norm) > direction_threshold:
            return 'down' if dy_norm > 0 else 'up'
        else:
            # Diagonal movement, determine primary direction
            if abs(dx_norm) > abs(dy_norm):
                return 'right' if dx_norm > 0 else 'left'
            else:
                return 'down' if dy_norm > 0 else 'up'
    
    def _calculate_movement_confidence(
        self, 
        current_landmarks: Dict[str, Tuple[float, float]],
        prev_landmarks: Dict[str, Tuple[float, float]],
        magnitude: float
    ) -> float:
        """Calculate confidence score for detected movement."""
        try:
            # Base confidence from magnitude
            magnitude_confidence = min(magnitude / self.config.significant_movement_threshold, 1.0)
            
            # Landmark consistency confidence
            landmark_confidence = 0.0
            if prev_landmarks:
                # Calculate how many landmarks are consistent
                consistent_landmarks = 0
                total_landmarks = 0
                
                for name in current_landmarks:
                    if name in prev_landmarks:
                        total_landmarks += 1
                        # Check if landmark position is reasonable (not too far)
                        curr_pos = current_landmarks[name]
                        prev_pos = prev_landmarks[name]
                        distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                        
                        # If distance is reasonable (not too large), consider consistent
                        if distance < magnitude * 2:  # Allow some variation
                            consistent_landmarks += 1
                
                if total_landmarks > 0:
                    landmark_confidence = consistent_landmarks / total_landmarks
            
            # Combine confidence scores
            confidence = (magnitude_confidence * 0.6 + landmark_confidence * 0.4)
            
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _calculate_velocity(self, dx: float, dy: float, dt: float) -> Tuple[float, float]:
        """Calculate velocity vector."""
        if dt <= 0:
            return (0.0, 0.0)
        
        vx = dx / dt
        vy = dy / dt
        
        return (vx, vy)
    
    def _calculate_acceleration(self, current_velocity: Tuple[float, float], timestamp: float) -> Tuple[float, float]:
        """Calculate acceleration vector."""
        if not self.velocity_history:
            self.velocity_history.append((current_velocity, timestamp))
            return (0.0, 0.0)
        
        prev_velocity, prev_timestamp = self.velocity_history[-1]
        dt = timestamp - prev_timestamp
        
        if dt <= 0:
            return (0.0, 0.0)
        
        ax = (current_velocity[0] - prev_velocity[0]) / dt
        ay = (current_velocity[1] - prev_velocity[1]) / dt
        
        self.velocity_history.append((current_velocity, timestamp))
        
        return (ax, ay)
    
    def _calculate_head_pose(
        self, 
        landmarks: Dict[str, Tuple[float, float]], 
        frame_shape: Tuple[int, int, int]
    ) -> Optional[Dict[str, float]]:
        """Calculate head pose angles using face landmarks."""
        try:
            # Use 3D face landmarks for pose estimation
            # This is a simplified version - in practice, you'd use more sophisticated 3D pose estimation
            
            # Calculate head pose using eye and nose positions
            if all(key in landmarks for key in ['left_eye_center', 'right_eye_center', 'nose_tip']):
                left_eye = landmarks['left_eye_center']
                right_eye = landmarks['right_eye_center']
                nose = landmarks['nose_tip']
                
                # Calculate yaw (left-right rotation)
                eye_center_x = (left_eye[0] + right_eye[0]) / 2
                eye_center_y = (left_eye[1] + right_eye[1]) / 2
                
                # Estimate yaw from nose position relative to eye center
                yaw = np.arctan2(nose[0] - eye_center_x, nose[1] - eye_center_y)
                yaw_degrees = np.degrees(yaw)
                
                # Calculate pitch (up-down rotation) from eye positions
                eye_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
                expected_eye_distance = 60  # Expected distance in pixels for frontal face
                pitch = np.arctan2(eye_distance - expected_eye_distance, expected_eye_distance)
                pitch_degrees = np.degrees(pitch)
                
                # Calculate roll (tilt) from eye positions
                roll = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
                roll_degrees = np.degrees(roll)
                
                return {
                    'yaw': yaw_degrees,
                    'pitch': pitch_degrees,
                    'roll': roll_degrees,
                    'confidence': 0.8  # Simplified confidence
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Head pose calculation failed: {e}")
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
            logger.warning(f"Quality score calculation failed: {e}")
            return 0.5
    
    def _calculate_quality_metrics(
        self, 
        movements: List[HeadMovementData], 
        head_poses: List[Dict[str, float]], 
        frames_processed: int
    ) -> Dict[str, float]:
        """Calculate overall quality metrics for the detection."""
        try:
            metrics = {
                'total_movements': len(movements),
                'frames_processed': frames_processed,
                'movement_rate': len(movements) / max(frames_processed, 1),
                'avg_movement_confidence': 0.0,
                'avg_movement_magnitude': 0.0,
                'avg_quality_score': 0.0,
                'head_pose_count': len(head_poses)
            }
            
            if movements:
                metrics['avg_movement_confidence'] = np.mean([m.confidence for m in movements])
                metrics['avg_movement_magnitude'] = np.mean([m.magnitude for m in movements])
                metrics['avg_quality_score'] = np.mean([m.quality_score for m in movements])
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Quality metrics calculation failed: {e}")
            return {
                'total_movements': 0,
                'frames_processed': frames_processed,
                'movement_rate': 0.0,
                'avg_movement_confidence': 0.0,
                'avg_movement_magnitude': 0.0,
                'avg_quality_score': 0.0,
                'head_pose_count': 0
            }
    
    def _validate_movement_sequence(
        self, 
        movements: List[HeadMovementData], 
        expected_sequence: List[str]
    ) -> Dict[str, Any]:
        """Validate detected movements against expected sequence."""
        try:
            detected_sequence = [movement.direction for movement in movements]
            
            # Simple sequence validation
            correct_movements = 0
            for i, expected in enumerate(expected_sequence):
                if i < len(detected_sequence) and detected_sequence[i] == expected:
                    correct_movements += 1
            
            accuracy = correct_movements / len(expected_sequence) if expected_sequence else 0.0
            
            return {
                'accuracy': accuracy,
                'expected_sequence': expected_sequence,
                'detected_sequence': detected_sequence,
                'correct_movements': correct_movements,
                'total_expected': len(expected_sequence)
            }
            
        except Exception as e:
            logger.warning(f"Sequence validation failed: {e}")
            return {
                'accuracy': 0.0,
                'expected_sequence': expected_sequence,
                'detected_sequence': [],
                'correct_movements': 0,
                'total_expected': len(expected_sequence) if expected_sequence else 0
            }
    
    def _reset_tracking_state(self):
        """Reset tracking state for new video."""
        self.movement_history.clear()
        self.velocity_history.clear()
        self.head_pose_history.clear()
        logger.info("MediaPipe tracking state reset")
    
    def release(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'pose'):
            self.pose.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        logger.info("MediaPipe resources released")

def create_mediapipe_detector(config: Optional[MediaPipeConfig] = None) -> MediaPipeHeadDetector:
    """Create MediaPipe head detector with optional custom config."""
    return MediaPipeHeadDetector(config or MediaPipeConfig())
