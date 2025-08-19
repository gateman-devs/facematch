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
    """Simplified MediaPipe-based head movement detector."""
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 movement_threshold: float = 0.02,
                 max_history: int = 10,
                 debug_mode: bool = False):
        """
        Initialize the detector.
        
        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
            movement_threshold: Threshold for movement detection (normalized coordinates)
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
        
        # Movement detection parameters
        self.movement_threshold = movement_threshold
        self.debug_mode = debug_mode
        self.previous_pose = None
        self.movement_history = deque(maxlen=max_history)
        self.pose_history = deque(maxlen=max_history)
        
        # Statistics for adaptive threshold
        self.total_frames = 0
        self.frames_with_face = 0
        self.movement_magnitudes = []
        
        # Key facial landmarks for head pose estimation
        self.landmarks = {
            'nose_tip': 1,
            'chin': 175,
            'left_eye': 33,
            'right_eye': 263,
            'left_ear': 234,
            'right_ear': 454,
            'mouth_center': 13
        }
        
        logger.info("Simple MediaPipe detector initialized")
    
    def get_head_pose(self, landmarks, image_shape) -> Optional[Dict[str, float]]:
        """
        Calculate head pose from facial landmarks.
        
        Args:
            landmarks: MediaPipe face mesh landmarks
            image_shape: Image dimensions (height, width)
            
        Returns:
            Head pose data or None if landmarks are invalid
        """
        try:
            # Extract key landmarks
            nose_tip = landmarks[self.landmarks['nose_tip']]
            chin = landmarks[self.landmarks['chin']]
            left_eye = landmarks[self.landmarks['left_eye']]
            right_eye = landmarks[self.landmarks['right_eye']]
            
            # Calculate head center (between eyes)
            x_center = (left_eye.x + right_eye.x) / 2
            y_center = (left_eye.y + right_eye.y) / 2
            
            # Calculate additional pose metrics
            eye_distance = np.sqrt((right_eye.x - left_eye.x)**2 + (right_eye.y - left_eye.y)**2)
            face_height = abs(chin.y - nose_tip.y)
            
            pose_data = {
                'x': x_center,
                'y': y_center,
                'nose_y': nose_tip.y,
                'chin_y': chin.y,
                'eye_distance': eye_distance,
                'face_height': face_height,
                'confidence': 1.0  # Will be updated based on landmark quality
            }
            
            return pose_data
            
        except (IndexError, AttributeError) as e:
            logger.warning(f"Failed to extract head pose: {e}")
            return None
    
    def detect_movement(self, current_pose: Dict[str, float], timestamp: float) -> Optional[MovementResult]:
        """
        Detect movement between current and previous pose.
        
        Args:
            current_pose: Current head pose data
            timestamp: Current timestamp
            
        Returns:
            Movement result or None if no significant movement
        """
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
        
        # Store magnitude for adaptive threshold
        self.movement_magnitudes.append(magnitude)
        
        if self.debug_mode:
            logger.debug(f"Movement: dx={dx:.4f}, dy={dy:.4f}, magnitude={magnitude:.4f}, threshold={self.movement_threshold:.4f}")
        
        # Check if movement exceeds threshold
        if magnitude < self.movement_threshold:
            self.previous_pose = current_pose
            self.pose_history.append(current_pose)
            return None
        
        # Determine movement direction
        direction = None
        if abs(dx) > abs(dy):
            direction = 'right' if dx > 0 else 'left'
        else:
            direction = 'down' if dy > 0 else 'up'
        
        if self.debug_mode:
            logger.debug(f"Movement detected: {direction} (magnitude={magnitude:.4f})")
        
        # Calculate confidence based on movement consistency
        confidence = self._calculate_movement_confidence(direction, magnitude)
        
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
        
        return movement
    
    def _calculate_movement_confidence(self, direction: str, magnitude: float) -> float:
        """
        Calculate confidence score for detected movement.
        
        Args:
            direction: Movement direction
            magnitude: Movement magnitude
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Base confidence from magnitude
        base_confidence = min(magnitude / (self.movement_threshold * 2), 1.0)
        
        # Check consistency with recent movements
        if len(self.movement_history) > 0:
            recent_directions = [m.direction for m in list(self.movement_history)[-3:]]
            direction_consistency = sum(1 for d in recent_directions if d == direction) / len(recent_directions)
            
            # Penalize repetitive movements (likely noise)
            if direction_consistency > 0.8:
                base_confidence *= 0.7
        
        return min(max(base_confidence, 0.0), 1.0)
    
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
            
            # Calculate statistics
            face_detection_rate = self.frames_with_face / max(self.total_frames, 1)
            avg_magnitude = np.mean(self.movement_magnitudes) if self.movement_magnitudes else 0
            
            logger.info(f"Video processing completed: {len(movements)} movements in {processing_time:.3f}s")
            logger.info(f"Statistics: frames={self.total_frames}, faces_detected={self.frames_with_face}, "
                       f"face_rate={face_detection_rate:.3f}, avg_magnitude={avg_magnitude:.4f}")
            
            # Log movement summary
            self.log_movement_summary()
            
            # Adaptive threshold adjustment
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
        logger.info("Detector state reset")
    
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

def create_simple_detector(**kwargs) -> SimpleMediaPipeDetector:
    """Create a simple MediaPipe detector with optional parameters."""
    return SimpleMediaPipeDetector(**kwargs)
