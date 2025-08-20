"""
Dlib-based Head Movement Detector
Replaces MediaPipe with dlib facial landmarks and OpenCV for head pose estimation.
Uses 68-point facial landmarks and solvePnP for 3D head pose calculation.
"""

import logging
import time
import numpy as np
import cv2
import dlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import os

logger = logging.getLogger(__name__)

@dataclass
class MovementResult:
    """Result of movement detection."""
    direction: str
    confidence: float
    magnitude: float  # Degrees of rotation
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

class DlibHeadDetector:
    """
    Dlib-based head movement detector using 68-point facial landmarks and OpenCV solvePnP.
    Provides accurate head pose estimation for movement detection.
    """

    def __init__(self,
                 shape_predictor_path: str = "shape_predictor_68_face_landmarks.dat",
                 min_rotation_degrees: float = 15.0,
                 significant_rotation_degrees: float = 25.0,
                 min_confidence_threshold: float = 0.7,
                 max_history: int = 10,
                 debug_mode: bool = False):
        """
        Initialize the dlib-based detector.
        
        Args:
            shape_predictor_path: Path to dlib's 68-point shape predictor
            min_rotation_degrees: Minimum degrees of head rotation to count as movement
            significant_rotation_degrees: Degrees for significant movement classification
            min_confidence_threshold: Minimum confidence to keep a movement
            max_history: Maximum number of poses to keep in history
            debug_mode: Enable debug logging
        """
        # Initialize dlib face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        
        # Check if shape predictor file exists
        if not os.path.exists(shape_predictor_path):
            logger.warning(f"Shape predictor not found at {shape_predictor_path}")
            logger.info("Downloading shape predictor...")
            self._download_shape_predictor(shape_predictor_path)
        
        self.predictor = dlib.shape_predictor(shape_predictor_path)
        
        # 3D model points (generic face model for solvePnP)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),           # Nose tip
            (0.0, -330.0, -65.0),      # Chin
            (-225.0, 170.0, -135.0),   # Left eye corner
            (225.0, 170.0, -135.0),    # Right eye corner
            (-150.0, -150.0, -125.0),  # Left mouth corner
            (150.0, -150.0, -125.0)    # Right mouth corner
        ], dtype=np.float32)
        
        # Camera parameters (will be updated based on video dimensions)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        
        # Movement detection parameters
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
        self.max_consecutive_same_direction = 1
        
        # Quality parameters
        self.min_quality_score = 0.3
        self.min_face_size = 50
        
        logger.info(f"Dlib Head Detector initialized - Min rotation: {min_rotation_degrees}°, Significant: {significant_rotation_degrees}°")

    def _download_shape_predictor(self, path: str) -> None:
        """Download the shape predictor file if not present."""
        try:
            import urllib.request
            import ssl
            url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            logger.info(f"Downloading shape predictor from {url}")
            
            # Create SSL context that ignores certificate verification
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Download the compressed file
            compressed_path = path + ".bz2"
            with urllib.request.urlopen(url, context=ssl_context) as response:
                with open(compressed_path, 'wb') as f:
                    f.write(response.read())
            
            # Extract the file
            import bz2
            with bz2.open(compressed_path, 'rb') as source, open(path, 'wb') as target:
                target.write(source.read())
            
            # Clean up compressed file
            os.remove(compressed_path)
            logger.info("Shape predictor downloaded and extracted successfully")
            
        except Exception as e:
            logger.error(f"Failed to download shape predictor: {e}")
            raise RuntimeError(f"Could not download shape predictor: {e}")

    def _update_camera_matrix(self, img_width: int, img_height: int) -> None:
        """Update camera matrix based on image dimensions."""
        focal_length = max(img_width, img_height)
        self.camera_matrix = np.array([
            [focal_length, 0, img_width / 2],
            [0, focal_length, img_height / 2],
            [0, 0, 1]
        ], dtype=np.float32)

    def _extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract 68-point facial landmarks using dlib."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            
            if len(faces) == 0:
                return None
            
            # Use the first detected face
            face = faces[0]
            
            # Get landmarks
            landmarks = self.predictor(gray, face)
            landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])
            
            return landmarks
            
        except Exception as e:
            logger.error(f"Error extracting landmarks: {e}")
            return None

    def _get_head_pose(self, landmarks: np.ndarray, frame_shape: Tuple[int, int]) -> Optional[Tuple[float, float, float]]:
        """Calculate head pose using solvePnP."""
        try:
            # Update camera matrix if needed
            if self.camera_matrix is None:
                self._update_camera_matrix(frame_shape[1], frame_shape[0])
            
            # Select specific landmarks for pose estimation
            image_points = np.array([
                landmarks[30],  # Nose tip
                landmarks[8],   # Chin
                landmarks[36],  # Left eye corner
                landmarks[45],  # Right eye corner
                landmarks[48],  # Left mouth corner
                landmarks[54]   # Right mouth corner
            ], dtype=np.float32)
            
            # Solve PnP
            success, rvec, tvec = cv2.solvePnP(
                self.model_points, 
                image_points, 
                self.camera_matrix, 
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return None
            
            # Convert rotation vector to rotation matrix
            rmat = cv2.Rodrigues(rvec)[0]
            
            # Extract Euler angles
            # Pitch (X-axis rotation)
            pitch = np.arctan2(rmat[2][1], rmat[2][2])
            
            # Yaw (Y-axis rotation)
            yaw = -np.arctan2(rmat[2][0], np.sqrt(rmat[2][1]**2 + rmat[2][2]**2))
            
            # Roll (Z-axis rotation)
            roll = np.arctan2(rmat[1][0], rmat[0][0])
            
            # Convert to degrees
            pitch_deg = np.degrees(pitch)
            yaw_deg = np.degrees(yaw)
            roll_deg = np.degrees(roll)
            
            return pitch_deg, yaw_deg, roll_deg
            
        except Exception as e:
            logger.error(f"Error calculating head pose: {e}")
            return None

    def _calculate_quality_score(self, landmarks: np.ndarray, frame_shape: Tuple[int, int]) -> float:
        """Calculate quality score based on face size and landmark visibility."""
        try:
            # Calculate face bounding box
            x_coords = landmarks[:, 0]
            y_coords = landmarks[:, 1]
            
            face_width = np.max(x_coords) - np.min(x_coords)
            face_height = np.max(y_coords) - np.min(y_coords)
            face_size = min(face_width, face_height)
            
            # Normalize by image size
            min_dim = min(frame_shape[0], frame_shape[1])
            normalized_size = face_size / min_dim
            
            # Quality score based on face size (0-1)
            quality_score = min(normalized_size / 0.3, 1.0)  # 30% of image is considered good
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0

    def _detect_movement(self, current_pose: Tuple[float, float, float], 
                        previous_pose: Optional[Tuple[float, float, float]]) -> Optional[MovementResult]:
        """Detect head movement based on pose changes."""
        if previous_pose is None:
            return None
        
        try:
            pitch, yaw, roll = current_pose
            prev_pitch, prev_yaw, prev_roll = previous_pose
            
            # Calculate rotation differences
            pitch_diff = abs(pitch - prev_pitch)
            yaw_diff = abs(yaw - prev_yaw)
            roll_diff = abs(roll - prev_roll)
            
            # Determine primary movement direction
            max_diff = max(pitch_diff, yaw_diff, roll_diff)
            
            if max_diff < self.min_rotation_degrees:
                return None
            
            # Determine direction based on largest change
            if yaw_diff == max_diff:
                direction = "Left" if yaw > prev_yaw else "Right"
                magnitude = yaw_diff
            elif pitch_diff == max_diff:
                direction = "Up" if pitch > prev_pitch else "Down"
                magnitude = pitch_diff
            else:
                # Roll movement (less common for head movements)
                direction = "Left" if roll > prev_roll else "Right"
                magnitude = roll_diff
            
            # Check if movement is significant enough
            if magnitude < self.significant_rotation_degrees:
                return None
            
            # Prevent consecutive same direction movements
            if direction == self.last_direction:
                self.consecutive_count += 1
                if self.consecutive_count > self.max_consecutive_same_direction:
                    return None
            else:
                self.consecutive_count = 0
                self.last_direction = direction
            
            # Calculate confidence based on magnitude
            confidence = min(magnitude / 45.0, 1.0)  # 45 degrees = 100% confidence
            
            return MovementResult(
                direction=direction,
                confidence=confidence,
                magnitude=magnitude,
                timestamp=time.time(),
                pose_data={
                    'pitch': pitch,
                    'yaw': yaw,
                    'roll': roll,
                    'pitch_diff': pitch_diff,
                    'yaw_diff': yaw_diff,
                    'roll_diff': roll_diff
                },
                rotation_degrees={
                    'pitch': pitch,
                    'yaw': yaw,
                    'roll': roll
                }
            )
            
        except Exception as e:
            logger.error(f"Error detecting movement: {e}")
            return None

    def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process a single frame and return pose data."""
        try:
            # Extract landmarks
            landmarks = self._extract_landmarks(frame)
            if landmarks is None:
                return None
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(landmarks, frame.shape)
            if quality_score < self.min_quality_score:
                return None
            
            # Get head pose
            pose = self._get_head_pose(landmarks, frame.shape)
            if pose is None:
                return None
            
            # Store pose in history
            self.pose_history.append(pose)
            
            # Detect movement
            movement = self._detect_movement(pose, self.previous_pose)
            if movement:
                self.movement_history.append(movement)
                if self.debug_mode:
                    logger.debug(f"Detected movement: {movement.direction} ({movement.magnitude:.1f}°)")
            
            self.previous_pose = pose
            
            return {
                'pose': pose,
                'landmarks': landmarks,
                'quality_score': quality_score,
                'movement': movement
            }
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return None

    def process_video(self, video_path: str, max_frames: Optional[int] = None) -> DetectionResult:
        """Process video file and detect head movements."""
        start_time = time.time()
        movements = []
        frames_processed = 0
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return DetectionResult(
                    success=False,
                    movements=[],
                    processing_time=0,
                    frames_processed=0,
                    error=f"Could not open video: {video_path}"
                )
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if max_frames and frame_count >= max_frames:
                    break
                
                result = self.process_frame(frame)
                if result and result.get('movement'):
                    movements.append(result['movement'])
                
                frames_processed += 1
                frame_count += 1
                
                if self.debug_mode and frame_count % 30 == 0:  # Log every 30 frames
                    logger.debug(f"Processed {frame_count} frames, detected {len(movements)} movements")
            
            cap.release()
            
            processing_time = time.time() - start_time
            
            return DetectionResult(
                success=True,
                movements=movements,
                processing_time=processing_time,
                frames_processed=frames_processed
            )
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return DetectionResult(
                success=False,
                movements=[],
                processing_time=time.time() - start_time,
                frames_processed=frames_processed,
                error=str(e)
            )

    def get_movements(self) -> List[MovementResult]:
        """Get all detected movements."""
        return list(self.movement_history)

    def reset(self) -> None:
        """Reset detector state."""
        self.previous_pose = None
        self.movement_history.clear()
        self.pose_history.clear()
        self.last_direction = None
        self.consecutive_count = 0

def create_dlib_detector(config: Optional[Dict[str, Any]] = None) -> DlibHeadDetector:
    """Create a dlib head detector with optional configuration."""
    if config is None:
        config = {}
    
    return DlibHeadDetector(
        shape_predictor_path=config.get('shape_predictor_path', 'shape_predictor_68_face_landmarks.dat'),
        min_rotation_degrees=config.get('min_rotation_degrees', 15.0),
        significant_rotation_degrees=config.get('significant_rotation_degrees', 25.0),
        min_confidence_threshold=config.get('min_confidence_threshold', 0.7),
        max_history=config.get('max_history', 10),
        debug_mode=config.get('debug_mode', False)
    )
