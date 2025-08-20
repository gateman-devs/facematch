"""
Simple Liveness Detection System
Randomly chooses between smile detection and head movement detection for mobile devices.
"""

import cv2
import numpy as np
import logging
import random
import time
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import mediapipe as mp
from .image_utils import get_image_processor
from .movement_config import (
    MovementThresholdConfig, 
    AdaptiveThresholdCalculator, 
    MovementValidationConfig,
    create_default_movement_config,
    create_adaptive_calculator,
    create_movement_validation_config
)
from .movement_confidence import (
    MovementConfidenceScorer,
    ConfidenceConfig,
    MovementData,
    create_default_confidence_config,
    create_confidence_scorer
)
from .flexible_sequence_validator import (
    FlexibleSequenceValidator,
    ValidationConfig,
    create_flexible_sequence_validator,
    create_default_validation_config,
    create_center_to_center_validation_config
)
from .motion_consistency import (
    MotionConsistencyValidator,
    MotionConsistencyConfig,
    create_motion_consistency_validator,
    create_default_motion_consistency_config
)
from .enhanced_anti_spoofing import (
    EnhancedAntiSpoofingEngine,
    AntiSpoofingConfig,
    AntiSpoofingResult,
    create_enhanced_anti_spoofing_engine,
    create_default_anti_spoofing_config
)
# Advanced head pose functionality removed for stability

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleLivenessDetector:
    """Simple liveness detection using smile and head movement."""
    
    def __init__(self, movement_config: Optional[MovementThresholdConfig] = None):
        """Initialize the simple liveness detector."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Challenge types - Only head movement for Gateman
        self.challenge_types = ['head_movement']
        
        # Initialize enhanced movement threshold configuration
        self.movement_config = movement_config or create_default_movement_config()
        self.adaptive_calculator = create_adaptive_calculator(self.movement_config)
        self.validation_config = create_movement_validation_config()
        
        # Initialize comprehensive movement confidence scoring system
        self.confidence_scorer = create_confidence_scorer()
        
        # Initialize flexible sequence validator for enhanced sequence validation with center-to-center config
        self.sequence_validator = create_flexible_sequence_validator(create_center_to_center_validation_config())
        
        # Initialize motion consistency validator for detecting artificial motion and video artifacts
        self.motion_consistency_validator = create_motion_consistency_validator(create_default_motion_consistency_config())
        
        # Initialize enhanced anti-spoofing engine for comprehensive detection
        self.enhanced_anti_spoofing = create_enhanced_anti_spoofing_engine(create_default_anti_spoofing_config())
        
        # Cache for adaptive thresholds to avoid recalculation
        self._adaptive_thresholds_cache = None
        self._last_face_size = None
        self._last_quality_metrics = None
        
        # Smile detection landmarks (mouth corners and center)
        self.smile_landmarks = {
            'left_mouth': 61,   # Left corner of mouth
            'right_mouth': 291, # Right corner of mouth
            'top_lip': 13,      # Top lip center
            'bottom_lip': 14    # Bottom lip center
        }
        
        # Head movement landmarks (nose and face boundary) - LEGACY
        self.head_landmarks = {
            'nose_tip': 1,
            'left_face': 234,
            'right_face': 454,
            'top_face': 10,
            'bottom_face': 152
        }
        
        # Initialize advanced 6DoF head pose estimator for accurate direction detection
        # self.head_pose_estimator = get_advanced_head_pose_estimator()  # Temporarily disabled
        self.head_pose_estimator = None  # Fallback to legacy method
        
        # Create faces directory for saving captured images
        self.faces_dir = os.getenv('FACES_DIR', "/app/faces")
        try:
            os.makedirs(self.faces_dir, exist_ok=True)
        except OSError:
            # Fallback to current directory if /app is not writable
            self.faces_dir = "./faces"
            os.makedirs(self.faces_dir, exist_ok=True)
        
        # Initialize image processor for URL/base64 handling
        self.image_processor = get_image_processor()
        
        logger.info("SimpleLivenessDetector initialized with Enhanced Anti-Spoofing + Face Capture + Face Comparison + Configurable Movement Thresholds")
    
    def perform_image_liveness_check(self, image_input: str) -> Dict:
        """
        Perform liveness check on a static image (URL or base64).
        This is a basic liveness assessment based on image quality and face detection.
        """
        try:
            # Process image input (URL or base64)
            image = self.image_processor.process_image_input(image_input)
            if image is None:
                return {
                    'success': False,
                    'passed': False,
                    'error': 'Failed to load image from provided input',
                    'liveness_score': 0.0
                }
            
            # Validate image
            is_valid, validation_message = self.image_processor.validate_image_for_face_detection(image)
            if not is_valid:
                return {
                    'success': False,
                    'passed': False,
                    'error': f'Image validation failed: {validation_message}',
                    'liveness_score': 0.0
                }
            
            # Resize if needed
            image = self.image_processor.resize_image_if_needed(image)
            
            # Perform enhanced anti-spoofing analysis
            anti_spoof_result = self.enhanced_anti_spoofing.analyze_frame(image)
            liveness_score = anti_spoof_result.overall_score
            
            # Check for face detection
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            face_detected = results.multi_face_landmarks is not None
            if not face_detected:
                return {
                    'success': True,
                    'passed': False,
                    'error': 'No face detected in the image',
                    'liveness_score': liveness_score,
                    'anti_spoofing': {
                        'overall_score': anti_spoof_result.overall_score,
                        'texture_score': anti_spoof_result.texture_score,
                        'lighting_score': anti_spoof_result.lighting_score,
                        'color_score': anti_spoof_result.color_score,
                        'edge_score': anti_spoof_result.edge_score,
                        'screen_reflection_score': anti_spoof_result.screen_reflection_score,
                        'passed': anti_spoof_result.passed,
                        'confidence': anti_spoof_result.confidence,
                        'analysis_details': anti_spoof_result.analysis_details
                    },
                    'face_detected': False
                }
            
            # Basic liveness threshold (lower for static images)
            liveness_threshold = 0.25
            passed = liveness_score >= liveness_threshold and face_detected
            
            return {
                'success': True,
                'passed': passed,
                'liveness_score': liveness_score,
                'anti_spoofing': {
                    'overall_score': anti_spoof_result.overall_score,
                    'texture_score': anti_spoof_result.texture_score,
                    'lighting_score': anti_spoof_result.lighting_score,
                    'color_score': anti_spoof_result.color_score,
                    'edge_score': anti_spoof_result.edge_score,
                    'screen_reflection_score': anti_spoof_result.screen_reflection_score,
                    'passed': anti_spoof_result.passed,
                    'confidence': anti_spoof_result.confidence,
                    'analysis_details': anti_spoof_result.analysis_details
                },
                'face_detected': face_detected,
                'threshold': liveness_threshold,
                'validation_message': validation_message
            }
            
        except Exception as e:
            logger.error(f"Error in image liveness check: {e}")
            return {
                'success': False,
                'passed': False,
                'error': f'Image liveness check failed: {str(e)}',
                'liveness_score': 0.0
            }
    
    def compare_faces(self, image1_input: str, image2_input: str, threshold: float = 0.6) -> Dict:
        """
        Compare two faces from different sources (URL, base64, or file path).
        
        Args:
            image1_input: First image (URL, base64, or file path)
            image2_input: Second image (URL, base64, or file path)  
            threshold: Similarity threshold (0.0-1.0)
            
        Returns:
            Dictionary with comparison results
        """
        try:
            # Process both images
            image1 = self._load_image_for_comparison(image1_input)
            image2 = self._load_image_for_comparison(image2_input)
            
            if image1 is None:
                return {
                    'success': False,
                    'match': False,
                    'error': 'Failed to load first image',
                    'similarity_score': 0.0
                }
            
            if image2 is None:
                return {
                    'success': False,
                    'match': False,
                    'error': 'Failed to load second image',
                    'similarity_score': 0.0
                }
            
            # Extract face embeddings using MediaPipe
            embedding1 = self._extract_face_embedding(image1)
            embedding2 = self._extract_face_embedding(image2)
            
            if embedding1 is None:
                return {
                    'success': False,
                    'match': False,
                    'error': 'No face detected in first image',
                    'similarity_score': 0.0
                }
            
            if embedding2 is None:
                return {
                    'success': False,
                    'match': False,
                    'error': 'No face detected in second image',
                    'similarity_score': 0.0
                }
            
            # Calculate similarity
            similarity = self._calculate_face_similarity(embedding1, embedding2)
            match = similarity >= threshold
            
            return {
                'success': True,
                'match': match,
                'similarity_score': float(similarity),
                'threshold': threshold,
                'confidence': 'high' if similarity > 0.8 else 'medium' if similarity > 0.6 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error comparing faces: {e}")
            return {
                'success': False,
                'match': False,
                'error': f'Face comparison failed: {str(e)}',
                'similarity_score': 0.0
            }
    
    def _load_image_for_comparison(self, image_input: str) -> Optional[np.ndarray]:
        """Load image from various sources for face comparison"""
        if os.path.isfile(image_input):
            # It's a file path
            return cv2.imread(image_input)
        else:
            # It's URL or base64
            return self.image_processor.process_image_input(image_input)
    
    def _extract_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding using MediaPipe landmarks as a simple representation.
        Note: This is a basic implementation. For production, consider using
        more sophisticated face recognition models like FaceNet or ArcFace.
        """
        try:
            # Resize image for consistent processing
            image = self.image_processor.resize_image_if_needed(image, 512)
            
            # Convert to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return None
            
            # Use the first detected face
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract key landmarks as simple embedding
            key_landmarks = [
                1, 2, 5, 6, 10, 151, 9, 10, 152, 175,  # Face contour
                33, 7, 163, 144, 145, 153, 154, 155, 133, 173,  # Eyes
                19, 20, 94, 125, 141, 235, 31, 228, 229, 230,  # Nose
                61, 84, 17, 314, 405, 320, 307, 375, 321, 308   # Mouth
            ]
            
            # Create embedding vector from landmark coordinates
            embedding = []
            for landmark_idx in key_landmarks:
                if landmark_idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[landmark_idx]
                    embedding.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(embedding, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting face embedding: {e}")
            return None
    
    def _calculate_face_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate similarity between two face embeddings"""
        try:
            # Normalize embeddings
            embedding1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
            embedding2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1_norm, embedding2_norm)
            
            # Convert to 0-1 range
            similarity = (similarity + 1) / 2
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def save_face_image(self, frame: np.ndarray, session_id: str) -> str:
        """Save a face image for internal use."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"face_{session_id}_{timestamp}_{unique_id}.jpg"
        filepath = os.path.join(self.faces_dir, filename)
        
        # Save the frame
        cv2.imwrite(filepath, frame)
        logger.info(f"Saved face image: {filepath}")
        
        # Return local filepath only
        return filepath
    

    
    def _calculate_video_quality_metrics(self, frame: np.ndarray) -> Dict[str, float]:
        """Calculate video quality metrics for adaptive threshold calculation."""
        try:
            # Convert to grayscale for analysis
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Calculate brightness (mean pixel value)
            brightness = float(np.mean(gray))
            
            # Calculate contrast (standard deviation)
            contrast = float(np.std(gray))
            
            # Calculate sharpness using Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = float(laplacian.var())
            
            return {
                'brightness': brightness,
                'contrast': contrast,
                'sharpness': sharpness
            }
        except Exception as e:
            logger.warning(f"Failed to calculate video quality metrics: {e}")
            # Return default values
            return {
                'brightness': 128.0,
                'contrast': 50.0,
                'sharpness': 100.0
            }
    
    def _calculate_face_size(self, face_landmarks, frame_shape: Tuple[int, int, int]) -> Tuple[int, int]:
        """Calculate face bounding box size from MediaPipe landmarks."""
        try:
            height, width = frame_shape[:2]
            
            # Get all landmark coordinates
            x_coords = [landmark.x * width for landmark in face_landmarks.landmark]
            y_coords = [landmark.y * height for landmark in face_landmarks.landmark]
            
            # Calculate bounding box
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            face_width = int(max_x - min_x)
            face_height = int(max_y - min_y)
            
            return (face_width, face_height)
        except Exception as e:
            logger.warning(f"Failed to calculate face size: {e}")
            # Return default face size (roughly 1/4 of frame)
            height, width = frame_shape[:2]
            return (width // 4, height // 4)
    
    def _get_adaptive_thresholds(self, frame: np.ndarray, face_landmarks) -> Dict[str, float]:
        """Get adaptive thresholds based on current frame and face detection."""
        try:
            # Calculate current metrics
            quality_metrics = self._calculate_video_quality_metrics(frame)
            face_size = self._calculate_face_size(face_landmarks, frame.shape)
            frame_dimensions = (frame.shape[1], frame.shape[0])  # (width, height)
            
            # Check if we can use cached thresholds
            if (self._adaptive_thresholds_cache is not None and 
                self._last_face_size == face_size and 
                self._last_quality_metrics == quality_metrics):
                return self._adaptive_thresholds_cache
            
            # Calculate new adaptive thresholds
            adaptive_thresholds = self.adaptive_calculator.calculate_adaptive_thresholds(
                face_size=face_size,
                video_quality_metrics=quality_metrics,
                frame_dimensions=frame_dimensions
            )
            
            # Cache the results
            self._adaptive_thresholds_cache = adaptive_thresholds
            self._last_face_size = face_size
            self._last_quality_metrics = quality_metrics
            
            return adaptive_thresholds
            
        except Exception as e:
            logger.warning(f"Failed to calculate adaptive thresholds: {e}")
            # Return base thresholds as fallback
            return self.adaptive_calculator._get_base_thresholds()
    
    def generate_challenge(self) -> Dict:
        """Generate a head movement challenge with anti-spoofing validation."""
        # Generate random head movement sequence - ALWAYS 4 directions
        directions = ['up', 'down', 'left', 'right']
        sequence_length = 4  # Always exactly 4 directions
        movement_sequence = [random.choice(directions) for _ in range(sequence_length)]
        
        instruction = f"Move your head in this sequence, starting and ending at center: {' → '.join(movement_sequence)}"
        
        return {
            'type': 'head_movement',
            'instruction': instruction,
            'duration': 30,  # Flexible duration - no time constraints
            'description': 'Move your head in the exact sequence shown, starting and ending at center position',
            'movement_sequence': movement_sequence,
            'direction_duration': None,  # No time constraints
            'anti_spoofing': True,  # Enable enhanced validation
            'movement_type': 'center_to_center'  # New movement type
        }
    
    def validate_video_challenge(self, video_path: str, challenge_type: str, movement_sequence: List[str] = None, session_id: str = None, reference_image: str = None) -> Dict:
        """
        Validate a video against the specified challenge type.
        
        Args:
            video_path: Path to the video file
            challenge_type: Type of challenge ('smile' or 'head_movement')
            
        Returns:
            Dictionary with validation results
        """
        # Performance timing tracking
        processing_start_time = time.time()
        video_open_start_time = None
        video_open_end_time = None
        frame_processing_start_time = None
        frame_processing_end_time = None
        face_comparison_start_time = None
        face_comparison_end_time = None
        
        try:
            logger.info(f"Validating {challenge_type} challenge: {video_path}")
            
            # Time video opening
            video_open_start_time = time.time()
            cap = cv2.VideoCapture(video_path)
            video_open_end_time = time.time()
            video_open_duration = video_open_end_time - video_open_start_time
            
            if not cap.isOpened():
                return {
                    'success': False,
                    'error': 'Failed to open video file',
                    'passed': False
                }
            
            # Get video properties with validation
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Validate FPS
            if fps <= 0 or fps > 120:
                fps = 30  # Default fallback
                logger.warning(f"Invalid FPS detected, using default: 30")
            
            # Validate frame count - handle negative or unrealistic values
            if total_frames <= 0 or total_frames > 1000000:
                logger.warning(f"Invalid frame count detected ({total_frames}), will count frames manually")
                total_frames = None
                duration = 0  # Will be calculated later
            else:
                duration = total_frames / fps
            
            logger.info(f"Video properties - FPS: {fps}, Expected frames: {total_frames}, Expected duration: {duration:.1f}s")
            
            # Log video dimensions for debugging
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Video dimensions: {width}x{height}")
            
            if challenge_type == 'head_movement':
                # Time frame processing
                frame_processing_start_time = time.time()
                validation_result = self._validate_head_movement_challenge(cap, fps, movement_sequence, session_id)
                frame_processing_end_time = time.time()
                frame_processing_duration = frame_processing_end_time - frame_processing_start_time
                
                # Add face comparison if reference image provided
                if reference_image and validation_result.get('success') and validation_result.get('face_image_path'):
                    face_comparison_start_time = time.time()
                    face_comparison = self.compare_faces(reference_image, validation_result['face_image_path'])
                    face_comparison_end_time = time.time()
                    face_comparison_duration = face_comparison_end_time - face_comparison_start_time
                    
                    validation_result['face_comparison'] = face_comparison
                    validation_result['face_comparison_time'] = face_comparison_duration
                    
                    # Update overall pass status to include face comparison
                    if validation_result.get('passed'):
                        validation_result['passed'] = face_comparison.get('match', False)
                        validation_result['face_match_required'] = True
                
                # Add timing information to result
                total_processing_time = time.time() - processing_start_time
                validation_result['video_open_time'] = video_open_duration
                validation_result['frame_processing_time'] = frame_processing_duration
                validation_result['total_processing_time'] = total_processing_time
                
                # Log detailed performance metrics
                logger.info(f"VIDEO_PROCESSING_DETAILS - session_id: {session_id}, "
                           f"video_open_time: {video_open_duration:.3f}s, "
                           f"frame_processing_time: {frame_processing_duration:.3f}s, "
                           f"face_comparison_time: {face_comparison_duration if face_comparison_start_time else 0:.3f}s, "
                           f"total_processing_time: {total_processing_time:.3f}s")
                
                return validation_result
            else:
                return {
                    'success': False,
                    'error': f'Unknown challenge type: {challenge_type}. Only head_movement is supported.',
                    'passed': False
                }
                
        except Exception as e:
            logger.error(f"Error validating challenge: {e}")
            return {
                'success': False,
                'error': str(e),
                'passed': False
            }
        finally:
            if 'cap' in locals():
                cap.release()
    
    def _validate_smile_challenge(self, cap: cv2.VideoCapture, fps: float) -> Dict:
        """Validate smile detection in video."""
        smile_frames = 0
        total_frames = 0
        smile_ratios = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            total_frames += 1
            
            # Safety check to prevent infinite loops
            if total_frames > 10000:
                logger.warning(f"Processed {total_frames} frames, stopping for safety")
                break
            
            # Validate frame
            if frame is None or frame.size == 0:
                continue
            
            # Process every 2nd frame for better movement detection
            if total_frames % 2 != 0:
                continue
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    smile_ratio = self._calculate_smile_ratio(face_landmarks, frame.shape)
                    smile_ratios.append(smile_ratio)
                    
                    # Consider it a smile if ratio is above threshold
                    if smile_ratio > 0.02:  # Adjust threshold as needed
                        smile_frames += 1
        
        logger.info(f"Processed {total_frames} total frames, found {len(smile_ratios)} valid face detections")
        
        # Calculate results
        processed_frames = len(smile_ratios)
        smile_percentage = (smile_frames / processed_frames * 100) if processed_frames > 0 else 0
        
        # Check if we have sufficient data
        if processed_frames < 10:
            return {
                'success': True,
                'passed': False,
                'challenge_type': 'smile',
                'error': 'Insufficient face detection in video - could not analyze smile reliably',
                'processed_frames': int(processed_frames),
                'total_frames_processed': int(total_frames)
            }
        
        # Require at least 60% of frames to show smiling
        passed = smile_percentage >= 60 and processed_frames >= 10
        
        avg_smile_ratio = float(np.mean(smile_ratios)) if smile_ratios else 0.0
        
        return {
            'success': True,
            'passed': bool(passed),  # Convert numpy.bool to Python bool
            'challenge_type': 'smile',
            'smile_percentage': float(smile_percentage),
            'avg_smile_ratio': avg_smile_ratio,
            'processed_frames': int(processed_frames),
            'details': {
                'smile_frames': int(smile_frames),
                'total_processed': int(processed_frames),
                'threshold_met': bool(smile_percentage >= 60)
            }
        }
    
    def _validate_head_movement_challenge(self, cap: cv2.VideoCapture, fps: float, movement_sequence: List[str] = None, session_id: str = None) -> Dict:
        """
        Validate head movement sequence. Uses advanced 6DoF head pose estimation if available,
        otherwise falls back to enhanced legacy tracking with anti-spoofing.
        """
        # Check if advanced head pose estimator is available
        if self.head_pose_estimator is not None:
            return self._validate_head_movement_advanced(cap, fps, movement_sequence)
        else:
            return self._validate_head_movement_legacy(cap, fps, movement_sequence, session_id)
    
    def _validate_head_movement_advanced(self, cap: cv2.VideoCapture, fps: float, movement_sequence: List[str] = None) -> Dict:
        """
        Validate head movement sequence using advanced 6DoF head pose estimation.
        Much more accurate than legacy nose position tracking.
        """
        head_pose_data_with_time = []
        total_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            total_frames += 1
            
            # Safety check to prevent infinite loops
            if total_frames > 10000:
                logger.warning(f"Processed {total_frames} frames, stopping for safety")
                break
            
            # Validate frame
            if frame is None or frame.size == 0:
                continue
            
            # Process every 2nd frame for better accuracy (less aggressive than every 3rd)
            if total_frames % 2 != 0:
                continue
            
            # Calculate timestamp
            timestamp = total_frames / fps
            
            # Use advanced 6DoF head pose estimation
            pose_data = self.head_pose_estimator.estimate_head_pose(frame)
            
            if pose_data and pose_data['confidence'] > 0.3:  # Minimum confidence threshold
                # Detect gaze direction from head pose
                direction, direction_confidence = self.head_pose_estimator.detect_gaze_direction(pose_data)
                
                head_pose_data_with_time.append({
                    'timestamp': float(timestamp),
                    'pitch': float(pose_data['pitch']),
                    'yaw': float(pose_data['yaw']),
                    'roll': float(pose_data['roll']),
                    'confidence': float(pose_data['confidence']),
                    'direction': str(direction),
                    'direction_confidence': float(direction_confidence)
                })
        
        logger.info(f"Advanced Head Pose: Processed {total_frames} total frames, found {len(head_pose_data_with_time)} valid pose detections")
        
        if len(head_pose_data_with_time) < 10:
            return {
                'success': True,
                'passed': False,
                'challenge_type': 'head_movement',
                'error': 'Insufficient head pose detection in video - could not track head movement reliably',
                'processed_frames': int(len(head_pose_data_with_time)),
                'total_frames_processed': int(total_frames),
                'method': 'Advanced 6DoF Head Pose Estimation'
            }
        
        # If no sequence provided, fall back to basic movement validation
        if not movement_sequence:
            movement_analysis = self._analyze_advanced_head_movement(head_pose_data_with_time)
            passed = (movement_analysis['direction_changes'] >= 2 and 
                     movement_analysis['avg_confidence'] > 0.5)
            
            return {
                'success': True,
                'passed': bool(passed),
                'challenge_type': 'head_movement',
                'details': movement_analysis,
                'processed_frames': int(len(head_pose_data_with_time)),
                'method': 'Advanced 6DoF Head Pose Estimation'
            }
        
        # Validate specific directional sequence using advanced pose data
        sequence_validation = self._validate_advanced_movement_sequence(head_pose_data_with_time, movement_sequence)
        
        return {
            'success': True,
            'passed': bool(sequence_validation['passed']),
            'challenge_type': 'head_movement',
            'expected_sequence': movement_sequence,
            'detected_sequence': sequence_validation['detected_sequence'],
            'sequence_accuracy': float(sequence_validation['accuracy']),
            'processed_frames': int(len(head_pose_data_with_time)),
            'details': sequence_validation,
            'method': 'Advanced 6DoF Head Pose Estimation'
        }
    
    def _validate_head_movement_legacy(self, cap: cv2.VideoCapture, fps: float, movement_sequence: List[str] = None, session_id: str = None) -> Dict:
        """
        Enhanced head movement validation with anti-spoofing and face capture.
        Uses nose position tracking with advanced security features.
        """
        # Performance timing tracking
        frame_processing_start_time = time.time()
        frame_read_start_time = None
        frame_read_end_time = None
        mediapipe_start_time = None
        mediapipe_end_time = None
        movement_analysis_start_time = None
        movement_analysis_end_time = None
        
        nose_positions_with_time = []
        anti_spoofing_scores = []
        total_frames = 0
        face_saved = False
        best_face_frame = None
        best_face_confidence = 0
        
        # Collect frames and landmarks for motion consistency analysis
        collected_frames = []
        collected_landmarks = []
        frame_timestamps = []
        
        while True:
            # Time frame reading
            frame_read_start_time = time.time()
            ret, frame = cap.read()
            frame_read_end_time = time.time()
            
            if not ret:
                break
                
            total_frames += 1
            
            # Safety check to prevent infinite loops
            if total_frames > 10000:
                logger.warning(f"Processed {total_frames} frames, stopping for safety")
                break
            
            # Validate frame
            if frame is None or frame.size == 0:
                continue
            
            # Process every 2nd frame for better movement detection
            if total_frames % 2 != 0:
                continue
            
            # Calculate timestamp
            timestamp = total_frames / fps
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply basic preprocessing to improve face detection
            # Resize if frame is too large (MediaPipe works better with moderate sizes)
            h, w = rgb_frame.shape[:2]
            if w > 640:
                scale = 640 / w
                new_w, new_h = int(w * scale), int(h * scale)
                rgb_frame = cv2.resize(rgb_frame, (new_w, new_h))
            
            # Time MediaPipe processing
            mediapipe_start_time = time.time()
            results = self.face_mesh.process(rgb_frame)
            mediapipe_end_time = time.time()
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    nose_pos = self._get_nose_position(face_landmarks, frame.shape)
                    if nose_pos:
                        # Record nose position
                        nose_positions_with_time.append({
                            'timestamp': float(timestamp),
                            'x': float(nose_pos[0]),
                            'y': float(nose_pos[1])
                        })
                        
                        # Collect frame and landmarks for motion consistency analysis
                        collected_frames.append(frame.copy())
                        collected_landmarks.append(face_landmarks)
                        frame_timestamps.append(timestamp)
                        
                        # Perform enhanced anti-spoofing analysis
                        anti_spoof_result = self.enhanced_anti_spoofing.analyze_frame(frame)
                        anti_spoofing_scores.append(anti_spoof_result)
                        
                        # Save best quality face image for matching
                        face_confidence = anti_spoof_result.overall_score
                        if face_confidence > best_face_confidence and session_id:
                            best_face_confidence = face_confidence
                            best_face_frame = frame.copy()
            else:
                # Log when face is not detected for debugging
                if total_frames % 30 == 0:  # Log every 30th frame to avoid spam
                    logger.debug(f"No face detected in frame {total_frames}")
        
        # Save the best face image if we have one
        face_image_path = None
        if best_face_frame is not None and session_id:
            try:
                face_image_path = self.save_face_image(best_face_frame, session_id)
                face_saved = True
                logger.info(f"Saved best face image with confidence: {best_face_confidence:.3f}")
            except Exception as e:
                logger.warning(f"Failed to save face image: {e}")
        
        logger.info(f"Enhanced Validation: Processed {total_frames} total frames, found {len(nose_positions_with_time)} valid face positions")
        
        # Calculate enhanced anti-spoofing metrics
        avg_anti_spoofing = {}
        if anti_spoofing_scores:
            # Extract scores from AntiSpoofingResult objects
            overall_scores = [result.overall_score for result in anti_spoofing_scores]
            texture_scores = [result.texture_score for result in anti_spoofing_scores]
            lighting_scores = [result.lighting_score for result in anti_spoofing_scores]
            color_scores = [result.color_score for result in anti_spoofing_scores]
            edge_scores = [result.edge_score for result in anti_spoofing_scores]
            screen_scores = [result.screen_reflection_score for result in anti_spoofing_scores]
            confidence_scores = [result.confidence for result in anti_spoofing_scores]
            
            avg_anti_spoofing = {
                'overall_score': float(np.mean(overall_scores)),
                'texture_score': float(np.mean(texture_scores)),
                'lighting_score': float(np.mean(lighting_scores)),
                'color_score': float(np.mean(color_scores)),
                'edge_score': float(np.mean(edge_scores)),
                'screen_reflection_score': float(np.mean(screen_scores)),
                'confidence': float(np.mean(confidence_scores)),
                'passed': float(np.mean([result.passed for result in anti_spoofing_scores])) > 0.5
            }
        
        # Determine if anti-spoofing validation passed - more lenient threshold
        anti_spoof_passed = avg_anti_spoofing.get('overall_score', 0) >= 0.2
        
        # Perform motion consistency analysis if we have sufficient frames
        motion_consistency_result = None
        motion_consistency_passed = True  # Default to pass if analysis not performed
        
        if len(collected_frames) >= 5:
            try:
                logger.info(f"Performing motion consistency analysis on {len(collected_frames)} frames")
                motion_consistency_result = self.motion_consistency_validator.analyze_motion_consistency(
                    collected_frames, collected_landmarks, frame_timestamps
                )
                
                # Motion consistency validation passes if overall score is above threshold
                motion_consistency_passed = motion_consistency_result.is_natural_motion
                
                logger.info(f"Motion consistency analysis: score={motion_consistency_result.overall_score:.3f}, "
                           f"natural_motion={motion_consistency_result.is_natural_motion}, "
                           f"pattern_score={motion_consistency_result.pattern_score:.3f}, "
                           f"edge_score={motion_consistency_result.edge_score:.3f}, "
                           f"blur_score={motion_consistency_result.blur_score:.3f}")
                
            except Exception as e:
                logger.warning(f"Motion consistency analysis failed: {e}")
                motion_consistency_result = None
                motion_consistency_passed = True  # Don't fail validation due to analysis error
        else:
            logger.info("Insufficient frames for motion consistency analysis")
        
        if len(nose_positions_with_time) < 5:
            return {
                'success': True,
                'passed': False,
                'challenge_type': 'head_movement',
                'error': 'Insufficient face detection in video - could not track head movement reliably',
                'processed_frames': int(len(nose_positions_with_time)),
                'total_frames_processed': int(total_frames),
                'method': 'Enhanced Anti-Spoofing + Head Movement + Motion Consistency',
                'anti_spoofing': avg_anti_spoofing,
                'anti_spoof_passed': anti_spoof_passed,
                'motion_consistency_passed': motion_consistency_passed,
                'face_saved': face_saved,
                'face_image_path': face_image_path
            }
        
        # If no sequence provided, fall back to basic movement validation
        if not movement_sequence:
            movement_analysis_start_time = time.time()
            movement_analysis = self._analyze_basic_head_movement(nose_positions_with_time)
            movement_analysis_end_time = time.time()
            movement_analysis_duration = movement_analysis_end_time - movement_analysis_start_time
            
            # More lenient thresholds for better head movement detection
            movement_passed = (movement_analysis['horizontal_range'] > 30 and 
                             movement_analysis['movement_changes'] >= 1)
            
            # Overall validation requires both movement and anti-spoofing to pass
            overall_passed = movement_passed and anti_spoof_passed
            
            # Convert all numpy types to Python types
            converted_analysis = {}
            for key, value in movement_analysis.items():
                if isinstance(value, np.floating):
                    converted_analysis[key] = float(value)
                elif isinstance(value, np.integer):
                    converted_analysis[key] = int(value)
                else:
                    converted_analysis[key] = value
            
            # Calculate total processing time
            total_frame_processing_time = time.time() - frame_processing_start_time
            
            # Log detailed frame processing performance
            logger.info(f"FRAME_PROCESSING_DETAILS - session_id: {session_id}, "
                       f"total_frames: {total_frames}, processed_frames: {len(nose_positions_with_time)}, "
                       f"movement_analysis_time: {movement_analysis_duration:.3f}s, "
                       f"total_frame_processing_time: {total_frame_processing_time:.3f}s, "
                       f"avg_time_per_frame: {total_frame_processing_time/total_frames if total_frames > 0 else 0:.3f}s")
            
            # Prepare motion consistency results for fallback case
            motion_consistency_details = {}
            if motion_consistency_result:
                motion_consistency_details = {
                    'overall_score': float(motion_consistency_result.overall_score),
                    'pattern_score': float(motion_consistency_result.pattern_score),
                    'edge_score': float(motion_consistency_result.edge_score),
                    'blur_score': float(motion_consistency_result.blur_score),
                    'is_natural_motion': bool(motion_consistency_result.is_natural_motion),
                    'confidence': float(motion_consistency_result.confidence),
                    'details': motion_consistency_result.details
                }
            
            # Update overall passed to include motion consistency
            overall_passed = movement_passed and anti_spoof_passed and motion_consistency_passed
            
            return {
                'success': True,
                'passed': bool(overall_passed),
                'challenge_type': 'head_movement',
                'details': converted_analysis,
                'processed_frames': int(len(nose_positions_with_time)),
                'method': 'Enhanced Anti-Spoofing + Head Movement + Motion Consistency',
                'anti_spoofing': avg_anti_spoofing,
                'anti_spoof_passed': anti_spoof_passed,
                'movement_passed': movement_passed,
                'motion_consistency': motion_consistency_details,
                'motion_consistency_passed': motion_consistency_passed,
                'face_saved': face_saved,
                'face_image_path': face_image_path,
                'frame_processing_time': total_frame_processing_time,
                'movement_analysis_time': movement_analysis_duration
            }
        
        # Calculate adaptive thresholds based on the best face frame and quality
        adaptive_thresholds = None
        if best_face_frame is not None and results.multi_face_landmarks:
            try:
                # Use the last detected face landmarks for threshold calculation
                adaptive_thresholds = self._get_adaptive_thresholds(best_face_frame, results.multi_face_landmarks[0])
                logger.info(f"Using adaptive thresholds: min_movement={adaptive_thresholds.get('min_movement_threshold', 'N/A'):.1f}, "
                           f"face_scale={adaptive_thresholds.get('face_scale_factor', 'N/A'):.2f}, "
                           f"quality_scale={adaptive_thresholds.get('quality_scale_factor', 'N/A'):.2f}")
            except Exception as e:
                logger.warning(f"Failed to calculate adaptive thresholds: {e}")
                adaptive_thresholds = None
        
        # Calculate frame context for confidence scoring
        frame_context = None
        if best_face_frame is not None:
            frame_context = self._calculate_video_quality_metrics(best_face_frame)
        
        # Validate specific directional sequence
        sequence_validation = self._validate_movement_sequence(nose_positions_with_time, movement_sequence, adaptive_thresholds, frame_context)
        
        # Overall validation requires sequence, anti-spoofing, and motion consistency to pass
        sequence_passed = sequence_validation['passed']
        overall_passed = sequence_passed and anti_spoof_passed and motion_consistency_passed
        
        # Convert all numpy types in sequence validation
        converted_validation = {}
        for key, value in sequence_validation.items():
            if isinstance(value, np.floating):
                converted_validation[key] = float(value)
            elif isinstance(value, np.integer):
                converted_validation[key] = int(value)
            elif isinstance(value, (np.bool_, bool)):
                converted_validation[key] = bool(value)
            elif isinstance(value, list):
                # Convert list items if they contain numpy types
                converted_validation[key] = [float(x) if isinstance(x, np.floating) else 
                                           int(x) if isinstance(x, np.integer) else 
                                           bool(x) if isinstance(x, (np.bool_, bool)) else x 
                                           for x in value]
            else:
                converted_validation[key] = value
        
        # Prepare motion consistency results for return
        motion_consistency_details = {}
        if motion_consistency_result:
            motion_consistency_details = {
                'overall_score': float(motion_consistency_result.overall_score),
                'pattern_score': float(motion_consistency_result.pattern_score),
                'edge_score': float(motion_consistency_result.edge_score),
                'blur_score': float(motion_consistency_result.blur_score),
                'is_natural_motion': bool(motion_consistency_result.is_natural_motion),
                'confidence': float(motion_consistency_result.confidence),
                'details': motion_consistency_result.details
            }
        
        return {
            'success': True,
            'passed': bool(overall_passed),
            'challenge_type': 'head_movement',
            'expected_sequence': movement_sequence,
            'detected_sequence': converted_validation['detected_sequence'],
            'sequence_accuracy': float(converted_validation['accuracy']),
            'processed_frames': int(len(nose_positions_with_time)),
            'details': converted_validation,
            'method': 'Enhanced Anti-Spoofing + Head Movement + Motion Consistency',
            'anti_spoofing': avg_anti_spoofing,
            'anti_spoof_passed': anti_spoof_passed,
            'sequence_passed': sequence_passed,
            'motion_consistency': motion_consistency_details,
            'motion_consistency_passed': motion_consistency_passed,
            'face_saved': face_saved,
            'face_image_path': face_image_path
        }
    
    def _calculate_smile_ratio(self, face_landmarks, frame_shape: Tuple[int, int, int]) -> float:
        """Calculate smile ratio based on mouth landmarks."""
        h, w = frame_shape[:2]
        
        # Get mouth corner and lip positions
        left_mouth = face_landmarks.landmark[self.smile_landmarks['left_mouth']]
        right_mouth = face_landmarks.landmark[self.smile_landmarks['right_mouth']]
        top_lip = face_landmarks.landmark[self.smile_landmarks['top_lip']]
        bottom_lip = face_landmarks.landmark[self.smile_landmarks['bottom_lip']]
        
        # Convert to pixel coordinates
        left_x, left_y = int(left_mouth.x * w), int(left_mouth.y * h)
        right_x, right_y = int(right_mouth.x * w), int(right_mouth.y * h)
        top_x, top_y = int(top_lip.x * w), int(top_lip.y * h)
        bottom_x, bottom_y = int(bottom_lip.x * w), int(bottom_lip.y * h)
        
        # Calculate mouth width and height
        mouth_width = abs(right_x - left_x)
        mouth_height = abs(bottom_y - top_y)
        
        # Smile ratio: wider mouth relative to height indicates smile
        if mouth_height > 0:
            smile_ratio = mouth_width / mouth_height
            # Normalize by face width for consistency
            face_width = w  # Simplified
            return smile_ratio / face_width
        
        return 0.0
    
    def _get_nose_position(self, face_landmarks, frame_shape: Tuple[int, int, int]) -> Optional[Tuple[float, float]]:
        """Get normalized nose position with proper coordinate validation."""
        h, w = frame_shape[:2]
        
        nose_tip = face_landmarks.landmark[self.head_landmarks['nose_tip']]
        
        # Validate nose position is within reasonable bounds
        # MediaPipe returns normalized coordinates [0, 1]
        if 0 <= nose_tip.x <= 1 and 0 <= nose_tip.y <= 1:
            return (nose_tip.x, nose_tip.y)  # Return normalized coordinates
        else:
            logger.warning(f"Invalid nose position detected: x={nose_tip.x}, y={nose_tip.y}")
            return None
    
    def _normalize_coordinates(self, x: float, y: float, frame_width: int, frame_height: int) -> Tuple[float, float]:
        """Normalize pixel coordinates to [0, 1] range."""
        return (x / frame_width, y / frame_height)
    
    def _denormalize_coordinates(self, norm_x: float, norm_y: float, frame_width: int, frame_height: int) -> Tuple[float, float]:
        """Convert normalized coordinates back to pixel coordinates."""
        return (norm_x * frame_width, norm_y * frame_height)
    
    def _calculate_movement_magnitude(self, start_pos: Dict, end_pos: Dict, frame_width: int = 640, frame_height: int = 480) -> Dict:
        """Calculate movement magnitude and direction with proper coordinate handling."""
        # Calculate deltas in normalized coordinates
        dx_norm = end_pos['x'] - start_pos['x']
        dy_norm = end_pos['y'] - start_pos['y']
        
        # Convert to pixel coordinates for magnitude calculation
        dx_pixels = dx_norm * frame_width
        dy_pixels = dy_norm * frame_height
        
        # Calculate magnitude
        magnitude = np.sqrt(dx_pixels**2 + dy_pixels**2)
        
        return {
            'dx_norm': dx_norm,
            'dy_norm': dy_norm,
            'dx_pixels': dx_pixels,
            'dy_pixels': dy_pixels,
            'magnitude': magnitude,
            'abs_dx': abs(dx_pixels),
            'abs_dy': abs(dy_pixels)
        }
    
    def _analyze_basic_head_movement(self, nose_positions_with_time: List[Dict]) -> Dict:
        """Analyze basic head movement patterns (backward compatibility)."""
        if len(nose_positions_with_time) < 3:
            return {
                'horizontal_range': 0,
                'vertical_range': 0,
                'movement_changes': 0,
                'total_movement': 0
            }
        
        x_positions = [pos['x'] for pos in nose_positions_with_time]
        y_positions = [pos['y'] for pos in nose_positions_with_time]
        
        # Calculate ranges (in normalized coordinates, then convert to pixels)
        horizontal_range = (max(x_positions) - min(x_positions)) * 640  # Assume 640px width
        vertical_range = (max(y_positions) - min(y_positions)) * 480    # Assume 480px height
        
        # Count direction changes (left-right movement)
        direction_changes = 0
        if len(x_positions) >= 3:
            for i in range(2, len(x_positions)):
                # Check if direction changed
                prev_direction = x_positions[i-1] - x_positions[i-2]
                curr_direction = x_positions[i] - x_positions[i-1]
                
                if prev_direction * curr_direction < 0:  # Sign change = direction change
                    direction_changes += 1
        
        # Calculate total movement
        total_movement = 0
        for i in range(1, len(nose_positions_with_time)):
            dx = (nose_positions_with_time[i]['x'] - nose_positions_with_time[i-1]['x']) * 640
            dy = (nose_positions_with_time[i]['y'] - nose_positions_with_time[i-1]['y']) * 480
            total_movement += np.sqrt(dx*dx + dy*dy)
        
        return {
            'horizontal_range': float(horizontal_range),
            'vertical_range': float(vertical_range),
            'movement_changes': int(direction_changes),
            'total_movement': float(total_movement),
            'smoothness': float(total_movement / len(nose_positions_with_time)) if nose_positions_with_time else 0.0
        }
    
    def _analyze_advanced_head_movement(self, head_pose_data_with_time: List[Dict]) -> Dict:
        """
        Analyze head movement patterns using advanced 6DoF pose data.
        Much more accurate than legacy nose position analysis.
        """
        if not head_pose_data_with_time:
            return {
                'direction_changes': 0,
                'avg_confidence': 0.0,
                'pitch_range': 0.0,
                'yaw_range': 0.0,
                'roll_range': 0.0,
                'dominant_directions': []
            }
        
        # Extract pose angles
        pitches = [data['pitch'] for data in head_pose_data_with_time]
        yaws = [data['yaw'] for data in head_pose_data_with_time]
        rolls = [data['roll'] for data in head_pose_data_with_time]
        confidences = [data['confidence'] for data in head_pose_data_with_time]
        directions = [data['direction'] for data in head_pose_data_with_time]
        
        # Calculate ranges
        pitch_range = float(max(pitches) - min(pitches))
        yaw_range = float(max(yaws) - min(yaws))
        roll_range = float(max(rolls) - min(rolls))
        avg_confidence = float(np.mean(confidences))
        
        # Count direction changes
        direction_changes = 0
        prev_direction = None
        for direction in directions:
            if direction != 'center' and direction != prev_direction:
                direction_changes += 1
                prev_direction = direction
        
        # Find dominant directions
        from collections import Counter
        direction_counts = Counter([d for d in directions if d != 'center'])
        dominant_directions = [dir for dir, count in direction_counts.most_common(4)]
        
        return {
            'direction_changes': int(direction_changes),
            'avg_confidence': float(avg_confidence),
            'pitch_range': float(pitch_range),
            'yaw_range': float(yaw_range),
            'roll_range': float(roll_range),
            'dominant_directions': dominant_directions,
            'total_measurements': len(head_pose_data_with_time)
        }

    def _validate_advanced_movement_sequence(self, head_pose_data_with_time: List[Dict], expected_sequence: List[str]) -> Dict:
        """
        Validate movement sequence using advanced 6DoF head pose data.
        Much more accurate than legacy nose position validation.
        """
        if not head_pose_data_with_time or not expected_sequence:
            return {
                'passed': False,
                'accuracy': 0.0,
                'detected_sequence': [],
                'expected_sequence': expected_sequence,
                'segment_accuracies': [],
                'total_duration': 0.0,
                'direction_duration': 0.0,
                'method': 'Advanced 6DoF Head Pose Validation'
            }
        
        # NON-TIME-BASED ANALYSIS: Analyze all movements detected in the video
        # No time constraints - just check if the expected sequence was performed
        total_duration = head_pose_data_with_time[-1]['timestamp'] - head_pose_data_with_time[0]['timestamp']
        
        detected_sequence = []
        segment_accuracies = []
        
        # Extract all detected movements from the video
        all_detected_movements = [data['direction'] for data in head_pose_data_with_time if data.get('direction') != 'center']
        all_movement_confidences = [data['direction_confidence'] for data in head_pose_data_with_time if data.get('direction') != 'center']
        
        # Apply post-processing improvements:
        # 1. Deduplicate consecutive movements (only record 1 direction)
        # 2. Reverse the order (last to first)
        # 3. Remove opposite directions and duplicates
        processed_movements, processed_confidences = self._post_process_movements(all_detected_movements, all_movement_confidences)
        
        # For each expected direction, find the best matching detected movement
        for expected_direction in expected_sequence:
            # Find movements that match the expected direction
            matching_movements = []
            matching_confidences = []
            
            for i, detected_dir in enumerate(processed_movements):
                if detected_dir == expected_direction:
                    matching_movements.append(detected_dir)
                    matching_confidences.append(processed_confidences[i])
            
            if matching_movements:
                # Use the highest confidence match
                best_confidence = max(matching_confidences)
                detected_sequence.append(expected_direction)
                segment_accuracies.append(float(best_confidence))
            else:
                # No matching movement found
                detected_sequence.append('not_detected')
                segment_accuracies.append(0.0)
        
        # Calculate overall accuracy
        overall_accuracy = float(np.mean(segment_accuracies)) if segment_accuracies else 0.0
        
        # Pass if accuracy is above threshold (60% for non-time-based method)
        # More lenient since we're not enforcing time constraints
        passed = overall_accuracy >= 0.6 and len([acc for acc in segment_accuracies if acc > 0.3]) >= len(expected_sequence) * 0.5
        
        return {
            'passed': bool(passed),
            'accuracy': float(overall_accuracy),
            'detected_sequence': detected_sequence,
            'expected_sequence': expected_sequence,
            'segment_accuracies': [float(acc) for acc in segment_accuracies],
            'total_duration': float(total_duration),
            'direction_duration': None,  # No time constraints
            'method': 'Non-Time-Based 6DoF Head Pose Validation'
        }

    def _detect_advanced_movement_direction(self, segment_data: List[Dict], expected_direction: str) -> Tuple[str, float]:
        """
        Detect movement direction using advanced 6DoF head pose analysis.
        Much more accurate than legacy pixel-based detection.
        """
        if len(segment_data) < 3:
            return 'none', 0.0
        
        # Use the direction detection from the head pose estimator for each frame
        # and aggregate the results
        directions_in_segment = [data['direction'] for data in segment_data]
        direction_confidences = [data['direction_confidence'] for data in segment_data]
        
        # Count occurrences of each direction
        from collections import Counter
        direction_counts = Counter(directions_in_segment)
        
        # Remove 'center' from consideration unless it's the most common
        if 'center' in direction_counts and len(direction_counts) > 1:
            center_count = direction_counts['center']
            del direction_counts['center']
            
            # Only add center back if it's overwhelmingly the most common
            if not direction_counts or center_count > max(direction_counts.values()) * 2:
                direction_counts['center'] = center_count
        
        if not direction_counts:
            return 'center', float(np.mean(direction_confidences))
        
        # Get the most common direction
        most_common_direction = direction_counts.most_common(1)[0][0]
        
        # Calculate confidence based on:
        # 1. How consistently this direction was detected
        # 2. Average confidence of detections for this direction
        consistency = direction_counts[most_common_direction] / len(directions_in_segment)
        
        # Get average confidence for frames where this direction was detected
        direction_specific_confidences = [
            data['direction_confidence'] for data in segment_data 
            if data['direction'] == most_common_direction
        ]
        avg_direction_confidence = float(np.mean(direction_specific_confidences)) if direction_specific_confidences else 0.0
        
        # Combined confidence score
        final_confidence = (consistency * 0.6 + avg_direction_confidence * 0.4)
        
        return str(most_common_direction), float(final_confidence)

    def _post_process_movements(self, movements: List[str], confidences: List[float]) -> Tuple[List[str], List[float]]:
        """
        Post-process movements to apply the three improvements:
        1. Deduplicate consecutive movements (only record 1 direction)
        2. Reverse the order (last to first)
        3. Remove opposite directions and duplicates
        """
        if not movements:
            return [], []
        
        # Step 1: Deduplicate consecutive movements
        # If we see up, up, up, just take 1 up. left, left, left, just take 1 left
        deduplicated_movements = []
        deduplicated_confidences = []
        current_direction = None
        
        for i, movement in enumerate(movements):
            if movement != current_direction:
                deduplicated_movements.append(movement)
                deduplicated_confidences.append(confidences[i])
                current_direction = movement
        
        # Step 2: Reverse the order (last to first)
        reversed_movements = list(reversed(deduplicated_movements))
        reversed_confidences = list(reversed(deduplicated_confidences))
        
        # Step 3: Remove opposite directions and duplicates
        # Don't generate left then right or up and up, etc.
        filtered_movements = []
        filtered_confidences = []
        opposite_pairs = [
            ('left', 'right'),
            ('right', 'left'),
            ('up', 'down'),
            ('down', 'up')
        ]
        
        for i, movement in enumerate(reversed_movements):
            # Check if this movement is opposite to the previous one
            if i > 0:
                prev_movement = filtered_movements[-1]
                is_opposite = False
                
                for dir1, dir2 in opposite_pairs:
                    if (prev_movement == dir1 and movement == dir2) or \
                       (prev_movement == dir2 and movement == dir1):
                        is_opposite = True
                        break
                
                if is_opposite:
                    continue
            
            # Check if this movement is the same as the previous one
            if i > 0 and movement == filtered_movements[-1]:
                continue
            
            filtered_movements.append(movement)
            filtered_confidences.append(reversed_confidences[i])
        
        return filtered_movements, filtered_confidences

    def _validate_movement_sequence(self, nose_positions_with_time: List[Dict], expected_sequence: List[str], adaptive_thresholds: Optional[Dict[str, float]] = None, frame_context: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Enhanced flexible sequence validation algorithm using center-to-center movement detection.
        Records any movement that starts from center, moves to a direction, and returns to center.
        This removes time-based constraints and focuses on natural movement patterns.
        
        This implementation addresses requirements:
        - Center-to-center movement patterns
        - No time-based constraints
        - Natural movement detection
        - Flexible sequence matching
        """
        if not nose_positions_with_time or not expected_sequence:
            return {
                'passed': False,
                'accuracy': 0.0,
                'detected_sequence': [],
                'error': 'Insufficient data for sequence validation'
            }
        
        # Detect all center-to-center movements throughout the video
        all_movements = self._detect_center_to_center_movements(nose_positions_with_time, adaptive_thresholds, frame_context)
        
        if not all_movements:
            logger.info("No significant movements detected in video")
            return {
                'passed': False,
                'accuracy': 0.0,
                'detected_sequence': [],
                'error': 'No significant movements detected'
            }
        
        logger.info(f"Detected {len(all_movements)} significant movements: {[m['direction'] for m in all_movements]}")
        
        # Use the enhanced flexible sequence validator for comprehensive validation
        validation_result = self.sequence_validator.validate_sequence(all_movements, expected_sequence)
        
        # Log detailed validation results
        if validation_result['passed']:
            logger.info(f"✓ Sequence validation PASSED using {validation_result['validation_strategy']} strategy")
            logger.info(f"  Accuracy: {validation_result['accuracy']:.3f}")
            logger.info(f"  Matched sequence: {validation_result['detected_sequence']}")
            
            # Log flexibility features used
            flexibility = validation_result.get('flexibility_features', {})
            if flexibility.get('timing_variations_handled'):
                logger.info("  ✓ Handled timing variations")
            if flexibility.get('extra_movements_filtered'):
                logger.info(f"  ✓ Filtered {validation_result['match_details']['extra_movements_ignored']} extra movements")
            if flexibility.get('pause_tolerance_applied'):
                logger.info("  ✓ Applied pause tolerance")
            if flexibility.get('speed_adaptation_applied'):
                logger.info("  ✓ Applied speed adaptation")
            if flexibility.get('sequence_found_anywhere'):
                logger.info("  ✓ Found sequence anywhere in video (not just at start)")
        else:
            logger.info(f"✗ Sequence validation FAILED: {validation_result.get('error', 'Unknown reason')}")
            logger.info(f"  Expected: {expected_sequence}")
            logger.info(f"  Detected: {validation_result['detected_sequence']}")
            
            # Log failure analysis if available
            failure_analysis = validation_result.get('failure_analysis', {})
            if failure_analysis:
                logger.info(f"  Failure analysis: {failure_analysis.get('reason', 'No details')}")
        
        return validation_result
    
    def _detect_movement_direction(self, segment_positions: List[Dict], adaptive_thresholds: Optional[Dict[str, float]] = None, frame_context: Optional[Dict[str, Any]] = None) -> Tuple[str, float]:
        """Detect the primary movement direction in a time segment with comprehensive confidence scoring."""
        if len(segment_positions) < 3:
            return 'none', 0.0
        
        # Use adaptive thresholds if provided, otherwise use base configuration
        if adaptive_thresholds is None:
            adaptive_thresholds = self.adaptive_calculator._get_base_thresholds()
        
        # Calculate overall movement from start to end
        start_pos = segment_positions[0]
        end_pos = segment_positions[-1]
        
        # Calculate movement deltas in normalized coordinates
        dx = end_pos['x'] - start_pos['x']
        dy = end_pos['y'] - start_pos['y']
        
        # Convert to pixel coordinates using adaptive frame scaling
        frame_scale_x = adaptive_thresholds.get('frame_scale_x', 1.0)
        frame_scale_y = adaptive_thresholds.get('frame_scale_y', 1.0)
        dx_pixels = dx * self.movement_config.base_frame_width * frame_scale_x
        dy_pixels = dy * self.movement_config.base_frame_height * frame_scale_y
        
        # Determine primary direction based on largest movement
        abs_dx = abs(dx_pixels)
        abs_dy = abs(dy_pixels)
        
        # Use adaptive minimum movement threshold
        min_movement = adaptive_thresholds.get('min_movement_threshold', self.movement_config.min_movement_threshold)
        
        if abs_dx < min_movement and abs_dy < min_movement:
            return 'none', 0.0
        
        # Use the new coordinate handling method for better accuracy
        movement_data = self._calculate_movement_magnitude(start_pos, end_pos, 
                                                         int(self.movement_config.base_frame_width * frame_scale_x),
                                                         int(self.movement_config.base_frame_height * frame_scale_y))
        
        # Determine direction with proper coordinate system handling
        if movement_data['abs_dx'] > movement_data['abs_dy']:
            # Horizontal movement - FIXED: Correct directional mapping
            # MediaPipe coordinates: when user moves right, x increases (dx > 0)
            # When user moves left, x decreases (dx < 0)
            if movement_data['dx_pixels'] > 0:
                direction = 'right'  # User moved right (x increased)
            else:
                direction = 'left'   # User moved left (x decreased)
        else:
            # Vertical movement - MediaPipe y increases downward
            if movement_data['dy_pixels'] > 0:
                direction = 'down'   # User moved down (y increased)
            else:
                direction = 'up'     # User moved up (y decreased)
        
        # Prepare movement data for comprehensive confidence scoring
        movement_info = {
            'direction': direction,
            'magnitude': movement_data['magnitude'],
            'dx_pixels': movement_data['dx_pixels'],
            'dy_pixels': movement_data['dy_pixels'],
            'dx_norm': movement_data['dx_norm'],
            'dy_norm': movement_data['dy_norm'],
            'timestamp': end_pos.get('timestamp', time.time()),
            'start_position': (start_pos['x'], start_pos['y']),
            'end_position': (end_pos['x'], end_pos['y']),
            'frame_indices': (0, len(segment_positions) - 1)
        }
        
        # Get historical movements for consistency analysis
        historical_movements = []
        if hasattr(self, '_recent_movements'):
            historical_movements = self._recent_movements[-10:]  # Last 10 movements
        
        # Calculate comprehensive confidence score
        enhanced_movement = self.confidence_scorer.calculate_comprehensive_confidence(
            movement_info, frame_context, historical_movements
        )
        
        # Store movement for future consistency analysis
        if not hasattr(self, '_recent_movements'):
            self._recent_movements = []
        self._recent_movements.append(movement_info)
        if len(self._recent_movements) > 20:  # Keep only recent movements
            self._recent_movements = self._recent_movements[-20:]
        
        return str(direction), float(enhanced_movement.confidence)

    def _analyze_head_movement(self, nose_positions: List[Tuple[float, float]]) -> Dict:
        """Analyze head movement patterns."""
        if len(nose_positions) < 3:
            return {
                'horizontal_range': 0,
                'vertical_range': 0,
                'movement_changes': 0,
                'total_movement': 0
            }
        
        x_positions = [pos[0] for pos in nose_positions]
        y_positions = [pos[1] for pos in nose_positions]
        
        # Calculate ranges (in normalized coordinates, then convert to pixels)
        horizontal_range = (max(x_positions) - min(x_positions)) * 640  # Assume 640px width
        vertical_range = (max(y_positions) - min(y_positions)) * 480    # Assume 480px height
        
        # Count direction changes (left-right movement)
        direction_changes = 0
        if len(x_positions) >= 3:
            for i in range(2, len(x_positions)):
                # Check if direction changed
                prev_direction = x_positions[i-1] - x_positions[i-2]
                curr_direction = x_positions[i] - x_positions[i-1]
                
                if prev_direction * curr_direction < 0:  # Sign change = direction change
                    direction_changes += 1
        
        # Calculate total movement
        total_movement = 0
        for i in range(1, len(nose_positions)):
            dx = (nose_positions[i][0] - nose_positions[i-1][0]) * 640
            dy = (nose_positions[i][1] - nose_positions[i-1][1]) * 480
            total_movement += np.sqrt(dx*dx + dy*dy)
        
        return {
            'horizontal_range': horizontal_range,
            'vertical_range': vertical_range,
            'movement_changes': direction_changes,
            'total_movement': total_movement,
            'smoothness': total_movement / len(nose_positions) if nose_positions else 0
        }

    def _detect_all_movements(self, nose_positions_with_time: List[Dict], adaptive_thresholds: Optional[Dict[str, float]] = None, frame_context: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Detect all significant movements throughout the video with comprehensive confidence scoring."""
        if len(nose_positions_with_time) < 10:
            return []
        
        # Use adaptive thresholds if provided, otherwise use base configuration
        if adaptive_thresholds is None:
            adaptive_thresholds = self.adaptive_calculator._get_base_thresholds()
        
        # Reset confidence scorer for new video analysis
        self.confidence_scorer.reset_temporal_state()
        
        movements = []
        enhanced_movements = []
        window_size = self.movement_config.window_size  # Configurable window size
        min_movement_threshold = adaptive_thresholds.get('significant_movement_threshold', self.movement_config.significant_movement_threshold)
        step_size = max(1, int(window_size * self.movement_config.step_size_ratio))  # Configurable step size
        
        # Get frame scaling factors
        frame_scale_x = adaptive_thresholds.get('frame_scale_x', 1.0)
        frame_scale_y = adaptive_thresholds.get('frame_scale_y', 1.0)
        frame_width = int(self.movement_config.base_frame_width * frame_scale_x)
        frame_height = int(self.movement_config.base_frame_height * frame_scale_y)
        
        for i in range(0, len(nose_positions_with_time) - window_size, step_size):
            window_positions = nose_positions_with_time[i:i + window_size]
            
            if len(window_positions) < 5:
                continue
            
            # Calculate movement using the new coordinate handling method
            start_pos = window_positions[0]
            end_pos = window_positions[-1]
            
            movement_data = self._calculate_movement_magnitude(start_pos, end_pos, frame_width, frame_height)
            
            # Validate movement magnitude using the validation config
            is_valid, validation_reason = self.validation_config.validate_movement_magnitude(
                movement_data['magnitude'], 
                (frame_width, frame_height), 
                adaptive_thresholds
            )
            
            if not is_valid:
                continue
            
            # Determine direction with proper coordinate system handling
            if movement_data['abs_dx'] > movement_data['abs_dy']:
                # Horizontal movement - FIXED: Correct directional mapping
                # MediaPipe coordinates: when user moves right, x increases (dx > 0)
                # When user moves left, x decreases (dx < 0)
                if movement_data['dx_pixels'] > 0:
                    direction = 'right'  # User moved right (x increased)
                else:
                    direction = 'left'   # User moved left (x decreased)
            else:
                # Vertical movement - MediaPipe y increases downward
                if movement_data['dy_pixels'] > 0:
                    direction = 'down'   # User moved down (y increased)
                else:
                    direction = 'up'     # User moved up (y decreased)
            
            # Prepare movement data for comprehensive confidence scoring
            movement_info = {
                'direction': direction,
                'magnitude': movement_data['magnitude'],
                'dx_pixels': movement_data['dx_pixels'],
                'dy_pixels': movement_data['dy_pixels'],
                'dx_norm': movement_data['dx_norm'],
                'dy_norm': movement_data['dy_norm'],
                'timestamp': end_pos.get('timestamp', time.time()),
                'start_position': (start_pos['x'], start_pos['y']),
                'end_position': (end_pos['x'], end_pos['y']),
                'frame_indices': (i, i + window_size)
            }
            
            # Calculate comprehensive confidence score
            enhanced_movement = self.confidence_scorer.calculate_comprehensive_confidence(
                movement_info, frame_context, movements
            )
            
            # Only add if confidence is high enough
            if enhanced_movement.confidence > self.movement_config.min_confidence_threshold:
                # Convert enhanced movement back to dictionary format for compatibility
                movement_dict = {
                    'direction': enhanced_movement.direction,
                    'confidence': enhanced_movement.confidence,
                    'start_time': start_pos['timestamp'],
                    'end_time': end_pos['timestamp'],
                    'magnitude': enhanced_movement.magnitude,
                    'start_index': i,
                    'end_index': i + window_size,
                    'dx_pixels': enhanced_movement.dx_pixels,
                    'dy_pixels': enhanced_movement.dy_pixels,
                    'dx_norm': enhanced_movement.dx_norm,
                    'dy_norm': enhanced_movement.dy_norm,
                    'raw_confidence': enhanced_movement.raw_confidence,
                    'temporal_confidence': enhanced_movement.temporal_confidence,
                    'consistency_score': enhanced_movement.consistency_score
                }
                
                movements.append(movement_dict)
                enhanced_movements.append(enhanced_movement)
        
        # Apply confidence-based filtering using the comprehensive scoring system
        filtered_enhanced_movements = self.confidence_scorer.filter_movements_by_confidence(enhanced_movements)
        
        # Convert filtered enhanced movements back to dictionary format
        filtered_movements = []
        for enhanced_mov in filtered_enhanced_movements:
            movement_dict = {
                'direction': enhanced_mov.direction,
                'confidence': enhanced_mov.confidence,
                'start_time': enhanced_mov.timestamp,
                'end_time': enhanced_mov.timestamp,
                'magnitude': enhanced_mov.magnitude,
                'start_index': enhanced_mov.frame_indices[0],
                'end_index': enhanced_mov.frame_indices[1],
                'dx_pixels': enhanced_mov.dx_pixels,
                'dy_pixels': enhanced_mov.dy_pixels,
                'dx_norm': enhanced_mov.dx_norm,
                'dy_norm': enhanced_mov.dy_norm,
                'raw_confidence': enhanced_mov.raw_confidence,
                'temporal_confidence': enhanced_mov.temporal_confidence,
                'consistency_score': enhanced_mov.consistency_score
            }
            filtered_movements.append(movement_dict)
        
        # Additional temporal filtering to remove movements that are too close together
        final_movements = []
        for i, movement in enumerate(filtered_movements):
            if i == 0:
                final_movements.append(movement)
                continue
            
            time_diff = movement['start_time'] - final_movements[-1]['end_time']
            direction_diff = movement['direction'] != final_movements[-1]['direction']
            
            # Add movement if it's either far enough in time OR in a different direction
            if time_diff > 0.3 or direction_diff:  # Reduced time threshold and added direction check
                final_movements.append(movement)
        
        logger.info(f"Movement detection summary: {len(movements)} raw movements, "
                   f"{len(filtered_movements)} after confidence filtering, "
                   f"{len(final_movements)} after temporal filtering")
        
        return final_movements

    def _detect_center_to_center_movements(self, nose_positions_with_time: List[Dict], adaptive_thresholds: Optional[Dict[str, float]] = None, frame_context: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Detect head movements based on center-to-center patterns.
        Records any movement that starts from center, moves to a direction, and returns to center.
        This removes time-based constraints and focuses on movement patterns.
        """
        if len(nose_positions_with_time) < 10:
            return []
        
        # Use adaptive thresholds if provided, otherwise use base configuration
        if adaptive_thresholds is None:
            adaptive_thresholds = self.adaptive_calculator._get_base_thresholds()
        
        # Reset confidence scorer for new video analysis
        self.confidence_scorer.reset_temporal_state()
        
        movements = []
        
        # Calculate center position (average of all positions)
        all_x = [pos['x'] for pos in nose_positions_with_time]
        all_y = [pos['y'] for pos in nose_positions_with_time]
        center_x = np.mean(all_x)
        center_y = np.mean(all_y)
        
        # Define center tolerance (how close to center is considered "center")
        center_tolerance = 0.05  # 5% of frame size
        
        # Get frame scaling factors
        frame_scale_x = adaptive_thresholds.get('frame_scale_x', 1.0)
        frame_scale_y = adaptive_thresholds.get('frame_scale_y', 1.0)
        frame_width = int(self.movement_config.base_frame_width * frame_scale_x)
        frame_height = int(self.movement_config.base_frame_height * frame_scale_y)
        
        # Convert center tolerance to normalized coordinates
        center_tolerance_x = center_tolerance / frame_scale_x
        center_tolerance_y = center_tolerance / frame_scale_y
        
        logger.info(f"Center position: ({center_x:.3f}, {center_y:.3f}), tolerance: ({center_tolerance_x:.3f}, {center_tolerance_y:.3f})")
        
        # Find center-to-center movement patterns
        i = 0
        while i < len(nose_positions_with_time):
            # Look for a position that's close to center (start of movement)
            if self._is_near_center(nose_positions_with_time[i], center_x, center_y, center_tolerance_x, center_tolerance_y):
                # Found potential start of movement, look for the pattern
                movement_pattern = self._find_center_to_center_pattern(
                    nose_positions_with_time, i, center_x, center_y, 
                    center_tolerance_x, center_tolerance_y, adaptive_thresholds, frame_context
                )
                
                if movement_pattern:
                    movements.append(movement_pattern)
                    # Skip to the end of this movement pattern
                    i = movement_pattern['end_index'] + 1
                else:
                    i += 1
            else:
                i += 1
        
        logger.info(f"Detected {len(movements)} center-to-center movements")
        return movements
    
    def _is_near_center(self, position: Dict, center_x: float, center_y: float, tolerance_x: float, tolerance_y: float) -> bool:
        """Check if a position is near the center."""
        dx = abs(position['x'] - center_x)
        dy = abs(position['y'] - center_y)
        return dx <= tolerance_x and dy <= tolerance_y
    
    def _find_center_to_center_pattern(self, positions: List[Dict], start_index: int, center_x: float, center_y: float, 
                                     tolerance_x: float, tolerance_y: float, adaptive_thresholds: Dict, 
                                     frame_context: Optional[Dict] = None) -> Optional[Dict]:
        """
        Find a complete center-to-center movement pattern starting from start_index.
        Pattern: Center -> Direction -> Center
        """
        if start_index >= len(positions) - 5:  # Need at least 5 positions for a pattern
            return None
        
        # Get frame scaling factors
        frame_scale_x = adaptive_thresholds.get('frame_scale_x', 1.0)
        frame_scale_y = adaptive_thresholds.get('frame_scale_y', 1.0)
        frame_width = int(self.movement_config.base_frame_width * frame_scale_x)
        frame_height = int(self.movement_config.base_frame_height * frame_scale_y)
        
        # Find the maximum movement point (furthest from center)
        max_movement_index = start_index
        max_distance = 0
        
        for i in range(start_index, len(positions)):
            pos = positions[i]
            distance = np.sqrt((pos['x'] - center_x)**2 + (pos['y'] - center_y)**2)
            
            if distance > max_distance:
                max_distance = distance
                max_movement_index = i
            
            # If we return to center, stop looking
            if self._is_near_center(pos, center_x, center_y, tolerance_x, tolerance_y):
                if i > start_index + 2:  # Need at least some movement
                    break
        
        # Check if we found a significant movement
        min_movement_threshold = adaptive_thresholds.get('min_movement_threshold', self.movement_config.min_movement_threshold)
        min_movement_norm = min_movement_threshold / frame_width  # Convert to normalized coordinates
        
        if max_distance < min_movement_norm:
            return None
        
        # Find the end of the pattern (return to center)
        end_index = max_movement_index
        for i in range(max_movement_index, len(positions)):
            if self._is_near_center(positions[i], center_x, center_y, tolerance_x, tolerance_y):
                end_index = i
                break
        
        # Calculate movement from start to max movement point
        start_pos = positions[start_index]
        max_pos = positions[max_movement_index]
        
        movement_data = self._calculate_movement_magnitude(start_pos, max_pos, frame_width, frame_height)
        
        # Determine direction
        if movement_data['abs_dx'] > movement_data['abs_dy']:
            # Horizontal movement
            if movement_data['dx_pixels'] > 0:
                direction = 'right'
            else:
                direction = 'left'
        else:
            # Vertical movement
            if movement_data['dy_pixels'] > 0:
                direction = 'down'
            else:
                direction = 'up'
        
        # Prepare movement data for confidence scoring
        movement_info = {
            'direction': direction,
            'magnitude': movement_data['magnitude'],
            'dx_pixels': movement_data['dx_pixels'],
            'dy_pixels': movement_data['dy_pixels'],
            'dx_norm': movement_data['dx_norm'],
            'dy_norm': movement_data['dy_norm'],
            'timestamp': max_pos.get('timestamp', time.time()),
            'start_position': (start_pos['x'], start_pos['y']),
            'end_position': (max_pos['x'], max_pos['y']),
            'frame_indices': (start_index, max_movement_index)
        }
        
        # Calculate confidence score
        enhanced_movement = self.confidence_scorer.calculate_comprehensive_confidence(
            movement_info, frame_context, []
        )
        
        # Only return if confidence is high enough
        if enhanced_movement.confidence > self.movement_config.min_confidence_threshold:
            return {
                'direction': enhanced_movement.direction,
                'confidence': enhanced_movement.confidence,
                'start_time': start_pos['timestamp'],
                'end_time': positions[end_index]['timestamp'],
                'magnitude': enhanced_movement.magnitude,
                'start_index': start_index,
                'end_index': end_index,
                'max_movement_index': max_movement_index,
                'dx_pixels': enhanced_movement.dx_pixels,
                'dy_pixels': enhanced_movement.dy_pixels,
                'dx_norm': enhanced_movement.dx_norm,
                'dy_norm': enhanced_movement.dy_norm,
                'raw_confidence': enhanced_movement.raw_confidence,
                'temporal_confidence': enhanced_movement.temporal_confidence,
                'consistency_score': enhanced_movement.consistency_score,
                'pattern_type': 'center_to_center'
            }
        
        return None




# Global detector instance
simple_liveness_detector = None

def initialize_simple_liveness_detector(movement_config: Optional[MovementThresholdConfig] = None) -> SimpleLivenessDetector:
    """Initialize global simple liveness detector with optional movement configuration."""
    global simple_liveness_detector
    simple_liveness_detector = SimpleLivenessDetector(movement_config)
    return simple_liveness_detector

def get_simple_liveness_detector() -> Optional[SimpleLivenessDetector]:
    """Get the global simple liveness detector instance."""
    global simple_liveness_detector
    return simple_liveness_detector 