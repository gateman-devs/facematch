"""
Simple Liveness Detection System
Randomly chooses between smile detection and head movement detection for mobile devices.
"""

import cv2
import numpy as np
import logging
import random
import time
import tempfile
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import mediapipe as mp
from .image_utils import get_image_processor
# Advanced head pose functionality removed for stability

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleLivenessDetector:
    """Simple liveness detection using smile and head movement."""
    
    def __init__(self):
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
        self.faces_dir = "/app/faces"
        os.makedirs(self.faces_dir, exist_ok=True)
        
        # Initialize image processor for URL/base64 handling
        self.image_processor = get_image_processor()
        
        logger.info("SimpleLivenessDetector initialized with Enhanced Anti-Spoofing + Face Capture + Face Comparison")
    
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
            
            # Perform anti-spoofing analysis
            anti_spoof_features = self.analyze_anti_spoofing_features(image)
            liveness_score = anti_spoof_features.get('overall_liveness_score', 0.0)
            
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
                    'anti_spoofing': anti_spoof_features,
                    'face_detected': False
                }
            
            # Basic liveness threshold (lower for static images)
            liveness_threshold = 0.25
            passed = liveness_score >= liveness_threshold and face_detected
            
            return {
                'success': True,
                'passed': passed,
                'liveness_score': liveness_score,
                'anti_spoofing': anti_spoof_features,
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
    
    def analyze_anti_spoofing_features(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Analyze frame for anti-spoofing features including texture analysis,
        lighting consistency, and motion patterns.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Texture Analysis using Local Binary Patterns
        texture_score = self._calculate_texture_variance(gray)
        
        # 2. Lighting Analysis
        lighting_score = self._analyze_lighting_distribution(gray)
        
        # 3. Edge Density Analysis  
        edge_score = self._calculate_edge_density(gray)
        
        # 4. Color Distribution Analysis
        color_score = self._analyze_color_distribution(frame)
        
        # 5. Motion Blur Analysis
        blur_score = self._calculate_motion_blur(gray)
        
        return {
            'texture_score': float(texture_score),
            'lighting_score': float(lighting_score),
            'edge_score': float(edge_score),
            'color_score': float(color_score),
            'blur_score': float(blur_score),
            'overall_liveness_score': float(np.mean([texture_score, lighting_score, edge_score, color_score, blur_score]))
        }
    
    def _calculate_texture_variance(self, gray_frame: np.ndarray) -> float:
        """Calculate texture variance to detect flat/printed surfaces."""
        # Use Laplacian variance to measure texture
        laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
        
        # Normalize to 0-1 range (higher = more texture = more likely real)
        normalized_score = min(laplacian_var / 1000.0, 1.0)
        return normalized_score
    
    def _analyze_lighting_distribution(self, gray_frame: np.ndarray) -> float:
        """Analyze lighting distribution to detect screen/print reflections."""
        # Calculate histogram
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        
        # Check for unnatural lighting peaks (screens often have specific brightness)
        hist_normalized = hist.flatten() / hist.sum()
        
        # Real faces have more distributed lighting
        entropy = -np.sum(hist_normalized * np.log(hist_normalized + 1e-7))
        
        # Normalize entropy score (higher entropy = more natural lighting)
        normalized_score = min(entropy / 8.0, 1.0)
        return normalized_score
    
    def _calculate_edge_density(self, gray_frame: np.ndarray) -> float:
        """Calculate edge density - real faces have natural edge patterns."""
        # Apply Canny edge detection
        edges = cv2.Canny(gray_frame, 50, 150)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / edges.size
        
        # Real faces typically have moderate edge density
        # Too high = noisy/artificial, too low = flat/blurred
        optimal_range = (0.02, 0.15)
        if optimal_range[0] <= edge_density <= optimal_range[1]:
            score = 1.0
        else:
            # Penalize values outside optimal range
            distance = min(abs(edge_density - optimal_range[0]), abs(edge_density - optimal_range[1]))
            score = max(0.0, 1.0 - distance * 10)
        
        return score
    
    def _analyze_color_distribution(self, frame: np.ndarray) -> float:
        """Analyze color distribution for natural skin tones."""
        # Convert to HSV for better skin tone analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define skin tone ranges in HSV
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])
        
        # Create mask for skin-like colors
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Calculate percentage of skin-like pixels
        skin_percentage = np.sum(skin_mask > 0) / skin_mask.size
        
        # Real faces should have reasonable skin tone distribution
        if 0.1 <= skin_percentage <= 0.7:
            score = 1.0
        else:
            score = max(0.0, 1.0 - abs(skin_percentage - 0.4) * 2)
        
        return score
    
    def _calculate_motion_blur(self, gray_frame: np.ndarray) -> float:
        """Calculate motion blur to detect video artifacts."""
        # Use gradient magnitude to detect blur
        grad_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Higher gradient magnitude = less blur = more likely real
        mean_gradient = np.mean(gradient_magnitude)
        
        # Normalize (real faces typically have gradients in this range)
        normalized_score = min(mean_gradient / 100.0, 1.0)
        return normalized_score
    
    def generate_challenge(self) -> Dict:
        """Generate a head movement challenge with anti-spoofing validation."""
        # Generate random head movement sequence - ALWAYS 4 directions
        directions = ['up', 'down', 'left', 'right']
        sequence_length = 4  # Always exactly 4 directions
        movement_sequence = [random.choice(directions) for _ in range(sequence_length)]
        
        duration = len(movement_sequence) * 3.5  # 3.5 seconds per direction (2s look + 1.5s center)
        instruction = f"Follow the directional prompts: {' → '.join(movement_sequence)}"
        
        return {
            'type': 'head_movement',
            'instruction': instruction,
            'duration': duration,
            'description': 'Move your head in the exact sequence shown',
            'movement_sequence': movement_sequence,
            'direction_duration': 3.5,
            'anti_spoofing': True  # Enable enhanced validation
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
        try:
            logger.info(f"Validating {challenge_type} challenge: {video_path}")
            
            # Open video
            cap = cv2.VideoCapture(video_path)
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
            
            if challenge_type == 'head_movement':
                validation_result = self._validate_head_movement_challenge(cap, fps, movement_sequence, session_id)
                
                # Add face comparison if reference image provided
                if reference_image and validation_result.get('success') and validation_result.get('face_image_path'):
                    face_comparison = self.compare_faces(reference_image, validation_result['face_image_path'])
                    validation_result['face_comparison'] = face_comparison
                    
                    # Update overall pass status to include face comparison
                    if validation_result.get('passed'):
                        validation_result['passed'] = face_comparison.get('match', False)
                        validation_result['face_match_required'] = True
                
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
            
            # Process every 3rd frame for performance
            if total_frames % 3 != 0:
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
        nose_positions_with_time = []
        anti_spoofing_scores = []
        total_frames = 0
        face_saved = False
        best_face_frame = None
        best_face_confidence = 0
        
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
            
            # Process every 3rd frame for performance
            if total_frames % 3 != 0:
                continue
            
            # Calculate timestamp
            timestamp = total_frames / fps
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
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
                        
                        # Perform anti-spoofing analysis
                        anti_spoof_features = self.analyze_anti_spoofing_features(frame)
                        anti_spoofing_scores.append(anti_spoof_features)
                        
                        # Save best quality face image for matching
                        face_confidence = anti_spoof_features['overall_liveness_score']
                        if face_confidence > best_face_confidence and session_id:
                            best_face_confidence = face_confidence
                            best_face_frame = frame.copy()
        
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
        
        # Calculate anti-spoofing metrics
        avg_anti_spoofing = {}
        if anti_spoofing_scores:
            for key in anti_spoofing_scores[0].keys():
                avg_anti_spoofing[key] = float(np.mean([score[key] for score in anti_spoofing_scores]))
        
        # Determine if anti-spoofing validation passed
        anti_spoof_passed = avg_anti_spoofing.get('overall_liveness_score', 0) >= 0.3
        
        if len(nose_positions_with_time) < 10:
            return {
                'success': True,
                'passed': False,
                'challenge_type': 'head_movement',
                'error': 'Insufficient face detection in video - could not track head movement reliably',
                'processed_frames': int(len(nose_positions_with_time)),
                'total_frames_processed': int(total_frames),
                'method': 'Enhanced Anti-Spoofing + Head Movement',
                'anti_spoofing': avg_anti_spoofing,
                'anti_spoof_passed': anti_spoof_passed,
                'face_saved': face_saved,
                'face_image_path': face_image_path
            }
        
        # If no sequence provided, fall back to basic movement validation
        if not movement_sequence:
            movement_analysis = self._analyze_basic_head_movement(nose_positions_with_time)
            movement_passed = (movement_analysis['horizontal_range'] > 50 and 
                             movement_analysis['movement_changes'] >= 2)
            
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
            
            return {
                'success': True,
                'passed': bool(overall_passed),
                'challenge_type': 'head_movement',
                'details': converted_analysis,
                'processed_frames': int(len(nose_positions_with_time)),
                'method': 'Enhanced Anti-Spoofing + Head Movement',
                'anti_spoofing': avg_anti_spoofing,
                'anti_spoof_passed': anti_spoof_passed,
                'movement_passed': movement_passed,
                'face_saved': face_saved,
                'face_image_path': face_image_path
            }
        
        # Validate specific directional sequence
        sequence_validation = self._validate_movement_sequence(nose_positions_with_time, movement_sequence)
        
        # Overall validation requires both sequence and anti-spoofing to pass
        sequence_passed = sequence_validation['passed']
        overall_passed = sequence_passed and anti_spoof_passed
        
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
        
        return {
            'success': True,
            'passed': bool(overall_passed),
            'challenge_type': 'head_movement',
            'expected_sequence': movement_sequence,
            'detected_sequence': converted_validation['detected_sequence'],
            'sequence_accuracy': float(converted_validation['accuracy']),
            'processed_frames': int(len(nose_positions_with_time)),
            'details': converted_validation,
            'method': 'Enhanced Anti-Spoofing + Head Movement',
            'anti_spoofing': avg_anti_spoofing,
            'anti_spoof_passed': anti_spoof_passed,
            'sequence_passed': sequence_passed,
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
        """Get normalized nose position."""
        h, w = frame_shape[:2]
        
        nose_tip = face_landmarks.landmark[self.head_landmarks['nose_tip']]
        return (nose_tip.x, nose_tip.y)  # Return normalized coordinates
    
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
        
        # Calculate total video duration
        total_duration = head_pose_data_with_time[-1]['timestamp'] - head_pose_data_with_time[0]['timestamp']
        direction_duration = total_duration / len(expected_sequence)
        
        # Each direction cycle has two phases: direction (2s) + look center (1.5s)
        step_duration = 2.0  # Analyze 2 seconds when user should be looking in direction
        
        detected_sequence = []
        segment_accuracies = []
        
        # Analyze each time segment (only the first half of each cycle when looking in direction)
        for i, expected_direction in enumerate(expected_sequence):
            start_time = head_pose_data_with_time[0]['timestamp'] + (i * direction_duration)
            end_time = start_time + step_duration  # Only analyze first half of the cycle
            
            # Get pose data for this time segment (when user should be looking in direction)
            segment_data = [
                data for data in head_pose_data_with_time 
                if start_time <= data['timestamp'] <= end_time
            ]
            
            if len(segment_data) < 3:
                detected_sequence.append('insufficient_data')
                segment_accuracies.append(0.0)
                continue
            
            # Detect movement direction in this segment using advanced pose analysis
            detected_direction, confidence = self._detect_advanced_movement_direction(segment_data, expected_direction)
            detected_sequence.append(detected_direction)
            
            # Calculate accuracy for this segment
            if detected_direction == expected_direction:
                segment_accuracies.append(float(confidence))
            else:
                # Partial credit for close directions or high confidence in wrong direction
                partial_credit = max(0.0, confidence * 0.3)  # 30% credit for confident wrong detection
                segment_accuracies.append(float(partial_credit))
        
        # Calculate overall accuracy
        overall_accuracy = float(np.mean(segment_accuracies)) if segment_accuracies else 0.0
        
        # Pass if accuracy is above threshold (70% for advanced method)
        passed = overall_accuracy >= 0.7 and len([acc for acc in segment_accuracies if acc > 0.5]) >= len(expected_sequence) * 0.6
        
        return {
            'passed': bool(passed),
            'accuracy': float(overall_accuracy),
            'detected_sequence': detected_sequence,
            'expected_sequence': expected_sequence,
            'segment_accuracies': [float(acc) for acc in segment_accuracies],
            'total_duration': float(total_duration),
            'direction_duration': float(direction_duration),
            'method': 'Advanced 6DoF Head Pose Validation'
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

    def _validate_movement_sequence(self, nose_positions_with_time: List[Dict], expected_sequence: List[str]) -> Dict:
        """Validate that head movement follows the expected directional sequence."""
        if not nose_positions_with_time or not expected_sequence:
            return {
                'passed': False,
                'accuracy': 0.0,
                'detected_sequence': [],
                'error': 'Insufficient data for sequence validation'
            }
        
        # Calculate total video duration
        total_duration = nose_positions_with_time[-1]['timestamp'] - nose_positions_with_time[0]['timestamp']
        direction_duration = total_duration / len(expected_sequence)
        
        # Each direction cycle now has two phases: direction (2s) + look center (1.5s)
        step_duration = 2.0  # Analyze 2 seconds when user should be looking in direction
        
        detected_sequence = []
        sequence_accuracies = []
        
        # Analyze each time segment (only the first half of each cycle)
        for i, expected_direction in enumerate(expected_sequence):
            start_time = nose_positions_with_time[0]['timestamp'] + (i * direction_duration)
            end_time = start_time + step_duration  # Only analyze first half of the cycle
            
            # Get nose positions for this time segment (when user should be looking in direction)
            segment_positions = [
                pos for pos in nose_positions_with_time 
                if start_time <= pos['timestamp'] <= end_time
            ]
            
            if len(segment_positions) < 3:
                detected_sequence.append('insufficient_data')
                sequence_accuracies.append(0.0)
                continue
            
            # Detect movement direction in this segment
            detected_direction, confidence = self._detect_movement_direction(segment_positions)
            detected_sequence.append(detected_direction)
            
            # Calculate accuracy for this segment
            if detected_direction == expected_direction:
                sequence_accuracies.append(confidence)
            else:
                sequence_accuracies.append(0.0)
        
        # Calculate overall accuracy
        overall_accuracy = float(np.mean(sequence_accuracies)) if sequence_accuracies else 0.0
        
        # Pass if accuracy is above threshold (lowered to 50% for better detection)
        passed = overall_accuracy >= 0.5 and len([acc for acc in sequence_accuracies if acc > 0]) >= len(expected_sequence) * 0.5
        
        return {
            'passed': bool(passed),
            'accuracy': float(overall_accuracy),
            'detected_sequence': detected_sequence,
            'expected_sequence': expected_sequence,
            'segment_accuracies': [float(acc) for acc in sequence_accuracies],
            'total_duration': float(total_duration),
            'direction_duration': float(direction_duration)
        }
    
    def _detect_movement_direction(self, segment_positions: List[Dict]) -> Tuple[str, float]:
        """Detect the primary movement direction in a time segment."""
        if len(segment_positions) < 3:
            return 'none', 0.0
        
        start_pos = segment_positions[0]
        end_pos = segment_positions[-1]
        
        # Calculate movement deltas
        dx = end_pos['x'] - start_pos['x']
        dy = end_pos['y'] - start_pos['y']
        
        # Convert to pixel coordinates for better thresholds
        dx_pixels = dx * 640
        dy_pixels = dy * 480
        
        # Determine primary direction based on largest movement
        abs_dx = abs(dx_pixels)
        abs_dy = abs(dy_pixels)
        
        # Minimum movement threshold (pixels)
        min_movement = 20
        
        if abs_dx < min_movement and abs_dy < min_movement:
            return 'none', 0.0
        
        # Determine direction
        if abs_dx > abs_dy:
            # Horizontal movement - flip for camera mirror effect
            # When user moves left, camera sees them move right (dx > 0), so we flip it
            if dx_pixels > 0:
                direction = 'left'  # User moved left (camera saw right movement)
            else:
                direction = 'right'  # User moved right (camera saw left movement)
            confidence = min(abs_dx / 100, 1.0)  # Scale confidence based on movement magnitude
        else:
            # Vertical movement
            if dy_pixels > 0:
                direction = 'down'
            else:
                direction = 'up'
            confidence = min(abs_dy / 100, 1.0)
        
        return str(direction), float(confidence)

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


# Global detector instance
simple_liveness_detector = None

def initialize_simple_liveness_detector() -> SimpleLivenessDetector:
    """Initialize global simple liveness detector."""
    global simple_liveness_detector
    simple_liveness_detector = SimpleLivenessDetector()
    return simple_liveness_detector

def get_simple_liveness_detector() -> Optional[SimpleLivenessDetector]:
    """Get the global simple liveness detector instance."""
    global simple_liveness_detector
    return simple_liveness_detector 