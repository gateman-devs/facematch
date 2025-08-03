"""
Face Detection Module using MTCNN
Provides single face detection with robust error handling and quality metrics.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2
from mtcnn import MTCNN
import tensorflow as tf
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDetector:
    """MTCNN-based face detector optimized for single face detection."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize face detector.
        
        Args:
            model_path: Path to MTCNN model (optional, uses default if None)
        """
        self.detector = None
        self.model_path = model_path
        self._initialize_detector()
    
    def _initialize_detector(self) -> None:
        """Initialize MTCNN detector with error handling."""
        try:
            # Set TensorFlow to use CPU for MTCNN (more stable)
            tf.config.set_visible_devices([], 'GPU')
            # Initialize MTCNN with correct parameters
            self.detector = MTCNN(device='CPU:0')
            logger.info("MTCNN face detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MTCNN detector: {e}")
            raise RuntimeError(f"Face detector initialization failed: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for face detection.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR and convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # Convert RGBA to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Ensure uint8 format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        return image
    
    def detect_faces(self, image: np.ndarray) -> Dict:
        """
        Detect faces in the input image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing detection results
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Detect faces
            detections = self.detector.detect_faces(processed_image)
            detection_time = time.time() - start_time
            
            # Extract face information
            face_count = len(detections)
            faces = []
            
            for detection in detections:
                face_info = {
                    'bbox': detection['box'],  # [x, y, width, height]
                    'confidence': detection['confidence'],
                    'keypoints': detection['keypoints']
                }
                faces.append(face_info)
            
            result = {
                'success': True,
                'face_count': face_count,
                'faces': faces,
                'image_shape': processed_image.shape,
                'detection_time': detection_time,
                'quality_metrics': {
                    'avg_confidence': np.mean([f['confidence'] for f in faces]) if faces else 0.0,
                    'max_confidence': max([f['confidence'] for f in faces]) if faces else 0.0
                }
            }
            
            logger.info(f"Face detection completed: {face_count} faces found in {detection_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'face_count': 0,
                'faces': [],
                'detection_time': time.time() - start_time
            }
    
    def validate_single_face(self, detection_result: Dict) -> Dict:
        """
        Validate that exactly one face is detected.
        
        Args:
            detection_result: Result from detect_faces
            
        Returns:
            Validation result with single face info or error
        """
        if not detection_result['success']:
            return {
                'valid': False,
                'error': f"Detection failed: {detection_result.get('error', 'Unknown error')}",
                'face_count': 0
            }
        
        face_count = detection_result['face_count']
        
        if face_count == 0:
            return {
                'valid': False,
                'error': "No face detected in image",
                'face_count': 0
            }
        elif face_count > 1:
            return {
                'valid': False,
                'error': f"Multiple faces detected ({face_count}). Only single face images are allowed.",
                'face_count': face_count
            }
        else:
            # Exactly one face found
            face = detection_result['faces'][0]
            return {
                'valid': True,
                'face': face,
                'face_count': 1,
                'confidence': face['confidence'],
                'bbox': face['bbox'],
                'keypoints': face['keypoints']
            }
    
    def extract_face_region(self, image: np.ndarray, bbox: List[int], padding: float = 0.3) -> np.ndarray:
        """
        Extract face region from image with padding.
        
        Args:
            image: Input image
            bbox: Bounding box [x, y, width, height]
            padding: Padding factor (0.3 = 30% padding)
            
        Returns:
            Extracted face region
        """
        x, y, w, h = bbox
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        # Calculate padded coordinates
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)
        
        # Extract face region
        face_region = image[y1:y2, x1:x2]
        
        return face_region
    
    def align_face(self, image: np.ndarray, keypoints: Dict) -> np.ndarray:
        """
        Align face based on eye positions.
        
        Args:
            image: Input image
            keypoints: Facial keypoints from MTCNN
            
        Returns:
            Aligned face image
        """
        try:
            # Get eye coordinates
            left_eye = keypoints['left_eye']
            right_eye = keypoints['right_eye']
            
            # Calculate angle between eyes
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Calculate center point between eyes
            center_x = (left_eye[0] + right_eye[0]) // 2
            center_y = (left_eye[1] + right_eye[1]) // 2
            center = (center_x, center_y)
            
            # Create rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply rotation
            aligned_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
            
            return aligned_image
            
        except Exception as e:
            logger.warning(f"Face alignment failed: {e}. Returning original image.")
            return image

def load_image_from_array(image_array: np.ndarray) -> np.ndarray:
    """
    Load and validate image from numpy array.
    
    Args:
        image_array: Image as numpy array
        
    Returns:
        Processed image array
    """
    if image_array is None:
        raise ValueError("Image array is None")
    
    if len(image_array.shape) not in [2, 3]:
        raise ValueError(f"Invalid image shape: {image_array.shape}")
    
    return image_array

def validate_image_quality(image: np.ndarray) -> Dict:
    """
    Validate image quality for face detection.
    
    Args:
        image: Input image
        
    Returns:
        Quality validation results
    """
    height, width = image.shape[:2]
    
    # Minimum resolution check
    min_size = 224
    if height < min_size or width < min_size:
        return {
            'valid': False,
            'error': f"Image too small: {width}x{height}. Minimum size: {min_size}x{min_size}",
            'resolution': (width, height)
        }
    
    # Maximum resolution check (to prevent memory issues)
    max_size = 2048
    if height > max_size or width > max_size:
        return {
            'valid': False,
            'error': f"Image too large: {width}x{height}. Maximum size: {max_size}x{max_size}",
            'resolution': (width, height)
        }
    
    # Brightness check
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    brightness = np.mean(gray)
    
    if brightness < 30:
        return {
            'valid': False,
            'error': f"Image too dark (brightness: {brightness:.1f})",
            'brightness': brightness
        }
    
    if brightness > 225:
        return {
            'valid': False,
            'error': f"Image too bright (brightness: {brightness:.1f})",
            'brightness': brightness
        }
    
    return {
        'valid': True,
        'resolution': (width, height),
        'brightness': brightness
    } 