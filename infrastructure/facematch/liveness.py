"""
Liveness Detection Module using CRMNET
Provides anti-spoofing detection to identify real vs fake faces.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)

class LivenessDetector:
    """CRMNET-based liveness detector for anti-spoofing."""
    
    def __init__(self, model_path: str):
        """
        Initialize liveness detector.
        
        Args:
            model_path: Path to CRMNET ONNX model
        """
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_names = None
        self.input_shape = None
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize ONNX Runtime session for CRMNET model."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"CRMNET model not found at: {self.model_path}")
            
            # Configure ONNX Runtime session
            providers = ['CPUExecutionProvider']
            
            # Try to use GPU if available
            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
            
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Get model input/output information
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            self.input_shape = self.session.get_inputs()[0].shape
            
            logger.info(f"CRMNET liveness detector initialized successfully")
            logger.info(f"Input shape: {self.input_shape}, Input name: {self.input_name}")
            logger.info(f"Output names: {self.output_names}")
            
        except Exception as e:
            logger.error(f"Failed to initialize CRMNET model: {e}")
            raise RuntimeError(f"Liveness detector initialization failed: {e}")
    
    def preprocess_face(self, face_image: np.ndarray, target_size: Tuple[int, int] = (192, 192)) -> np.ndarray:
        """
        Preprocess face image for liveness detection.
        
        Args:
            face_image: Face region image
            target_size: Target size for model input
            
        Returns:
            Preprocessed image tensor
        """
        try:
            # Resize to target size
            if face_image.shape[:2] != target_size:
                face_image = cv2.resize(face_image, target_size, interpolation=cv2.INTER_LINEAR)
            
            # Ensure RGB format
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                # Convert BGR to RGB if needed
                if face_image.max() <= 1.0:
                    face_image = (face_image * 255).astype(np.uint8)
                # Assume the image is already in correct format
            
            # Normalize to [0, 1]
            if face_image.dtype == np.uint8:
                face_image = face_image.astype(np.float32) / 255.0
            
            # Normalize to [-1, 1] (common for many models)
            face_image = (face_image - 0.5) * 2.0
            
            # Add batch dimension and reorder to NCHW if needed
            if len(self.input_shape) == 4 and self.input_shape[1] == 3:  # NCHW format
                face_image = np.transpose(face_image, (2, 0, 1))  # HWC to CHW
                face_image = np.expand_dims(face_image, axis=0)  # Add batch dimension
            else:  # NHWC format
                face_image = np.expand_dims(face_image, axis=0)  # Add batch dimension
            
            return face_image.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Face preprocessing failed: {e}")
            raise ValueError(f"Face preprocessing error: {e}")
    
    def detect_liveness(self, face_image: np.ndarray) -> Dict:
        """
        Detect liveness/anti-spoofing for a face image.
        
        Args:
            face_image: Face region image
            
        Returns:
            Liveness detection results
        """
        start_time = time.time()
        
        try:
            # Preprocess face image
            input_tensor = self.preprocess_face(face_image)
            
            # Run inference
            inputs = {self.input_name: input_tensor}
            outputs = self.session.run(self.output_names, inputs)
            
            inference_time = time.time() - start_time
            
            # Process outputs (assuming binary classification: real vs fake)
            # Output format may vary depending on the specific CRMNET model
            if len(outputs) == 1:
                # Single output (probability or logits)
                raw_output = outputs[0]
                if raw_output.shape[-1] == 2:  # Two classes: [fake, real]
                    probabilities = self._softmax(raw_output[0])
                    fake_prob = float(probabilities[0])
                    real_prob = float(probabilities[1])
                elif raw_output.shape[-1] == 1:  # Single value (real probability)
                    real_prob = float(raw_output[0][0])
                    fake_prob = 1.0 - real_prob
                else:
                    # Unknown format, try to extract meaningful values
                    real_prob = float(raw_output[0].max())
                    fake_prob = 1.0 - real_prob
            else:
                # Multiple outputs, use the first one
                raw_output = outputs[0]
                real_prob = float(raw_output[0].max())
                fake_prob = 1.0 - real_prob
            
            # Calculate liveness score (real probability)
            liveness_score = real_prob
            
            # Determine if live based on threshold
            is_live = liveness_score > 0.5
            confidence = max(real_prob, fake_prob)
            
            result = {
                'success': True,
                'is_live': is_live,
                'liveness_score': liveness_score,
                'confidence': confidence,
                'real_probability': real_prob,
                'fake_probability': fake_prob,
                'inference_time': inference_time,
                'quality_metrics': {
                    'confidence_threshold': 0.5,
                    'decision_margin': abs(real_prob - fake_prob)
                }
            }
            
            logger.info(f"Liveness detection completed: {is_live} (score: {liveness_score:.3f}) in {inference_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Liveness detection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'is_live': False,
                'liveness_score': 0.0,
                'confidence': 0.0,
                'inference_time': time.time() - start_time
            }
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Apply softmax to logits."""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
    
    def detect_liveness_with_quality_check(self, face_image: np.ndarray, min_face_size: int = 64) -> Dict:
        """
        Detect liveness with additional quality checks.
        
        Args:
            face_image: Face region image
            min_face_size: Minimum face size for reliable detection
            
        Returns:
            Enhanced liveness detection results
        """
        # Check face image quality
        height, width = face_image.shape[:2]
        
        if height < min_face_size or width < min_face_size:
            return {
                'success': False,
                'error': f"Face too small for liveness detection: {width}x{height}. Minimum size: {min_face_size}x{min_face_size}",
                'is_live': False,
                'liveness_score': 0.0,
                'confidence': 0.0
            }
        
        # Check image brightness and contrast
        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY) if len(face_image.shape) == 3 else face_image
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        quality_warnings = []
        
        if brightness < 40:
            quality_warnings.append("Face image too dark")
        elif brightness > 200:
            quality_warnings.append("Face image too bright")
        
        if contrast < 20:
            quality_warnings.append("Face image has low contrast")
        
        # Perform liveness detection
        result = self.detect_liveness(face_image)
        
        # Add quality information
        if result['success']:
            result['quality_warnings'] = quality_warnings
            result['quality_metrics'].update({
                'face_size': (width, height),
                'brightness': brightness,
                'contrast': contrast
            })
        
        return result
    
    def batch_detect_liveness(self, face_images: List[np.ndarray]) -> List[Dict]:
        """
        Detect liveness for multiple face images.
        
        Args:
            face_images: List of face region images
            
        Returns:
            List of liveness detection results
        """
        results = []
        
        for i, face_image in enumerate(face_images):
            try:
                result = self.detect_liveness_with_quality_check(face_image)
                result['image_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Liveness detection failed for image {i}: {e}")
                results.append({
                    'success': False,
                    'error': str(e),
                    'image_index': i,
                    'is_live': False,
                    'liveness_score': 0.0,
                    'confidence': 0.0
                })
        
        return results

def validate_liveness_model(model_path: str) -> Dict:
    """
    Validate CRMNET model file.
    
    Args:
        model_path: Path to ONNX model file
        
    Returns:
        Validation results
    """
    try:
        if not os.path.exists(model_path):
            return {
                'valid': False,
                'error': f"Model file not found: {model_path}"
            }
        
        # Check file size (should be > 1MB for a meaningful model)
        file_size = os.path.getsize(model_path)
        if file_size < 1024 * 1024:  # 1MB
            return {
                'valid': False,
                'error': f"Model file too small: {file_size} bytes"
            }
        
        # Try to load the model
        try:
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            input_shape = session.get_inputs()[0].shape
            output_names = [output.name for output in session.get_outputs()]
            
            return {
                'valid': True,
                'file_size': file_size,
                'input_shape': input_shape,
                'output_names': output_names
            }
        except Exception as e:
            return {
                'valid': False,
                'error': f"Failed to load model: {e}"
            }
            
    except Exception as e:
        return {
            'valid': False,
            'error': f"Model validation error: {e}"
        } 