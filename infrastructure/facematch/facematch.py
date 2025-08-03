"""
Face Recognition and Matching Module using ArcFace
Provides face embedding extraction and similarity comparison for face recognition.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Configure logging
logger = logging.getLogger(__name__)

class FaceRecognizer:
    """ArcFace-based face recognizer for embedding extraction and matching."""
    
    def __init__(self, model_path: str):
        """
        Initialize face recognizer.
        
        Args:
            model_path: Path to ArcFace ONNX model
        """
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_names = None
        self.input_shape = None
        self.embedding_size = None
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize ONNX Runtime session for ArcFace model."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"ArcFace model not found at: {self.model_path}")
            
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
            self.embedding_size = self.session.get_outputs()[0].shape[-1]
            
            logger.info(f"ArcFace face recognizer initialized successfully")
            logger.info(f"Input shape: {self.input_shape}, Input name: {self.input_name}")
            logger.info(f"Output names: {self.output_names}, Embedding size: {self.embedding_size}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ArcFace model: {e}")
            raise RuntimeError(f"Face recognizer initialization failed: {e}")
    
    def preprocess_face(self, face_image: np.ndarray, target_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
        """
        Preprocess face image for ArcFace model.
        
        Args:
            face_image: Face region image
            target_size: Target size for model input (typically 112x112 for ArcFace)
            
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
            
            # Normalize to [0, 1]
            if face_image.dtype == np.uint8:
                face_image = face_image.astype(np.float32) / 255.0
            
            # Apply standard normalization for ArcFace
            # Mean and std values commonly used for face recognition
            mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            face_image = (face_image - mean) / std
            
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
    
    def extract_embedding(self, face_image: np.ndarray) -> Dict:
        """
        Extract face embedding from a face image.
        
        Args:
            face_image: Face region image
            
        Returns:
            Face embedding extraction results
        """
        start_time = time.time()
        
        try:
            # Preprocess face image
            input_tensor = self.preprocess_face(face_image)
            
            # Run inference
            inputs = {self.input_name: input_tensor}
            outputs = self.session.run(self.output_names, inputs)
            
            inference_time = time.time() - start_time
            
            # Extract embedding (first output)
            raw_embedding = outputs[0][0]  # Remove batch dimension
            
            # Normalize embedding (L2 normalization is common for face embeddings)
            embedding = normalize(raw_embedding.reshape(1, -1), norm='l2')[0]
            
            # Calculate embedding quality metrics
            embedding_norm = np.linalg.norm(raw_embedding)
            embedding_mean = np.mean(embedding)
            embedding_std = np.std(embedding)
            
            result = {
                'success': True,
                'embedding': embedding.astype(np.float32),
                'raw_embedding': raw_embedding.astype(np.float32),
                'embedding_size': len(embedding),
                'inference_time': inference_time,
                'quality_metrics': {
                    'embedding_norm': float(embedding_norm),
                    'embedding_mean': float(embedding_mean),
                    'embedding_std': float(embedding_std)
                }
            }
            
            logger.info(f"Embedding extraction completed: {len(embedding)}D vector in {inference_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'embedding': None,
                'embedding_size': 0,
                'inference_time': time.time() - start_time
            }
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> Dict:
        """
        Calculate similarity between two face embeddings.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Similarity calculation results
        """
        try:
            # Ensure embeddings are normalized
            emb1_normalized = normalize(embedding1.reshape(1, -1), norm='l2')[0]
            emb2_normalized = normalize(embedding2.reshape(1, -1), norm='l2')[0]
            
            # Calculate cosine similarity
            cosine_sim = np.dot(emb1_normalized, emb2_normalized)
            
            # Calculate Euclidean distance
            euclidean_dist = np.linalg.norm(emb1_normalized - emb2_normalized)
            
            # Calculate Manhattan distance
            manhattan_dist = np.sum(np.abs(emb1_normalized - emb2_normalized))
            
            # Convert cosine similarity to cosine distance
            cosine_dist = 1.0 - cosine_sim
            
            result = {
                'cosine_similarity': float(cosine_sim),
                'cosine_distance': float(cosine_dist),
                'euclidean_distance': float(euclidean_dist),
                'manhattan_distance': float(manhattan_dist),
                'similarity_score': float(cosine_sim)  # Use cosine similarity as primary score
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return {
                'error': str(e),
                'cosine_similarity': 0.0,
                'cosine_distance': 1.0,
                'euclidean_distance': float('inf'),
                'manhattan_distance': float('inf'),
                'similarity_score': 0.0
            }
    
    def compare_faces(self, face1: np.ndarray, face2: np.ndarray, threshold: float = 0.6) -> Dict:
        """
        Compare two face images and determine if they match.
        
        Args:
            face1: First face image
            face2: Second face image
            threshold: Similarity threshold for match decision
            
        Returns:
            Face comparison results
        """
        start_time = time.time()
        
        try:
            # Extract embeddings for both faces
            emb1_result = self.extract_embedding(face1)
            emb2_result = self.extract_embedding(face2)
            
            if not emb1_result['success']:
                return {
                    'success': False,
                    'error': f"Failed to extract embedding from first face: {emb1_result.get('error', 'Unknown error')}",
                    'match': False,
                    'similarity_score': 0.0,
                    'total_time': time.time() - start_time
                }
            
            if not emb2_result['success']:
                return {
                    'success': False,
                    'error': f"Failed to extract embedding from second face: {emb2_result.get('error', 'Unknown error')}",
                    'match': False,
                    'similarity_score': 0.0,
                    'total_time': time.time() - start_time
                }
            
            # Calculate similarity
            similarity_result = self.calculate_similarity(
                emb1_result['embedding'], 
                emb2_result['embedding']
            )
            
            if 'error' in similarity_result:
                return {
                    'success': False,
                    'error': f"Similarity calculation failed: {similarity_result['error']}",
                    'match': False,
                    'similarity_score': 0.0,
                    'total_time': time.time() - start_time
                }
            
            # Determine match based on threshold
            similarity_score = similarity_result['similarity_score']
            is_match = similarity_score >= threshold
            confidence = abs(similarity_score - threshold) + threshold
            
            total_time = time.time() - start_time
            
            result = {
                'success': True,
                'match': is_match,
                'similarity_score': similarity_score,
                'confidence': min(confidence, 1.0),
                'threshold': threshold,
                'distance_metrics': {
                    'cosine_similarity': similarity_result['cosine_similarity'],
                    'cosine_distance': similarity_result['cosine_distance'],
                    'euclidean_distance': similarity_result['euclidean_distance'],
                    'manhattan_distance': similarity_result['manhattan_distance']
                },
                'embedding_quality': {
                    'face1_quality': emb1_result['quality_metrics'],
                    'face2_quality': emb2_result['quality_metrics']
                },
                'timing': {
                    'face1_embedding_time': emb1_result['inference_time'],
                    'face2_embedding_time': emb2_result['inference_time'],
                    'total_time': total_time
                }
            }
            
            logger.info(f"Face comparison completed: {'MATCH' if is_match else 'NO MATCH'} "
                       f"(similarity: {similarity_score:.3f}, threshold: {threshold}) in {total_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Face comparison failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'match': False,
                'similarity_score': 0.0,
                'total_time': time.time() - start_time
            }
    
    def batch_extract_embeddings(self, face_images: List[np.ndarray]) -> List[Dict]:
        """
        Extract embeddings for multiple face images.
        
        Args:
            face_images: List of face images
            
        Returns:
            List of embedding extraction results
        """
        results = []
        
        for i, face_image in enumerate(face_images):
            try:
                result = self.extract_embedding(face_image)
                result['image_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Embedding extraction failed for image {i}: {e}")
                results.append({
                    'success': False,
                    'error': str(e),
                    'image_index': i,
                    'embedding': None,
                    'embedding_size': 0
                })
        
        return results

def validate_arcface_model(model_path: str) -> Dict:
    """
    Validate ArcFace model file.
    
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
        
        # Check file size (ArcFace models are typically 10MB+)
        file_size = os.path.getsize(model_path)
        if file_size < 5 * 1024 * 1024:  # 5MB
            return {
                'valid': False,
                'error': f"Model file too small: {file_size} bytes"
            }
        
        # Try to load the model
        try:
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            input_shape = session.get_inputs()[0].shape
            output_shape = session.get_outputs()[0].shape
            
            # Validate expected input/output shapes for face recognition
            if len(input_shape) != 4:
                return {
                    'valid': False,
                    'error': f"Invalid input shape: {input_shape}. Expected 4D tensor."
                }
            
            if len(output_shape) != 2:
                return {
                    'valid': False,
                    'error': f"Invalid output shape: {output_shape}. Expected 2D tensor (batch, embedding_size)."
                }
            
            embedding_size = output_shape[-1]
            if embedding_size < 128:  # Most face recognition models have 128+ dimensional embeddings
                return {
                    'valid': False,
                    'error': f"Embedding size too small: {embedding_size}. Expected at least 128 dimensions."
                }
            
            return {
                'valid': True,
                'file_size': file_size,
                'input_shape': input_shape,
                'output_shape': output_shape,
                'embedding_size': embedding_size
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

def calculate_match_confidence(similarity_score: float, threshold: float) -> float:
    """
    Calculate match confidence based on similarity score and threshold.
    
    Args:
        similarity_score: Computed similarity score
        threshold: Decision threshold
        
    Returns:
        Confidence value between 0 and 1
    """
    if similarity_score >= threshold:
        # For matches, confidence increases with distance from threshold
        return min(1.0, threshold + (similarity_score - threshold) * 2)
    else:
        # For non-matches, confidence increases with distance from threshold
        return min(1.0, (threshold - similarity_score) * 2)

def adaptive_threshold(embedding_quality1: Dict, embedding_quality2: Dict, base_threshold: float = 0.6) -> float:
    """
    Calculate adaptive threshold based on embedding quality.
    
    Args:
        embedding_quality1: Quality metrics for first embedding
        embedding_quality2: Quality metrics for second embedding
        base_threshold: Base threshold value
        
    Returns:
        Adjusted threshold
    """
    # Calculate quality score based on embedding statistics
    std1 = embedding_quality1.get('embedding_std', 0.5)
    std2 = embedding_quality2.get('embedding_std', 0.5)
    
    # Lower standard deviation generally indicates more confident embeddings
    avg_std = (std1 + std2) / 2
    
    # Adjust threshold based on quality (higher std = higher threshold needed)
    if avg_std > 0.3:
        adjustment = min(0.1, (avg_std - 0.3) * 0.5)
        return min(0.8, base_threshold + adjustment)
    else:
        adjustment = max(-0.05, (0.3 - avg_std) * 0.2)
        return max(0.4, base_threshold + adjustment) 