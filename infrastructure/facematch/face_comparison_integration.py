"""
Face Comparison Integration Module
Provides seamless integration between face comparison and liveness detection with comprehensive error handling and logging.
Optimized for concurrent processing and maximum performance.
"""

import logging
import time
import asyncio
from typing import Dict, Optional, Any, Union, Tuple, List
from dataclasses import dataclass
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class FaceComparisonResult:
    """Standardized face comparison result with comprehensive details."""
    success: bool
    match: bool
    similarity_score: float
    threshold: float
    confidence: str
    processing_time: float
    error: Optional[str] = None
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    quality_metrics: Optional[Dict[str, Any]] = None

@dataclass
class IntegratedLivenessResult:
    """Combined liveness and face comparison result."""
    liveness_success: bool
    liveness_passed: bool
    liveness_score: float
    face_comparison_success: bool
    face_comparison_match: bool
    face_comparison_score: float
    overall_passed: bool
    processing_time: float
    error: Optional[str] = None
    error_code: Optional[str] = None
    liveness_details: Optional[Dict[str, Any]] = None
    face_comparison_details: Optional[Dict[str, Any]] = None

class FaceComparisonIntegrator:
    """
    Handles seamless integration between face comparison and liveness detection.
    Provides comprehensive error handling and logging for all face comparison operations.
    Optimized for concurrent processing and maximum performance.
    """
    
    def __init__(self, face_recognizer=None, unified_detector=None):
        """
        Initialize the face comparison integrator.
        
        Args:
            face_recognizer: ArcFace-based face recognizer instance
            unified_detector: Unified liveness detector instance
        """
        self.face_recognizer = face_recognizer
        self.unified_detector = unified_detector
        self.default_threshold = 0.6
        self.max_processing_time = 30.0  # Maximum allowed processing time in seconds
        
        # Performance optimization settings
        self.max_workers = 4  # Number of concurrent workers for image processing
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self._lock = threading.Lock()  # Thread safety for shared resources
        
        # Image cache for repeated comparisons (LRU cache)
        self._image_cache = {}
        self._cache_max_size = 100
        
        logger.info("Face comparison integrator initialized with concurrent processing optimizations")
    
    def compare_faces_with_validation(
        self,
        image1: Union[str, np.ndarray],
        image2: Union[str, np.ndarray],
        threshold: Optional[float] = None,
        enable_quality_check: bool = True,
        session_id: Optional[str] = None
    ) -> FaceComparisonResult:
        """
        Perform face comparison with comprehensive validation and error handling.
        Optimized for concurrent processing and maximum performance.
        
        Args:
            image1: First image (path, URL, base64, or numpy array)
            image2: Second image (path, URL, base64, or numpy array)
            threshold: Similarity threshold (uses default if None)
            enable_quality_check: Whether to perform image quality validation
            session_id: Session ID for logging context
            
        Returns:
            Comprehensive face comparison result
        """
        start_time = time.time()
        threshold = threshold or self.default_threshold
        
        # Log face comparison initiation
        self._log_face_comparison_start(session_id, threshold, enable_quality_check)
        
        try:
            # Validate inputs
            validation_result = self._validate_comparison_inputs(image1, image2, threshold)
            if not validation_result['valid']:
                return self._create_error_result(
                    error=validation_result['error'],
                    error_code="INPUT_VALIDATION_ERROR",
                    threshold=threshold,
                    processing_time=time.time() - start_time
                )
            
            # Check if face recognizer is available
            if not self.face_recognizer:
                return self._create_error_result(
                    error="Face recognition service not available",
                    error_code="SERVICE_UNAVAILABLE",
                    threshold=threshold,
                    processing_time=time.time() - start_time
                )
            
            # Concurrent image loading and preprocessing
            image_load_start = time.time()
            face1_array, face2_array = self._concurrent_load_and_preprocess_images(image1, image2)
            image_load_time = time.time() - image_load_start
            
            if face1_array is None:
                return self._create_error_result(
                    error="Failed to convert first image to numpy array",
                    error_code="IMAGE_CONVERSION_ERROR",
                    threshold=threshold,
                    processing_time=time.time() - start_time
                )
            
            if face2_array is None:
                return self._create_error_result(
                    error="Failed to convert second image to numpy array",
                    error_code="IMAGE_CONVERSION_ERROR",
                    threshold=threshold,
                    processing_time=time.time() - start_time
                )
            
            # Concurrent embedding extraction
            embedding_start = time.time()
            emb1_result, emb2_result = self._concurrent_extract_embeddings(face1_array, face2_array)
            embedding_time = time.time() - embedding_start
            
            # Check embedding extraction results
            if not emb1_result['success']:
                return self._create_error_result(
                    error=f"Failed to extract embedding from first face: {emb1_result.get('error', 'Unknown error')}",
                    error_code="EMBEDDING_EXTRACTION_ERROR",
                    threshold=threshold,
                    processing_time=time.time() - start_time
                )
            
            if not emb2_result['success']:
                return self._create_error_result(
                    error=f"Failed to extract embedding from second face: {emb2_result.get('error', 'Unknown error')}",
                    error_code="EMBEDDING_EXTRACTION_ERROR",
                    threshold=threshold,
                    processing_time=time.time() - start_time
                )
            
            # Calculate similarity (this is fast, no need for concurrency)
            similarity_start = time.time()
            similarity_result = self.face_recognizer.calculate_similarity(
                emb1_result['embedding'], 
                emb2_result['embedding']
            )
            similarity_time = time.time() - similarity_start
            
            if 'error' in similarity_result:
                return self._create_error_result(
                    error=f"Similarity calculation failed: {similarity_result['error']}",
                    error_code="SIMILARITY_CALCULATION_ERROR",
                    threshold=threshold,
                    processing_time=time.time() - start_time
                )
            
            # Determine match based on threshold
            similarity_score = similarity_result['similarity_score']
            is_match = similarity_score >= threshold
            confidence = abs(similarity_score - threshold) + threshold
            
            processing_time = time.time() - start_time
            
            # Check for timeout
            if processing_time > self.max_processing_time:
                self._log_face_comparison_timeout(session_id, processing_time)
                return self._create_error_result(
                    error=f"Face comparison timed out after {processing_time:.2f}s",
                    error_code="PROCESSING_TIMEOUT",
                    threshold=threshold,
                    processing_time=processing_time
                )
            
            # Create successful result with detailed timing
            result = FaceComparisonResult(
                success=True,
                match=is_match,
                similarity_score=similarity_score,
                threshold=threshold,
                confidence=self._calculate_confidence_level(similarity_score, threshold),
                processing_time=processing_time,
                details={
                    'distance_metrics': similarity_result,
                    'embedding_quality': {
                        'face1_quality': emb1_result['quality_metrics'],
                        'face2_quality': emb2_result['quality_metrics']
                    },
                    'timing': {
                        'image_load_time': image_load_time,
                        'embedding_time': embedding_time,
                        'similarity_time': similarity_time,
                        'face1_embedding_time': emb1_result['inference_time'],
                        'face2_embedding_time': emb2_result['inference_time'],
                        'total_time': processing_time
                    }
                },
                quality_metrics=self._extract_quality_metrics({
                    'embedding_quality': {
                        'face1_quality': emb1_result['quality_metrics'],
                        'face2_quality': emb2_result['quality_metrics']
                    },
                    'timing': {
                        'face1_embedding_time': emb1_result['inference_time'],
                        'face2_embedding_time': emb2_result['inference_time']
                    }
                })
            )
            
            # Log successful comparison with performance metrics
            self._log_face_comparison_success(session_id, result)
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Unexpected error in face comparison: {str(e)}"
            self._log_face_comparison_error(session_id, error_msg, processing_time)
            
            return self._create_error_result(
                error=error_msg,
                error_code="UNEXPECTED_ERROR",
                threshold=threshold,
                processing_time=processing_time
            )
    
    def _concurrent_load_and_preprocess_images(
        self, 
        image1: Union[str, np.ndarray], 
        image2: Union[str, np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load and preprocess both images concurrently for maximum performance.
        
        Args:
            image1: First image input
            image2: Second image input
            
        Returns:
            Tuple of preprocessed numpy arrays
        """
        try:
            # Submit both image processing tasks concurrently
            future1 = self.thread_pool.submit(self._convert_to_numpy_array, image1)
            future2 = self.thread_pool.submit(self._convert_to_numpy_array, image2)
            
            # Wait for both tasks to complete
            face1_array = future1.result(timeout=10.0)  # 10 second timeout per image
            face2_array = future2.result(timeout=10.0)
            
            return face1_array, face2_array
            
        except Exception as e:
            logger.error(f"Concurrent image loading failed: {e}")
            return None, None
    
    def _concurrent_extract_embeddings(
        self, 
        face1_array: np.ndarray, 
        face2_array: np.ndarray
    ) -> Tuple[Dict, Dict]:
        """
        Extract embeddings for both faces concurrently.
        
        Args:
            face1_array: First face image array
            face2_array: Second face image array
            
        Returns:
            Tuple of embedding extraction results
        """
        try:
            # Submit both embedding extraction tasks concurrently
            future1 = self.thread_pool.submit(self.face_recognizer.extract_embedding, face1_array)
            future2 = self.thread_pool.submit(self.face_recognizer.extract_embedding, face2_array)
            
            # Wait for both tasks to complete
            emb1_result = future1.result(timeout=15.0)  # 15 second timeout per embedding
            emb2_result = future2.result(timeout=15.0)
            
            return emb1_result, emb2_result
            
        except Exception as e:
            logger.error(f"Concurrent embedding extraction failed: {e}")
            return {'success': False, 'error': str(e)}, {'success': False, 'error': str(e)}
    
    def _convert_to_numpy_array(self, image_input: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Convert image input (URL, base64, or numpy array) to numpy array.
        Optimized with caching for repeated images.
        
        Args:
            image_input: Image input (URL, base64, or numpy array)
            
        Returns:
            Numpy array or None if conversion failed
        """
        try:
            # If already a numpy array, return as is
            if isinstance(image_input, np.ndarray):
                return image_input
            
            # If it's a string, check cache first
            if isinstance(image_input, str):
                # Create cache key (simple hash for performance)
                cache_key = hash(image_input) % 1000000
                
                with self._lock:
                    if cache_key in self._image_cache:
                        logger.debug("Image cache hit")
                        return self._image_cache[cache_key]
                
                # Import here to avoid circular imports
                from .image_utils import image_processor
                
                # Use the image processor to convert string to numpy array
                numpy_array = image_processor.process_image_input(image_input)
                
                if numpy_array is not None:
                    # Convert BGR to RGB if needed (OpenCV uses BGR, but face recognition expects RGB)
                    if len(numpy_array.shape) == 3 and numpy_array.shape[2] == 3:
                        numpy_array = cv2.cvtColor(numpy_array, cv2.COLOR_BGR2RGB)
                    
                    # Cache the result
                    with self._lock:
                        if len(self._image_cache) >= self._cache_max_size:
                            # Remove oldest entry (simple FIFO)
                            oldest_key = next(iter(self._image_cache))
                            del self._image_cache[oldest_key]
                        self._image_cache[cache_key] = numpy_array
                
                return numpy_array
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to convert image input to numpy array: {e}")
            return None
    
    def batch_compare_faces(
        self, 
        image_pairs: List[Tuple[Union[str, np.ndarray], Union[str, np.ndarray]]],
        threshold: Optional[float] = None,
        session_id: Optional[str] = None
    ) -> List[FaceComparisonResult]:
        """
        Perform batch face comparison for multiple image pairs concurrently.
        
        Args:
            image_pairs: List of (image1, image2) pairs
            threshold: Similarity threshold
            session_id: Session ID for logging
            
        Returns:
            List of face comparison results
        """
        start_time = time.time()
        threshold = threshold or self.default_threshold
        
        logger.info(f"Starting batch face comparison: {len(image_pairs)} pairs")
        
        try:
            # Submit all comparison tasks concurrently
            futures = []
            for i, (img1, img2) in enumerate(image_pairs):
                future = self.thread_pool.submit(
                    self.compare_faces_with_validation,
                    img1, img2, threshold, True, f"{session_id}_pair_{i}"
                )
                futures.append(future)
            
            # Collect results
            results = []
            for future in as_completed(futures, timeout=60.0):  # 60 second total timeout
                try:
                    result = future.result(timeout=30.0)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch comparison task failed: {e}")
                    results.append(self._create_error_result(
                        error=f"Batch task failed: {str(e)}",
                        error_code="BATCH_TASK_ERROR",
                        threshold=threshold,
                        processing_time=time.time() - start_time
                    ))
            
            total_time = time.time() - start_time
            logger.info(f"Batch face comparison completed: {len(results)} results in {total_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch face comparison failed: {e}")
            return [self._create_error_result(
                error=f"Batch comparison failed: {str(e)}",
                error_code="BATCH_ERROR",
                threshold=threshold,
                processing_time=time.time() - start_time
            ) for _ in image_pairs]
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.thread_pool.shutdown(wait=True)
            with self._lock:
                self._image_cache.clear()
            logger.info("Face comparison integrator resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass
    
    def integrate_with_liveness_detection(
        self,
        video_path: str,
        reference_image: str,
        movement_sequence: Optional[list] = None,
        liveness_threshold: float = 0.5,
        face_threshold: float = 0.6,
        session_id: Optional[str] = None
    ) -> IntegratedLivenessResult:
        """
        Perform integrated liveness detection with face comparison.
        
        Args:
            video_path: Path to video file for liveness detection
            reference_image: Reference image for face comparison
            movement_sequence: Expected movement sequence for liveness
            liveness_threshold: Threshold for liveness detection
            face_threshold: Threshold for face comparison
            session_id: Session ID for logging context
            
        Returns:
            Integrated liveness and face comparison result
        """
        start_time = time.time()
        
        # Log integration start
        self._log_integration_start(session_id, video_path, reference_image)
        
        try:
            # Validate inputs
            if not self.unified_detector:
                return self._create_integration_error_result(
                    error="Liveness detection service not available",
                    error_code="LIVENESS_SERVICE_UNAVAILABLE",
                    processing_time=time.time() - start_time
                )
            
            # Perform video liveness detection
            liveness_result = self.unified_detector.detect_video_liveness(
                video_path=video_path,
                movement_sequence=movement_sequence,
                session_id=session_id,
                reference_image=reference_image
            )
            
            # Extract face image from liveness result for comparison
            face_image_path = getattr(liveness_result, 'face_image_path', None)
            
            # Initialize face comparison variables
            face_comparison_success = False
            face_comparison_match = False
            face_comparison_score = 0.0
            face_comparison_details = None
            
            # Perform face comparison if we have both images
            if face_image_path and reference_image:
                face_comparison_result = self.compare_faces_with_validation(
                    reference_image,
                    face_image_path,
                    threshold=face_threshold,
                    session_id=session_id
                )
                
                face_comparison_success = face_comparison_result.success
                face_comparison_match = face_comparison_result.match
                face_comparison_score = face_comparison_result.similarity_score
                face_comparison_details = {
                    'threshold': face_comparison_result.threshold,
                    'confidence': face_comparison_result.confidence,
                    'error': face_comparison_result.error,
                    'error_code': face_comparison_result.error_code,
                    'details': face_comparison_result.details
                }
            elif reference_image:
                # Reference image provided but no face extracted from video
                face_comparison_details = {
                    'error': 'No face image extracted from video for comparison',
                    'error_code': 'NO_FACE_EXTRACTED'
                }
            
            # Determine overall pass status
            liveness_passed = liveness_result.success and liveness_result.passed
            overall_passed = liveness_passed
            
            # If face comparison was requested, both must pass
            if reference_image:
                overall_passed = liveness_passed and face_comparison_success and face_comparison_match
            
            processing_time = time.time() - start_time
            
            # Create integrated result
            result = IntegratedLivenessResult(
                liveness_success=liveness_result.success,
                liveness_passed=liveness_result.passed,
                liveness_score=liveness_result.liveness_score,
                face_comparison_success=face_comparison_success,
                face_comparison_match=face_comparison_match,
                face_comparison_score=face_comparison_score,
                overall_passed=overall_passed,
                processing_time=processing_time,
                liveness_details={
                    'confidence': liveness_result.confidence,
                    'error': liveness_result.error,
                    'detected_sequence': getattr(liveness_result, 'detected_sequence', None),
                    'expected_sequence': getattr(liveness_result, 'expected_sequence', None),
                    'sequence_accuracy': getattr(liveness_result, 'sequence_accuracy', None),
                    'details': getattr(liveness_result, 'details', None),
                    'movement_details': getattr(liveness_result, 'movement_details', None),
                    'anti_spoofing_details': getattr(liveness_result, 'anti_spoofing_details', None)
                },
                face_comparison_details=face_comparison_details
            )
            
            # Log integration result
            self._log_integration_success(session_id, result)
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Unexpected error in integrated liveness detection: {str(e)}"
            self._log_integration_error(session_id, error_msg, processing_time)
            
            return self._create_integration_error_result(
                error=error_msg,
                error_code="INTEGRATION_ERROR",
                processing_time=processing_time
            )
    
    def _validate_comparison_inputs(
        self, 
        image1: Any, 
        image2: Any, 
        threshold: float
    ) -> Dict[str, Any]:
        """Validate face comparison inputs."""
        if not image1:
            return {'valid': False, 'error': 'First image is required'}
        
        if not image2:
            return {'valid': False, 'error': 'Second image is required'}
        
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            return {'valid': False, 'error': 'Threshold must be a number between 0 and 1'}
        
        return {'valid': True}
    
    def _calculate_confidence_level(self, similarity_score: float, threshold: float) -> str:
        """Calculate confidence level based on similarity score and threshold."""
        if similarity_score >= threshold:
            # For matches, confidence increases with distance from threshold
            confidence_score = min(1.0, threshold + (similarity_score - threshold) * 2)
        else:
            # For non-matches, confidence increases with distance from threshold
            confidence_score = min(1.0, (threshold - similarity_score) * 2)
        
        if confidence_score >= 0.8:
            return 'high'
        elif confidence_score >= 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _extract_quality_metrics(self, comparison_result: Dict) -> Dict[str, Any]:
        """Extract quality metrics from comparison result."""
        quality_metrics = {}
        
        embedding_quality = comparison_result.get('embedding_quality', {})
        if embedding_quality:
            quality_metrics['embedding_quality'] = embedding_quality
        
        timing = comparison_result.get('timing', {})
        if timing:
            quality_metrics['timing'] = timing
        
        return quality_metrics
    
    def _create_error_result(
        self,
        error: str,
        error_code: str,
        threshold: float,
        processing_time: float
    ) -> FaceComparisonResult:
        """Create a face comparison error result."""
        return FaceComparisonResult(
            success=False,
            match=False,
            similarity_score=0.0,
            threshold=threshold,
            confidence='unknown',
            processing_time=processing_time,
            error=error,
            error_code=error_code
        )
    
    def _create_integration_error_result(
        self,
        error: str,
        error_code: str,
        processing_time: float
    ) -> IntegratedLivenessResult:
        """Create an integration error result."""
        return IntegratedLivenessResult(
            liveness_success=False,
            liveness_passed=False,
            liveness_score=0.0,
            face_comparison_success=False,
            face_comparison_match=False,
            face_comparison_score=0.0,
            overall_passed=False,
            processing_time=processing_time,
            error=error,
            error_code=error_code
        )
    
    # Comprehensive logging methods
    def _log_face_comparison_start(
        self, 
        session_id: Optional[str], 
        threshold: float, 
        enable_quality_check: bool
    ):
        """Log face comparison initiation."""
        logger.info(
            f"FACE_COMPARISON_START - session_id: {session_id}, "
            f"threshold: {threshold}, quality_check: {enable_quality_check}"
        )
    
    def _log_face_comparison_success(
        self, 
        session_id: Optional[str], 
        result: FaceComparisonResult
    ):
        """Log successful face comparison."""
        logger.info(
            f"FACE_COMPARISON_SUCCESS - session_id: {session_id}, "
            f"match: {result.match}, similarity: {result.similarity_score:.4f}, "
            f"threshold: {result.threshold}, confidence: {result.confidence}, "
            f"processing_time: {result.processing_time:.3f}s"
        )
    
    def _log_face_comparison_error(
        self, 
        session_id: Optional[str], 
        error: str, 
        processing_time: float
    ):
        """Log face comparison error."""
        logger.error(
            f"FACE_COMPARISON_ERROR - session_id: {session_id}, "
            f"error: {error}, processing_time: {processing_time:.3f}s"
        )
    
    def _log_face_comparison_timeout(
        self, 
        session_id: Optional[str], 
        processing_time: float
    ):
        """Log face comparison timeout."""
        logger.warning(
            f"FACE_COMPARISON_TIMEOUT - session_id: {session_id}, "
            f"processing_time: {processing_time:.3f}s, "
            f"max_allowed: {self.max_processing_time}s"
        )
    
    def _log_integration_start(
        self, 
        session_id: Optional[str], 
        video_path: str, 
        reference_image: str
    ):
        """Log integration process start."""
        logger.info(
            f"LIVENESS_FACE_INTEGRATION_START - session_id: {session_id}, "
            f"video_path: {video_path}, has_reference: {bool(reference_image)}"
        )
    
    def _log_integration_success(
        self, 
        session_id: Optional[str], 
        result: IntegratedLivenessResult
    ):
        """Log successful integration."""
        logger.info(
            f"LIVENESS_FACE_INTEGRATION_SUCCESS - session_id: {session_id}, "
            f"liveness_passed: {result.liveness_passed}, "
            f"face_match: {result.face_comparison_match}, "
            f"overall_passed: {result.overall_passed}, "
            f"liveness_score: {result.liveness_score:.4f}, "
            f"face_score: {result.face_comparison_score:.4f}, "
            f"processing_time: {result.processing_time:.3f}s"
        )
    
    def _log_integration_error(
        self, 
        session_id: Optional[str], 
        error: str, 
        processing_time: float
    ):
        """Log integration error."""
        logger.error(
            f"LIVENESS_FACE_INTEGRATION_ERROR - session_id: {session_id}, "
            f"error: {error}, processing_time: {processing_time:.3f}s"
        )

# Global integrator instance
_face_comparison_integrator = None

def get_face_comparison_integrator() -> Optional[FaceComparisonIntegrator]:
    """Get the global face comparison integrator instance."""
    return _face_comparison_integrator

def initialize_face_comparison_integrator(
    face_recognizer=None, 
    unified_detector=None
) -> FaceComparisonIntegrator:
    """Initialize the face comparison integrator with required components."""
    global _face_comparison_integrator
    _face_comparison_integrator = FaceComparisonIntegrator(
        face_recognizer=face_recognizer,
        unified_detector=unified_detector
    )
    return _face_comparison_integrator