"""
Unified Liveness Detection Interface
Consolidates multiple liveness detection implementations into a single, clean API.
"""

import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class LivenessMode(Enum):
    """Supported liveness detection modes."""
    CRMNET = "crmnet"  # CRMNET-based anti-spoofing
    VIDEO_MOVEMENT = "video_movement"  # Head movement validation
    IMAGE_ANALYSIS = "image_analysis"  # Static image liveness
    ENHANCED = "enhanced"  # Combined CRMNET + enhanced anti-spoofing
    COMPREHENSIVE = "comprehensive"  # All available methods

@dataclass
class LivenessResult:
    """Standardized liveness detection result."""
    success: bool
    passed: bool
    confidence: float
    liveness_score: float
    mode: LivenessMode
    processing_time: float
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    anti_spoofing_details: Optional[Dict[str, Any]] = None
    movement_details: Optional[Dict[str, Any]] = None
    quality_metrics: Optional[Dict[str, Any]] = None

@dataclass
class VideoLivenessResult(LivenessResult):
    """Extended result for video-based liveness detection."""
    detected_sequence: Optional[List[str]] = None
    expected_sequence: Optional[List[str]] = None
    sequence_accuracy: Optional[float] = None
    face_image_path: Optional[str] = None
    face_comparison: Optional[Dict[str, Any]] = None

class BaseLivenessDetector(ABC):
    """Abstract base class for liveness detectors."""
    
    @abstractmethod
    def detect_liveness(self, input_data: Any, **kwargs) -> LivenessResult:
        """Perform liveness detection on input data."""
        pass
    
    @abstractmethod
    def get_supported_modes(self) -> List[LivenessMode]:
        """Get list of supported liveness detection modes."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the detector is available and ready."""
        pass

class UnifiedLivenessDetector:
    """
    Unified interface for all liveness detection methods.
    Provides a clean API that consolidates multiple detection implementations.
    """
    
    def __init__(self):
        """Initialize the unified liveness detector."""
        self._detectors = {}
        self._default_mode = LivenessMode.IMAGE_ANALYSIS
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Initialize available liveness detectors."""
        try:
            # Import and initialize CRMNET detector
            from .liveness import LivenessDetector
            try:
                import os
                crmnet_path = os.getenv('CRMNET_MODEL_PATH', '/app/models/crmnet.onnx')
                if os.path.exists(crmnet_path):
                    crmnet_detector = LivenessDetector(crmnet_path)
                    self._detectors[LivenessMode.CRMNET] = CRMNETLivenessAdapter(crmnet_detector)
                    self._detectors[LivenessMode.ENHANCED] = CRMNETLivenessAdapter(crmnet_detector)
                    logger.info("CRMNET liveness detector initialized")
                else:
                    logger.warning(f"CRMNET model not found at {crmnet_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize CRMNET detector: {e}")
            
            # Import and initialize Simple/Video detector
            from .simple_liveness import SimpleLivenessDetector
            try:
                simple_detector = SimpleLivenessDetector()
                self._detectors[LivenessMode.VIDEO_MOVEMENT] = VideoLivenessAdapter(simple_detector)
                self._detectors[LivenessMode.IMAGE_ANALYSIS] = ImageLivenessAdapter(simple_detector)
                logger.info("Video and image liveness detectors initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize simple liveness detector: {e}")
            
            # Set default mode based on available detectors
            if LivenessMode.VIDEO_MOVEMENT in self._detectors:
                self._default_mode = LivenessMode.VIDEO_MOVEMENT
            elif LivenessMode.CRMNET in self._detectors:
                self._default_mode = LivenessMode.CRMNET
                
        except Exception as e:
            logger.error(f"Failed to initialize liveness detectors: {e}")
    
    def detect_liveness(
        self, 
        input_data: Union[str, np.ndarray], 
        mode: Optional[LivenessMode] = None,
        **kwargs
    ) -> LivenessResult:
        """
        Perform liveness detection using specified or default mode.
        
        Args:
            input_data: Image path, video path, or numpy array
            mode: Liveness detection mode to use
            **kwargs: Additional parameters for specific detectors
            
        Returns:
            Standardized liveness result
        """
        start_time = time.time()
        
        # Use default mode if not specified
        if mode is None:
            mode = self._default_mode
        
        # Check if requested mode is available
        if mode not in self._detectors:
            available_modes = list(self._detectors.keys())
            if not available_modes:
                return LivenessResult(
                    success=False,
                    passed=False,
                    confidence=0.0,
                    liveness_score=0.0,
                    mode=mode,
                    processing_time=time.time() - start_time,
                    error="No liveness detectors available"
                )
            
            # Fallback to first available mode
            mode = available_modes[0]
            logger.warning(f"Requested mode not available, using fallback: {mode}")
        
        try:
            detector = self._detectors[mode]
            result = detector.detect_liveness(input_data, **kwargs)
            result.processing_time = time.time() - start_time
            return result
            
        except Exception as e:
            logger.error(f"Liveness detection failed with mode {mode}: {e}")
            return LivenessResult(
                success=False,
                passed=False,
                confidence=0.0,
                liveness_score=0.0,
                mode=mode,
                processing_time=time.time() - start_time,
                error=str(e)
            )
    
    def detect_video_liveness(
        self,
        video_path: str,
        movement_sequence: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        reference_image: Optional[str] = None,
        **kwargs
    ) -> VideoLivenessResult:
        """
        Perform video-based liveness detection with movement validation.
        
        Args:
            video_path: Path to video file
            movement_sequence: Expected head movement sequence
            session_id: Session identifier for tracking
            reference_image: Reference image for face comparison
            **kwargs: Additional parameters
            
        Returns:
            Video liveness result with movement details
        """
        start_time = time.time()
        
        if LivenessMode.VIDEO_MOVEMENT not in self._detectors:
            return VideoLivenessResult(
                success=False,
                passed=False,
                confidence=0.0,
                liveness_score=0.0,
                mode=LivenessMode.VIDEO_MOVEMENT,
                processing_time=time.time() - start_time,
                error="Video liveness detector not available"
            )
        
        try:
            detector = self._detectors[LivenessMode.VIDEO_MOVEMENT]
            
            # Call the video validation method
            result = detector.validate_video_challenge(
                video_path=video_path,
                challenge_type='head_movement',
                movement_sequence=movement_sequence,
                session_id=session_id,
                reference_image=reference_image
            )
            
            # Convert to standardized format
            return VideoLivenessResult(
                success=result.get('success', False),
                passed=result.get('passed', False),
                confidence=result.get('confidence', 0.0),
                liveness_score=result.get('liveness_score', 0.0),
                mode=LivenessMode.VIDEO_MOVEMENT,
                processing_time=time.time() - start_time,
                error=result.get('error'),
                detected_sequence=result.get('detected_sequence'),
                expected_sequence=result.get('expected_sequence'),
                sequence_accuracy=result.get('sequence_accuracy'),
                face_image_path=result.get('face_image_path'),
                face_comparison=result.get('face_comparison'),
                details=result.get('details'),
                movement_details=result.get('movement_details'),
                anti_spoofing_details=result.get('anti_spoofing_details')
            )
            
        except Exception as e:
            logger.error(f"Video liveness detection failed: {e}")
            return VideoLivenessResult(
                success=False,
                passed=False,
                confidence=0.0,
                liveness_score=0.0,
                mode=LivenessMode.VIDEO_MOVEMENT,
                processing_time=time.time() - start_time,
                error=str(e)
            )
    
    def detect_image_liveness(
        self,
        image_input: Union[str, np.ndarray],
        **kwargs
    ) -> LivenessResult:
        """
        Perform static image liveness detection.
        
        Args:
            image_input: Image URL, base64, or numpy array
            **kwargs: Additional parameters
            
        Returns:
            Image liveness result
        """
        return self.detect_liveness(image_input, mode=LivenessMode.IMAGE_ANALYSIS, **kwargs)
    
    def compare_faces(
        self,
        image1: Union[str, np.ndarray],
        image2: Union[str, np.ndarray],
        threshold: float = 0.6,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare two faces for similarity.
        
        Args:
            image1: First image
            image2: Second image
            threshold: Similarity threshold
            **kwargs: Additional parameters
            
        Returns:
            Face comparison result
        """
        try:
            # Try to use face comparison integrator if available
            from .face_comparison_integration import get_face_comparison_integrator
            integrator = get_face_comparison_integrator()
            
            if integrator:
                result = integrator.compare_faces_with_validation(
                    image1=image1,
                    image2=image2,
                    threshold=threshold,
                    session_id=kwargs.get('session_id')
                )
                
                return {
                    'success': result.success,
                    'match': result.match,
                    'similarity_score': result.similarity_score,
                    'threshold': result.threshold,
                    'confidence': result.confidence,
                    'error': result.error,
                    'processing_time': result.processing_time
                }
            
            # Fallback to simple detector
            if LivenessMode.IMAGE_ANALYSIS in self._detectors:
                detector = self._detectors[LivenessMode.IMAGE_ANALYSIS]
                return detector.compare_faces(image1, image2, threshold)
            else:
                return {
                    'success': False,
                    'match': False,
                    'error': 'Face comparison not available',
                    'similarity_score': 0.0
                }
                
        except Exception as e:
            logger.error(f"Face comparison failed: {e}")
            return {
                'success': False,
                'match': False,
                'error': f'Face comparison error: {str(e)}',
                'similarity_score': 0.0
            }
    
    def generate_challenge(self) -> Dict[str, Any]:
        """
        Generate a liveness challenge.
        
        Returns:
            Challenge configuration
        """
        if LivenessMode.VIDEO_MOVEMENT in self._detectors:
            detector = self._detectors[LivenessMode.VIDEO_MOVEMENT]
            return detector.generate_challenge()
        else:
            return {
                'type': 'unavailable',
                'error': 'Challenge generation not available'
            }
    
    def get_available_modes(self) -> List[LivenessMode]:
        """Get list of available liveness detection modes."""
        return list(self._detectors.keys())
    
    def is_mode_available(self, mode: LivenessMode) -> bool:
        """Check if a specific mode is available."""
        return mode in self._detectors and self._detectors[mode].is_available()
    
    def get_detector_status(self) -> Dict[str, bool]:
        """Get status of all detectors."""
        return {
            mode.value: detector.is_available() 
            for mode, detector in self._detectors.items()
        }

class CRMNETLivenessAdapter(BaseLivenessDetector):
    """Adapter for CRMNET-based liveness detector."""
    
    def __init__(self, crmnet_detector):
        self.detector = crmnet_detector
    
    def detect_liveness(self, input_data: np.ndarray, **kwargs) -> LivenessResult:
        """Detect liveness using CRMNET model."""
        try:
            use_enhanced = kwargs.get('use_enhanced', True)
            
            if use_enhanced:
                result = self.detector.detect_liveness_with_enhanced_anti_spoofing(input_data)
            else:
                result = self.detector.detect_liveness_with_quality_check(input_data)
            
            return LivenessResult(
                success=result.get('success', False),
                passed=result.get('is_live', False),
                confidence=result.get('confidence', 0.0),
                liveness_score=result.get('liveness_score', 0.0),
                mode=LivenessMode.ENHANCED if use_enhanced else LivenessMode.CRMNET,
                processing_time=result.get('inference_time', 0.0),
                error=result.get('error'),
                details=result.get('crmnet_result'),
                anti_spoofing_details=result.get('enhanced_anti_spoofing'),
                quality_metrics=result.get('quality_metrics')
            )
            
        except Exception as e:
            return LivenessResult(
                success=False,
                passed=False,
                confidence=0.0,
                liveness_score=0.0,
                mode=LivenessMode.CRMNET,
                processing_time=0.0,
                error=str(e)
            )
    
    def get_supported_modes(self) -> List[LivenessMode]:
        return [LivenessMode.CRMNET, LivenessMode.ENHANCED]
    
    def is_available(self) -> bool:
        return self.detector is not None and hasattr(self.detector, 'session')

class VideoLivenessAdapter(BaseLivenessDetector):
    """Adapter for video-based liveness detector with MediaPipe support."""
    
    def __init__(self, simple_detector):
        self.detector = simple_detector
        # Initialize MediaPipe processor
        try:
            from .optimized_video_processor import create_optimized_processor
            self.mediapipe_processor = create_optimized_processor(use_mediapipe=True)
            self.use_mediapipe = True
            logger.info("VideoLivenessAdapter initialized with MediaPipe support")
        except Exception as e:
            logger.warning(f"Failed to initialize MediaPipe processor: {e}")
            self.mediapipe_processor = None
            self.use_mediapipe = False
    
    def detect_liveness(self, input_data: str, **kwargs) -> LivenessResult:
        """Detect liveness in video."""
        # This method is primarily for video files
        # For direct video liveness, use validate_video_challenge
        return LivenessResult(
            success=False,
            passed=False,
            confidence=0.0,
            liveness_score=0.0,
            mode=LivenessMode.VIDEO_MOVEMENT,
            processing_time=0.0,
            error="Use validate_video_challenge for video liveness detection"
        )
    
    def validate_video_challenge(self, **kwargs) -> Dict[str, Any]:
        """Validate video challenge using MediaPipe or fallback to simple detector."""
        try:
            # Try MediaPipe first if available
            if self.use_mediapipe and self.mediapipe_processor:
                logger.info("Using MediaPipe for video challenge validation")
                return self._validate_with_mediapipe(**kwargs)
            else:
                logger.info("Using legacy simple detector for video challenge validation")
                return self.detector.validate_video_challenge(**kwargs)
        except Exception as e:
            logger.warning(f"MediaPipe validation failed, falling back to legacy: {e}")
            return self.detector.validate_video_challenge(**kwargs)
    
    def _validate_with_mediapipe(self, **kwargs) -> Dict[str, Any]:
        """Validate video challenge using MediaPipe processor."""
        try:
            video_path = kwargs.get('video_path')
            movement_sequence = kwargs.get('movement_sequence')
            
            if not video_path:
                return {
                    'success': False,
                    'passed': False,
                    'error': 'No video path provided',
                    'confidence': 0.0,
                    'liveness_score': 0.0
                }
            
            # Process video with MediaPipe
            result = self.mediapipe_processor.process_video_for_liveness(
                video_path=video_path,
                expected_sequence=movement_sequence
            )
            
            if not result.success:
                return {
                    'success': False,
                    'passed': False,
                    'error': result.error or 'MediaPipe processing failed',
                    'confidence': 0.0,
                    'liveness_score': 0.0
                }
            
            # Analyze movements
            movements = result.movements
            if not movements:
                return {
                    'success': True,
                    'passed': False,
                    'error': 'No movements detected',
                    'confidence': 0.0,
                    'liveness_score': 0.0,
                    'detected_sequence': [],
                    'expected_sequence': movement_sequence,
                    'sequence_accuracy': 0.0
                }
            
            # Extract movement sequence
            detected_sequence = [movement['direction'] for movement in movements]
            
            # Calculate sequence accuracy if expected sequence provided
            sequence_accuracy = 0.0
            if movement_sequence and detected_sequence:
                correct_movements = 0
                for i, expected in enumerate(movement_sequence):
                    if i < len(detected_sequence) and detected_sequence[i] == expected:
                        correct_movements += 1
                sequence_accuracy = correct_movements / len(movement_sequence) if movement_sequence else 0.0
            
            # Calculate overall confidence
            movement_confidences = [movement.get('confidence', 0.0) for movement in movements]
            avg_confidence = sum(movement_confidences) / len(movement_confidences) if movement_confidences else 0.0
            
            # Determine if passed (at least 70% accuracy and good confidence)
            passed = sequence_accuracy >= 0.7 and avg_confidence >= 0.6
            
            return {
                'success': True,
                'passed': passed,
                'confidence': avg_confidence,
                'liveness_score': sequence_accuracy,
                'detected_sequence': detected_sequence,
                'expected_sequence': movement_sequence,
                'sequence_accuracy': sequence_accuracy,
                'movement_details': {
                    'total_movements': len(movements),
                    'movements': movements,
                    'avg_confidence': avg_confidence,
                    'processing_time': result.processing_time,
                    'frames_processed': result.frames_processed
                },
                'details': {
                    'mediapipe_used': True,
                    'performance_metrics': result.performance_metrics,
                    'optimization_applied': result.optimization_applied
                }
            }
            
        except Exception as e:
            logger.error(f"MediaPipe validation failed: {e}")
            return {
                'success': False,
                'passed': False,
                'error': f'MediaPipe validation error: {str(e)}',
                'confidence': 0.0,
                'liveness_score': 0.0
            }
    
    def generate_challenge(self) -> Dict[str, Any]:
        """Generate movement challenge."""
        return self.detector.generate_challenge()
    
    def get_supported_modes(self) -> List[LivenessMode]:
        return [LivenessMode.VIDEO_MOVEMENT]
    
    def is_available(self) -> bool:
        return self.detector is not None

class ImageLivenessAdapter(BaseLivenessDetector):
    """Adapter for image-based liveness detector."""
    
    def __init__(self, simple_detector):
        self.detector = simple_detector
    
    def detect_liveness(self, input_data: str, **kwargs) -> LivenessResult:
        """Detect liveness in static image."""
        try:
            result = self.detector.perform_image_liveness_check(input_data)
            
            return LivenessResult(
                success=result.get('success', False),
                passed=result.get('passed', False),
                confidence=result.get('confidence', 0.0),
                liveness_score=result.get('liveness_score', 0.0),
                mode=LivenessMode.IMAGE_ANALYSIS,
                processing_time=0.0,  # Will be set by caller
                error=result.get('error'),
                anti_spoofing_details=result.get('anti_spoofing'),
                details={
                    'face_detected': result.get('face_detected', False),
                    'threshold': result.get('threshold'),
                    'validation_message': result.get('validation_message')
                }
            )
            
        except Exception as e:
            return LivenessResult(
                success=False,
                passed=False,
                confidence=0.0,
                liveness_score=0.0,
                mode=LivenessMode.IMAGE_ANALYSIS,
                processing_time=0.0,
                error=str(e)
            )
    
    def compare_faces(self, image1: str, image2: str, threshold: float = 0.6) -> Dict[str, Any]:
        """Compare faces using simple detector."""
        return self.detector.compare_faces(image1, image2, threshold)
    
    def get_supported_modes(self) -> List[LivenessMode]:
        return [LivenessMode.IMAGE_ANALYSIS]
    
    def is_available(self) -> bool:
        return self.detector is not None

# Global instance
_unified_detector = None

def get_unified_liveness_detector() -> UnifiedLivenessDetector:
    """Get the global unified liveness detector instance."""
    global _unified_detector
    if _unified_detector is None:
        _unified_detector = UnifiedLivenessDetector()
    return _unified_detector

def initialize_unified_liveness_detector() -> UnifiedLivenessDetector:
    """Initialize and return the unified liveness detector."""
    global _unified_detector
    _unified_detector = UnifiedLivenessDetector()
    return _unified_detector