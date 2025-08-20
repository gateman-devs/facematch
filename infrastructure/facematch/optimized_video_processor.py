"""
Optimized Video Processor for Liveness Detection

This module provides an optimized video processing pipeline that integrates
performance optimizations with the existing liveness detection system.
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2
from dataclasses import dataclass

from .performance_optimizer import (
    VideoProcessingOptimizer,
    OptimizationConfig,
    create_default_optimization_config,
    create_performance_optimized_config,
    create_memory_optimized_config
)
from .face_detection import FaceDetector
from .movement_confidence import MovementConfidenceScorer, create_confidence_scorer
from .flexible_sequence_validator import FlexibleSequenceValidator
from .dlib_head_detector import DlibHeadDetector, create_dlib_detector

logger = logging.getLogger(__name__)

@dataclass
class VideoProcessingResult:
    """Result of optimized video processing."""
    success: bool
    movements: List[Dict[str, Any]]
    validation_result: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    frames_processed: int = 0
    frames_skipped: int = 0
    optimization_applied: Optional[Dict[str, bool]] = None

class OptimizedVideoProcessor:
    """
    Optimized video processor that combines performance optimizations
    with liveness detection capabilities.
    """
    
    def __init__(
        self, 
        face_detector: Optional[FaceDetector] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        use_performance_mode: bool = True,
        use_dlib: bool = True
    ):
        """
        Initialize optimized video processor.
        
        Args:
            face_detector: Face detector instance (will create if None)
            optimization_config: Optimization configuration
            use_performance_mode: Use performance-optimized settings
            use_dlib: Use dlib for head movement detection
        """
        # Initialize face detector
        self.face_detector = face_detector or FaceDetector()
        
        # Setup optimization configuration
        if optimization_config is None:
            if use_performance_mode:
                self.optimization_config = create_performance_optimized_config()
            else:
                self.optimization_config = create_default_optimization_config()
        else:
            self.optimization_config = optimization_config
        
        # Initialize dlib head detector if enabled
        self.use_dlib = use_dlib
        if self.use_dlib:
            self.dlib_detector = create_dlib_detector({
                'min_rotation_degrees': 15.0,
                'significant_rotation_degrees': 25.0,
                'debug_mode': False
            })
            logger.info("Dlib head detector initialized")
        else:
            self.dlib_detector = None
        
        # Initialize optimizer and supporting components
        self.optimizer = VideoProcessingOptimizer(self.optimization_config)
        self.confidence_scorer = create_confidence_scorer()
        self.sequence_validator = FlexibleSequenceValidator()
        
        # Processing state
        self.last_face_landmarks = None
        self.movement_history = []
        
        logger.info("OptimizedVideoProcessor initialized with performance optimizations")
    
    def process_video_for_liveness(
        self, 
        video_path: str, 
        expected_sequence: Optional[List[str]] = None
    ) -> VideoProcessingResult:
        """
        Process video for liveness detection with full optimization.
        
        Args:
            video_path: Path to video file
            expected_sequence: Expected movement sequence for validation
            
        Returns:
            Video processing result with performance metrics
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting optimized video processing: {video_path}")
            
            # Validate video file
            if not os.path.exists(video_path):
                return VideoProcessingResult(
                    success=False,
                    movements=[],
                    error=f"Video file not found: {video_path}",
                    processing_time=time.time() - start_time
                )
            
            # Use dlib detector if available
            if self.use_dlib and self.dlib_detector:
                logger.info("Using dlib for head movement detection")
                return self._process_with_dlib(video_path, expected_sequence, start_time)
            else:
                logger.info("Using legacy processing pipeline")
                return self._process_with_legacy_pipeline(video_path, expected_sequence, start_time)
            
        except Exception as e:
            logger.error(f"Optimized video processing failed: {e}")
            return VideoProcessingResult(
                success=False,
                movements=[],
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    def _process_with_dlib(
        self, 
        video_path: str, 
        expected_sequence: Optional[List[str]], 
        start_time: float
    ) -> VideoProcessingResult:
        """Process video using dlib head movement detection."""
        try:
            # Use dlib detector
            dlib_result = self.dlib_detector.process_video(video_path)
            
            if not dlib_result.success:
                return VideoProcessingResult(
                    success=False,
                    movements=[],
                    error=dlib_result.error or "Dlib processing failed",
                    processing_time=time.time() - start_time
                )
            
            # Convert dlib movements to standard format
            movements = []
            for dlib_movement in dlib_result.movements:
                movement = {
                    'direction': dlib_movement.direction,
                    'confidence': dlib_movement.confidence,
                    'magnitude': dlib_movement.magnitude,
                    'timestamp': dlib_movement.timestamp,
                    'start_position': (0, 0),  # dlib doesn't track positions, use default
                    'end_position': (0, 0),
                    'dx_pixels': 0,  # dlib doesn't provide pixel movement
                    'dy_pixels': 0,
                    'dx_norm': 0,
                    'dy_norm': 0,
                    'frame_indices': (0, 1),  # Placeholder
                    'pose_data': dlib_movement.pose_data,
                    'rotation_degrees': dlib_movement.rotation_degrees
                }
                movements.append(movement)
            
            # Validate sequence if provided
            validation_result = None
            if expected_sequence and movements:
                validation_result = self.sequence_validator.validate_sequence(
                    movements, expected_sequence
                )
            
            processing_time = time.time() - start_time
            
            # Create performance metrics
            performance_metrics = {
                'frames_processed': dlib_result.frames_processed,
                'processing_time': dlib_result.processing_time,
                'movements_detected': len(movements),
                'detector_type': 'dlib'
            }
            
            return VideoProcessingResult(
                success=True,
                movements=movements,
                validation_result=validation_result,
                performance_metrics=performance_metrics,
                processing_time=processing_time,
                frames_processed=dlib_result.frames_processed,
                optimization_applied={'dlib_detection': True}
            )
            
        except Exception as e:
            logger.error(f"Dlib processing failed: {e}")
            return VideoProcessingResult(
                success=False,
                movements=[],
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    def _process_with_legacy_pipeline(
        self, 
        video_path: str, 
        expected_sequence: Optional[List[str]], 
        start_time: float
    ) -> VideoProcessingResult:
        """Process video using legacy processing pipeline."""
        try:
            # Define processing pipeline
            processing_pipeline = [
                self._detect_faces_in_frame,
                self._analyze_movement_in_frame,
                self._enhance_movement_confidence
            ]
            
            # Process video with optimization
            optimization_result = self.optimizer.optimize_video_processing(
                video_path, processing_pipeline
            )
            
            if not optimization_result['success']:
                return VideoProcessingResult(
                    success=False,
                    movements=[],
                    error=optimization_result.get('error', 'Video processing failed'),
                    performance_metrics=optimization_result.get('performance_metrics'),
                    processing_time=time.time() - start_time
                )
            
            # Extract movement data from results
            movements = self._extract_movements_from_results(optimization_result['results'])
            
            # Validate sequence if expected sequence provided
            validation_result = None
            if expected_sequence and movements:
                validation_result = self.sequence_validator.validate_sequence(
                    movements, expected_sequence
                )
            
            # Calculate final processing time
            total_processing_time = time.time() - start_time
            
            # Extract performance metrics
            performance_metrics = optimization_result.get('performance_metrics', {})
            extraction_metadata = optimization_result.get('extraction_metadata', {})
            
            return VideoProcessingResult(
                success=True,
                movements=movements,
                validation_result=validation_result,
                performance_metrics=performance_metrics,
                processing_time=total_processing_time,
                frames_processed=extraction_metadata.get('frames_extracted', 0),
                frames_skipped=extraction_metadata.get('frames_skipped', 0),
                optimization_applied=optimization_result.get('optimization_applied')
            )
            
        except Exception as e:
            logger.error(f"Legacy processing failed: {e}")
            return VideoProcessingResult(
                success=False,
                movements=[],
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    def process_frame_sequence_optimized(
        self, 
        frames: List[np.ndarray]
    ) -> VideoProcessingResult:
        """
        Process a sequence of frames with optimization.
        
        Args:
            frames: List of video frames
            
        Returns:
            Processing result with movements and performance metrics
        """
        start_time = time.time()
        
        try:
            # Process frames with optimization
            optimization_result = self.optimizer.optimize_frame_processing_pipeline(
                frames, self.face_detector, self
            )
            
            if not optimization_result['success']:
                return VideoProcessingResult(
                    success=False,
                    movements=[],
                    error=optimization_result.get('error', 'Frame processing failed'),
                    performance_metrics=optimization_result.get('performance_metrics'),
                    processing_time=time.time() - start_time
                )
            
            # Extract movements from successful results
            successful_results = optimization_result.get('successful_results', [])
            movements = [result.get('movement_data') for result in successful_results if result.get('movement_data')]
            
            return VideoProcessingResult(
                success=True,
                movements=movements,
                performance_metrics=optimization_result.get('performance_metrics'),
                processing_time=time.time() - start_time,
                frames_processed=len(frames)
            )
            
        except Exception as e:
            logger.error(f"Optimized frame sequence processing failed: {e}")
            return VideoProcessingResult(
                success=False,
                movements=[],
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    def _detect_faces_in_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect faces in a single frame (optimized)."""
        try:
            # Use optimized face detection
            detection_result = self.face_detector.detect_faces(frame)
            
            if detection_result['success'] and detection_result['face_count'] == 1:
                # Extract face landmarks for movement analysis
                face = detection_result['faces'][0]
                landmarks = self._extract_face_landmarks(face)
                
                return {
                    'success': True,
                    'frame': frame,
                    'face_detection': detection_result,
                    'landmarks': landmarks,
                    'timestamp': time.time()
                }
            else:
                return {
                    'success': False,
                    'error': 'No single face detected',
                    'face_detection': detection_result
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _analyze_movement_in_frame(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze movement in frame data (optimized)."""
        if not frame_data.get('success'):
            return frame_data
        
        try:
            current_landmarks = frame_data['landmarks']
            
            if self.last_face_landmarks is not None:
                # Calculate movement
                movement_data = self._calculate_movement(
                    self.last_face_landmarks, 
                    current_landmarks,
                    frame_data['timestamp']
                )
                
                frame_data['movement_data'] = movement_data
                frame_data['has_movement'] = movement_data is not None
            else:
                frame_data['movement_data'] = None
                frame_data['has_movement'] = False
            
            # Update landmarks for next frame
            self.last_face_landmarks = current_landmarks
            
            return frame_data
            
        except Exception as e:
            frame_data['success'] = False
            frame_data['error'] = str(e)
            return frame_data
    
    def _enhance_movement_confidence(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance movement confidence using advanced scoring (optimized)."""
        if not frame_data.get('success') or not frame_data.get('has_movement'):
            return frame_data
        
        try:
            movement_data = frame_data['movement_data']
            
            # Calculate comprehensive confidence
            enhanced_movement = self.confidence_scorer.calculate_comprehensive_confidence(
                movement_data,
                frame_context=self._extract_frame_context(frame_data['frame']),
                historical_movements=self.movement_history
            )
            
            # Update movement history
            self.movement_history.append(movement_data)
            if len(self.movement_history) > 20:  # Keep last 20 movements
                self.movement_history.pop(0)
            
            # Update frame data with enhanced movement
            frame_data['enhanced_movement'] = enhanced_movement
            frame_data['movement_data'] = {
                'direction': enhanced_movement.direction,
                'confidence': enhanced_movement.confidence,
                'magnitude': enhanced_movement.magnitude,
                'timestamp': enhanced_movement.timestamp,
                'start_position': enhanced_movement.start_position,
                'end_position': enhanced_movement.end_position
            }
            
            return frame_data
            
        except Exception as e:
            logger.warning(f"Movement confidence enhancement failed: {e}")
            return frame_data
    
    def _extract_face_landmarks(self, face_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract face landmarks from face detection result."""
        try:
            keypoints = face_data.get('keypoints', {})
            bbox = face_data.get('bbox', [0, 0, 0, 0])
            
            # Calculate face center from bbox
            x, y, w, h = bbox
            face_center = (x + w/2, y + h/2)
            
            # Use keypoints if available, otherwise use face center
            landmarks = {
                'nose': keypoints.get('nose', face_center),
                'left_eye': keypoints.get('left_eye', (x + w*0.3, y + h*0.4)),
                'right_eye': keypoints.get('right_eye', (x + w*0.7, y + h*0.4)),
                'left_mouth': keypoints.get('mouth_left', (x + w*0.3, y + h*0.7)),
                'right_mouth': keypoints.get('mouth_right', (x + w*0.7, y + h*0.7)),
                'face_center': face_center,
                'bbox': bbox
            }
            
            return landmarks
            
        except Exception as e:
            logger.warning(f"Landmark extraction failed: {e}")
            return {'face_center': (0, 0), 'bbox': [0, 0, 0, 0]}
    
    def _calculate_movement(
        self, 
        prev_landmarks: Dict[str, Any], 
        curr_landmarks: Dict[str, Any],
        timestamp: float
    ) -> Optional[Dict[str, Any]]:
        """Calculate movement between two sets of landmarks."""
        try:
            # Use face center for movement calculation
            prev_center = prev_landmarks['face_center']
            curr_center = curr_landmarks['face_center']
            
            # Calculate movement vector
            dx = curr_center[0] - prev_center[0]
            dy = curr_center[1] - prev_center[1]
            
            # Calculate magnitude
            magnitude = np.sqrt(dx**2 + dy**2)
            
            # Determine direction (with threshold)
            direction_threshold = 5.0  # pixels
            
            if magnitude < direction_threshold:
                return None  # No significant movement
            
            # Determine primary direction
            if abs(dx) > abs(dy):
                direction = 'right' if dx > 0 else 'left'
            else:
                direction = 'down' if dy > 0 else 'up'
            
            # Calculate basic confidence based on magnitude
            confidence = min(magnitude / 50.0, 1.0)  # Normalize to 0-1
            
            return {
                'direction': direction,
                'magnitude': magnitude,
                'confidence': confidence,
                'dx_pixels': dx,
                'dy_pixels': dy,
                'dx_norm': dx / magnitude if magnitude > 0 else 0,
                'dy_norm': dy / magnitude if magnitude > 0 else 0,
                'timestamp': timestamp,
                'start_position': prev_center,
                'end_position': curr_center,
                'frame_indices': (0, 1)  # Placeholder
            }
            
        except Exception as e:
            logger.warning(f"Movement calculation failed: {e}")
            return None
    
    def _extract_frame_context(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extract frame quality context for confidence scoring."""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
            
            # Calculate quality metrics
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Calculate sharpness using Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            
            return {
                'brightness': brightness,
                'contrast': contrast,
                'sharpness': sharpness,
                'frame_size': frame.shape[:2]
            }
            
        except Exception as e:
            logger.warning(f"Frame context extraction failed: {e}")
            return {
                'brightness': 128.0,
                'contrast': 50.0,
                'sharpness': 100.0,
                'frame_size': (480, 640)
            }
    
    def _extract_movements_from_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract movement data from processing results."""
        movements = []
        
        for result in results:
            if (result.get('success') and 
                result.get('has_movement') and 
                result.get('movement_data')):
                
                movement_data = result['movement_data']
                
                # Ensure required fields are present
                if all(key in movement_data for key in ['direction', 'confidence', 'magnitude']):
                    movements.append(movement_data)
        
        return movements
    
    def reset_processing_state(self):
        """Reset processing state for new video."""
        self.last_face_landmarks = None
        self.movement_history.clear()
        self.confidence_scorer.reset_temporal_state()
        logger.info("Processing state reset")
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics and performance metrics."""
        return {
            'optimizer_config': {
                'frame_skipping_enabled': self.optimization_config.enable_frame_skipping,
                'target_fps': self.optimization_config.target_fps,
                'max_frames_to_process': self.optimization_config.max_frames_to_process,
                'concurrent_processing_enabled': self.optimization_config.enable_concurrent_processing,
                'memory_optimization_enabled': self.optimization_config.enable_memory_optimization,
                'max_memory_usage_mb': self.optimization_config.max_memory_usage_mb
            },
            'confidence_scorer_stats': self.confidence_scorer.get_confidence_statistics(),
            'movement_history_size': len(self.movement_history)
        }

def create_optimized_processor(
    face_detector: Optional[FaceDetector] = None,
    performance_mode: str = 'balanced',
    use_dlib: bool = True
) -> OptimizedVideoProcessor:
    """
    Create optimized video processor with predefined configuration.
    
    Args:
        face_detector: Face detector instance
        performance_mode: 'performance', 'memory', or 'balanced'
        use_dlib: Use dlib for head movement detection
        
    Returns:
        Configured optimized video processor
    """
    if performance_mode == 'performance':
        config = create_performance_optimized_config()
    elif performance_mode == 'memory':
        config = create_memory_optimized_config()
    else:  # balanced
        config = create_default_optimization_config()
    
    return OptimizedVideoProcessor(
        face_detector=face_detector,
        optimization_config=config,
        use_performance_mode=(performance_mode == 'performance'),
        use_dlib=use_dlib
    )

async def process_video_async(
    video_path: str,
    expected_sequence: Optional[List[str]] = None,
    performance_mode: str = 'balanced',
    use_dlib: bool = True
) -> VideoProcessingResult:
    """
    Asynchronously process video for liveness detection.
    
    Args:
        video_path: Path to video file
        expected_sequence: Expected movement sequence
        performance_mode: Performance optimization mode
        use_dlib: Use dlib for head movement detection
        
    Returns:
        Video processing result
    """
    processor = create_optimized_processor(
        performance_mode=performance_mode,
        use_dlib=use_dlib
    )
    
    # Run in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, 
        processor.process_video_for_liveness,
        video_path,
        expected_sequence
    )
    
    return result