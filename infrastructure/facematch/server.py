"""
Face Recognition API Server
Production-grade FastAPI server for face matching, liveness detection, and recognition.
"""

import asyncio
import logging
import os
import time
import tempfile
import shutil
import json
from contextlib import asynccontextmanager
from typing import Dict, Optional, Any, List, Tuple
import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field, field_validator

from .face_detection import FaceDetector, validate_image_quality
from .liveness import LivenessDetector, validate_liveness_model
from .facematch import FaceRecognizer, validate_arcface_model, adaptive_threshold
from .image_utils import load_image_pair, validate_image_input
from .simple_liveness import initialize_simple_liveness_detector, get_simple_liveness_detector
from .unified_liveness_detector import (
    UnifiedLivenessDetector, 
    LivenessMode, 
    LivenessResult, 
    VideoLivenessResult,
    get_unified_liveness_detector,
    initialize_unified_liveness_detector
)
# Advanced head pose removed - was causing startup issues
from .redis_manager import initialize_session_manager, get_session_manager
from .optimized_video_processor import create_optimized_processor, process_video_async
from .concurrent_video_manager import get_global_video_manager, initialize_video_manager
from .face_comparison_integration import (
    FaceComparisonIntegrator,
    initialize_face_comparison_integrator,
    get_face_comparison_integrator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instances
face_detector = None
liveness_detector = None
face_recognizer = None
simple_liveness_detector = None
unified_liveness_detector = None
face_comparison_integrator = None

def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

# Configuration
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.6'))
MAX_IMAGE_SIZE = int(os.getenv('MAX_IMAGE_SIZE', str(10 * 1024 * 1024)))  # 10MB
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '30'))
MAX_IMAGE_DIMENSION = int(os.getenv('MAX_IMAGE_DIMENSION', '1024'))

class FaceMatchRequest(BaseModel):
    """Request model for face matching."""
    image1: str = Field(..., description="First image (URL or base64)")
    image2: str = Field(..., description="Second image (URL or base64)")
    threshold: Optional[float] = Field(SIMILARITY_THRESHOLD, ge=0.0, le=1.0, 
                                     description="Similarity threshold for match decision")
    verbose: Optional[bool] = Field(False, description="If true, return detailed analysis. If false, return minimal result.")
    
    @field_validator('image1', 'image2')
    @classmethod
    def validate_image_sources(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Image must be a non-empty string")
        return v.strip()

def round_scores_in_response(obj: Any) -> Any:
    """Round numerical scores to 2 decimal places for API response only."""
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            # Round specific score fields to 2 decimal places
            score_fields = [
                'similarity_score', 'confidence', 'liveness_score', 
                'threshold', 'detection_confidence', 'face_confidence',
                'quality_score', 'embedding_confidence'
            ]
            if key in score_fields and isinstance(value, (int, float)):
                result[key] = round(float(value), 2)
            else:
                result[key] = round_scores_in_response(value)
        return result
    elif isinstance(obj, list):
        return [round_scores_in_response(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(round_scores_in_response(item) for item in obj)
    else:
        return obj

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: float
    models: Dict[str, bool]
    version: str = "1.0.0"
    details: Optional[Dict] = None

class FaceMatchResponse(BaseModel):
    """Response model for face matching."""
    success: bool
    match: Optional[bool] = None
    similarity_score: Optional[float] = None
    confidence: Optional[float] = None
    threshold: Optional[float] = None
    
    # Detailed results
    liveness_results: Optional[Dict] = None
    face_detection_results: Optional[Dict] = None
    recognition_results: Optional[Dict] = None
    
    # Performance metrics
    processing_time: Optional[float] = None
    image_loading_time: Optional[float] = None
    
    # Error information
    error: Optional[str] = None
    error_details: Optional[Dict] = None

class LivenessChallengeUrlResponse(BaseModel):
    """Response model for liveness challenge URL creation."""
    success: bool
    challenge_url: Optional[str] = None
    session_id: Optional[str] = None
    expires_in: Optional[int] = None  # seconds until expiration
    error: Optional[str] = None

class LivenessInitiateResponse(BaseModel):
    """Response model for liveness challenge initiation with screen data."""
    success: bool
    session_id: Optional[str] = None
    sequence: Optional[List[int]] = None
    area_duration: Optional[float] = None
    total_duration: Optional[float] = None
    screen_areas: Optional[List[Dict]] = None  # Real screen coordinates
    instructions: Optional[str] = None
    error: Optional[str] = None

class LivenessSubmitResponse(BaseModel):
    """Response model for liveness challenge submission."""
    success: bool
    session_id: Optional[str] = None
    result: Optional[str] = None  # "pass" or "fail"
    overall_accuracy: Optional[float] = None
    sequence_results: Optional[List[Dict]] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None

class ImageLivenessRequest(BaseModel):
    """Request model for image liveness check."""
    image: str = Field(..., description="Image URL or base64 encoded image")
    
    @field_validator('image')
    @classmethod
    def validate_image(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Image field cannot be empty')
        return v.strip()

class ImageLivenessResponse(BaseModel):
    """Response model for image liveness check."""
    success: bool
    passed: bool
    liveness_score: float
    face_detected: bool
    processing_time: Optional[float] = None
    anti_spoofing: Optional[Dict] = None
    error: Optional[str] = None

class FaceComparisonRequest(BaseModel):
    """Request model for face comparison."""
    image1: str = Field(..., description="First image (URL or base64)")
    image2: str = Field(..., description="Second image (URL or base64)")
    threshold: Optional[float] = Field(0.6, ge=0.0, le=1.0, description="Similarity threshold (0.0-1.0)")
    
    @field_validator('image1', 'image2')
    @classmethod
    def validate_images(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Image fields cannot be empty')
        return v.strip()

class BatchFaceComparisonRequest(BaseModel):
    """Request model for batch face comparison."""
    image_pairs: List[Tuple[str, str]] = Field(..., description="List of (image1, image2) pairs")
    threshold: Optional[float] = Field(0.6, ge=0.0, le=1.0, description="Similarity threshold (0.0-1.0)")
    
    @field_validator('image_pairs')
    @classmethod
    def validate_image_pairs(cls, v):
        if not v or len(v) == 0:
            raise ValueError('Image pairs list cannot be empty')
        if len(v) > 10:  # Limit batch size for performance
            raise ValueError('Maximum 10 image pairs allowed per batch')
        return v

class FaceComparisonResponse(BaseModel):
    """Response model for face comparison."""
    success: bool
    match: bool
    similarity_score: float
    threshold: float
    confidence: str
    processing_time: Optional[float] = None
    error: Optional[str] = None

class BatchFaceComparisonResponse(BaseModel):
    """Response model for batch face comparison."""
    success: bool
    results: List[FaceComparisonResponse]
    total_pairs: int
    successful_comparisons: int
    failed_comparisons: int
    total_processing_time: float
    average_processing_time: float
    error: Optional[str] = None

class VideoChallengeWithComparisonRequest(BaseModel):
    """Request model for video challenge with optional face comparison."""
    reference_image: Optional[str] = Field(None, description="Reference image for face comparison (URL or base64)")
    enable_face_comparison: Optional[bool] = Field(False, description="Enable face comparison with reference image")
    third_party_id: Optional[str] = Field(None, description="3rd party identifier for result tracking")
    
    @field_validator('reference_image')
    @classmethod
    def validate_reference_image(cls, v):
        if v is not None and len(v.strip()) == 0:
            raise ValueError('Reference image cannot be empty if provided')
        return v.strip() if v else None
    
    @field_validator('third_party_id')
    @classmethod
    def validate_third_party_id(cls, v):
        if v is not None and len(v.strip()) == 0:
            raise ValueError('Third party ID cannot be empty if provided')
        return v.strip() if v else None

class LivenessResultResponse(BaseModel):
    """Response model for fetching liveness results."""
    success: bool
    found: bool
    session_id: Optional[str] = None
    third_party_id: Optional[str] = None
    result: Optional[str] = None  # "pass" or "fail"
    passed: Optional[bool] = None
    challenge_type: Optional[str] = None
    processing_time: Optional[float] = None
    stored_at: Optional[float] = None
    expires_at: Optional[float] = None
    details: Optional[Dict] = None
    error: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources."""
    global face_detector, liveness_detector, face_recognizer, simple_liveness_detector, unified_liveness_detector
    
    logger.info("Initializing Face Recognition API with Simple Liveness...")
    
    try:
        # Model paths
        models_dir = os.path.join(os.path.dirname(__file__), '../../models')
        mtcnn_path = os.path.join(models_dir, 'mtcnn.pb')
        arcface_path = os.path.join(models_dir, 'arcface_resnet50.onnx')
        crmnet_path = os.path.join(models_dir, 'crmnet.onnx')
        
        # Validate model files
        model_validations = {}
        
        # Initialize face detector (MTCNN)
        try:
            face_detector = FaceDetector()
            model_validations['mtcnn'] = True
            logger.info("MTCNN face detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MTCNN: {e}")
            model_validations['mtcnn'] = False
        
        # Initialize liveness detector (CRMNET)
        if os.path.exists(crmnet_path):
            validation = validate_liveness_model(crmnet_path)
            if validation['valid']:
                try:
                    liveness_detector = LivenessDetector(crmnet_path)
                    model_validations['crmnet'] = True
                    logger.info("CRMNET liveness detector initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize CRMNET: {e}")
                    model_validations['crmnet'] = False
            else:
                logger.error(f"CRMNET validation failed: {validation['error']}")
                model_validations['crmnet'] = False
        else:
            logger.warning(f"CRMNET model not found at: {crmnet_path}")
            model_validations['crmnet'] = False
        
        # Initialize face recognizer (ArcFace)
        if os.path.exists(arcface_path):
            validation = validate_arcface_model(arcface_path)
            if validation['valid']:
                try:
                    face_recognizer = FaceRecognizer(arcface_path)
                    model_validations['arcface'] = True
                    logger.info("ArcFace face recognizer initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize ArcFace: {e}")
                    model_validations['arcface'] = False
            else:
                logger.error(f"ArcFace validation failed: {validation['error']}")
                model_validations['arcface'] = False
        else:
            logger.warning(f"ArcFace model not found at: {arcface_path}")
            model_validations['arcface'] = False
        
        # Initialize simple liveness detector
        try:
            simple_liveness_detector = initialize_simple_liveness_detector()
            model_validations['simple_liveness'] = True
            logger.info("Simple liveness detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize simple liveness detector: {e}")
            model_validations['simple_liveness'] = False
        
        # Initialize unified liveness detector
        try:
            unified_liveness_detector = initialize_unified_liveness_detector()
            model_validations['unified_liveness'] = True
            logger.info("Unified liveness detector initialized successfully")
            
            # Log available modes
            available_modes = unified_liveness_detector.get_available_modes()
            logger.info(f"Available liveness modes: {[mode.value for mode in available_modes]}")
        except Exception as e:
            logger.error(f"Failed to initialize unified liveness detector: {e}")
            model_validations['unified_liveness'] = False
        
        # Initialize advanced 6DoF head pose estimator (temporarily disabled)
        # try:
        #     initialize_advanced_head_pose_estimator()
        #     model_validations['advanced_head_pose'] = True
        #     logger.info("Advanced 6DoF Head Pose Estimator initialized successfully")
        # except Exception as e:
        #     logger.error(f"Failed to initialize advanced head pose estimator: {e}")
        #     model_validations['advanced_head_pose'] = False
        model_validations['advanced_head_pose'] = False  # Temporarily disabled
        logger.info("Advanced 6DoF Head Pose Estimator temporarily disabled, using legacy method")
        
        # Initialize Redis session manager
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            session_ttl = int(os.getenv('SESSION_TTL', '300'))  # 5 minutes
            initialize_session_manager(redis_url, session_ttl)
            logger.info("Redis session manager initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis session manager: {e}")
        
        # Initialize performance-optimized video processing manager
        try:
            max_concurrent_videos = int(os.getenv('MAX_CONCURRENT_VIDEOS', '4'))
            max_queue_size = int(os.getenv('MAX_QUEUE_SIZE', '20'))
            memory_limit_mb = float(os.getenv('MEMORY_LIMIT_MB', '4096'))
            
            initialize_video_manager(
                max_concurrent_videos=max_concurrent_videos,
                max_queue_size=max_queue_size,
                memory_limit_mb=memory_limit_mb
            )
            logger.info(f"Concurrent video manager initialized: {max_concurrent_videos} workers, "
                       f"{max_queue_size} queue size, {memory_limit_mb}MB memory limit")
        except Exception as e:
            logger.warning(f"Failed to initialize concurrent video manager: {e}")
        
        # Initialize face comparison integrator
        try:
            global face_comparison_integrator
            face_comparison_integrator = initialize_face_comparison_integrator(
                face_recognizer=face_recognizer,
                unified_detector=unified_liveness_detector
            )
            logger.info("Face comparison integrator initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize face comparison integrator: {e}")
        
        # Log initialization summary
        initialized_models = sum(model_validations.values())
        total_models = len(model_validations)
        logger.info(f"Model initialization complete: {initialized_models}/{total_models} models loaded")
        
        if initialized_models == 0:
            logger.error("No models could be initialized! Please check model files.")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    finally:
        logger.info("Shutting down Face Recognition API...")

# Create FastAPI app
app = FastAPI(
    title="Face Recognition API",
    description="Production-grade face recognition system with liveness detection",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/health", response_model=HealthResponse)
async def health_check(verbose: bool = False):
    """Health check endpoint with optional verbose details."""
    models_status = {
        'mtcnn': face_detector is not None,
        'crmnet': liveness_detector is not None,
        'arcface': face_recognizer is not None,
        'simple_liveness': get_simple_liveness_detector() is not None,
        'unified_liveness': unified_liveness_detector is not None
    }
    
    # Add unified liveness detector status details
    if unified_liveness_detector:
        models_status['unified_liveness_modes'] = unified_liveness_detector.get_detector_status()
    
    response_data = {
        'status': "healthy" if any(models_status.values()) else "degraded",
        'timestamp': time.time(),
        'models': models_status,
        'version': "1.0.0"
    }
    
    # Add detailed model info if verbose=True
    if verbose:
        response_data['details'] = {
            'total_models': len(models_status),
            'loaded_models': sum(models_status.values()),
            'model_paths': {
                'mtcnn': 'Built-in MTCNN',
                'crmnet': '../../models/crmnet.onnx',
                'arcface': '../../models/arcface_resnet50.onnx',
                'eye_tracker': 'MediaPipe FaceMesh'
            },
            'session_stats': get_session_manager().get_session_stats() if get_session_manager() else None
        }
    
    return HealthResponse(**response_data)

@app.post("/match", response_model=FaceMatchResponse)
async def match_faces(request: FaceMatchRequest):
    """
    Compare two face images for similarity with liveness detection.
    
    This endpoint performs comprehensive face analysis including:
    - Face detection (single face validation)
    - Liveness detection (anti-spoofing)
    - Face recognition and similarity calculation
    """
    start_time = time.time()
    
    try:
        # Validate required models
        if not face_detector:
            raise HTTPException(
                status_code=503, 
                detail="Face detection model not available"
            )
        
        if not face_recognizer:
            raise HTTPException(
                status_code=503, 
                detail="Face recognition model not available"
            )
        
        # Validate input sources
        validation1 = validate_image_input(request.image1)
        validation2 = validate_image_input(request.image2)
        
        if not validation1['valid']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image1: {validation1['error']}"
            )
        
        if not validation2['valid']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image2: {validation2['error']}"
            )
        
        # Load images concurrently
        logger.info("Loading images...")
        image_load_start = time.time()
        
        image1_result, image2_result = await load_image_pair(
            request.image1, 
            request.image2,
            max_file_size=MAX_IMAGE_SIZE,
            timeout=REQUEST_TIMEOUT
        )
        
        image_loading_time = time.time() - image_load_start
        
        # Check image loading results
        if not image1_result['success']:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load image1: {image1_result['error']}"
            )
        
        if not image2_result['success']:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load image2: {image2_result['error']}"
            )
        
        # Resize images if needed
        image1 = resize_image_if_needed(image1_result['image'], MAX_IMAGE_DIMENSION)
        image2 = resize_image_if_needed(image2_result['image'], MAX_IMAGE_DIMENSION)
        
        # Validate image quality
        quality1 = validate_image_quality(image1)
        quality2 = validate_image_quality(image2)
        
        if not quality1['valid']:
            raise HTTPException(
                status_code=400,
                detail=f"Image1 quality issue: {quality1['error']}"
            )
        
        if not quality2['valid']:
            raise HTTPException(
                status_code=400,
                detail=f"Image2 quality issue: {quality2['error']}"
            )
        
        # Detect faces
        logger.info("Detecting faces...")
        detection1 = face_detector.detect_faces(image1)
        detection2 = face_detector.detect_faces(image2)
        
        # Validate single face requirement
        validation1 = face_detector.validate_single_face(detection1)
        validation2 = face_detector.validate_single_face(detection2)
        
        if not validation1['valid']:
            error_data = {
                'success': False,
                'error': f"Image1 face detection issue: {validation1['error']}",
                'processing_time': float(time.time() - start_time),
                'image_loading_time': float(image_loading_time)
            }
            
            # Add detailed results only if verbose=True
            if request.verbose:
                error_data['face_detection_results'] = convert_numpy_types({
                    'image1': detection1,
                    'image2': detection2
                })
            
            # Round scores in response
            error_data = round_scores_in_response(error_data)
            return FaceMatchResponse(**error_data)
        
        if not validation2['valid']:
            error_data = {
                'success': False,
                'error': f"Image2 face detection issue: {validation2['error']}",
                'processing_time': float(time.time() - start_time),
                'image_loading_time': float(image_loading_time)
            }
            
            # Add detailed results only if verbose=True
            if request.verbose:
                error_data['face_detection_results'] = convert_numpy_types({
                    'image1': detection1,
                    'image2': detection2
                })
            
            # Round scores in response
            error_data = round_scores_in_response(error_data)
            return FaceMatchResponse(**error_data)
        
        # Extract face regions
        face1 = face_detector.extract_face_region(image1, validation1['bbox'])
        face2 = face_detector.extract_face_region(image2, validation2['bbox'])
        
        # Align faces
        face1_aligned = face_detector.align_face(face1, validation1['keypoints'])
        face2_aligned = face_detector.align_face(face2, validation2['keypoints'])
        
        # Perform liveness detection if available
        liveness_results = None
        if liveness_detector:
            logger.info("Performing liveness detection...")
            try:
                liveness1 = liveness_detector.detect_liveness_with_quality_check(face1_aligned)
                liveness2 = liveness_detector.detect_liveness_with_quality_check(face2_aligned)
                
                liveness_results = {
                    'image1': liveness1,
                    'image2': liveness2
                }
                
                # Check liveness results
                if liveness1['success'] and not liveness1['is_live']:
                    error_data = {
                        'success': False,
                        'error': f"Image1 liveness check failed: Not a live face (score: {liveness1['liveness_score']:.3f})",
                        'processing_time': float(time.time() - start_time),
                        'image_loading_time': float(image_loading_time)
                    }
                    
                    # Add detailed results only if verbose=True
                    if request.verbose:
                        error_data.update({
                            'liveness_results': convert_numpy_types(liveness_results),
                            'face_detection_results': convert_numpy_types({
                                'image1': detection1,
                                'image2': detection2
                            })
                        })
                    
                    # Round scores in response
                    error_data = round_scores_in_response(error_data)
                    return FaceMatchResponse(**error_data)
                
                if liveness2['success'] and not liveness2['is_live']:
                    error_data = {
                        'success': False,
                        'error': f"Image2 liveness check failed: Not a live face (score: {liveness2['liveness_score']:.3f})",
                        'processing_time': float(time.time() - start_time),
                        'image_loading_time': float(image_loading_time)
                    }
                    
                    # Add detailed results only if verbose=True
                    if request.verbose:
                        error_data.update({
                            'liveness_results': convert_numpy_types(liveness_results),
                            'face_detection_results': convert_numpy_types({
                                'image1': detection1,
                                'image2': detection2
                            })
                        })
                    
                    # Round scores in response
                    error_data = round_scores_in_response(error_data)
                    return FaceMatchResponse(**error_data)
                
            except Exception as e:
                logger.warning(f"Liveness detection failed: {e}")
                liveness_results = {
                    'error': str(e),
                    'image1': {'success': False, 'error': str(e)},
                    'image2': {'success': False, 'error': str(e)}
                }
        
        # Perform face recognition
        logger.info("Performing face recognition...")
        recognition_result = face_recognizer.compare_faces(
            face1_aligned, 
            face2_aligned, 
            threshold=request.threshold
        )
        
        if not recognition_result['success']:
            error_data = {
                'success': False,
                'error': f"Face recognition failed: {recognition_result['error']}",
                'processing_time': float(time.time() - start_time),
                'image_loading_time': float(image_loading_time)
            }
            
            # Add detailed results only if verbose=True
            if request.verbose:
                error_data.update({
                    'liveness_results': convert_numpy_types(liveness_results),
                    'face_detection_results': convert_numpy_types({
                        'image1': detection1,
                        'image2': detection2
                    }),
                    'recognition_results': convert_numpy_types(recognition_result)
                })
            
            # Round scores in response
            error_data = round_scores_in_response(error_data)
            return FaceMatchResponse(**error_data)
        
        # Calculate final processing time
        total_processing_time = time.time() - start_time
        
        # Prepare response - convert numpy types to Python types (keep full precision during processing)
        base_response_data = {
            'success': True,
            'match': recognition_result['match'],
            'similarity_score': float(recognition_result['similarity_score']),
            'confidence': float(recognition_result['confidence']),
            'threshold': float(request.threshold),
            'processing_time': float(total_processing_time),
            'image_loading_time': float(image_loading_time)
        }
        
        # Add detailed results only if verbose=True
        if request.verbose:
            base_response_data.update({
                'liveness_results': convert_numpy_types(liveness_results),
                'face_detection_results': convert_numpy_types({
                    'image1': {
                        'face_count': detection1['face_count'],
                        'confidence': validation1['confidence'],
                        'detection_time': detection1['detection_time']
                    },
                    'image2': {
                        'face_count': detection2['face_count'],
                        'confidence': validation2['confidence'],
                        'detection_time': detection2['detection_time']
                    }
                }),
                'recognition_results': convert_numpy_types({
                    'distance_metrics': recognition_result['distance_metrics'],
                    'embedding_quality': recognition_result['embedding_quality'],
                    'timing': recognition_result['timing']
                })
            })
        
        # Round scores to 2 decimal places only in the final response
        response_data = round_scores_in_response(base_response_data)
        response = FaceMatchResponse(**response_data)
        
        # Log result (keep full precision in logs for debugging)
        match_status = "MATCH" if recognition_result['match'] else "NO MATCH"
        logger.info(f"Face matching completed: {match_status} "
                   f"(similarity: {recognition_result['similarity_score']:.6f}, "
                   f"threshold: {request.threshold}, verbose: {request.verbose}) in {total_processing_time:.3f}s")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in face matching: {e}")
        error_data = {
            'success': False,
            'error': f"Internal server error: {str(e)}",
            'processing_time': time.time() - start_time
        }
        
        # Round scores in response (even for errors)
        error_data = round_scores_in_response(error_data)
        return FaceMatchResponse(**error_data)

@app.post("/process-video-optimized")
async def process_video_optimized(
    video: UploadFile = File(...),
    expected_sequence: Optional[str] = Form(None),
    performance_mode: Optional[str] = Form("balanced")
):
    """
    Process video for liveness detection with performance optimizations.
    
    This endpoint uses advanced performance optimizations including:
    - Optimized frame processing pipeline
    - Efficient memory management
    - Concurrent processing capabilities
    - Performance monitoring
    """
    start_time = time.time()
    
    try:
        # Validate video file
        if not video.filename or not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(
                status_code=400,
                detail="Invalid video format. Supported formats: mp4, avi, mov, mkv"
            )
        
        # Parse expected sequence
        sequence = None
        if expected_sequence:
            try:
                import json
                sequence = json.loads(expected_sequence)
                if not isinstance(sequence, list):
                    raise ValueError("Expected sequence must be a list")
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid expected_sequence format: {e}"
                )
        
        # Validate performance mode
        if performance_mode not in ['performance', 'memory', 'balanced']:
            performance_mode = 'balanced'
        
        # Save uploaded video to temporary file
        temp_video_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                temp_video_path = temp_file.name
                shutil.copyfileobj(video.file, temp_file)
            
            # Process video with optimizations
            result = await process_video_async(
                video_path=temp_video_path,
                expected_sequence=sequence,
                performance_mode=performance_mode
            )
            
            # Prepare response
            response_data = {
                'success': result.success,
                'processing_time': result.processing_time,
                'frames_processed': result.frames_processed,
                'frames_skipped': result.frames_skipped,
                'performance_mode': performance_mode
            }
            
            if result.success:
                response_data.update({
                    'movements_detected': len(result.movements),
                    'movements': result.movements,
                    'validation_result': result.validation_result,
                    'performance_metrics': result.performance_metrics,
                    'optimization_applied': result.optimization_applied
                })
            else:
                response_data['error'] = result.error
            
            # Convert numpy types for JSON serialization
            response_data = convert_numpy_types(response_data)
            
            logger.info(f"Optimized video processing completed: {result.success} "
                       f"({len(result.movements) if result.movements else 0} movements) "
                       f"in {result.processing_time:.3f}s")
            
            return JSONResponse(content=response_data)
            
        finally:
            # Clean up temporary file
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.unlink(temp_video_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary video file: {e}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Optimized video processing failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
        )

@app.post("/submit-video-concurrent")
async def submit_video_concurrent(
    video: UploadFile = File(...),
    expected_sequence: Optional[str] = Form(None),
    priority: Optional[int] = Form(2)
):
    """
    Submit video for concurrent processing with queue management.
    
    This endpoint submits video to a concurrent processing queue and returns
    immediately with a request ID for tracking.
    """
    try:
        # Validate video file
        if not video.filename or not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(
                status_code=400,
                detail="Invalid video format. Supported formats: mp4, avi, mov, mkv"
            )
        
        # Parse expected sequence
        sequence = None
        if expected_sequence:
            try:
                import json
                sequence = json.loads(expected_sequence)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid expected_sequence format: {e}"
                )
        
        # Validate priority
        if priority not in [1, 2, 3]:
            priority = 2
        
        # Save uploaded video to temporary file
        temp_video_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                temp_video_path = temp_file.name
                shutil.copyfileobj(video.file, temp_file)
            
            # Submit to concurrent processing manager
            video_manager = get_global_video_manager()
            request_id = video_manager.submit_video_processing(
                video_path=temp_video_path,
                expected_sequence=sequence,
                priority=priority
            )
            
            return JSONResponse(content={
                'success': True,
                'request_id': request_id,
                'status': 'queued',
                'priority': priority,
                'message': 'Video submitted for processing'
            })
            
        except ValueError as e:
            # Queue full or memory limit exceeded
            if temp_video_path and os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
            
            raise HTTPException(
                status_code=503,
                detail=str(e)
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video submission failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Video submission failed: {e}"
        )

@app.get("/video-processing-status/{request_id}")
async def get_video_processing_status(request_id: str):
    """Get status of a video processing request."""
    try:
        video_manager = get_global_video_manager()
        status = video_manager.get_request_status(request_id)
        
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"Failed to get processing status: {e}")
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'error': str(e)
            }
        )

@app.get("/video-processing-result/{request_id}")
async def get_video_processing_result(request_id: str):
    """Get result of a completed video processing request."""
    try:
        video_manager = get_global_video_manager()
        result = video_manager.get_result(request_id)
        
        if result is None:
            return JSONResponse(
                status_code=404,
                content={
                    'success': False,
                    'error': 'Result not found or processing not completed'
                }
            )
        
        # Convert result to response format
        response_data = {
            'success': result.success,
            'processing_time': result.processing_time,
            'frames_processed': result.frames_processed,
            'frames_skipped': result.frames_skipped
        }
        
        if result.success:
            response_data.update({
                'movements_detected': len(result.movements),
                'movements': result.movements,
                'validation_result': result.validation_result,
                'performance_metrics': result.performance_metrics,
                'optimization_applied': result.optimization_applied
            })
        else:
            response_data['error'] = result.error
        
        # Convert numpy types for JSON serialization
        response_data = convert_numpy_types(response_data)
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Failed to get processing result: {e}")
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'error': str(e)
            }
        )

@app.get("/processing-statistics")
async def get_processing_statistics():
    """Get current processing statistics and performance metrics."""
    try:
        video_manager = get_global_video_manager()
        stats = video_manager.get_processing_statistics()
        
        return JSONResponse(content={
            'success': True,
            'statistics': stats,
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"Failed to get processing statistics: {e}")
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'error': str(e)
            }
        )

@app.post("/create-simple-challenge")
async def create_simple_challenge():
    """
    Create a new simple liveness challenge.
    Randomly chooses between smile and head movement challenges.
    """
    try:
        # Get unified liveness detector
        detector = get_unified_liveness_detector()
        if not detector:
            raise HTTPException(
                status_code=503,
                detail="Unified liveness detector not available"
            )
        
        # Generate random challenge using unified interface
        challenge = detector.generate_challenge()
        
        # Create simple session (no Redis needed for simple version)
        session_id = f"simple_{int(time.time() * 1000)}"
        
        # Create mobile-friendly challenge URL
        base_url = os.getenv('BASE_URL', 'http://localhost:8000')
        challenge_url = f"{base_url}/static/simple.html?session={session_id}&type={challenge['type']}"
        
        # Add movement sequence to URL for head movement challenges
        if challenge['type'] == 'head_movement' and 'movement_sequence' in challenge:
            import urllib.parse
            sequence_param = urllib.parse.quote(json.dumps(challenge['movement_sequence']))
            direction_duration = challenge.get('direction_duration', 3.5)
            challenge_url += f"&sequence={sequence_param}&direction_duration={direction_duration}"
        
        response_data = {
            'success': True,
            'challenge_url': challenge_url,
            'session_id': session_id,
            'challenge_type': challenge['type'],
            'instruction': challenge['instruction'],
            'description': challenge['description'],
            'duration': challenge['duration']
        }
        
        # Include movement sequence in response for head movement challenges
        if challenge['type'] == 'head_movement' and 'movement_sequence' in challenge:
            response_data['movement_sequence'] = challenge['movement_sequence']
            response_data['direction_duration'] = challenge.get('direction_duration', 3.5)
        
        logger.info(f"Created simple {challenge['type']} challenge: {session_id}")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create simple challenge: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create simple challenge: {str(e)}"
        )

@app.post("/submit-simple-challenge")
async def submit_simple_challenge(
    session_id: str = Form(...),
    challenge_type: str = Form(...),
    video: UploadFile = File(...),
    movement_sequence: str = Form(None),  # JSON string of movement sequence
    third_party_id: str = Form(None)  # Optional 3rd party identifier
):
    """
    Submit video for simple liveness validation.
    Validates either smile or head movement challenge.
    """
    # Performance timing tracking
    request_start_time = time.time()
    video_receive_start_time = None
    video_receive_end_time = None
    processing_start_time = None
    processing_end_time = None
    
    # Log request start
    logger.info(f"VIDEO_LIVENESS_REQUEST_START - session_id: {session_id}, challenge_type: {challenge_type}, file_size: {video.size if hasattr(video, 'size') else 'unknown'}")
    
    try:
        # Get unified liveness detector
        detector = get_unified_liveness_detector()
        if not detector:
            raise HTTPException(
                status_code=503,
                detail="Unified liveness detector not available"
            )
        
        # Validate challenge type - Only head_movement for Gateman
        if challenge_type != 'head_movement':
            raise HTTPException(
                status_code=400,
                detail="Invalid challenge type. Only 'head_movement' is supported for Gateman Liveness Check"
            )
        
        # Validate video file
        max_size = 50 * 1024 * 1024  # 50MB
        if not video.content_type or not video.content_type.startswith('video/'):
            raise HTTPException(
                status_code=400,
                detail="Uploaded file must be a video"
            )
        
        # Start video file reception timing
        video_receive_start_time = time.time()
        logger.info(f"VIDEO_FILE_RECEIVE_START - session_id: {session_id}")
        
        content = await video.read()
        video_receive_end_time = time.time()
        video_receive_duration = video_receive_end_time - video_receive_start_time
        
        logger.info(f"VIDEO_FILE_RECEIVE_COMPLETE - session_id: {session_id}, duration: {video_receive_duration:.3f}s, size: {len(content)} bytes")
        
        if len(content) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"Video file too large: {len(content)} bytes (maximum 50MB)"
            )
        
        # Save video to temporary file
        temp_save_start = time.time()
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
            temp_video_path = temp_file.name
            temp_file.write(content)
        temp_save_duration = time.time() - temp_save_start
        
        logger.info(f"VIDEO_FILE_SAVE_COMPLETE - session_id: {session_id}, duration: {temp_save_duration:.3f}s, path: {temp_video_path}")
        
        try:
            # Parse movement sequence if provided
            parsed_movement_sequence = None
            if movement_sequence:
                try:
                    import json
                    parsed_movement_sequence = json.loads(movement_sequence)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid movement sequence JSON: {movement_sequence}")
            
            # Validate challenge
            logger.info(f"Processing {challenge_type} challenge for session {session_id}")
            if challenge_type == 'head_movement' and parsed_movement_sequence:
                logger.info(f"Expected movement sequence: {parsed_movement_sequence}")
            
            # Check if face comparison is enabled for this session
            reference_image = None
            session_manager = get_session_manager()
            if session_manager:
                session_data = session_manager.get_session(session_id)
                if session_data and session_data.get('enable_face_comparison'):
                    reference_image = session_data.get('reference_image')
            
            # Start video processing timing
            processing_start_time = time.time()
            logger.info(f"VIDEO_PROCESSING_START - session_id: {session_id}, challenge_type: {challenge_type}, "
                       f"has_reference_image: {bool(reference_image)}")
            
            # Use face comparison integrator for comprehensive liveness + face comparison
            integrator = get_face_comparison_integrator()
            if integrator and reference_image:
                # Use integrated approach when face comparison is required
                integrated_result = integrator.integrate_with_liveness_detection(
                    video_path=temp_video_path,
                    reference_image=reference_image,
                    movement_sequence=parsed_movement_sequence,
                    session_id=session_id
                )
                
                # Convert integrated result to validation result format
                validation_result = {
                    'success': integrated_result.liveness_success,
                    'passed': integrated_result.overall_passed,
                    'confidence': integrated_result.liveness_score,
                    'liveness_score': integrated_result.liveness_score,
                    'error': integrated_result.error,
                    'detected_sequence': integrated_result.liveness_details.get('detected_sequence') if integrated_result.liveness_details else None,
                    'expected_sequence': integrated_result.liveness_details.get('expected_sequence') if integrated_result.liveness_details else None,
                    'sequence_accuracy': integrated_result.liveness_details.get('sequence_accuracy') if integrated_result.liveness_details else None,
                    'face_image_path': None,  # Will be set from liveness details if available
                    'face_comparison': {
                        'success': integrated_result.face_comparison_success,
                        'match': integrated_result.face_comparison_match,
                        'similarity_score': integrated_result.face_comparison_score,
                        'details': integrated_result.face_comparison_details
                    } if integrated_result.face_comparison_details else None,
                    'details': integrated_result.liveness_details,
                    'movement_details': integrated_result.liveness_details.get('movement_details') if integrated_result.liveness_details else None,
                    'anti_spoofing_details': integrated_result.liveness_details.get('anti_spoofing_details') if integrated_result.liveness_details else None
                }
            else:
                # Use standard unified detector for liveness only
                validation_result = detector.detect_video_liveness(
                    video_path=temp_video_path,
                    movement_sequence=parsed_movement_sequence,
                    session_id=session_id,
                    reference_image=reference_image
                )
                
                # Convert VideoLivenessResult to dict format for compatibility
                validation_result = {
                    'success': validation_result.success,
                    'passed': validation_result.passed,
                    'confidence': validation_result.confidence,
                    'liveness_score': validation_result.liveness_score,
                    'error': validation_result.error,
                    'detected_sequence': validation_result.detected_sequence,
                    'expected_sequence': validation_result.expected_sequence,
                    'sequence_accuracy': validation_result.sequence_accuracy,
                    'face_image_path': validation_result.face_image_path,
                    'face_comparison': validation_result.face_comparison,
                    'details': validation_result.details,
                    'movement_details': validation_result.movement_details,
                    'anti_spoofing_details': validation_result.anti_spoofing_details
                }
            
            processing_end_time = time.time()
            processing_duration = processing_end_time - processing_start_time
            
            logger.info(f"VIDEO_PROCESSING_COMPLETE - session_id: {session_id}, duration: {processing_duration:.3f}s, success: {validation_result.get('success', False)}")
            
            total_time = time.time() - request_start_time
            
            if validation_result['success']:
                result = "pass" if validation_result['passed'] else "fail"
                
                response_data = {
                    'success': True,
                    'session_id': session_id,
                    'result': result,
                    'challenge_type': challenge_type,
                    'processing_time': processing_duration,
                    'total_time': total_time,
                    'details': validation_result
                }
                
                # Log comprehensive performance metrics
                logger.info(f"VIDEO_LIVENESS_COMPLETE - session_id: {session_id}, result: {result}, "
                           f"total_time: {total_time:.3f}s, processing_time: {processing_duration:.3f}s, "
                           f"video_receive_time: {video_receive_duration:.3f}s, "
                           f"file_save_time: {temp_save_duration:.3f}s")
                
                logger.info(f"Simple {challenge_type} challenge {result} for session {session_id}")
                
                # Save result to Redis with 1-hour expiration
                try:
                    if session_manager:
                        session_manager.save_liveness_result(
                            session_id=session_id,
                            result_data=response_data,
                            third_party_id=third_party_id
                        )
                        logger.info(f"Saved liveness result for session {session_id}")
                except Exception as save_error:
                    logger.warning(f"Failed to save result for session {session_id}: {save_error}")
                
                # Clean up Redis session data after completion
                try:
                    if session_manager:
                        session_manager.delete_session(session_id)
                        logger.info(f"Cleaned up session data for {session_id}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup session {session_id}: {cleanup_error}")
                
                return response_data
            else:
                # Save failed result to Redis as well
                failed_response_data = {
                    'success': True,
                    'session_id': session_id,
                    'result': 'fail',
                    'challenge_type': challenge_type,
                    'processing_time': processing_duration,
                    'total_time': total_time,
                    'details': validation_result
                }
                
                # Log failed performance metrics
                logger.warning(f"VIDEO_LIVENESS_FAILED - session_id: {session_id}, "
                              f"total_time: {total_time:.3f}s, processing_time: {processing_duration:.3f}s, "
                              f"error: {validation_result.get('error', 'Unknown error')}")
                
                try:
                    if session_manager:
                        session_manager.save_liveness_result(
                            session_id=session_id,
                            result_data=failed_response_data,
                            third_party_id=third_party_id
                        )
                        logger.info(f"Saved failed liveness result for session {session_id}")
                except Exception as save_error:
                    logger.warning(f"Failed to save failed result for session {session_id}: {save_error}")
                
                # Clean up Redis session data after failure too
                try:
                    if session_manager:
                        session_manager.delete_session(session_id)
                        logger.info(f"Cleaned up session data for failed session {session_id}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup failed session {session_id}: {cleanup_error}")
                
                raise HTTPException(
                    status_code=400,
                    detail=validation_result.get('error', 'Validation failed')
                )
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
        
    except HTTPException:
        # Log HTTP exception timing
        if request_start_time:
            total_time = time.time() - request_start_time
            logger.error(f"VIDEO_LIVENESS_HTTP_ERROR - session_id: {session_id}, total_time: {total_time:.3f}s")
        raise
    except Exception as e:
        # Log general exception timing
        if request_start_time:
            total_time = time.time() - request_start_time
            logger.error(f"VIDEO_LIVENESS_EXCEPTION - session_id: {session_id}, total_time: {total_time:.3f}s, error: {str(e)}")
        
        logger.error(f"Failed to submit simple challenge: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit simple challenge: {str(e)}"
        )

@app.post("/image-liveness-check", response_model=ImageLivenessResponse)
async def image_liveness_check(request: ImageLivenessRequest):
    """
    Perform liveness check on a static image from URL or base64.
    """
    start_time = time.time()
    
    try:
        # Get unified liveness detector
        detector = get_unified_liveness_detector()
        if not detector:
            raise HTTPException(
                status_code=503,
                detail="Unified liveness detector not available"
            )
        
        logger.info(f"Processing image liveness check for image input")
        
        # Perform image liveness check using unified interface
        result = detector.detect_image_liveness(request.image)
        
        processing_time = time.time() - start_time
        
        if result.success:
            return ImageLivenessResponse(
                success=True,
                passed=result.passed,
                liveness_score=result.liveness_score,
                face_detected=result.details.get('face_detected', False) if result.details else False,
                processing_time=processing_time,
                anti_spoofing=result.anti_spoofing_details or {},
                error=result.error
            )
        else:
            return ImageLivenessResponse(
                success=False,
                passed=False,
                liveness_score=result.liveness_score,
                face_detected=False,
                processing_time=processing_time,
                error=result.error or 'Image liveness check failed'
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process image liveness check: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process image liveness check: {str(e)}"
        )

@app.post("/face-comparison", response_model=FaceComparisonResponse)
async def face_comparison(request: FaceComparisonRequest):
    """
    Compare two faces from URLs or base64 images.
    Optimized with concurrent processing for maximum performance.
    """
    start_time = time.time()
    
    try:
        # Get face comparison integrator
        integrator = get_face_comparison_integrator()
        if not integrator:
            logger.error("Face comparison integrator not available")
            raise HTTPException(
                status_code=503,
                detail="Face comparison service not available"
            )
        
        logger.info(f"Processing optimized face comparison with threshold {request.threshold}")
        
        # Perform face comparison using optimized integrator with concurrent processing
        result = integrator.compare_faces_with_validation(
            image1=request.image1,
            image2=request.image2,
            threshold=request.threshold,
            enable_quality_check=True,
            session_id=None  # No session ID for direct face comparison
        )
        
        # Convert integrator result to API response
        if result.success:
            # Log detailed performance metrics
            timing_details = result.details.get('timing', {}) if result.details else {}
            image_load_time = timing_details.get('image_load_time', 0)
            embedding_time = timing_details.get('embedding_time', 0)
            similarity_time = timing_details.get('similarity_time', 0)
            
            logger.info(f"Optimized face comparison completed successfully: "
                       f"match={result.match}, similarity={result.similarity_score:.4f}, "
                       f"confidence={result.confidence}, total_time={result.processing_time:.3f}s, "
                       f"image_load={image_load_time:.3f}s, embedding={embedding_time:.3f}s, "
                       f"similarity={similarity_time:.3f}s")
            
            return FaceComparisonResponse(
                success=True,
                match=result.match,
                similarity_score=result.similarity_score,
                threshold=result.threshold,
                confidence=result.confidence,
                processing_time=result.processing_time
            )
        else:
            logger.warning(f"Face comparison failed: {result.error} (code: {result.error_code})")
            return FaceComparisonResponse(
                success=False,
                match=False,
                similarity_score=result.similarity_score,
                threshold=result.threshold,
                confidence=result.confidence,
                processing_time=result.processing_time,
                error=result.error
            )
            
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Unexpected error in face comparison endpoint: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=500,
            detail=f"Face comparison failed: {str(e)}"
        )

@app.post("/batch-face-comparison", response_model=BatchFaceComparisonResponse)
async def batch_face_comparison(request: BatchFaceComparisonRequest):
    """
    Compare a batch of face image pairs for similarity.
    """
    start_time = time.time()
    
    try:
        # Get face comparison integrator
        integrator = get_face_comparison_integrator()
        if not integrator:
            logger.error("Face comparison integrator not available")
            raise HTTPException(
                status_code=503,
                detail="Face comparison service not available"
            )
        
        logger.info(f"Processing batch face comparison for {len(request.image_pairs)} pairs")
        
        # Prepare image pairs for processing
        processed_pairs = []
        for i, (image1_url, image2_url) in enumerate(request.image_pairs):
            try:
                # Load images concurrently
                image1_result = await load_image_pair(image1_url, None, max_file_size=MAX_IMAGE_SIZE, timeout=REQUEST_TIMEOUT)
                image2_result = await load_image_pair(image2_url, None, max_file_size=MAX_IMAGE_SIZE, timeout=REQUEST_TIMEOUT)

                if not image1_result['success']:
                    processed_pairs.append({
                        'success': False,
                        'error': f"Failed to load image1 for pair {i}: {image1_result['error']}"
                    })
                    continue
                if not image2_result['success']:
                    processed_pairs.append({
                        'success': False,
                        'error': f"Failed to load image2 for pair {i}: {image2_result['error']}"
                    })
                    continue

                # Resize images if needed
                image1 = resize_image_if_needed(image1_result['image'], MAX_IMAGE_DIMENSION)
                image2 = resize_image_if_needed(image2_result['image'], MAX_IMAGE_DIMENSION)

                # Validate image quality
                quality1 = validate_image_quality(image1)
                quality2 = validate_image_quality(image2)

                if not quality1['valid']:
                    processed_pairs.append({
                        'success': False,
                        'error': f"Image1 quality issue for pair {i}: {quality1['error']}"
                    })
                    continue
                if not quality2['valid']:
                    processed_pairs.append({
                        'success': False,
                        'error': f"Image2 quality issue for pair {i}: {quality2['error']}"
                    })
                    continue

                # Detect faces
                detection1 = face_detector.detect_faces(image1)
                detection2 = face_detector.detect_faces(image2)

                # Validate single face requirement
                validation1 = face_detector.validate_single_face(detection1)
                validation2 = face_detector.validate_single_face(detection2)

                if not validation1['valid']:
                    processed_pairs.append({
                        'success': False,
                        'error': f"Image1 face detection issue for pair {i}: {validation1['error']}"
                    })
                    continue
                if not validation2['valid']:
                    processed_pairs.append({
                        'success': False,
                        'error': f"Image2 face detection issue for pair {i}: {validation2['error']}"
                    })
                    continue

                # Extract face regions
                face1 = face_detector.extract_face_region(image1, validation1['bbox'])
                face2 = face_detector.extract_face_region(image2, validation2['bbox'])

                # Align faces
                face1_aligned = face_detector.align_face(face1, validation1['keypoints'])
                face2_aligned = face_detector.align_face(face2, validation2['keypoints'])

                # Perform liveness detection if available
                liveness_results = None
                if liveness_detector:
                    try:
                        liveness1 = liveness_detector.detect_liveness_with_quality_check(face1_aligned)
                        liveness2 = liveness_detector.detect_liveness_with_quality_check(face2_aligned)
                        liveness_results = {
                            'image1': liveness1,
                            'image2': liveness2
                        }
                    except Exception as e:
                        logger.warning(f"Liveness detection failed for pair {i}: {e}")
                        liveness_results = {
                            'error': str(e),
                            'image1': {'success': False, 'error': str(e)},
                            'image2': {'success': False, 'error': str(e)}
                        }

                # Perform face recognition
                recognition_result = face_recognizer.compare_faces(
                    face1_aligned, 
                    face2_aligned, 
                    threshold=request.threshold
                )

                if not recognition_result['success']:
                    processed_pairs.append({
                        'success': False,
                        'error': f"Face recognition failed for pair {i}: {recognition_result['error']}"
                    })
                    continue

                processed_pairs.append({
                    'success': True,
                    'match': recognition_result['match'],
                    'similarity_score': float(recognition_result['similarity_score']),
                    'confidence': float(recognition_result['confidence']),
                    'threshold': float(request.threshold),
                    'processing_time': float(time.time() - start_time), # This is a bit off, should be per pair
                    'image_loading_time': float(image1_result['loading_time'] + image2_result['loading_time'])
                })

            except Exception as e:
                processed_pairs.append({
                    'success': False,
                    'error': f"An unexpected error occurred for pair {i}: {str(e)}"
                })

        # Calculate total processing time and average processing time
        total_processing_time = time.time() - start_time
        average_processing_time = total_processing_time / len(request.image_pairs) if request.image_pairs else 0

        # Prepare final response
        response_data = {
            'success': True,
            'results': processed_pairs,
            'total_pairs': len(request.image_pairs),
            'successful_comparisons': sum(1 for p in processed_pairs if p['success']),
            'failed_comparisons': sum(1 for p in processed_pairs if not p['success']),
            'total_processing_time': total_processing_time,
            'average_processing_time': average_processing_time
        }

        # Convert numpy types for JSON serialization
        response_data = convert_numpy_types(response_data)

        logger.info(f"Batch face comparison completed for {len(request.image_pairs)} pairs. "
                   f"Total processing time: {total_processing_time:.3f}s, "
                   f"Average processing time: {average_processing_time:.3f}s")

        return BatchFaceComparisonResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch face comparison failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'error': str(e),
                'total_processing_time': time.time() - start_time
            }
        )

@app.post("/create-challenge-with-comparison")
async def create_challenge_with_comparison(request: VideoChallengeWithComparisonRequest):
    """
    Create a video liveness challenge with optional face comparison.
    """
    try:
        detector = get_unified_liveness_detector()
        if not detector:
            raise HTTPException(
                status_code=503,
                detail="Unified liveness detector not available"
            )
        
        # Generate challenge using unified interface
        challenge = detector.generate_challenge()
        session_id = f"simple_{int(time.time() * 1000)}"
        base_url = os.getenv('BASE_URL', 'http://localhost:8000')
        
        # Build challenge URL
        challenge_url = f"{base_url}/static/simple.html?session={session_id}&type={challenge['type']}"
        
        if challenge['type'] == 'head_movement' and 'movement_sequence' in challenge:
            import urllib.parse
            sequence_param = urllib.parse.quote(json.dumps(challenge['movement_sequence']))
            direction_duration = challenge.get('direction_duration', 3.5)
            challenge_url += f"&sequence={sequence_param}&direction_duration={direction_duration}"
        
        # Store reference image in session if provided
        session_manager = get_session_manager()
        if session_manager and request.reference_image and request.enable_face_comparison:
            # Create a new session first
            created_session = session_manager.create_session()
            # Use the generated session ID instead of our custom one
            actual_session_id = created_session.session_id
            
            # Update session with our custom data
            session_data = {
                'challenge_type': challenge['type'],
                'reference_image': request.reference_image,
                'enable_face_comparison': True,
                'third_party_id': request.third_party_id,
                'created_at': time.time()
            }
            session_manager.update_session_data(actual_session_id, session_data)
            
            # Use the actual session ID for the response
            session_id = actual_session_id
        
        response_data = {
            'success': True,
            'challenge_url': challenge_url,
            'session_id': session_id,
            'challenge_type': challenge['type'],
            'instruction': challenge['instruction'],
            'description': challenge['description'],
            'duration': challenge['duration'],
            'face_comparison_enabled': request.enable_face_comparison and request.reference_image is not None
        }
        
        if challenge['type'] == 'head_movement' and 'movement_sequence' in challenge:
            response_data['movement_sequence'] = challenge['movement_sequence']
            response_data['direction_duration'] = challenge.get('direction_duration', 3.5)
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create challenge with comparison: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create challenge with comparison: {str(e)}"
        )

@app.get("/liveness-result/{session_id}", response_model=LivenessResultResponse)
async def get_liveness_result(
    session_id: str,
    third_party_id: str = None
):
    """
    Fetch stored liveness challenge result by session ID.
    Supports optional 3rd party ID for result segregation.
    """
    try:
        session_manager = get_session_manager()
        if not session_manager:
            raise HTTPException(
                status_code=503,
                detail="Session manager not available"
            )
        
        logger.info(f"Fetching liveness result for session {session_id}, third_party: {third_party_id}")
        
        # Get result from Redis
        result_data = session_manager.get_liveness_result(session_id, third_party_id)
        
        if result_data:
            return LivenessResultResponse(
                success=True,
                found=True,
                session_id=result_data.get('session_id'),
                third_party_id=result_data.get('third_party_id'),
                result=result_data.get('result'),
                passed=result_data.get('result') == 'pass',
                challenge_type=result_data.get('challenge_type'),
                processing_time=result_data.get('processing_time'),
                stored_at=result_data.get('stored_at'),
                expires_at=result_data.get('expires_at'),
                details=result_data.get('details')
            )
        else:
            return LivenessResultResponse(
                success=True,
                found=False,
                session_id=session_id,
                third_party_id=third_party_id,
                error="No result found for the provided session ID"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch liveness result: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch liveness result: {str(e)}"
        )

# Old complex liveness endpoint removed - using simple challenges now

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Gateman Liveness Check API",
        "version": "2.1.0",
        "description": "Enhanced liveness detection with optimized face comparison and anti-spoofing",
        "endpoints": {
            "health": "/health",
            "face_matching": "/match",
            "image_liveness": "/image-liveness-check",
            "face_comparison": "/face-comparison",
            "batch_face_comparison": "/batch-face-comparison",
            "simple_challenge": "/create-simple-challenge",
            "challenge_with_comparison": "/create-challenge-with-comparison",
            "submit_challenge": "/submit-simple-challenge",
            "get_result": "/liveness-result/{session_id}",
            "docs": "/docs"
        },
        "features": [
            "Static image liveness detection",
            "Optimized face comparison with concurrent processing",
            "Batch face comparison (up to 10 pairs)",
            "Video liveness challenges",
            "Advanced anti-spoofing detection",
            "Head movement validation",
            "Redis session management",
            "Liveness result storage (1-hour expiration)",
            "3rd party integration support",
            "Automatic session cleanup",
            "Performance optimizations: concurrent image loading, parallel embedding extraction, image caching"
        ],
        "performance_optimizations": {
            "concurrent_image_loading": "Load and preprocess images simultaneously",
            "parallel_embedding_extraction": "Extract face embeddings concurrently",
            "image_caching": "Cache processed images for repeated comparisons",
            "batch_processing": "Process multiple face comparisons in parallel",
            "thread_pool_execution": "Use ThreadPoolExecutor for CPU-intensive tasks",
            "optimized_pipeline": "Streamlined preprocessing and comparison pipeline"
        }
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if app.debug else "An unexpected error occurred"
        }
    )

def run_server(host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
    """Run the FastAPI server."""
    uvicorn.run(
        "infrastructure.facematch.server:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Face Recognition API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, debug=args.debug) 