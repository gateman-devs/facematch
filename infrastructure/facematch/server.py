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
from typing import Dict, Optional, Any, List
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
# Advanced head pose removed - was causing startup issues
from .redis_manager import initialize_session_manager, get_session_manager

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

class FaceComparisonResponse(BaseModel):
    """Response model for face comparison."""
    success: bool
    match: bool
    similarity_score: float
    threshold: float
    confidence: str
    processing_time: Optional[float] = None
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
    global face_detector, liveness_detector, face_recognizer, simple_liveness_detector
    
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
        'simple_liveness': get_simple_liveness_detector() is not None
    }
    
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

@app.post("/create-simple-challenge")
async def create_simple_challenge():
    """
    Create a new simple liveness challenge.
    Randomly chooses between smile and head movement challenges.
    """
    try:
        # Get simple liveness detector
        detector = get_simple_liveness_detector()
        if not detector:
            raise HTTPException(
                status_code=503,
                detail="Simple liveness detector not available"
            )
        
        # Generate random challenge
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
        # Get simple liveness detector
        detector = get_simple_liveness_detector()
        if not detector:
            raise HTTPException(
                status_code=503,
                detail="Simple liveness detector not available"
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
            logger.info(f"VIDEO_PROCESSING_START - session_id: {session_id}, challenge_type: {challenge_type}")
            
            validation_result = detector.validate_video_challenge(
                temp_video_path, 
                challenge_type, 
                parsed_movement_sequence,
                session_id,
                reference_image
            )
            
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
        # Get simple liveness detector
        detector = get_simple_liveness_detector()
        if not detector:
            raise HTTPException(
                status_code=503,
                detail="Simple liveness detector not available"
            )
        
        logger.info(f"Processing image liveness check for image input")
        
        # Perform image liveness check
        result = detector.perform_image_liveness_check(request.image)
        
        processing_time = time.time() - start_time
        
        if result['success']:
            return ImageLivenessResponse(
                success=True,
                passed=result['passed'],
                liveness_score=result['liveness_score'],
                face_detected=result.get('face_detected', False),
                processing_time=processing_time,
                anti_spoofing=result.get('anti_spoofing', {}),
                error=result.get('error')
            )
        else:
            return ImageLivenessResponse(
                success=False,
                passed=False,
                liveness_score=result.get('liveness_score', 0.0),
                face_detected=False,
                processing_time=processing_time,
                error=result.get('error', 'Image liveness check failed')
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
    """
    start_time = time.time()
    
    try:
        # Get simple liveness detector (which has face comparison functionality)
        detector = get_simple_liveness_detector()
        if not detector:
            raise HTTPException(
                status_code=503,
                detail="Face comparison service not available"
            )
        
        logger.info(f"Processing face comparison with threshold {request.threshold}")
        
        # Perform face comparison
        result = detector.compare_faces(request.image1, request.image2, request.threshold)
        
        processing_time = time.time() - start_time
        
        if result['success']:
            return FaceComparisonResponse(
                success=True,
                match=result['match'],
                similarity_score=result['similarity_score'],
                threshold=result['threshold'],
                confidence=result.get('confidence', 'unknown'),
                processing_time=processing_time
            )
        else:
            return FaceComparisonResponse(
                success=False,
                match=False,
                similarity_score=result.get('similarity_score', 0.0),
                threshold=request.threshold,
                confidence='unknown',
                processing_time=processing_time,
                error=result.get('error', 'Face comparison failed')
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to perform face comparison: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to perform face comparison: {str(e)}"
        )

@app.post("/create-challenge-with-comparison")
async def create_challenge_with_comparison(request: VideoChallengeWithComparisonRequest):
    """
    Create a video liveness challenge with optional face comparison.
    """
    try:
        detector = get_simple_liveness_detector()
        if not detector:
            raise HTTPException(
                status_code=503,
                detail="Simple liveness detector not available"
            )
        
        # Generate challenge
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
        "version": "2.0.0",
        "description": "Enhanced liveness detection with face comparison and anti-spoofing",
        "endpoints": {
            "health": "/health",
            "face_matching": "/match",
            "image_liveness": "/image-liveness-check",
            "face_comparison": "/face-comparison",
            "simple_challenge": "/create-simple-challenge",
            "challenge_with_comparison": "/create-challenge-with-comparison",
            "submit_challenge": "/submit-simple-challenge",
            "get_result": "/liveness-result/{session_id}",
            "docs": "/docs"
        },
        "features": [
            "Static image liveness detection",
            "Face comparison (URL/base64 support)",
            "Video liveness challenges",
            "Advanced anti-spoofing detection",
            "Head movement validation",
            "Redis session management",
            "Liveness result storage (1-hour expiration)",
            "3rd party integration support",
            "Automatic session cleanup"
        ]
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