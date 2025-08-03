"""
Face Recognition API Server
Production-grade FastAPI server for face matching, liveness detection, and recognition.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional, Any
import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from .face_detection import FaceDetector, validate_image_quality
from .liveness import LivenessDetector, validate_liveness_model
from .facematch import FaceRecognizer, validate_arcface_model, adaptive_threshold
from .image_utils import load_image_pair, validate_image_input, resize_image_if_needed

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources."""
    global face_detector, liveness_detector, face_recognizer
    
    logger.info("Initializing Face Recognition API...")
    
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
        'arcface': face_recognizer is not None
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
                'arcface': '../../models/arcface_resnet50.onnx'
            }
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

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Face Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "match": "/match",
            "docs": "/docs"
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