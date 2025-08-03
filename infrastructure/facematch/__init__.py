"""
FaceMatch Package
Production-grade face recognition system with liveness detection.
"""

__version__ = "1.0.0"
__author__ = "FaceMatch Team"
__description__ = "Production-grade face recognition system with MTCNN, ArcFace, and CRMNET"

from .face_detection import FaceDetector, validate_image_quality
from .liveness import LivenessDetector, validate_liveness_model
from .facematch import FaceRecognizer, validate_arcface_model
from .image_utils import ImageLoader, load_image_pair, validate_image_input
from .server import app, run_server

__all__ = [
    'FaceDetector',
    'LivenessDetector', 
    'FaceRecognizer',
    'ImageLoader',
    'load_image_pair',
    'validate_image_input',
    'validate_image_quality',
    'validate_liveness_model',
    'validate_arcface_model',
    'app',
    'run_server'
] 