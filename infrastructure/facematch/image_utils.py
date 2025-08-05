"""
Image Utilities Module
Provides async image loading from URLs and base64 data with comprehensive error handling.
"""

import asyncio
import base64
import io
import logging
import time
from typing import Dict, Optional, Tuple, Union
import numpy as np
import cv2
import aiohttp
from PIL import Image
import requests
import tempfile
import os

# Configure logging
logger = logging.getLogger(__name__)

class ImageLoader:
    """Async image loader for URLs and base64 data."""
    
    def __init__(self, max_file_size: int = 10 * 1024 * 1024, timeout: int = 30):
        """
        Initialize image loader.
        
        Args:
            max_file_size: Maximum file size in bytes (default: 10MB)
            timeout: Request timeout in seconds
        """
        self.max_file_size = max_file_size
        self.timeout = timeout
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers={'User-Agent': 'FaceMatch/1.0'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def load_from_url(self, url: str) -> Dict:
        """
        Load image from URL asynchronously.
        
        Args:
            url: Image URL
            
        Returns:
            Image loading result
        """
        start_time = time.time()
        
        try:
            if not self.session:
                raise RuntimeError("ImageLoader not initialized. Use async context manager.")
            
            logger.info(f"Loading image from URL: {url}")
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    return {
                        'success': False,
                        'error': f"HTTP {response.status}: {response.reason}",
                        'url': url,
                        'load_time': time.time() - start_time
                    }
                
                # Check content length
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > self.max_file_size:
                    return {
                        'success': False,
                        'error': f"File too large: {content_length} bytes (max: {self.max_file_size})",
                        'url': url,
                        'load_time': time.time() - start_time
                    }
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if not any(ct in content_type for ct in ['image/', 'jpeg', 'jpg', 'png', 'gif', 'bmp', 'webp']):
                    return {
                        'success': False,
                        'error': f"Invalid content type: {content_type}",
                        'url': url,
                        'load_time': time.time() - start_time
                    }
                
                # Read image data
                image_data = await response.read()
                
                if len(image_data) > self.max_file_size:
                    return {
                        'success': False,
                        'error': f"File too large: {len(image_data)} bytes (max: {self.max_file_size})",
                        'url': url,
                        'load_time': time.time() - start_time
                    }
                
                # Convert to numpy array
                image_array = self._bytes_to_array(image_data)
                
                if image_array is None:
                    return {
                        'success': False,
                        'error': "Failed to decode image data",
                        'url': url,
                        'load_time': time.time() - start_time
                    }
                
                load_time = time.time() - start_time
                
                result = {
                    'success': True,
                    'image': image_array,
                    'url': url,
                    'load_time': load_time,
                    'file_size': len(image_data),
                    'content_type': content_type,
                    'image_shape': image_array.shape
                }
                
                logger.info(f"Image loaded successfully from URL: {url} "
                           f"({image_array.shape}) in {load_time:.3f}s")
                
                return result
                
        except asyncio.TimeoutError:
            return {
                'success': False,
                'error': f"Request timeout after {self.timeout}s",
                'url': url,
                'load_time': time.time() - start_time
            }
        except Exception as e:
            logger.error(f"Failed to load image from URL {url}: {e}")
            return {
                'success': False,
                'error': str(e),
                'url': url,
                'load_time': time.time() - start_time
            }
    
    def load_from_base64(self, base64_data: str) -> Dict:
        """
        Load image from base64 string.
        
        Args:
            base64_data: Base64 encoded image data (with or without data URL prefix)
            
        Returns:
            Image loading result
        """
        start_time = time.time()
        
        try:
            logger.info("Loading image from base64 data")
            
            # Remove data URL prefix if present
            if base64_data.startswith('data:'):
                # Format: data:image/jpeg;base64,/9j/4AAQSkZJRgABA...
                header, data = base64_data.split(',', 1)
                base64_data = data
            
            # Decode base64
            try:
                image_bytes = base64.b64decode(base64_data)
            except Exception as e:
                return {
                    'success': False,
                    'error': f"Invalid base64 data: {e}",
                    'load_time': time.time() - start_time
                }
            
            # Check file size
            if len(image_bytes) > self.max_file_size:
                return {
                    'success': False,
                    'error': f"File too large: {len(image_bytes)} bytes (max: {self.max_file_size})",
                    'load_time': time.time() - start_time
                }
            
            # Convert to numpy array
            image_array = self._bytes_to_array(image_bytes)
            
            if image_array is None:
                return {
                    'success': False,
                    'error': "Failed to decode image data",
                    'load_time': time.time() - start_time
                }
            
            load_time = time.time() - start_time
            
            result = {
                'success': True,
                'image': image_array,
                'load_time': load_time,
                'file_size': len(image_bytes),
                'image_shape': image_array.shape
            }
            
            logger.info(f"Image loaded successfully from base64 data "
                       f"({image_array.shape}) in {load_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to load image from base64: {e}")
            return {
                'success': False,
                'error': str(e),
                'load_time': time.time() - start_time
            }
    
    def _bytes_to_array(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Convert image bytes to numpy array.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Image as numpy array or None if failed
        """
        try:
            # Try PIL first (handles more formats)
            try:
                pil_image = Image.open(io.BytesIO(image_bytes))
                # Convert to RGB if needed
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                # Convert to numpy array
                image_array = np.array(pil_image)
                return image_array
            except Exception:
                pass
            
            # Fallback to OpenCV
            try:
                nparr = np.frombuffer(image_bytes, np.uint8)
                image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image_array is not None:
                    # Convert BGR to RGB
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                return image_array
            except Exception:
                pass
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to convert bytes to array: {e}")
            return None

async def load_image_pair(source1: str, source2: str, 
                         max_file_size: int = 10 * 1024 * 1024, 
                         timeout: int = 30) -> Tuple[Dict, Dict]:
    """
    Load two images concurrently.
    
    Args:
        source1: First image source (URL or base64)
        source2: Second image source (URL or base64)
        max_file_size: Maximum file size in bytes
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of image loading results
    """
    async with ImageLoader(max_file_size=max_file_size, timeout=timeout) as loader:
        # Determine source types
        is_url1 = source1.startswith(('http://', 'https://'))
        is_url2 = source2.startswith(('http://', 'https://'))
        
        # Create tasks for concurrent loading
        tasks = []
        
        if is_url1:
            tasks.append(loader.load_from_url(source1))
        else:
            tasks.append(asyncio.create_task(
                asyncio.get_event_loop().run_in_executor(
                    None, loader.load_from_base64, source1
                )
            ))
        
        if is_url2:
            tasks.append(loader.load_from_url(source2))
        else:
            tasks.append(asyncio.create_task(
                asyncio.get_event_loop().run_in_executor(
                    None, loader.load_from_base64, source2
                )
            ))
        
        # Wait for both images to load
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        result1 = results[0] if not isinstance(results[0], Exception) else {
            'success': False,
            'error': str(results[0]),
            'load_time': 0.0
        }
        
        result2 = results[1] if not isinstance(results[1], Exception) else {
            'success': False,
            'error': str(results[1]),
            'load_time': 0.0
        }
        
        return result1, result2

def validate_image_input(source: str) -> Dict:
    """
    Validate image input source.
    
    Args:
        source: Image source (URL or base64)
        
    Returns:
        Validation result
    """
    if not source or not isinstance(source, str):
        return {
            'valid': False,
            'error': "Image source must be a non-empty string",
            'source_type': 'unknown'
        }
    
    source = source.strip()
    
    if source.startswith(('http://', 'https://')):
        # URL validation
        if len(source) > 2048:
            return {
                'valid': False,
                'error': "URL too long (max 2048 characters)",
                'source_type': 'url'
            }
        
        return {
            'valid': True,
            'source_type': 'url',
            'url': source
        }
    
    elif source.startswith('data:'):
        # Data URL validation
        if not ',base64,' in source and not ',base64;' in source:
            return {
                'valid': False,
                'error': "Data URL must be base64 encoded",
                'source_type': 'data_url'
            }
        
        return {
            'valid': True,
            'source_type': 'data_url'
        }
    
    else:
        # Assume base64 data
        if len(source) < 100:
            return {
                'valid': False,
                'error': "Base64 data too short",
                'source_type': 'base64'
            }
        
        # Basic base64 validation
        try:
            base64.b64decode(source[:100])  # Test first 100 chars
            return {
                'valid': True,
                'source_type': 'base64'
            }
        except Exception:
            return {
                'valid': False,
                'error': "Invalid base64 data",
                'source_type': 'base64'
            }

class ImageProcessor:
    """Utility class for processing images from various sources (URL, base64, file)"""
    
    def __init__(self):
        self.max_image_size = 10 * 1024 * 1024  # 10MB
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
    def process_image_input(self, image_input: str) -> Optional[np.ndarray]:
        """
        Process image from URL or base64 string and return OpenCV image array.
        
        Args:
            image_input: URL string or base64 encoded image
            
        Returns:
            OpenCV image array (BGR format) or None if failed
        """
        try:
            if self._is_url(image_input):
                return self._load_image_from_url(image_input)
            elif self._is_base64(image_input):
                return self._load_image_from_base64(image_input)
            else:
                logger.error("Invalid image input: must be URL or base64")
                return None
                
        except Exception as e:
            logger.error(f"Error processing image input: {e}")
            return None
    
    def _is_url(self, text: str) -> bool:
        """Check if text is a valid URL"""
        return text.startswith(('http://', 'https://'))
    
    def _is_base64(self, text: str) -> bool:
        """Check if text is a valid base64 string"""
        try:
            # Remove data URL prefix if present
            if text.startswith('data:image'):
                text = text.split(',')[1]
            
            # Try to decode base64
            decoded = base64.b64decode(text, validate=True)
            return len(decoded) > 0
        except Exception:
            return False
    
    def _load_image_from_url(self, url: str) -> Optional[np.ndarray]:
        """Download and load image from URL"""
        try:
            logger.info(f"Downloading image from URL: {url[:100]}...")
            
            # Set headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                logger.error(f"Invalid content type: {content_type}")
                return None
            
            # Check file size
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.max_image_size:
                logger.error(f"Image too large: {content_length} bytes")
                return None
            
            # Read image data
            image_data = b''
            for chunk in response.iter_content(chunk_size=8192):
                image_data += chunk
                if len(image_data) > self.max_image_size:
                    logger.error("Image too large during download")
                    return None
            
            # Convert to OpenCV format
            return self._bytes_to_opencv(image_data)
            
        except requests.RequestException as e:
            logger.error(f"Error downloading image: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing downloaded image: {e}")
            return None
    
    def _load_image_from_base64(self, base64_string: str) -> Optional[np.ndarray]:
        """Load image from base64 string"""
        try:
            # Remove data URL prefix if present
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            
            # Check size
            if len(image_data) > self.max_image_size:
                logger.error(f"Base64 image too large: {len(image_data)} bytes")
                return None
            
            # Convert to OpenCV format
            return self._bytes_to_opencv(image_data)
            
        except Exception as e:
            logger.error(f"Error processing base64 image: {e}")
            return None
    
    def _bytes_to_opencv(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Convert image bytes to OpenCV format"""
        try:
            # Use PIL to handle various formats
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert PIL to OpenCV (RGB to BGR)
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Validate image
            if opencv_image.size == 0:
                logger.error("Empty image after conversion")
                return None
            
            logger.info(f"Successfully loaded image: {opencv_image.shape}")
            return opencv_image
            
        except Exception as e:
            logger.error(f"Error converting bytes to OpenCV: {e}")
            return None
    
    def validate_image_for_face_detection(self, image: np.ndarray) -> Tuple[bool, str]:
        """
        Validate image for face detection requirements.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            if image is None or image.size == 0:
                return False, "Invalid or empty image"
            
            height, width = image.shape[:2]
            
            # Check minimum dimensions
            if width < 100 or height < 100:
                return False, f"Image too small: {width}x{height} (minimum 100x100)"
            
            # Check maximum dimensions
            if width > 4000 or height > 4000:
                return False, f"Image too large: {width}x{height} (maximum 4000x4000)"
            
            # Check aspect ratio (not too extreme)
            aspect_ratio = width / height
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                return False, f"Extreme aspect ratio: {aspect_ratio:.2f}"
            
            return True, "Image validation passed"
            
        except Exception as e:
            return False, f"Error validating image: {e}"
    
    def save_temp_image(self, image: np.ndarray, prefix: str = "temp_image") -> str:
        """Save image to temporary file and return path"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False, prefix=prefix) as temp_file:
                cv2.imwrite(temp_file.name, image)
                return temp_file.name
        except Exception as e:
            logger.error(f"Error saving temporary image: {e}")
            raise
    
    def resize_image_if_needed(self, image: np.ndarray, max_dimension: int = 1024) -> np.ndarray:
        """Resize image if it's too large, maintaining aspect ratio"""
        try:
            height, width = image.shape[:2]
            
            if max(width, height) <= max_dimension:
                return image
            
            # Calculate new dimensions
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            
            return resized
            
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return image


# Global instance
image_processor = ImageProcessor()


def get_image_processor() -> ImageProcessor:
    """Get the global image processor instance"""
    return image_processor 