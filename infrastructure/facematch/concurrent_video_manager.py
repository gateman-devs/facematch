"""
Concurrent Video Processing Manager

This module provides concurrent processing capabilities for handling multiple
video liveness detection requests simultaneously while maintaining performance
and resource limits.

Requirements addressed:
- 7.4: Maintain performance without degradation during concurrent processing
- 7.5: Not exceed reasonable memory limits during processing
"""

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid
from collections import defaultdict, deque
import psutil

from .optimized_video_processor import OptimizedVideoProcessor, VideoProcessingResult, create_optimized_processor
from .performance_optimizer import OptimizationConfig, create_memory_optimized_config

logger = logging.getLogger(__name__)

@dataclass
class ProcessingRequest:
    """Video processing request."""
    request_id: str
    video_path: str
    expected_sequence: Optional[List[str]] = None
    priority: int = 1  # 1=high, 2=normal, 3=low
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[VideoProcessingResult] = None
    error: Optional[str] = None

@dataclass
class ConcurrentProcessingStats:
    """Statistics for concurrent processing."""
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    active_requests: int = 0
    queued_requests: int = 0
    average_processing_time: float = 0.0
    peak_memory_usage: float = 0.0
    current_memory_usage: float = 0.0
    throughput_per_minute: float = 0.0

class ConcurrentVideoManager:
    """
    Manager for concurrent video processing with resource management
    and performance optimization.
    """
    
    def __init__(
        self,
        max_concurrent_videos: int = 4,
        max_queue_size: int = 20,
        memory_limit_mb: float = 4096.0,
        enable_priority_queue: bool = True
    ):
        """
        Initialize concurrent video manager.
        
        Args:
            max_concurrent_videos: Maximum number of videos to process simultaneously
            max_queue_size: Maximum number of requests in queue
            memory_limit_mb: Memory limit in MB for all concurrent processing
            enable_priority_queue: Enable priority-based request processing
        """
        self.max_concurrent_videos = max_concurrent_videos
        self.max_queue_size = max_queue_size
        self.memory_limit_mb = memory_limit_mb
        self.enable_priority_queue = enable_priority_queue
        
        # Processing infrastructure
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_videos)
        self.request_queue = deque()
        self.active_requests: Dict[str, ProcessingRequest] = {}
        self.completed_requests: Dict[str, ProcessingRequest] = {}
        
        # Resource monitoring
        self.process = psutil.Process()
        self.stats = ConcurrentProcessingStats()
        self.processing_times = deque(maxlen=100)  # Keep last 100 processing times
        
        # Thread safety
        self.queue_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        
        # Background monitoring
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info(f"ConcurrentVideoManager initialized: max_concurrent={max_concurrent_videos}, "
                   f"queue_size={max_queue_size}, memory_limit={memory_limit_mb}MB")
    
    def submit_video_processing(
        self,
        video_path: str,
        expected_sequence: Optional[List[str]] = None,
        priority: int = 2,
        request_id: Optional[str] = None
    ) -> str:
        """
        Submit video for processing.
        
        Args:
            video_path: Path to video file
            expected_sequence: Expected movement sequence
            priority: Request priority (1=high, 2=normal, 3=low)
            request_id: Optional custom request ID
            
        Returns:
            Request ID for tracking
            
        Raises:
            ValueError: If queue is full or memory limit exceeded
        """
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        # Check queue capacity
        with self.queue_lock:
            if len(self.request_queue) >= self.max_queue_size:
                raise ValueError(f"Queue is full ({self.max_queue_size} requests)")
        
        # Check memory usage
        current_memory = self._get_memory_usage()
        if current_memory > self.memory_limit_mb * 0.9:  # 90% threshold
            raise ValueError(f"Memory usage too high: {current_memory:.1f}MB > {self.memory_limit_mb * 0.9:.1f}MB")
        
        # Create processing request
        request = ProcessingRequest(
            request_id=request_id,
            video_path=video_path,
            expected_sequence=expected_sequence,
            priority=priority
        )
        
        # Add to queue
        with self.queue_lock:
            if self.enable_priority_queue:
                # Insert based on priority (lower number = higher priority)
                inserted = False
                for i, existing_request in enumerate(self.request_queue):
                    if request.priority < existing_request.priority:
                        self.request_queue.insert(i, request)
                        inserted = True
                        break
                if not inserted:
                    self.request_queue.append(request)
            else:
                self.request_queue.append(request)
            
            self.stats.total_requests += 1
            self.stats.queued_requests += 1
        
        # Try to start processing immediately if capacity available
        self._try_start_next_request()
        
        logger.info(f"Video processing request submitted: {request_id} (priority: {priority})")
        return request_id
    
    def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """
        Get status of a processing request.
        
        Args:
            request_id: Request ID
            
        Returns:
            Request status information
        """
        # Check active requests
        if request_id in self.active_requests:
            request = self.active_requests[request_id]
            return {
                'status': 'processing',
                'request_id': request_id,
                'created_at': request.created_at,
                'started_at': request.started_at,
                'processing_time': time.time() - request.started_at if request.started_at else 0,
                'priority': request.priority
            }
        
        # Check completed requests
        if request_id in self.completed_requests:
            request = self.completed_requests[request_id]
            status = 'completed' if request.result and request.result.success else 'failed'
            
            return {
                'status': status,
                'request_id': request_id,
                'created_at': request.created_at,
                'started_at': request.started_at,
                'completed_at': request.completed_at,
                'processing_time': request.completed_at - request.started_at if request.started_at and request.completed_at else 0,
                'priority': request.priority,
                'result': request.result,
                'error': request.error
            }
        
        # Check queued requests
        with self.queue_lock:
            for i, request in enumerate(self.request_queue):
                if request.request_id == request_id:
                    return {
                        'status': 'queued',
                        'request_id': request_id,
                        'created_at': request.created_at,
                        'queue_position': i + 1,
                        'priority': request.priority
                    }
        
        return {
            'status': 'not_found',
            'request_id': request_id
        }
    
    def get_result(self, request_id: str) -> Optional[VideoProcessingResult]:
        """
        Get processing result for a completed request.
        
        Args:
            request_id: Request ID
            
        Returns:
            Processing result or None if not completed
        """
        if request_id in self.completed_requests:
            return self.completed_requests[request_id].result
        return None
    
    def cancel_request(self, request_id: str) -> bool:
        """
        Cancel a queued or active request.
        
        Args:
            request_id: Request ID
            
        Returns:
            True if cancelled, False if not found or already completed
        """
        # Try to remove from queue
        with self.queue_lock:
            for i, request in enumerate(self.request_queue):
                if request.request_id == request_id:
                    del self.request_queue[i]
                    self.stats.queued_requests -= 1
                    logger.info(f"Request cancelled from queue: {request_id}")
                    return True
        
        # Cannot cancel active requests (would need more complex future tracking)
        logger.warning(f"Cannot cancel active or completed request: {request_id}")
        return False
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        with self.stats_lock:
            stats_dict = {
                'total_requests': self.stats.total_requests,
                'completed_requests': self.stats.completed_requests,
                'failed_requests': self.stats.failed_requests,
                'active_requests': self.stats.active_requests,
                'queued_requests': self.stats.queued_requests,
                'average_processing_time': self.stats.average_processing_time,
                'peak_memory_usage_mb': self.stats.peak_memory_usage,
                'current_memory_usage_mb': self.stats.current_memory_usage,
                'throughput_per_minute': self.stats.throughput_per_minute,
                'queue_capacity_used': len(self.request_queue) / self.max_queue_size,
                'processing_capacity_used': len(self.active_requests) / self.max_concurrent_videos,
                'memory_usage_percent': (self.stats.current_memory_usage / self.memory_limit_mb) * 100
            }
        
        return stats_dict
    
    def wait_for_completion(self, request_id: str, timeout: Optional[float] = None) -> Optional[VideoProcessingResult]:
        """
        Wait for a request to complete.
        
        Args:
            request_id: Request ID
            timeout: Maximum time to wait in seconds
            
        Returns:
            Processing result or None if timeout/error
        """
        start_time = time.time()
        
        while True:
            # Check if completed
            if request_id in self.completed_requests:
                return self.completed_requests[request_id].result
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Wait timeout for request: {request_id}")
                return None
            
            # Sleep briefly before checking again
            time.sleep(0.1)
    
    def shutdown(self, wait: bool = True):
        """
        Shutdown the concurrent video manager.
        
        Args:
            wait: Whether to wait for active requests to complete
        """
        logger.info("Shutting down ConcurrentVideoManager...")
        
        self.monitoring_active = False
        
        if wait:
            # Wait for active requests to complete
            while self.active_requests:
                time.sleep(0.1)
        
        self.executor.shutdown(wait=wait)
        
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
        
        logger.info("ConcurrentVideoManager shutdown complete")
    
    def _try_start_next_request(self):
        """Try to start the next queued request if capacity available."""
        with self.queue_lock:
            if (len(self.active_requests) < self.max_concurrent_videos and 
                self.request_queue and
                self._check_memory_capacity()):
                
                request = self.request_queue.popleft()
                self.stats.queued_requests -= 1
                self.stats.active_requests += 1
                
                # Move to active requests
                self.active_requests[request.request_id] = request
                request.started_at = time.time()
                
                # Submit to executor
                future = self.executor.submit(self._process_video_request, request)
                future.add_done_callback(lambda f: self._handle_request_completion(request.request_id, f))
                
                logger.info(f"Started processing request: {request.request_id}")
    
    def _process_video_request(self, request: ProcessingRequest) -> VideoProcessingResult:
        """Process a single video request."""
        try:
            # Create processor with memory-optimized config for concurrent processing
            config = create_memory_optimized_config()
            processor = OptimizedVideoProcessor(optimization_config=config)
            
            # Process video
            result = processor.process_video_for_liveness(
                request.video_path,
                request.expected_sequence
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Video processing failed for request {request.request_id}: {e}")
            return VideoProcessingResult(
                success=False,
                movements=[],
                error=str(e)
            )
    
    def _handle_request_completion(self, request_id: str, future):
        """Handle completion of a processing request."""
        try:
            # Get result from future
            result = future.result()
            
            # Move request from active to completed
            request = self.active_requests.pop(request_id)
            request.completed_at = time.time()
            request.result = result
            
            if not result.success:
                request.error = result.error
                self.stats.failed_requests += 1
            else:
                self.stats.completed_requests += 1
            
            # Update statistics
            processing_time = request.completed_at - request.started_at
            self.processing_times.append(processing_time)
            
            with self.stats_lock:
                self.stats.active_requests -= 1
                if self.processing_times:
                    self.stats.average_processing_time = sum(self.processing_times) / len(self.processing_times)
            
            # Store completed request (with cleanup of old requests)
            self.completed_requests[request_id] = request
            self._cleanup_old_completed_requests()
            
            logger.info(f"Request completed: {request_id} (success: {result.success}, time: {processing_time:.2f}s)")
            
        except Exception as e:
            logger.error(f"Error handling request completion {request_id}: {e}")
            
            # Handle error case
            if request_id in self.active_requests:
                request = self.active_requests.pop(request_id)
                request.completed_at = time.time()
                request.error = str(e)
                self.completed_requests[request_id] = request
                
                with self.stats_lock:
                    self.stats.active_requests -= 1
                    self.stats.failed_requests += 1
        
        finally:
            # Try to start next request
            self._try_start_next_request()
    
    def _cleanup_old_completed_requests(self):
        """Clean up old completed requests to prevent memory buildup."""
        max_completed_requests = 100
        
        if len(self.completed_requests) > max_completed_requests:
            # Remove oldest requests
            sorted_requests = sorted(
                self.completed_requests.items(),
                key=lambda x: x[1].completed_at or 0
            )
            
            requests_to_remove = len(self.completed_requests) - max_completed_requests
            for i in range(requests_to_remove):
                request_id = sorted_requests[i][0]
                del self.completed_requests[request_id]
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _check_memory_capacity(self) -> bool:
        """Check if there's enough memory capacity for another request."""
        current_memory = self._get_memory_usage()
        return current_memory < self.memory_limit_mb * 0.8  # 80% threshold
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Update memory statistics
                current_memory = self._get_memory_usage()
                
                with self.stats_lock:
                    self.stats.current_memory_usage = current_memory
                    self.stats.peak_memory_usage = max(self.stats.peak_memory_usage, current_memory)
                    
                    # Calculate throughput (requests per minute)
                    if self.processing_times:
                        recent_times = list(self.processing_times)[-10:]  # Last 10 requests
                        if recent_times:
                            avg_time = sum(recent_times) / len(recent_times)
                            self.stats.throughput_per_minute = 60.0 / avg_time if avg_time > 0 else 0.0
                
                time.sleep(1.0)  # Monitor every second
                
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")

# Global manager instance
_global_manager: Optional[ConcurrentVideoManager] = None

def get_global_video_manager() -> ConcurrentVideoManager:
    """Get or create global video manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = ConcurrentVideoManager()
    return _global_manager

def initialize_video_manager(
    max_concurrent_videos: int = 4,
    max_queue_size: int = 20,
    memory_limit_mb: float = 4096.0
) -> ConcurrentVideoManager:
    """Initialize global video manager with custom settings."""
    global _global_manager
    if _global_manager is not None:
        _global_manager.shutdown(wait=False)
    
    _global_manager = ConcurrentVideoManager(
        max_concurrent_videos=max_concurrent_videos,
        max_queue_size=max_queue_size,
        memory_limit_mb=memory_limit_mb
    )
    
    return _global_manager

def shutdown_video_manager():
    """Shutdown global video manager."""
    global _global_manager
    if _global_manager is not None:
        _global_manager.shutdown()
        _global_manager = None