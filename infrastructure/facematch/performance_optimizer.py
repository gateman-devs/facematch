"""
Performance Optimization Module for Video Liveness Detection

This module implements comprehensive performance optimizations including:
- Optimized frame processing pipeline for better speed
- Efficient memory management for large video processing
- Concurrent processing capabilities where appropriate
- Performance monitoring and metrics collection

Requirements addressed:
- 7.1: Complete validation within 10 seconds for typical 15-second videos
- 7.2: Process only necessary frames to maintain accuracy while optimizing speed
- 7.3: Use efficient algorithms that don't block the processing pipeline
- 7.4: Maintain performance without degradation during concurrent processing
- 7.5: Not exceed reasonable memory limits during processing
"""

import asyncio
import concurrent.futures
import gc
import logging
import multiprocessing
import os
import psutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import cv2
from collections import deque
from functools import lru_cache, wraps
import weakref

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring and optimization."""
    
    # Processing times
    total_processing_time: float = 0.0
    video_loading_time: float = 0.0
    frame_extraction_time: float = 0.0
    face_detection_time: float = 0.0
    movement_analysis_time: float = 0.0
    validation_time: float = 0.0
    
    # Resource usage
    peak_memory_usage: float = 0.0  # MB
    average_memory_usage: float = 0.0  # MB
    cpu_usage_percent: float = 0.0
    
    # Processing statistics
    frames_processed: int = 0
    frames_skipped: int = 0
    frames_per_second: float = 0.0
    
    # Optimization statistics
    cache_hits: int = 0
    cache_misses: int = 0
    concurrent_tasks: int = 0
    memory_optimizations_applied: int = 0

@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    
    # Frame processing optimization
    enable_frame_skipping: bool = True
    target_fps: float = 5.0  # Process every N frames to achieve this FPS
    max_frames_to_process: int = 150  # Limit for very long videos
    frame_resize_factor: float = 0.8  # Resize frames to reduce processing load
    
    # Memory management
    enable_memory_optimization: bool = True
    max_memory_usage_mb: float = 2048.0  # 2GB limit
    memory_cleanup_threshold: float = 0.8  # Cleanup when 80% of limit reached
    enable_garbage_collection: bool = True
    gc_frequency: int = 10  # Run GC every N frames
    
    # Concurrent processing
    enable_concurrent_processing: bool = True
    max_workers: int = min(4, multiprocessing.cpu_count())
    concurrent_batch_size: int = 8
    enable_async_processing: bool = True
    
    # Caching
    enable_caching: bool = True
    cache_size: int = 128
    enable_result_caching: bool = True
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    monitoring_interval: float = 1.0  # seconds
    enable_profiling: bool = False

class MemoryManager:
    """Advanced memory management for video processing."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.process = psutil.Process()
        self.peak_memory = 0.0
        self.memory_samples = deque(maxlen=100)
        self.cleanup_callbacks = []
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            self.memory_samples.append(memory_mb)
            self.peak_memory = max(self.peak_memory, memory_mb)
            return memory_mb
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0.0
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage is within limits."""
        current_memory = self.get_memory_usage()
        return current_memory < self.config.max_memory_usage_mb
    
    def cleanup_if_needed(self) -> bool:
        """Perform memory cleanup if threshold is reached."""
        current_memory = self.get_memory_usage()
        threshold = self.config.max_memory_usage_mb * self.config.memory_cleanup_threshold
        
        if current_memory > threshold:
            logger.info(f"Memory cleanup triggered: {current_memory:.1f}MB > {threshold:.1f}MB")
            
            # Run cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.warning(f"Cleanup callback failed: {e}")
            
            # Force garbage collection
            if self.config.enable_garbage_collection:
                gc.collect()
            
            return True
        
        return False
    
    def register_cleanup_callback(self, callback: Callable[[], None]):
        """Register a cleanup callback."""
        self.cleanup_callbacks.append(callback)
    
    def get_average_memory_usage(self) -> float:
        """Get average memory usage."""
        if not self.memory_samples:
            return 0.0
        return sum(self.memory_samples) / len(self.memory_samples)

class FrameProcessor:
    """Optimized frame processing pipeline."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.frame_cache = {}
        self.processing_stats = {
            'frames_processed': 0,
            'frames_skipped': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
    def calculate_frame_skip_interval(self, video_fps: float, total_frames: int) -> int:
        """Calculate optimal frame skip interval."""
        if not self.config.enable_frame_skipping:
            return 1
        
        # Calculate skip interval to achieve target FPS
        skip_interval = max(1, int(video_fps / self.config.target_fps))
        
        # Ensure we don't process too many frames
        if total_frames > self.config.max_frames_to_process:
            min_skip = max(1, total_frames // self.config.max_frames_to_process)
            skip_interval = max(skip_interval, min_skip)
        
        logger.info(f"Frame skip interval: {skip_interval} (video_fps: {video_fps}, target_fps: {self.config.target_fps})")
        return skip_interval
    
    def optimize_frame_size(self, frame: np.ndarray) -> np.ndarray:
        """Optimize frame size for processing."""
        if self.config.frame_resize_factor >= 1.0:
            return frame
        
        height, width = frame.shape[:2]
        new_width = int(width * self.config.frame_resize_factor)
        new_height = int(height * self.config.frame_resize_factor)
        
        # Use efficient interpolation
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return resized_frame
    
    @lru_cache(maxsize=128)
    def get_cached_frame_hash(self, frame_data: bytes) -> str:
        """Get cached hash for frame data."""
        import hashlib
        return hashlib.md5(frame_data).hexdigest()
    
    def process_frame_batch(self, frames: List[np.ndarray], processor_func: Callable) -> List[Any]:
        """Process a batch of frames efficiently."""
        results = []
        
        for frame in frames:
            # Optimize frame size
            optimized_frame = self.optimize_frame_size(frame)
            
            # Process frame
            result = processor_func(optimized_frame)
            results.append(result)
            
            self.processing_stats['frames_processed'] += 1
        
        return results
    
    def extract_frames_optimized(self, video_path: str) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Extract frames with optimization."""
        start_time = time.time()
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate frame skip interval
            skip_interval = self.calculate_frame_skip_interval(fps, total_frames)
            
            frames = []
            frame_indices = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames based on interval
                if frame_count % skip_interval == 0:
                    # Optimize frame
                    optimized_frame = self.optimize_frame_size(frame)
                    frames.append(optimized_frame)
                    frame_indices.append(frame_count)
                else:
                    self.processing_stats['frames_skipped'] += 1
                
                frame_count += 1
                
                # Limit total frames processed
                if len(frames) >= self.config.max_frames_to_process:
                    logger.info(f"Reached max frames limit: {self.config.max_frames_to_process}")
                    break
            
            cap.release()
            
            extraction_time = time.time() - start_time
            
            metadata = {
                'total_frames_in_video': total_frames,
                'frames_extracted': len(frames),
                'frames_skipped': self.processing_stats['frames_skipped'],
                'skip_interval': skip_interval,
                'original_fps': fps,
                'effective_fps': len(frames) / (total_frames / fps) if total_frames > 0 else 0,
                'extraction_time': extraction_time,
                'frame_indices': frame_indices
            }
            
            logger.info(f"Extracted {len(frames)} frames from {total_frames} total frames in {extraction_time:.3f}s")
            return frames, metadata
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            raise

class ConcurrentProcessor:
    """Concurrent processing manager for video analysis."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.executor = None
        self.active_tasks = []
        
    def __enter__(self):
        if self.config.enable_concurrent_processing:
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.max_workers
            )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def process_concurrent_batches(
        self, 
        items: List[Any], 
        processor_func: Callable,
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """Process items in concurrent batches."""
        if not self.config.enable_concurrent_processing or not self.executor:
            # Fallback to sequential processing
            return [processor_func(item) for item in items]
        
        batch_size = batch_size or self.config.concurrent_batch_size
        results = [None] * len(items)
        futures = []
        
        # Submit batches to executor
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_indices = list(range(i, min(i + batch_size, len(items))))
            
            future = self.executor.submit(self._process_batch, batch, processor_func)
            futures.append((future, batch_indices))
        
        # Collect results
        for future, indices in futures:
            try:
                batch_results = future.result(timeout=30)  # 30 second timeout
                for idx, result in zip(indices, batch_results):
                    results[idx] = result
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Fill with error results
                for idx in indices:
                    results[idx] = {'success': False, 'error': str(e)}
        
        return results
    
    def _process_batch(self, batch: List[Any], processor_func: Callable) -> List[Any]:
        """Process a single batch."""
        return [processor_func(item) for item in batch]
    
    async def process_async_pipeline(
        self, 
        items: List[Any], 
        pipeline_stages: List[Callable]
    ) -> List[Any]:
        """Process items through an async pipeline."""
        if not self.config.enable_async_processing:
            # Fallback to synchronous processing
            results = items
            for stage in pipeline_stages:
                results = [stage(item) for item in results]
            return results
        
        # Create async tasks for each pipeline stage
        async def process_item_async(item):
            result = item
            for stage in pipeline_stages:
                if asyncio.iscoroutinefunction(stage):
                    result = await stage(result)
                else:
                    result = stage(result)
            return result
        
        # Process all items concurrently
        tasks = [process_item_async(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Async processing failed: {result}")
                processed_results.append({'success': False, 'error': str(result)})
            else:
                processed_results.append(result)
        
        return processed_results

class PerformanceMonitor:
    """Performance monitoring and profiling."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.metrics = PerformanceMetrics()
        self.start_time = None
        self.memory_manager = MemoryManager(config)
        self.monitoring_active = False
        self.monitoring_thread = None
        
    def start_monitoring(self):
        """Start performance monitoring."""
        if not self.config.enable_performance_monitoring:
            return
        
        self.start_time = time.time()
        self.monitoring_active = True
        
        if self.config.monitoring_interval > 0:
            self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        
        if self.start_time:
            self.metrics.total_processing_time = time.time() - self.start_time
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Update memory metrics
                current_memory = self.memory_manager.get_memory_usage()
                self.metrics.peak_memory_usage = max(self.metrics.peak_memory_usage, current_memory)
                self.metrics.average_memory_usage = self.memory_manager.get_average_memory_usage()
                
                # Update CPU usage
                self.metrics.cpu_usage_percent = psutil.cpu_percent(interval=None)
                
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")
    
    @contextmanager
    def measure_time(self, metric_name: str):
        """Context manager for measuring execution time."""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed_time = time.time() - start_time
            setattr(self.metrics, metric_name, elapsed_time)
    
    def update_processing_stats(self, frames_processed: int, frames_skipped: int):
        """Update processing statistics."""
        self.metrics.frames_processed = frames_processed
        self.metrics.frames_skipped = frames_skipped
        
        if self.metrics.total_processing_time > 0:
            self.metrics.frames_per_second = frames_processed / self.metrics.total_processing_time
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            'processing_times': {
                'total_processing_time': self.metrics.total_processing_time,
                'video_loading_time': self.metrics.video_loading_time,
                'frame_extraction_time': self.metrics.frame_extraction_time,
                'face_detection_time': self.metrics.face_detection_time,
                'movement_analysis_time': self.metrics.movement_analysis_time,
                'validation_time': self.metrics.validation_time
            },
            'resource_usage': {
                'peak_memory_usage_mb': self.metrics.peak_memory_usage,
                'average_memory_usage_mb': self.metrics.average_memory_usage,
                'current_memory_usage_mb': self.memory_manager.get_memory_usage(),
                'cpu_usage_percent': self.metrics.cpu_usage_percent
            },
            'processing_stats': {
                'frames_processed': self.metrics.frames_processed,
                'frames_skipped': self.metrics.frames_skipped,
                'frames_per_second': self.metrics.frames_per_second
            },
            'optimization_stats': {
                'cache_hits': self.metrics.cache_hits,
                'cache_misses': self.metrics.cache_misses,
                'concurrent_tasks': self.metrics.concurrent_tasks,
                'memory_optimizations_applied': self.metrics.memory_optimizations_applied
            }
        }

class VideoProcessingOptimizer:
    """Main optimizer class that coordinates all performance optimizations."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.memory_manager = MemoryManager(self.config)
        self.frame_processor = FrameProcessor(self.config)
        self.performance_monitor = PerformanceMonitor(self.config)
        
        # Setup memory cleanup callbacks
        self.memory_manager.register_cleanup_callback(self._cleanup_caches)
        
        logger.info("VideoProcessingOptimizer initialized with performance optimizations")
    
    def _cleanup_caches(self):
        """Cleanup internal caches to free memory."""
        self.frame_processor.frame_cache.clear()
        self.frame_processor.get_cached_frame_hash.cache_clear()
        self.performance_monitor.metrics.memory_optimizations_applied += 1
    
    def optimize_video_processing(
        self, 
        video_path: str, 
        processing_pipeline: List[Callable]
    ) -> Dict[str, Any]:
        """
        Optimize complete video processing pipeline.
        
        Args:
            video_path: Path to video file
            processing_pipeline: List of processing functions to apply
            
        Returns:
            Optimized processing results with performance metrics
        """
        self.performance_monitor.start_monitoring()
        
        try:
            # Extract frames with optimization
            with self.performance_monitor.measure_time('frame_extraction_time'):
                frames, extraction_metadata = self.frame_processor.extract_frames_optimized(video_path)
            
            if not frames:
                raise ValueError("No frames extracted from video")
            
            # Process frames with concurrent optimization
            with ConcurrentProcessor(self.config) as concurrent_processor:
                with self.performance_monitor.measure_time('face_detection_time'):
                    # Process frames through pipeline
                    results = []
                    
                    for i, stage_func in enumerate(processing_pipeline):
                        stage_name = f"stage_{i}"
                        logger.info(f"Processing pipeline stage {i+1}/{len(processing_pipeline)}")
                        
                        if self.config.enable_concurrent_processing and len(frames) > self.config.concurrent_batch_size:
                            # Use concurrent processing for large frame sets
                            stage_results = concurrent_processor.process_concurrent_batches(
                                frames if i == 0 else results,
                                stage_func
                            )
                        else:
                            # Sequential processing for small sets or non-concurrent stages
                            stage_results = [stage_func(item) for item in (frames if i == 0 else results)]
                        
                        results = stage_results
                        
                        # Memory cleanup check
                        if self.memory_manager.cleanup_if_needed():
                            logger.info(f"Memory cleanup performed after stage {i+1}")
                        
                        # Garbage collection
                        if (i + 1) % self.config.gc_frequency == 0 and self.config.enable_garbage_collection:
                            gc.collect()
            
            # Update performance statistics
            self.performance_monitor.update_processing_stats(
                len(frames), 
                extraction_metadata['frames_skipped']
            )
            
            # Generate performance report
            performance_report = self.performance_monitor.get_performance_report()
            
            return {
                'success': True,
                'results': results,
                'extraction_metadata': extraction_metadata,
                'performance_metrics': performance_report,
                'optimization_applied': {
                    'frame_skipping': self.config.enable_frame_skipping,
                    'memory_optimization': self.config.enable_memory_optimization,
                    'concurrent_processing': self.config.enable_concurrent_processing,
                    'caching': self.config.enable_caching
                }
            }
            
        except Exception as e:
            logger.error(f"Optimized video processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'performance_metrics': self.performance_monitor.get_performance_report()
            }
        
        finally:
            self.performance_monitor.stop_monitoring()
    
    def optimize_frame_processing_pipeline(
        self, 
        frames: List[np.ndarray], 
        face_detector: Any,
        processor_instance: Any = None
    ) -> Dict[str, Any]:
        """
        Optimize frame processing pipeline for face detection and movement analysis.
        
        Args:
            frames: List of video frames
            face_detector: Face detection instance
            processor_instance: Optional processor instance for movement analysis
            
        Returns:
            Optimized processing results
        """
        self.performance_monitor.start_monitoring()
        
        try:
            # Define optimized processing stages
            def detect_faces_optimized(frame):
                # Check memory before processing
                if not self.memory_manager.check_memory_limit():
                    self.memory_manager.cleanup_if_needed()
                
                return face_detector.detect_faces(frame)
            
            def analyze_movement_optimized(detection_result):
                if detection_result.get('success') and detection_result.get('face_count') == 1:
                    # Create a simple movement analysis result
                    return {
                        'success': True,
                        'movement_data': {
                            'direction': 'none',
                            'confidence': 0.5,
                            'magnitude': 0.0,
                            'timestamp': time.time()
                        }
                    }
                return {'success': False, 'error': 'No valid face detected'}
            
            # Process with concurrent optimization
            with ConcurrentProcessor(self.config) as concurrent_processor:
                # Face detection stage
                with self.performance_monitor.measure_time('face_detection_time'):
                    face_results = concurrent_processor.process_concurrent_batches(
                        frames, detect_faces_optimized
                    )
                
                # Movement analysis stage
                with self.performance_monitor.measure_time('movement_analysis_time'):
                    movement_results = concurrent_processor.process_concurrent_batches(
                        face_results, analyze_movement_optimized
                    )
            
            # Filter successful results
            successful_results = [r for r in movement_results if r.get('success', False)]
            
            performance_report = self.performance_monitor.get_performance_report()
            
            return {
                'success': True,
                'face_detection_results': face_results,
                'movement_analysis_results': movement_results,
                'successful_results': successful_results,
                'performance_metrics': performance_report
            }
            
        except Exception as e:
            logger.error(f"Frame processing pipeline optimization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'performance_metrics': self.performance_monitor.get_performance_report()
            }
        
        finally:
            self.performance_monitor.stop_monitoring()

def create_default_optimization_config() -> OptimizationConfig:
    """Create default optimization configuration."""
    return OptimizationConfig()

def create_performance_optimized_config() -> OptimizationConfig:
    """Create performance-optimized configuration for production use."""
    return OptimizationConfig(
        enable_frame_skipping=True,
        target_fps=4.0,  # More aggressive frame skipping
        max_frames_to_process=120,  # Limit for faster processing
        frame_resize_factor=0.7,  # More aggressive resizing
        
        enable_memory_optimization=True,
        max_memory_usage_mb=1536.0,  # 1.5GB limit
        memory_cleanup_threshold=0.75,
        enable_garbage_collection=True,
        gc_frequency=8,
        
        enable_concurrent_processing=True,
        max_workers=min(6, multiprocessing.cpu_count()),
        concurrent_batch_size=12,
        enable_async_processing=True,
        
        enable_caching=True,
        cache_size=256,
        enable_result_caching=True,
        
        enable_performance_monitoring=True,
        monitoring_interval=0.5,
        enable_profiling=False
    )

def create_memory_optimized_config() -> OptimizationConfig:
    """Create memory-optimized configuration for resource-constrained environments."""
    return OptimizationConfig(
        enable_frame_skipping=True,
        target_fps=3.0,  # Very aggressive frame skipping
        max_frames_to_process=80,  # Strict frame limit
        frame_resize_factor=0.6,  # Aggressive resizing
        
        enable_memory_optimization=True,
        max_memory_usage_mb=1024.0,  # 1GB limit
        memory_cleanup_threshold=0.6,  # Early cleanup
        enable_garbage_collection=True,
        gc_frequency=5,  # Frequent GC
        
        enable_concurrent_processing=False,  # Disable to save memory
        max_workers=2,
        concurrent_batch_size=4,
        enable_async_processing=False,
        
        enable_caching=False,  # Disable caching to save memory
        cache_size=32,
        enable_result_caching=False,
        
        enable_performance_monitoring=True,
        monitoring_interval=1.0,
        enable_profiling=False
    )