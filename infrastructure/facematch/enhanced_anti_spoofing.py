"""
Enhanced Anti-Spoofing Detection Engine
Provides comprehensive anti-spoofing protection using multiple detection techniques.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import cv2

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class AntiSpoofingConfig:
    """Configuration for anti-spoofing detection."""
    texture_threshold: float = 0.3
    lighting_threshold: float = 0.4
    color_threshold: float = 0.3
    motion_threshold: float = 0.4
    overall_threshold: float = 0.35
    
    # Advanced thresholds for enhanced detection
    laplacian_variance_threshold: float = 100.0
    screen_reflection_threshold: float = 0.6
    skin_tone_variance_threshold: float = 0.15
    lighting_uniformity_threshold: float = 0.7

@dataclass
class AntiSpoofingResult:
    """Result of anti-spoofing analysis."""
    overall_score: float
    texture_score: float
    lighting_score: float
    color_score: float
    motion_score: float
    edge_score: float
    passed: bool
    confidence: float
    
    # Enhanced detection results
    laplacian_variance: float
    screen_reflection_score: float
    skin_tone_score: float
    lighting_uniformity: float
    
    # Detailed analysis
    analysis_details: Dict[str, Any]
    processing_time: float

class EnhancedAntiSpoofingEngine:
    """
    Enhanced anti-spoofing engine with comprehensive detection capabilities.
    
    Implements multiple detection techniques:
    1. Texture analysis using Laplacian variance for flat surface detection
    2. Lighting pattern analysis to detect screen reflections and artificial lighting
    3. Color distribution analysis for natural skin tone validation
    4. Motion consistency analysis for realistic movement patterns
    5. Edge density analysis for image quality assessment
    """
    
    def __init__(self, config: Optional[AntiSpoofingConfig] = None):
        """
        Initialize enhanced anti-spoofing engine.
        
        Args:
            config: Anti-spoofing configuration parameters
        """
        self.config = config or AntiSpoofingConfig()
        logger.info("Enhanced Anti-Spoofing Engine initialized")
    
    def analyze_frame(self, frame: np.ndarray) -> AntiSpoofingResult:
        """
        Perform comprehensive anti-spoofing analysis on a single frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            AntiSpoofingResult with detailed analysis
        """
        start_time = time.time()
        
        try:
            # Convert to grayscale for certain analyses
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. Enhanced texture analysis using Laplacian variance
            texture_score, laplacian_variance = self.calculate_texture_score(gray)
            
            # 2. Advanced lighting pattern analysis
            lighting_score, lighting_uniformity = self.analyze_lighting_patterns(gray)
            
            # 3. Enhanced color distribution analysis
            color_score, skin_tone_score = self.validate_color_distribution(frame)
            
            # 4. Edge density analysis
            edge_score = self.calculate_edge_density(gray)
            
            # 5. Screen reflection detection
            screen_reflection_score = self.detect_screen_reflections(frame)
            
            # Calculate overall score with weighted components
            overall_score = self._calculate_weighted_score(
                texture_score, lighting_score, color_score, edge_score, screen_reflection_score
            )
            
            # Determine if frame passes anti-spoofing checks
            passed = overall_score >= self.config.overall_threshold
            confidence = self._calculate_confidence(overall_score, [
                texture_score, lighting_score, color_score, edge_score, screen_reflection_score
            ])
            
            processing_time = time.time() - start_time
            
            # Create detailed analysis
            analysis_details = {
                'texture_analysis': {
                    'laplacian_variance': laplacian_variance,
                    'flat_surface_detected': laplacian_variance < self.config.laplacian_variance_threshold
                },
                'lighting_analysis': {
                    'uniformity': lighting_uniformity,
                    'artificial_lighting_detected': lighting_uniformity > self.config.lighting_uniformity_threshold
                },
                'color_analysis': {
                    'skin_tone_variance': skin_tone_score,
                    'unnatural_colors_detected': skin_tone_score < self.config.skin_tone_variance_threshold
                },
                'screen_detection': {
                    'reflection_score': screen_reflection_score,
                    'screen_detected': screen_reflection_score > self.config.screen_reflection_threshold
                }
            }
            
            return AntiSpoofingResult(
                overall_score=overall_score,
                texture_score=texture_score,
                lighting_score=lighting_score,
                color_score=color_score,
                motion_score=0.0,  # Will be calculated in video analysis
                edge_score=edge_score,
                passed=passed,
                confidence=confidence,
                laplacian_variance=laplacian_variance,
                screen_reflection_score=screen_reflection_score,
                skin_tone_score=skin_tone_score,
                lighting_uniformity=lighting_uniformity,
                analysis_details=analysis_details,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Anti-spoofing analysis failed: {e}")
            processing_time = time.time() - start_time
            
            return AntiSpoofingResult(
                overall_score=0.0,
                texture_score=0.0,
                lighting_score=0.0,
                color_score=0.0,
                motion_score=0.0,
                edge_score=0.0,
                passed=False,
                confidence=0.0,
                laplacian_variance=0.0,
                screen_reflection_score=0.0,
                skin_tone_score=0.0,
                lighting_uniformity=0.0,
                analysis_details={'error': str(e)},
                processing_time=processing_time
            )
    
    def calculate_texture_score(self, gray_frame: np.ndarray) -> Tuple[float, float]:
        """
        Calculate texture score using Laplacian variance for flat surface detection.
        
        Args:
            gray_frame: Grayscale input frame
            
        Returns:
            Tuple of (normalized_score, raw_laplacian_variance)
        """
        try:
            # Calculate Laplacian variance - measures texture/sharpness
            laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
            laplacian_variance = laplacian.var()
            
            # Normalize score (higher variance = more texture = more likely real)
            # Flat surfaces (photos, screens) have low Laplacian variance
            if laplacian_variance >= self.config.laplacian_variance_threshold:
                normalized_score = min(laplacian_variance / 1000.0, 1.0)
            else:
                # Penalize low variance (flat surfaces)
                normalized_score = laplacian_variance / self.config.laplacian_variance_threshold * 0.3
            
            return float(normalized_score), float(laplacian_variance)
            
        except Exception as e:
            logger.error(f"Texture analysis failed: {e}")
            return 0.0, 0.0
    
    def analyze_lighting_patterns(self, gray_frame: np.ndarray) -> Tuple[float, float]:
        """
        Analyze lighting patterns to detect screen reflections and artificial lighting.
        
        Args:
            gray_frame: Grayscale input frame
            
        Returns:
            Tuple of (lighting_score, lighting_uniformity)
        """
        try:
            # 1. Calculate histogram for brightness distribution analysis
            hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
            hist_normalized = hist.flatten() / (hist.sum() + 1e-8)
            
            # 2. Calculate entropy (measure of lighting distribution)
            entropy = -np.sum(hist_normalized * np.log(hist_normalized + 1e-8))
            
            # 3. Detect artificial lighting peaks (screens often have specific brightness levels)
            # Look for unusual peaks in brightness distribution
            peak_threshold = np.max(hist_normalized) * 0.8
            peaks = np.where(hist_normalized > peak_threshold)[0]
            
            # 4. Calculate lighting uniformity
            # Real faces have varied lighting, screens tend to be more uniform
            std_dev = np.std(gray_frame)
            mean_brightness = np.mean(gray_frame)
            lighting_uniformity = std_dev / (mean_brightness + 1e-8)
            
            # 5. Screen reflection detection using gradient analysis
            # Screens often have horizontal/vertical patterns
            grad_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate directional gradient dominance
            horizontal_energy = np.sum(np.abs(grad_x))
            vertical_energy = np.sum(np.abs(grad_y))
            total_energy = horizontal_energy + vertical_energy
            
            if total_energy > 0:
                directional_bias = abs(horizontal_energy - vertical_energy) / total_energy
            else:
                directional_bias = 0.0
            
            # Calculate final lighting score
            # Higher entropy = more natural lighting
            entropy_score = min(entropy / 8.0, 1.0)
            
            # Penalize too many brightness peaks (artificial lighting)
            peak_penalty = min(len(peaks) / 10.0, 0.5)
            
            # Penalize excessive directional bias (screen patterns)
            bias_penalty = min(directional_bias * 2.0, 0.3)
            
            lighting_score = max(0.0, entropy_score - peak_penalty - bias_penalty)
            
            return float(lighting_score), float(lighting_uniformity)
            
        except Exception as e:
            logger.error(f"Lighting analysis failed: {e}")
            return 0.0, 0.0
    
    def validate_color_distribution(self, frame: np.ndarray) -> Tuple[float, float]:
        """
        Analyze color distribution for natural skin tone validation.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (color_score, skin_tone_variance)
        """
        try:
            # Convert to different color spaces for comprehensive analysis
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            
            # 1. Enhanced skin tone detection in HSV space
            # Define multiple skin tone ranges to account for diversity
            skin_ranges = [
                # Light skin tones
                ([0, 20, 70], [20, 255, 255]),
                # Medium skin tones  
                ([0, 30, 60], [25, 255, 255]),
                # Darker skin tones
                ([0, 40, 30], [30, 255, 200])
            ]
            
            skin_masks = []
            for lower, upper in skin_ranges:
                lower_skin = np.array(lower)
                upper_skin = np.array(upper)
                mask = cv2.inRange(hsv, lower_skin, upper_skin)
                skin_masks.append(mask)
            
            # Combine all skin masks
            combined_skin_mask = np.zeros_like(skin_masks[0])
            for mask in skin_masks:
                combined_skin_mask = cv2.bitwise_or(combined_skin_mask, mask)
            
            # Calculate skin percentage
            skin_percentage = np.sum(combined_skin_mask > 0) / combined_skin_mask.size
            
            # 2. Analyze color variance in LAB space (better for skin tone analysis)
            l_channel = lab[:, :, 0]  # Lightness
            a_channel = lab[:, :, 1]  # Green-Red
            b_channel = lab[:, :, 2]  # Blue-Yellow
            
            # Calculate color variance
            l_variance = np.var(l_channel)
            a_variance = np.var(a_channel)
            b_variance = np.var(b_channel)
            
            # Natural faces have moderate color variance
            color_variance = (l_variance + a_variance + b_variance) / 3.0
            
            # 3. Detect unnatural color casts (screens often have color bias)
            mean_b, mean_g, mean_r = np.mean(frame, axis=(0, 1))
            color_balance = np.std([mean_b, mean_g, mean_r]) / (np.mean([mean_b, mean_g, mean_r]) + 1e-8)
            
            # 4. Calculate final color score
            # Optimal skin percentage range
            if 0.15 <= skin_percentage <= 0.65:
                skin_score = 1.0
            else:
                skin_score = max(0.0, 1.0 - abs(skin_percentage - 0.4) * 2.5)
            
            # Normalize color variance (moderate variance is natural)
            variance_score = min(color_variance / 1000.0, 1.0)
            if variance_score > 0.8:  # Too much variance might indicate noise
                variance_score = max(0.5, 1.0 - (variance_score - 0.8) * 2.0)
            
            # Color balance score (natural faces have balanced colors)
            balance_score = min(color_balance * 2.0, 1.0)
            
            # Combine scores
            color_score = (skin_score * 0.5 + variance_score * 0.3 + balance_score * 0.2)
            
            return float(color_score), float(color_variance)
            
        except Exception as e:
            logger.error(f"Color analysis failed: {e}")
            return 0.0, 0.0
    
    def calculate_edge_density(self, gray_frame: np.ndarray) -> float:
        """
        Calculate edge density for image quality assessment.
        
        Args:
            gray_frame: Grayscale input frame
            
        Returns:
            Edge density score
        """
        try:
            # Apply Canny edge detection with multiple thresholds for robustness
            edges_low = cv2.Canny(gray_frame, 30, 100)
            edges_high = cv2.Canny(gray_frame, 50, 150)
            
            # Combine edge maps
            edges_combined = cv2.bitwise_or(edges_low, edges_high)
            
            # Calculate edge density
            edge_density = np.sum(edges_combined > 0) / edges_combined.size
            
            # Real faces typically have moderate edge density
            # Too high = noisy/artificial, too low = flat/blurred
            optimal_range = (0.02, 0.15)
            
            if optimal_range[0] <= edge_density <= optimal_range[1]:
                score = 1.0
            elif edge_density < optimal_range[0]:
                # Too few edges (flat surface, blur)
                score = edge_density / optimal_range[0] * 0.5
            else:
                # Too many edges (noise, artificial patterns)
                excess = edge_density - optimal_range[1]
                score = max(0.0, 1.0 - excess * 5.0)
            
            return float(score)
            
        except Exception as e:
            logger.error(f"Edge density calculation failed: {e}")
            return 0.0
    
    def detect_screen_reflections(self, frame: np.ndarray) -> float:
        """
        Detect screen reflections and artificial lighting patterns.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Screen reflection score (higher = more likely screen)
        """
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. Detect periodic patterns (common in screens)
            # Apply FFT to detect regular patterns
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Look for peaks in frequency domain (indicates regular patterns)
            # Screens often have regular pixel patterns
            center_y, center_x = np.array(magnitude_spectrum.shape) // 2
            
            # Create a mask to exclude DC component
            mask = np.ones_like(magnitude_spectrum)
            mask[center_y-5:center_y+5, center_x-5:center_x+5] = 0
            
            # Find peaks in frequency domain
            masked_spectrum = magnitude_spectrum * mask
            peak_threshold = np.mean(masked_spectrum) + 2 * np.std(masked_spectrum)
            peaks = np.sum(masked_spectrum > peak_threshold)
            
            # Normalize peak count
            total_pixels = magnitude_spectrum.size
            peak_ratio = peaks / total_pixels
            
            # 2. Detect moiré patterns (interference between camera and screen)
            # Apply Gabor filters to detect regular textures
            kernel_size = 21
            sigma = 3
            theta = 0  # Horizontal patterns
            lambd = 10  # Wavelength
            gamma = 0.5
            psi = 0
            
            gabor_kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambd, gamma, psi)
            gabor_response = cv2.filter2D(gray, cv2.CV_8UC3, gabor_kernel)
            
            # Calculate response strength
            gabor_energy = np.mean(np.abs(gabor_response))
            
            # 3. Detect uniform brightness regions (common in screens)
            # Calculate local standard deviation
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
            local_std = np.sqrt(local_variance)
            
            # Count pixels with very low local variation
            low_variation_pixels = np.sum(local_std < 5)
            uniformity_ratio = low_variation_pixels / gray.size
            
            # 4. Combine indicators
            # Higher values indicate more screen-like characteristics
            frequency_score = min(peak_ratio * 1000, 1.0)  # Normalize
            gabor_score = min(gabor_energy / 50.0, 1.0)    # Normalize
            uniformity_score = uniformity_ratio
            
            # Weighted combination
            screen_score = (frequency_score * 0.4 + gabor_score * 0.3 + uniformity_score * 0.3)
            
            return float(screen_score)
            
        except Exception as e:
            logger.error(f"Screen reflection detection failed: {e}")
            return 0.0
    
    def assess_motion_consistency(self, frames: List[np.ndarray]) -> float:
        """
        Assess motion consistency across multiple frames to detect artificial motion.
        
        Args:
            frames: List of consecutive frames
            
        Returns:
            Motion consistency score
        """
        try:
            if len(frames) < 2:
                return 0.5  # Neutral score for insufficient data
            
            motion_scores = []
            
            for i in range(1, len(frames)):
                prev_frame = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
                curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                
                # Calculate optical flow
                flow = cv2.calcOpticalFlowPyrLK(
                    prev_frame, curr_frame, None, None,
                    winSize=(15, 15),
                    maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                )
                
                # Analyze motion patterns
                if flow[0] is not None and len(flow[0]) > 0:
                    # Calculate motion magnitude
                    motion_vectors = flow[0] - flow[1] if flow[1] is not None else flow[0]
                    motion_magnitude = np.linalg.norm(motion_vectors, axis=1)
                    
                    # Natural motion has certain characteristics
                    mean_motion = np.mean(motion_magnitude)
                    motion_variance = np.var(motion_magnitude)
                    
                    # Score based on natural motion characteristics
                    # Real head movement has moderate, varied motion
                    if 1.0 <= mean_motion <= 10.0 and motion_variance > 0.5:
                        motion_score = 1.0
                    else:
                        motion_score = 0.5
                    
                    motion_scores.append(motion_score)
            
            return float(np.mean(motion_scores)) if motion_scores else 0.5
            
        except Exception as e:
            logger.error(f"Motion consistency assessment failed: {e}")
            return 0.5
    
    def _calculate_weighted_score(self, texture_score: float, lighting_score: float, 
                                color_score: float, edge_score: float, 
                                screen_score: float) -> float:
        """
        Calculate weighted overall anti-spoofing score.
        
        Args:
            texture_score: Texture analysis score
            lighting_score: Lighting analysis score
            color_score: Color distribution score
            edge_score: Edge density score
            screen_score: Screen reflection score (inverted for final calculation)
            
        Returns:
            Weighted overall score
        """
        # Invert screen score (lower screen detection = higher liveness)
        inverted_screen_score = 1.0 - screen_score
        
        # Weighted combination based on importance
        weights = {
            'texture': 0.25,
            'lighting': 0.20,
            'color': 0.20,
            'edge': 0.15,
            'screen': 0.20
        }
        
        overall_score = (
            texture_score * weights['texture'] +
            lighting_score * weights['lighting'] +
            color_score * weights['color'] +
            edge_score * weights['edge'] +
            inverted_screen_score * weights['screen']
        )
        
        return float(np.clip(overall_score, 0.0, 1.0))
    
    def _calculate_confidence(self, overall_score: float, individual_scores: List[float]) -> float:
        """
        Calculate confidence in the anti-spoofing decision.
        
        Args:
            overall_score: Overall anti-spoofing score
            individual_scores: List of individual component scores
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Confidence is higher when:
        # 1. Overall score is far from threshold
        # 2. Individual scores are consistent (low variance)
        
        threshold_distance = abs(overall_score - self.config.overall_threshold)
        score_variance = np.var(individual_scores)
        
        # Distance component (further from threshold = higher confidence)
        distance_confidence = min(threshold_distance * 2.0, 1.0)
        
        # Consistency component (lower variance = higher confidence)
        consistency_confidence = max(0.0, 1.0 - score_variance * 2.0)
        
        # Combine components
        confidence = (distance_confidence * 0.6 + consistency_confidence * 0.4)
        
        return float(np.clip(confidence, 0.0, 1.0))

def create_enhanced_anti_spoofing_engine(config: Optional[AntiSpoofingConfig] = None) -> EnhancedAntiSpoofingEngine:
    """
    Factory function to create enhanced anti-spoofing engine.
    
    Args:
        config: Optional configuration parameters
        
    Returns:
        Configured EnhancedAntiSpoofingEngine instance
    """
    return EnhancedAntiSpoofingEngine(config)

def create_default_anti_spoofing_config() -> AntiSpoofingConfig:
    """
    Create default anti-spoofing configuration.
    
    Returns:
        Default AntiSpoofingConfig instance
    """
    return AntiSpoofingConfig()