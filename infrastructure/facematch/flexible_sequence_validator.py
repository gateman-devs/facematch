"""
Enhanced Flexible Sequence Validation Algorithm

This module implements a comprehensive flexible sequence validation system that replaces
rigid time-based segmentation with advanced sliding window analysis. It provides tolerance
for extra movements, timing variations, and natural user behavior patterns.

Requirements addressed:
- 2.1: Timing variation tolerance
- 2.2: Extra movement filtering  
- 2.3: Pause tolerance
- 2.4: Speed adaptation
- 2.5: Sequence detection anywhere in video
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationStrategy(Enum):
    """Different validation strategies in order of preference."""
    EXACT_CONSECUTIVE = "exact_consecutive"
    FLEXIBLE_CONSECUTIVE = "flexible_consecutive"
    SLIDING_WINDOW_STRICT = "sliding_window_strict"
    SLIDING_WINDOW_FLEXIBLE = "sliding_window_flexible"
    SUBSEQUENCE_MATCHING = "subsequence_matching"
    ADAPTIVE_TEMPORAL = "adaptive_temporal"


@dataclass
class SequenceMatchResult:
    """Result of sequence matching operation."""
    found: bool
    accuracy: float
    strategy: str
    matched_movements: List[str]
    start_index: int = -1
    end_index: int = -1
    match_count: int = 0
    timing_tolerance_used: bool = False
    extra_movements_ignored: int = 0
    confidence_scores: List[float] = None
    temporal_gaps: List[float] = None
    speed_adaptation_used: bool = False
    pause_tolerance_used: bool = False


@dataclass
class ValidationConfig:
    """Configuration for flexible sequence validation."""
    # Timing tolerance settings
    max_timing_variation: float = 2.0  # Maximum timing variation in seconds
    pause_tolerance: float = 3.0  # Maximum pause duration in seconds
    min_movement_duration: float = 0.5  # Minimum movement duration
    
    # Speed adaptation settings
    speed_adaptation_enabled: bool = True
    min_speed_factor: float = 0.5  # Minimum speed (50% of normal)
    max_speed_factor: float = 2.0  # Maximum speed (200% of normal)
    
    # Extra movement tolerance
    max_extra_movements: int = 3  # Maximum extra movements to ignore
    extra_movement_penalty: float = 0.1  # Penalty per extra movement
    
    # Accuracy thresholds
    exact_match_threshold: float = 0.95
    flexible_match_threshold: float = 0.80
    minimum_match_threshold: float = 0.65
    
    # Confidence requirements
    min_movement_confidence: float = 0.3
    min_average_confidence: float = 0.5


class FlexibleSequenceValidator:
    """
    Enhanced flexible sequence validation algorithm that implements sliding window analysis
    to detect required sequences anywhere in the video with comprehensive tolerance for
    natural user behavior patterns.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize the flexible sequence validator."""
        self.config = config or ValidationConfig()
        self.validation_history = []
        
    def validate_sequence(self, detected_movements: List[Dict], expected_sequence: List[str]) -> Dict:
        """
        Main validation method that implements flexible sequence validation algorithm.
        
        Args:
            detected_movements: List of detected movement dictionaries with direction, confidence, timing
            expected_sequence: Expected sequence of movement directions
            
        Returns:
            Comprehensive validation result dictionary
        """
        start_time = time.time()
        
        # Input validation
        if not detected_movements or not expected_sequence:
            return self._create_failure_result(
                "Insufficient data for sequence validation",
                detected_movements, expected_sequence
            )
        
        logger.info(f"Starting flexible sequence validation: {len(detected_movements)} movements, "
                   f"expected sequence: {expected_sequence}")
        
        # Preprocess movements for better analysis
        processed_movements = self._preprocess_movements(detected_movements)
        
        # Filter movements by confidence threshold
        confident_movements = self._filter_by_confidence(processed_movements)
        
        if len(confident_movements) < len(expected_sequence):
            return self._create_failure_result(
                f"Insufficient confident movements: {len(confident_movements)} < {len(expected_sequence)}",
                detected_movements, expected_sequence
            )
        
        # Try multiple validation strategies in order of preference
        strategies = [
            (ValidationStrategy.EXACT_CONSECUTIVE, self._validate_exact_consecutive),
            (ValidationStrategy.FLEXIBLE_CONSECUTIVE, self._validate_flexible_consecutive),
            (ValidationStrategy.SLIDING_WINDOW_STRICT, self._validate_sliding_window_strict),
            (ValidationStrategy.SLIDING_WINDOW_FLEXIBLE, self._validate_sliding_window_flexible),
            (ValidationStrategy.ADAPTIVE_TEMPORAL, self._validate_adaptive_temporal),
            (ValidationStrategy.SUBSEQUENCE_MATCHING, self._validate_subsequence_matching)
        ]
        
        best_result = None
        
        for strategy, validator_func in strategies:
            try:
                result = validator_func(confident_movements, expected_sequence)
                result.strategy = strategy.value
                
                logger.info(f"Strategy {strategy.value}: found={result.found}, accuracy={result.accuracy:.3f}")
                
                if result.found:
                    # Use the first successful strategy with good accuracy
                    if result.accuracy >= self.config.exact_match_threshold:
                        logger.info(f"Excellent match found with {strategy.value}")
                        best_result = result
                        break
                    elif best_result is None or result.accuracy > best_result.accuracy:
                        best_result = result
                        
            except Exception as e:
                logger.warning(f"Strategy {strategy.value} failed: {e}")
                continue
        
        # Create final result
        processing_time = time.time() - start_time
        
        if best_result and best_result.found and best_result.accuracy >= self.config.minimum_match_threshold:
            return self._create_success_result(best_result, detected_movements, expected_sequence, processing_time)
        else:
            # Provide detailed failure analysis
            failure_analysis = self._analyze_failure(confident_movements, expected_sequence)
            return self._create_failure_result(
                failure_analysis['reason'], detected_movements, expected_sequence, 
                processing_time, failure_analysis
            )
    
    def _preprocess_movements(self, movements: List[Dict]) -> List[Dict]:
        """Preprocess movements to add temporal analysis and normalize data."""
        processed = []
        
        for i, movement in enumerate(movements):
            processed_movement = movement.copy()
            
            # Add temporal context
            if i > 0:
                time_gap = movement.get('start_time', 0) - movements[i-1].get('end_time', 0)
                processed_movement['time_gap_before'] = time_gap
            else:
                processed_movement['time_gap_before'] = 0.0
            
            # Calculate movement duration
            start_time = movement.get('start_time', 0)
            end_time = movement.get('end_time', start_time + 1.0)
            processed_movement['duration'] = end_time - start_time
            
            # Normalize confidence score
            confidence = movement.get('confidence', 0.5)
            processed_movement['normalized_confidence'] = max(0.0, min(1.0, confidence))
            
            processed.append(processed_movement)
        
        return processed
    
    def _filter_by_confidence(self, movements: List[Dict]) -> List[Dict]:
        """Filter movements by confidence threshold."""
        return [
            movement for movement in movements 
            if movement.get('normalized_confidence', 0.0) >= self.config.min_movement_confidence
        ]
    
    def _validate_exact_consecutive(self, movements: List[Dict], expected_sequence: List[str]) -> SequenceMatchResult:
        """Validate exact consecutive sequence match."""
        for start_idx in range(len(movements) - len(expected_sequence) + 1):
            match_result = self._check_consecutive_match(
                movements[start_idx:start_idx + len(expected_sequence)], 
                expected_sequence, 
                strict=True
            )
            
            if match_result.found and match_result.accuracy >= self.config.exact_match_threshold:
                match_result.start_index = start_idx
                match_result.end_index = start_idx + len(expected_sequence) - 1
                return match_result
        
        return SequenceMatchResult(found=False, accuracy=0.0, strategy="", matched_movements=[])
    
    def _validate_flexible_consecutive(self, movements: List[Dict], expected_sequence: List[str]) -> SequenceMatchResult:
        """Validate flexible consecutive sequence with tolerance for mismatches."""
        best_result = SequenceMatchResult(found=False, accuracy=0.0, strategy="", matched_movements=[])
        
        for start_idx in range(len(movements) - len(expected_sequence) + 1):
            match_result = self._check_consecutive_match(
                movements[start_idx:start_idx + len(expected_sequence)], 
                expected_sequence, 
                strict=False
            )
            
            if match_result.found and match_result.accuracy > best_result.accuracy:
                match_result.start_index = start_idx
                match_result.end_index = start_idx + len(expected_sequence) - 1
                match_result.timing_tolerance_used = True
                best_result = match_result
        
        return best_result
    
    def _validate_sliding_window_strict(self, movements: List[Dict], expected_sequence: List[str]) -> SequenceMatchResult:
        """Validate using sliding window with strict matching but allowing extra movements."""
        best_result = SequenceMatchResult(found=False, accuracy=0.0, strategy="", matched_movements=[])
        
        # Try different window sizes to accommodate extra movements
        max_window_size = min(len(movements), len(expected_sequence) + self.config.max_extra_movements)
        
        for window_size in range(len(expected_sequence), max_window_size + 1):
            for start_idx in range(len(movements) - window_size + 1):
                window_movements = movements[start_idx:start_idx + window_size]
                match_result = self._find_sequence_in_window(window_movements, expected_sequence, strict=True)
                
                if match_result.found and match_result.accuracy > best_result.accuracy:
                    match_result.start_index = start_idx
                    match_result.end_index = start_idx + window_size - 1
                    match_result.extra_movements_ignored = window_size - len(expected_sequence)
                    best_result = match_result
        
        return best_result
    
    def _validate_sliding_window_flexible(self, movements: List[Dict], expected_sequence: List[str]) -> SequenceMatchResult:
        """Validate using sliding window with flexible matching."""
        best_result = SequenceMatchResult(found=False, accuracy=0.0, strategy="", matched_movements=[])
        
        # Use larger windows for maximum flexibility
        max_window_size = min(len(movements), len(expected_sequence) * 2)
        
        for window_size in range(len(expected_sequence), max_window_size + 1):
            for start_idx in range(len(movements) - window_size + 1):
                window_movements = movements[start_idx:start_idx + window_size]
                match_result = self._find_sequence_in_window(window_movements, expected_sequence, strict=False)
                
                if match_result.found and match_result.accuracy > best_result.accuracy:
                    match_result.start_index = start_idx
                    match_result.end_index = start_idx + window_size - 1
                    match_result.extra_movements_ignored = window_size - len(expected_sequence)
                    match_result.timing_tolerance_used = True
                    best_result = match_result
        
        return best_result
    
    def _validate_adaptive_temporal(self, movements: List[Dict], expected_sequence: List[str]) -> SequenceMatchResult:
        """
        Validate using adaptive temporal analysis that accounts for natural timing variations,
        pauses, and speed differences.
        """
        # Analyze temporal patterns in the movements
        temporal_analysis = self._analyze_temporal_patterns(movements)
        
        # Adapt validation parameters based on detected patterns
        adapted_config = self._adapt_validation_config(temporal_analysis)
        
        # Find sequence matches with temporal tolerance
        best_result = SequenceMatchResult(found=False, accuracy=0.0, strategy="", matched_movements=[])
        
        for start_idx in range(len(movements) - len(expected_sequence) + 1):
            match_result = self._check_temporal_sequence_match(
                movements[start_idx:], expected_sequence, adapted_config
            )
            
            if match_result.found and match_result.accuracy > best_result.accuracy:
                match_result.start_index = start_idx
                match_result.speed_adaptation_used = True
                match_result.pause_tolerance_used = temporal_analysis['has_long_pauses']
                best_result = match_result
        
        return best_result
    
    def _validate_subsequence_matching(self, movements: List[Dict], expected_sequence: List[str]) -> SequenceMatchResult:
        """Validate using longest common subsequence matching."""
        movement_directions = [m['direction'] for m in movements]
        
        # Use dynamic programming to find the best subsequence match
        match_result = self._find_longest_common_subsequence(
            movement_directions, expected_sequence, movements
        )
        
        if match_result.found:
            match_result.timing_tolerance_used = True
            match_result.extra_movements_ignored = len(movements) - len(expected_sequence)
        
        return match_result 
   
    def _check_consecutive_match(self, movements: List[Dict], expected_sequence: List[str], strict: bool = True) -> SequenceMatchResult:
        """Check for consecutive sequence match with optional flexibility."""
        if len(movements) != len(expected_sequence):
            return SequenceMatchResult(found=False, accuracy=0.0, strategy="", matched_movements=[])
        
        match_count = 0
        matched_movements = []
        confidence_scores = []
        temporal_gaps = []
        
        for i, (movement, expected_direction) in enumerate(zip(movements, expected_sequence)):
            detected_direction = movement['direction']
            confidence = movement.get('normalized_confidence', 0.5)
            
            matched_movements.append(detected_direction)
            
            if detected_direction == expected_direction:
                match_count += 1
                confidence_scores.append(confidence)
            else:
                # In flexible mode, give partial credit for mismatches
                confidence_scores.append(0.2 if not strict else 0.0)
            
            # Track temporal gaps for pause analysis
            if i > 0:
                gap = movement.get('time_gap_before', 0.0)
                temporal_gaps.append(gap)
        
        # Calculate accuracy with timing and confidence considerations
        base_accuracy = match_count / len(expected_sequence)
        confidence_factor = sum(confidence_scores) / len(confidence_scores)
        
        # Apply timing penalty for excessive gaps (pauses)
        timing_penalty = 0.0
        if temporal_gaps:
            avg_gap = sum(temporal_gaps) / len(temporal_gaps)
            if avg_gap > self.config.pause_tolerance:
                timing_penalty = min(0.2, (avg_gap - self.config.pause_tolerance) * 0.1)
        
        final_accuracy = (base_accuracy * confidence_factor) - timing_penalty
        
        # Determine if match is successful
        threshold = self.config.exact_match_threshold if strict else self.config.flexible_match_threshold
        required_matches = len(expected_sequence) if strict else int(len(expected_sequence) * 0.75)
        
        found = match_count >= required_matches and final_accuracy >= threshold
        
        return SequenceMatchResult(
            found=found,
            accuracy=final_accuracy,
            strategy="",
            matched_movements=matched_movements,
            match_count=match_count,
            confidence_scores=confidence_scores,
            temporal_gaps=temporal_gaps,
            pause_tolerance_used=any(gap > self.config.pause_tolerance for gap in temporal_gaps)
        )
    
    def _find_sequence_in_window(self, window_movements: List[Dict], expected_sequence: List[str], strict: bool = True) -> SequenceMatchResult:
        """Find the expected sequence within a window of movements."""
        window_directions = [m['direction'] for m in window_movements]
        best_result = SequenceMatchResult(found=False, accuracy=0.0, strategy="", matched_movements=[])
        
        # Try all possible starting positions within the window
        for start_pos in range(len(window_directions) - len(expected_sequence) + 1):
            match_count = 0
            matched_movements = []
            confidence_scores = []
            
            for i, expected_direction in enumerate(expected_sequence):
                pos = start_pos + i
                if pos < len(window_directions):
                    detected_direction = window_directions[pos]
                    confidence = window_movements[pos].get('normalized_confidence', 0.5)
                    
                    matched_movements.append(detected_direction)
                    
                    if detected_direction == expected_direction:
                        match_count += 1
                        confidence_scores.append(confidence)
                    else:
                        confidence_scores.append(0.1 if not strict else 0.0)
            
            # Calculate accuracy
            base_accuracy = match_count / len(expected_sequence)
            confidence_factor = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            final_accuracy = base_accuracy * confidence_factor
            
            # Apply penalty for extra movements
            extra_movements = len(window_movements) - len(expected_sequence)
            if extra_movements > 0:
                penalty = min(0.2, extra_movements * self.config.extra_movement_penalty)
                final_accuracy -= penalty
            
            # Check if this is the best match so far
            threshold = self.config.exact_match_threshold if strict else self.config.flexible_match_threshold
            required_matches = len(expected_sequence) if strict else int(len(expected_sequence) * 0.75)
            
            if match_count >= required_matches and final_accuracy > best_result.accuracy:
                best_result = SequenceMatchResult(
                    found=final_accuracy >= threshold,
                    accuracy=final_accuracy,
                    strategy="",
                    matched_movements=matched_movements,
                    match_count=match_count,
                    confidence_scores=confidence_scores,
                    extra_movements_ignored=extra_movements
                )
        
        return best_result
    
    def _analyze_temporal_patterns(self, movements: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal patterns in the movement sequence."""
        if len(movements) < 2:
            return {
                'avg_duration': 1.0,
                'avg_gap': 0.5,
                'has_long_pauses': False,
                'speed_factor': 1.0,
                'timing_consistency': 1.0
            }
        
        durations = [m.get('duration', 1.0) for m in movements]
        gaps = [m.get('time_gap_before', 0.0) for m in movements[1:]]
        
        avg_duration = sum(durations) / len(durations)
        avg_gap = sum(gaps) / len(gaps) if gaps else 0.0
        
        # Detect long pauses
        has_long_pauses = any(gap > self.config.pause_tolerance for gap in gaps)
        
        # Estimate speed factor based on movement durations
        expected_duration = 1.5  # Expected duration per movement
        speed_factor = expected_duration / avg_duration if avg_duration > 0 else 1.0
        speed_factor = max(self.config.min_speed_factor, min(self.config.max_speed_factor, speed_factor))
        
        # Calculate timing consistency
        if len(durations) > 1:
            duration_std = np.std(durations)
            timing_consistency = max(0.0, 1.0 - (duration_std / avg_duration))
        else:
            timing_consistency = 1.0
        
        return {
            'avg_duration': avg_duration,
            'avg_gap': avg_gap,
            'has_long_pauses': has_long_pauses,
            'speed_factor': speed_factor,
            'timing_consistency': timing_consistency,
            'durations': durations,
            'gaps': gaps
        }
    
    def _adapt_validation_config(self, temporal_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt validation configuration based on temporal analysis."""
        adapted_config = {
            'timing_tolerance': self.config.max_timing_variation,
            'pause_tolerance': self.config.pause_tolerance,
            'speed_factor': temporal_analysis['speed_factor'],
            'confidence_adjustment': 1.0
        }
        
        # Adjust tolerance based on detected patterns
        if temporal_analysis['has_long_pauses']:
            adapted_config['pause_tolerance'] *= 1.5
            adapted_config['confidence_adjustment'] *= 0.9  # Slight penalty for pauses
        
        if temporal_analysis['timing_consistency'] < 0.7:
            adapted_config['timing_tolerance'] *= 1.3
            adapted_config['confidence_adjustment'] *= 0.95
        
        # Adjust for speed variations
        if temporal_analysis['speed_factor'] < 0.8 or temporal_analysis['speed_factor'] > 1.2:
            adapted_config['confidence_adjustment'] *= 0.9
        
        return adapted_config
    
    def _check_temporal_sequence_match(self, movements: List[Dict], expected_sequence: List[str], adapted_config: Dict[str, Any]) -> SequenceMatchResult:
        """Check sequence match with temporal adaptation."""
        if len(movements) < len(expected_sequence):
            return SequenceMatchResult(found=False, accuracy=0.0, strategy="", matched_movements=[])
        
        best_result = SequenceMatchResult(found=False, accuracy=0.0, strategy="", matched_movements=[])
        
        # Use dynamic window sizing based on temporal patterns
        max_search_length = min(len(movements), len(expected_sequence) * 2)
        
        current_pos = 0
        matched_movements = []
        confidence_scores = []
        sequence_index = 0
        
        while current_pos < len(movements) and sequence_index < len(expected_sequence):
            movement = movements[current_pos]
            expected_direction = expected_sequence[sequence_index]
            
            if movement['direction'] == expected_direction:
                # Found matching movement
                matched_movements.append(movement['direction'])
                confidence_scores.append(movement.get('normalized_confidence', 0.5))
                sequence_index += 1
                current_pos += 1
            else:
                # Check if we should skip this movement (extra movement tolerance)
                if len(matched_movements) < sequence_index:
                    # We're still looking for the next expected movement
                    current_pos += 1
                    if current_pos - len(matched_movements) > self.config.max_extra_movements:
                        break  # Too many extra movements
                else:
                    # Move to next movement
                    current_pos += 1
        
        # Calculate final accuracy
        if sequence_index == len(expected_sequence):
            base_accuracy = len(matched_movements) / len(expected_sequence)
            confidence_factor = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            speed_adjustment = adapted_config.get('confidence_adjustment', 1.0)
            
            final_accuracy = base_accuracy * confidence_factor * speed_adjustment
            
            return SequenceMatchResult(
                found=final_accuracy >= self.config.flexible_match_threshold,
                accuracy=final_accuracy,
                strategy="",
                matched_movements=matched_movements,
                match_count=len(matched_movements),
                confidence_scores=confidence_scores,
                end_index=current_pos - 1
            )
        
        return best_result
    
    def _find_longest_common_subsequence(self, detected_directions: List[str], expected_sequence: List[str], movements: List[Dict]) -> SequenceMatchResult:
        """Find longest common subsequence using dynamic programming."""
        m, n = len(detected_directions), len(expected_sequence)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if detected_directions[i-1] == expected_sequence[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        
        # Check if we have a good enough match
        if lcs_length >= len(expected_sequence) * 0.75:
            # Backtrack to find the actual subsequence
            matched_movements, confidence_scores = self._backtrack_lcs(
                dp, detected_directions, expected_sequence, movements, m, n
            )
            
            base_accuracy = lcs_length / len(expected_sequence)
            confidence_factor = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            final_accuracy = base_accuracy * confidence_factor
            
            return SequenceMatchResult(
                found=final_accuracy >= self.config.minimum_match_threshold,
                accuracy=final_accuracy,
                strategy="",
                matched_movements=matched_movements,
                match_count=lcs_length,
                confidence_scores=confidence_scores,
                start_index=0,
                end_index=len(detected_directions) - 1
            )
        
        return SequenceMatchResult(found=False, accuracy=0.0, strategy="", matched_movements=[])
    
    def _backtrack_lcs(self, dp: List[List[int]], detected: List[str], expected: List[str], 
                      movements: List[Dict], i: int, j: int) -> Tuple[List[str], List[float]]:
        """Backtrack through DP table to find actual LCS and confidences."""
        matched_movements = []
        confidence_scores = []
        
        while i > 0 and j > 0:
            if detected[i-1] == expected[j-1]:
                matched_movements.append(detected[i-1])
                confidence_scores.append(movements[i-1].get('normalized_confidence', 0.5))
                i -= 1
                j -= 1
            elif dp[i-1][j] > dp[i][j-1]:
                i -= 1
            else:
                j -= 1
        
        # Reverse to get correct order
        matched_movements.reverse()
        confidence_scores.reverse()
        
        return matched_movements, confidence_scores
    
    def _analyze_failure(self, movements: List[Dict], expected_sequence: List[str]) -> Dict[str, Any]:
        """Analyze why validation failed and provide detailed feedback."""
        analysis = {
            'reason': 'Unknown failure',
            'detected_movements': [m['direction'] for m in movements],
            'expected_sequence': expected_sequence,
            'movement_count': len(movements),
            'expected_count': len(expected_sequence),
            'confidence_issues': [],
            'timing_issues': [],
            'sequence_issues': []
        }
        
        # Check movement count
        if len(movements) < len(expected_sequence):
            analysis['reason'] = f"Insufficient movements: {len(movements)} < {len(expected_sequence)}"
            return analysis
        
        # Check confidence issues
        low_confidence_movements = [
            i for i, m in enumerate(movements) 
            if m.get('normalized_confidence', 0.0) < self.config.min_movement_confidence
        ]
        if low_confidence_movements:
            analysis['confidence_issues'] = low_confidence_movements
        
        # Check timing issues
        long_gaps = []
        for i, movement in enumerate(movements[1:], 1):
            gap = movement.get('time_gap_before', 0.0)
            if gap > self.config.pause_tolerance:
                long_gaps.append((i, gap))
        if long_gaps:
            analysis['timing_issues'] = long_gaps
        
        # Check sequence matching issues
        detected_dirs = [m['direction'] for m in movements]
        partial_matches = []
        for i in range(len(detected_dirs) - len(expected_sequence) + 1):
            window = detected_dirs[i:i + len(expected_sequence)]
            matches = sum(1 for d, e in zip(window, expected_sequence) if d == e)
            if matches > 0:
                partial_matches.append((i, matches, window))
        
        if partial_matches:
            best_partial = max(partial_matches, key=lambda x: x[1])
            analysis['sequence_issues'] = {
                'best_partial_match': best_partial,
                'all_partial_matches': partial_matches
            }
            analysis['reason'] = f"Best partial match: {best_partial[1]}/{len(expected_sequence)} movements"
        else:
            analysis['reason'] = "No matching subsequence found"
        
        return analysis
    
    def _create_success_result(self, match_result: SequenceMatchResult, original_movements: List[Dict], 
                             expected_sequence: List[str], processing_time: float) -> Dict:
        """Create comprehensive success result dictionary."""
        return {
            'success': True,
            'passed': True,
            'accuracy': float(match_result.accuracy),
            'detected_sequence': match_result.matched_movements,
            'expected_sequence': expected_sequence,
            'all_movements': [m['direction'] for m in original_movements],
            'validation_strategy': match_result.strategy,
            'match_details': {
                'start_index': match_result.start_index,
                'end_index': match_result.end_index,
                'match_count': match_result.match_count,
                'confidence_scores': match_result.confidence_scores or [],
                'timing_tolerance_used': match_result.timing_tolerance_used,
                'extra_movements_ignored': match_result.extra_movements_ignored,
                'speed_adaptation_used': match_result.speed_adaptation_used,
                'pause_tolerance_used': match_result.pause_tolerance_used
            },
            'performance_metrics': {
                'total_movements_detected': len(original_movements),
                'processing_time': processing_time,
                'average_confidence': sum(match_result.confidence_scores or [0]) / max(1, len(match_result.confidence_scores or []))
            },
            'flexibility_features': {
                'timing_variations_handled': match_result.timing_tolerance_used,
                'extra_movements_filtered': match_result.extra_movements_ignored > 0,
                'pause_tolerance_applied': match_result.pause_tolerance_used,
                'speed_adaptation_applied': match_result.speed_adaptation_used,
                'sequence_found_anywhere': match_result.start_index > 0 or match_result.end_index < len(original_movements) - 1
            }
        }
    
    def _create_failure_result(self, reason: str, original_movements: List[Dict], 
                             expected_sequence: List[str], processing_time: float = 0.0, 
                             failure_analysis: Optional[Dict] = None) -> Dict:
        """Create comprehensive failure result dictionary."""
        return {
            'success': True,  # Processing succeeded, but validation failed
            'passed': False,
            'accuracy': 0.0,
            'detected_sequence': [m['direction'] for m in original_movements],
            'expected_sequence': expected_sequence,
            'all_movements': [m['direction'] for m in original_movements],
            'error': reason,
            'validation_strategy': 'none',
            'failure_analysis': failure_analysis or {},
            'performance_metrics': {
                'total_movements_detected': len(original_movements),
                'processing_time': processing_time
            },
            'debug_info': {
                'movement_confidences': [m.get('confidence', 0.0) for m in original_movements],
                'movement_timings': [m.get('start_time', 0.0) for m in original_movements],
                'config_used': {
                    'min_confidence': self.config.min_movement_confidence,
                    'timing_tolerance': self.config.max_timing_variation,
                    'pause_tolerance': self.config.pause_tolerance,
                    'min_accuracy': self.config.minimum_match_threshold
                }
            }
        }


def create_flexible_sequence_validator(config: Optional[ValidationConfig] = None) -> FlexibleSequenceValidator:
    """Factory function to create a flexible sequence validator."""
    return FlexibleSequenceValidator(config)


def create_default_validation_config() -> ValidationConfig:
    """Create default validation configuration."""
    return ValidationConfig()


def create_strict_validation_config() -> ValidationConfig:
    """Create strict validation configuration for high-security scenarios."""
    return ValidationConfig(
        max_timing_variation=1.0,
        pause_tolerance=2.0,
        max_extra_movements=1,
        exact_match_threshold=0.98,
        flexible_match_threshold=0.90,
        minimum_match_threshold=0.80,
        min_movement_confidence=0.5,
        min_average_confidence=0.7
    )


def create_lenient_validation_config() -> ValidationConfig:
    """Create lenient validation configuration for better user experience."""
    return ValidationConfig(
        max_timing_variation=3.0,
        pause_tolerance=5.0,
        max_extra_movements=5,
        exact_match_threshold=0.85,
        flexible_match_threshold=0.70,
        minimum_match_threshold=0.55,
        min_movement_confidence=0.2,
        min_average_confidence=0.4
    )