#!/usr/bin/env python3
"""
Test script to specifically analyze up and down movement detection.
"""

import logging
import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add the infrastructure directory to the path
sys.path.insert(0, str(Path(__file__).parent / "infrastructure"))

from facematch.simple_mediapipe_detector import SimpleMediaPipeDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_up_down_movements():
    """Test up and down movement detection specifically."""
    
    print("=== Up/Down Movement Analysis ===\n")
    
    # Create detector with very low threshold
    detector = SimpleMediaPipeDetector(
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        movement_threshold=0.005,  # Very low threshold
        max_history=100,
        debug_mode=True
    )
    
    video_path = "test_video.mov"
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    print(f"📹 Analyzing up/down movements in: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Failed to open video")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Focus on the later part of the video where up/down movements occur
    # Based on debug output, up movements start around frame 270 (9.0s)
    # and down movements around frame 292 (9.7s)
    
    # Skip to frame 250 (around 8.3s)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 250)
    
    frame_count = 250
    pose_history = []
    movements = []
    
    # Process frames from 250 onwards
    while frame_count < 350:  # Process about 100 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        result = detector._process_frame(frame, frame_count)
        
        if result and result.get('pose'):
            pose = result['pose']
            pose_history.append({
                'frame': frame_count,
                'timestamp': frame_count / fps,
                'x': pose['x'],
                'y': pose['y'],
                'quality_score': pose.get('quality_score', 0)
            })
            
            # Calculate movement if we have previous pose
            if len(pose_history) > 1:
                prev_pose = pose_history[-2]
                dx = pose['x'] - prev_pose['x']
                dy = pose['y'] - prev_pose['y']
                magnitude = np.sqrt(dx**2 + dy**2)
                
                # Test movement detection
                if magnitude > 0.005:  # Very low threshold
                    direction = detector._detect_direction_improved(dx, dy, magnitude)
                    
                    movements.append({
                        'frame': frame_count,
                        'timestamp': frame_count / fps,
                        'dx': dx,
                        'dy': dy,
                        'magnitude': magnitude,
                        'direction': direction
                    })
                    
                    # Print significant movements
                    if magnitude > 0.008:  # Show significant movements
                        print(f"Frame {frame_count:3d} ({frame_count/fps:5.2f}s): "
                              f"dx={dx:7.4f}, dy={dy:7.4f}, mag={magnitude:6.4f}, "
                              f"direction={direction:5}")
        
        frame_count += 1
    
    cap.release()
    
    print(f"\n📊 Movement Analysis:")
    print(f"   Total movements detected: {len(movements)}")
    
    if movements:
        # Group by direction
        directions = [m['direction'] for m in movements]
        direction_counts = {}
        for direction in directions:
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        
        print(f"   Direction breakdown:")
        for direction, count in direction_counts.items():
            print(f"     {direction}: {count}")
        
        # Show up and down movements specifically
        up_movements = [m for m in movements if m['direction'] == 'up']
        down_movements = [m for m in movements if m['direction'] == 'down']
        
        print(f"\n🎯 Up movements ({len(up_movements)}):")
        for m in up_movements:
            print(f"   Frame {m['frame']:3d} ({m['timestamp']:5.2f}s): "
                  f"dy={m['dy']:7.4f}, magnitude={m['magnitude']:6.4f}")
        
        print(f"\n🎯 Down movements ({len(down_movements)}):")
        for m in down_movements:
            print(f"   Frame {m['frame']:3d} ({m['timestamp']:5.2f}s): "
                  f"dy={m['dy']:7.4f}, magnitude={m['magnitude']:6.4f}")
        
        # Test the main movement detection logic
        print(f"\n🧪 Testing main movement detection:")
        
        # Reset detector state
        detector._reset_state()
        
        # Process the same frames with main logic
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 250)
        
        frame_count = 250
        detected_movements = []
        
        while frame_count < 350:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = detector._process_frame(frame, frame_count)
            if result and result.get('pose'):
                timestamp = frame_count / fps
                movement = detector.detect_movement(result['pose'], timestamp)
                if movement:
                    detected_movements.append(movement)
                    print(f"   Main detection: Frame {frame_count:3d} ({timestamp:5.2f}s): "
                          f"{movement.direction} (mag={movement.magnitude:.4f})")
            
            frame_count += 1
        
        cap.release()
        
        print(f"\n📈 Main detection results:")
        print(f"   Movements detected by main logic: {len(detected_movements)}")
        if detected_movements:
            main_directions = [m.direction for m in detected_movements]
            print(f"   Directions: {main_directions}")
    
    detector.release()
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    test_up_down_movements()
