#!/usr/bin/env python3
"""
Debug script to analyze raw movement data and understand direction detection issues.
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
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_movement_detection():
    """Debug the movement detection by analyzing raw pose data."""
    
    print("=== Debug Movement Detection ===\n")
    
    # Create detector with very low threshold for debugging
    detector = SimpleMediaPipeDetector(
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        movement_threshold=0.005,  # Very low threshold to catch all movements
        max_history=100,
        debug_mode=True
    )
    
    video_path = "test_video.mov"
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    print(f"📹 Debugging video: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Failed to open video")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")
    print()
    
    # Process frames manually to debug
    frame_count = 0
    pose_history = []
    movement_data = []
    
    # Process every 2nd frame for debugging (15 FPS effective)
    frame_skip = 2
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            # Process frame
            result = detector._process_frame(frame, frame_count)
            
            if result and result.get('pose'):
                pose = result['pose']
                pose_history.append({
                    'frame': frame_count,
                    'timestamp': frame_count / fps,
                    'x': pose['x'],
                    'y': pose['y'],
                    'eye_distance': pose.get('eye_distance', 0),
                    'quality_score': pose.get('quality_score', 0)
                })
                
                # Calculate movement if we have previous pose
                if len(pose_history) > 1:
                    prev_pose = pose_history[-2]
                    dx = pose['x'] - prev_pose['x']
                    dy = pose['y'] - prev_pose['y']
                    magnitude = np.sqrt(dx**2 + dy**2)
                    
                    # Test both direction detection methods
                    direction_old = detector._detect_direction_angle(dx, dy)
                    direction_new = detector._detect_direction_improved(dx, dy, magnitude)
                    
                    movement_data.append({
                        'frame': frame_count,
                        'timestamp': frame_count / fps,
                        'dx': dx,
                        'dy': dy,
                        'magnitude': magnitude,
                        'direction_old': direction_old,
                        'direction_new': direction_new,
                        'threshold': detector.movement_threshold
                    })
                    
                    # Print significant movements
                    if magnitude > 0.005:  # Very low threshold for debugging
                        print(f"Frame {frame_count:3d} ({frame_count/fps:5.2f}s): "
                              f"dx={dx:7.4f}, dy={dy:7.4f}, mag={magnitude:6.4f}, "
                              f"old={direction_old or 'None':5}, new={direction_new or 'None':5}")
        
        frame_count += 1
        
        # Limit processing for debugging
        if frame_count > 300:
            break
    
    cap.release()
    
    print(f"\n📊 Movement Analysis Summary:")
    print(f"   Total poses tracked: {len(pose_history)}")
    print(f"   Total movements detected: {len(movement_data)}")
    
    if movement_data:
        print(f"\n📈 Movement Statistics:")
        
        # Group by direction
        directions_old = [m['direction_old'] for m in movement_data if m['direction_old']]
        directions_new = [m['direction_new'] for m in movement_data if m['direction_new']]
        
        print(f"   Old method directions: {directions_old}")
        print(f"   New method directions: {directions_new}")
        
        # Analyze movement magnitudes
        magnitudes = [m['magnitude'] for m in movement_data]
        dx_values = [m['dx'] for m in movement_data]
        dy_values = [m['dy'] for m in movement_data]
        
        print(f"   Magnitude range: {min(magnitudes):.4f} to {max(magnitudes):.4f}")
        print(f"   DX range: {min(dx_values):.4f} to {max(dx_values):.4f}")
        print(f"   DY range: {min(dy_values):.4f} to {max(dy_values):.4f}")
        
        # Show significant movements
        print(f"\n🎯 Significant Movements (>0.01):")
        significant = [m for m in movement_data if m['magnitude'] > 0.01]
        for m in significant:
            print(f"   Frame {m['frame']:3d}: {m['direction_new']:5} "
                  f"(dx={m['dx']:6.4f}, dy={m['dy']:6.4f}, mag={m['magnitude']:6.4f})")
    
    # Test direction detection with sample movements
    print(f"\n🧪 Direction Detection Test:")
    test_movements = [
        (0.02, 0.0, "right"),
        (-0.02, 0.0, "left"),
        (0.0, 0.02, "down"),
        (0.0, -0.02, "up"),
        (0.015, 0.015, "diagonal"),
        (-0.015, 0.015, "diagonal"),
    ]
    
    for dx, dy, expected in test_movements:
        magnitude = np.sqrt(dx**2 + dy**2)
        direction_old = detector._detect_direction_angle(dx, dy)
        direction_new = detector._detect_direction_improved(dx, dy, magnitude)
        print(f"   dx={dx:6.3f}, dy={dy:6.3f} -> old={direction_old:5}, new={direction_new:5} (expected: {expected})")
    
    detector.release()
    print("\n=== Debug Complete ===")

if __name__ == "__main__":
    debug_movement_detection()
