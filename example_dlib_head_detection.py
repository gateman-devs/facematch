#!/usr/bin/env python3
"""
Simple Example: Dlib Head Movement Detection
Basic implementation using dlib facial landmarks + OpenCV for head pose estimation.
"""

import cv2
import dlib
import numpy as np
import urllib.request
import tempfile
import os

def download_shape_predictor():
    """Download the shape predictor file if not present."""
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    
    if not os.path.exists(predictor_path):
        print("Downloading shape predictor...")
        try:
            import bz2
            url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            compressed_path = predictor_path + ".bz2"
            
            # Download compressed file
            urllib.request.urlretrieve(url, compressed_path)
            
            # Extract
            with bz2.open(compressed_path, 'rb') as source, open(predictor_path, 'wb') as target:
                target.write(source.read())
            
            # Clean up
            os.remove(compressed_path)
            print("Shape predictor downloaded successfully!")
            
        except Exception as e:
            print(f"Failed to download shape predictor: {e}")
            return None
    
    return predictor_path

def main():
    """Main function demonstrating dlib head movement detection."""
    
    # Download shape predictor
    predictor_path = download_shape_predictor()
    if not predictor_path:
        print("Could not get shape predictor. Exiting.")
        return
    
    # Initialize Dlib face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    # 3D model points (generic face model)
    model_points = np.array([
        (0.0, 0.0, 0.0),           # Nose tip
        (0.0, -330.0, -65.0),      # Chin
        (-225.0, 170.0, -135.0),   # Left eye corner
        (225.0, 170.0, -135.0),    # Right eye corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0)    # Right mouth corner
    ], dtype=np.float32)
    
    # Camera parameters (will be updated based on video dimensions)
    img_width, img_height = 640, 480
    focal_length = max(img_width, img_height)
    camera_matrix = np.array([
        [focal_length, 0, img_width/2],
        [0, focal_length, img_height/2],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))
    
    def get_head_pose(image, landmarks):
        """Calculate head pose from landmarks."""
        image_points = np.array([
            landmarks[30],  # Nose tip
            landmarks[8],   # Chin
            landmarks[36],  # Left eye
            landmarks[45],  # Right eye
            landmarks[48],  # Left mouth
            landmarks[54]   # Right mouth
        ], dtype=np.float32)
        
        success, rvec, tvec = cv2.solvePnP(
            model_points, 
            image_points, 
            camera_matrix, 
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            rmat = cv2.Rodrigues(rvec)[0]
            pitch = np.arctan2(rmat[2][1], rmat[2][2])
            yaw = -np.arctan2(rmat[2][0], np.sqrt(rmat[2][1]**2 + rmat[2][2]**2))
            return np.degrees(pitch), np.degrees(yaw)
        return None
    
    # Download and process video
    video_url = "https://res.cloudinary.com/themizehq/video/upload/v1755621958/IMG_6482.mov"
    
    print(f"Downloading video from {video_url}")
    with tempfile.NamedTemporaryFile(suffix='.mov', delete=False) as tmp_file:
        video_path = tmp_file.name
    
    try:
        # Create SSL context that ignores certificate verification
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Use the SSL context for the request
        with urllib.request.urlopen(video_url, context=ssl_context) as response:
            with open(video_path, 'wb') as f:
                f.write(response.read())
        print("Video downloaded successfully!")
        
        # Process video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Could not open video!")
            return
        
        frame_count = 0
        movements_detected = []
        
        print("Processing video frames...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame for consistent processing
            frame = cv2.resize(frame, (img_width, img_height))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = detector(gray)
            
            for face in faces:
                # Get landmarks
                landmarks = predictor(gray, face)
                landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])
                
                # Calculate head pose
                pose = get_head_pose(frame, landmarks)
                
                if pose:
                    pitch, yaw = pose
                    
                    # Determine movement direction
                    direction = "Neutral"
                    if abs(yaw) > abs(pitch):
                        if yaw > 20:
                            direction = "Left"
                        elif yaw < -20:
                            direction = "Right"
                    else:
                        if pitch > 20:
                            direction = "Up"
                        elif pitch < -20:
                            direction = "Down"
                    
                    # Only log significant movements
                    if direction != "Neutral":
                        movements_detected.append(direction)
                        print(f"Frame {frame_count}: Direction: {direction} (Yaw: {yaw:.1f}°, Pitch: {pitch:.1f}°)")
            
            frame_count += 1
            
            # Process only first 300 frames for demo
            if frame_count >= 300:
                break
        
        cap.release()
        
        # Summary
        print(f"\nProcessing completed!")
        print(f"Frames processed: {frame_count}")
        print(f"Movements detected: {len(movements_detected)}")
        print(f"Movement sequence: {movements_detected}")
        
        # Expected movements
        expected = ["Left", "Left", "Right", "Right", "Up", "Up", "Down", "Down", "Left", "Right", "Up", "Down"]
        print(f"Expected movements: {expected}")
        
        # Simple accuracy check
        if len(movements_detected) > 0:
            accuracy = min(len(movements_detected) / len(expected), 1.0)
            print(f"Detection rate: {accuracy:.1%}")
        
    except Exception as e:
        print(f"Error processing video: {e}")
        
    finally:
        # Clean up
        try:
            os.unlink(video_path)
            print("Cleaned up temporary video file")
        except:
            pass

if __name__ == "__main__":
    main()
