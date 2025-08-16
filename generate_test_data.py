#!/usr/bin/env python3
"""
Generate Sample Performance Data
Creates realistic performance log entries for testing the analysis system.
"""

import os
import random
import time
from datetime import datetime, timedelta

def generate_sample_logs():
    """Generate sample performance log entries."""
    
    # Ensure logs directory exists
    logs_dir = "logs"  # Use local logs directory for development
    os.makedirs(logs_dir, exist_ok=True)
    
    log_file = os.path.join(logs_dir, "facematch.log")
    
    # Generate 50 sample sessions
    sessions = []
    base_time = datetime.now() - timedelta(hours=2)
    
    for i in range(50):
        session_id = f"test_session_{i:03d}"
        
        # Generate realistic timing data
        request_start = base_time + timedelta(minutes=i*2, seconds=random.randint(0, 60))
        
        # File receive time (0.1 to 2.0 seconds)
        receive_time = random.uniform(0.1, 2.0)
        receive_complete = request_start + timedelta(seconds=receive_time)
        
        # Processing time (1.0 to 8.0 seconds)
        processing_time = random.uniform(1.0, 8.0)
        processing_start = receive_complete + timedelta(seconds=random.uniform(0.01, 0.1))
        processing_complete = processing_start + timedelta(seconds=processing_time)
        
        # Total time
        total_time = (processing_complete - request_start).total_seconds()
        
        # Success rate (90% success)
        success = random.random() > 0.1
        result = "pass" if success else "fail"
        
        # File size (1MB to 10MB)
        file_size = random.randint(1024*1024, 10*1024*1024)
        
        sessions.append({
            'session_id': session_id,
            'request_start': request_start,
            'receive_time': receive_time,
            'processing_time': processing_time,
            'total_time': total_time,
            'success': success,
            'result': result,
            'file_size': file_size
        })
    
    # Write log entries
    with open(log_file, 'w') as f:
        for session in sessions:
            # Request start
            f.write(f"{session['request_start'].strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - INFO - VIDEO_LIVENESS_REQUEST_START - session_id: {session['session_id']}, challenge_type: head_movement, file_size: {session['file_size']}\n")
            
            # File receive complete
            receive_complete_time = session['request_start'] + timedelta(seconds=session['receive_time'])
            f.write(f"{receive_complete_time.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - INFO - VIDEO_FILE_RECEIVE_COMPLETE - session_id: {session['session_id']}, duration: {session['receive_time']:.3f}s, size: {session['file_size']} bytes\n")
            
            # Processing start
            processing_start_time = receive_complete_time + timedelta(seconds=random.uniform(0.01, 0.1))
            f.write(f"{processing_start_time.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - INFO - VIDEO_PROCESSING_START - session_id: {session['session_id']}, challenge_type: head_movement\n")
            
            # Processing complete
            processing_complete_time = processing_start_time + timedelta(seconds=session['processing_time'])
            f.write(f"{processing_complete_time.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - INFO - VIDEO_PROCESSING_COMPLETE - session_id: {session['session_id']}, duration: {session['processing_time']:.3f}s, success: {session['success']}\n")
            
            if session['success']:
                # Liveness complete
                complete_time = processing_complete_time + timedelta(seconds=random.uniform(0.01, 0.1))
                file_save_time = random.uniform(0.01, 0.05)
                f.write(f"{complete_time.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - INFO - VIDEO_LIVENESS_COMPLETE - session_id: {session['session_id']}, result: {session['result']}, total_time: {session['total_time']:.3f}s, processing_time: {session['processing_time']:.3f}s, video_receive_time: {session['receive_time']:.3f}s, file_save_time: {file_save_time:.3f}s\n")
            else:
                # Liveness failed
                failed_time = processing_complete_time + timedelta(seconds=random.uniform(0.01, 0.1))
                f.write(f"{failed_time.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - WARNING - VIDEO_LIVENESS_FAILED - session_id: {session['session_id']}, total_time: {session['total_time']:.3f}s, processing_time: {session['processing_time']:.3f}s, error: Video processing failed\n")
    
    print(f"Generated {len(sessions)} sample log entries in {log_file}")
    print(f"Success rate: {sum(1 for s in sessions if s['success'])/len(sessions):.1%}")
    print(f"Average total time: {sum(s['total_time'] for s in sessions)/len(sessions):.2f}s")
    print(f"Average processing time: {sum(s['processing_time'] for s in sessions)/len(sessions):.2f}s")

if __name__ == "__main__":
    generate_sample_logs()
