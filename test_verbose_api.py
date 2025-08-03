#!/usr/bin/env python3
"""
Test script to demonstrate the verbose parameter and score rounding functionality.
"""

import requests
import json
import time
from typing import Dict

# Test image URLs (same as in test_api.py)
TEST_IMAGES = {
    "good_no_glare": "https://res.cloudinary.com/themizehq/image/upload/v1750725491/IMG_5680.jpg",
    "phone_screen": "https://res.cloudinary.com/themizehq/image/upload/v1750726459/Photo_on_24-06-2025_at_01.28.jpg"
}

def test_verbose_functionality(base_url: str = "http://localhost:8000"):
    """Test verbose parameter functionality."""
    
    print("=" * 80)
    print("Testing Verbose Parameter and Score Rounding Functionality")
    print("=" * 80)
    
    # Test data
    payload_base = {
        'image1': TEST_IMAGES["good_no_glare"],
        'image2': TEST_IMAGES["phone_screen"],
        'threshold': 0.6
    }
    
    print("\n1. Testing Health Endpoint with verbose parameter")
    print("-" * 50)
    
    # Health check - non-verbose
    print("\nHealth Check (verbose=false):")
    try:
        response = requests.get(f"{base_url}/health?verbose=false", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(json.dumps(health_data, indent=2))
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Health check - verbose
    print("\nHealth Check (verbose=true):")
    try:
        response = requests.get(f"{base_url}/health?verbose=true", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(json.dumps(health_data, indent=2))
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n2. Testing Face Match Endpoint with verbose parameter")
    print("-" * 50)
    
    # Face match - non-verbose
    print("\nFace Match (verbose=false) - Should return minimal response:")
    payload_non_verbose = {**payload_base, 'verbose': False}
    try:
        start_time = time.time()
        response = requests.post(f"{base_url}/match", json=payload_non_verbose, timeout=60)
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            match_data = response.json()
            print(f"Request Time: {request_time:.3f}s")
            print("Response Keys:", list(match_data.keys()))
            print(json.dumps(match_data, indent=2))
            
            # Check if scores are rounded to 2 decimal places
            if 'similarity_score' in match_data:
                score = match_data['similarity_score']
                decimal_places = len(str(score).split('.')[-1]) if '.' in str(score) else 0
                print(f"\n✓ Similarity score: {score} (decimal places: {decimal_places})")
                if decimal_places <= 2:
                    print("✓ Score properly rounded to 2 decimal places or less")
                else:
                    print("✗ Score not properly rounded")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Face match - verbose
    print("\nFace Match (verbose=true) - Should return detailed response:")
    payload_verbose = {**payload_base, 'verbose': True}
    try:
        start_time = time.time()
        response = requests.post(f"{base_url}/match", json=payload_verbose, timeout=60)
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            match_data = response.json()
            print(f"Request Time: {request_time:.3f}s")
            print("Response Keys:", list(match_data.keys()))
            print(json.dumps(match_data, indent=2))
            
            # Check if scores are rounded to 2 decimal places
            if 'similarity_score' in match_data:
                score = match_data['similarity_score']
                decimal_places = len(str(score).split('.')[-1]) if '.' in str(score) else 0
                print(f"\n✓ Similarity score: {score} (decimal places: {decimal_places})")
                if decimal_places <= 2:
                    print("✓ Score properly rounded to 2 decimal places or less")
                else:
                    print("✗ Score not properly rounded")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n3. Comparing Response Sizes")
    print("-" * 50)
    
    # Compare response sizes
    try:
        # Non-verbose response
        response_non_verbose = requests.post(f"{base_url}/match", json=payload_non_verbose, timeout=60)
        # Verbose response  
        response_verbose = requests.post(f"{base_url}/match", json=payload_verbose, timeout=60)
        
        if response_non_verbose.status_code == 200 and response_verbose.status_code == 200:
            data_non_verbose = response_non_verbose.json()
            data_verbose = response_verbose.json()
            
            print(f"Non-verbose response fields: {len(data_non_verbose.keys())}")
            print(f"Verbose response fields: {len(data_verbose.keys())}")
            print(f"Non-verbose response size: {len(response_non_verbose.text)} chars")
            print(f"Verbose response size: {len(response_verbose.text)} chars")
            
            # Show field differences
            verbose_only_fields = set(data_verbose.keys()) - set(data_non_verbose.keys())
            print(f"Fields only in verbose response: {list(verbose_only_fields)}")
            
        else:
            print("Could not compare responses due to errors")
            
    except Exception as e:
        print(f"Error comparing responses: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test verbose parameter and score rounding")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    
    args = parser.parse_args()
    test_verbose_functionality(args.api_url) 