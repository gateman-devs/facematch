#!/usr/bin/env python3
"""
Face Recognition API Test Script
Tests the API with various image combinations from the provided URLs.
"""

import requests
import json
import time
from typing import Dict, List, Tuple

# Test image URLs provided by the user
TEST_IMAGES = {
    "phone_screen": "https://res.cloudinary.com/themizehq/image/upload/v1750726459/Photo_on_24-06-2025_at_01.28.jpg",
    "good_with_glare": "https://res.cloudinary.com/themizehq/image/upload/v1753822320/IMG_6122.jpg",
    "poor_lighting": "https://res.cloudinary.com/themizehq/image/upload/v1753822320/IMG_6123.jpg",
    "good_no_glare": "https://res.cloudinary.com/themizehq/image/upload/v1750725491/IMG_5680.jpg",
    "different_face": "https://res.cloudinary.com/themizehq/image/upload/v1703815583/ba30e7e5-5518-4818-91f6-a1e3f8941932.jpg",
    "cartoon_1": "https://i.pinimg.com/736x/fc/4a/7c/fc4a7c9506315a726ae4e276cea929e3.jpg",
    "cartoon_2": "https://firewireblog.com/wp-content/uploads/2014/12/realistic-cartoon-characters-3d-real-life-34.jpg"
}

class FaceMatchTester:
    """Test client for Face Recognition API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'FaceMatch-TestClient/1.0'
        })
    
    def health_check(self) -> Dict:
        """Check API health status."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            return {
                'success': response.status_code == 200,
                'data': response.json() if response.status_code == 200 else None,
                'status_code': response.status_code,
                'error': None if response.status_code == 200 else response.text
            }
        except Exception as e:
            return {
                'success': False,
                'data': None,
                'status_code': None,
                'error': str(e)
            }
    
    def compare_faces(self, image1: str, image2: str, threshold: float = 0.6) -> Dict:
        """Compare two face images."""
        payload = {
            'image1': image1,
            'image2': image2,
            'threshold': threshold
        }
        
        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/match", 
                json=payload, 
                timeout=60
            )
            request_time = time.time() - start_time
            
            return {
                'success': response.status_code == 200,
                'data': response.json(),
                'status_code': response.status_code,
                'request_time': request_time,
                'error': None if response.status_code == 200 else response.text
            }
        except Exception as e:
            return {
                'success': False,
                'data': None,
                'status_code': None,
                'request_time': time.time() - start_time,
                'error': str(e)
            }

def print_separator(title: str = ""):
    """Print a formatted separator."""
    if title:
        print(f"\n{'='*20} {title} {'='*20}")
    else:
        print("="*60)

def print_test_result(result: Dict, test_name: str):
    """Print formatted test result."""
    print(f"\n🧪 Test: {test_name}")
    print(f"⏱️  Request Time: {result.get('request_time', 'N/A'):.3f}s")
    
    if result['success']:
        data = result['data']
        if data.get('success'):
            print(f"✅ Status: SUCCESS")
            print(f"🎯 Match: {'YES' if data.get('match') else 'NO'}")
            print(f"📊 Similarity: {data.get('similarity_score', 0):.3f}")
            print(f"🔒 Confidence: {data.get('confidence', 0):.3f}")
            
            # Liveness results
            liveness = data.get('liveness_results')
            if liveness:
                for i, (key, result) in enumerate(liveness.items(), 1):
                    if isinstance(result, dict) and result.get('success'):
                        live_status = "LIVE" if result.get('is_live') else "FAKE"
                        print(f"👁️  Image{i} Liveness: {live_status} ({result.get('liveness_score', 0):.3f})")
            
            # Processing time
            print(f"⚡ Processing Time: {data.get('processing_time', 0):.3f}s")
        else:
            print(f"❌ Status: FAILED")
            print(f"💬 Error: {data.get('error', 'Unknown error')}")
    else:
        print(f"❌ Status: REQUEST FAILED")
        print(f"💬 Error: {result.get('error', 'Unknown error')}")
        print(f"🔢 Status Code: {result.get('status_code', 'N/A')}")

def run_comprehensive_tests():
    """Run comprehensive test suite."""
    tester = FaceMatchTester()
    
    print_separator("Face Recognition API Test Suite")
    print("Testing with provided image URLs")
    print("Note: Some URLs may be inaccessible or return errors")
    
    # Health check
    print_separator("Health Check")
    health = tester.health_check()
    if health['success']:
        print("✅ API is healthy")
        models = health['data'].get('models', {})
        for model, status in models.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {model.upper()}: {'Available' if status else 'Not Available'}")
    else:
        print("❌ API health check failed")
        print(f"Error: {health.get('error')}")
        return
    
    # Test scenarios
    test_scenarios = [
        # Same person comparisons (should match)
        ("phone_screen", "good_no_glare", "Same person - different lighting"),
        ("good_with_glare", "good_no_glare", "Same person - with/without glare"),
        ("phone_screen", "good_with_glare", "Same person - phone vs direct"),
        
        # Different person comparisons (should not match)
        ("good_no_glare", "different_face", "Different people"),
        ("phone_screen", "different_face", "Different people - phone vs direct"),
        
        # Cartoon comparisons (testing edge cases)
        ("good_no_glare", "cartoon_1", "Real vs Cartoon"),
        ("cartoon_1", "cartoon_2", "Cartoon vs Cartoon"),
        
        # Quality comparisons
        ("poor_lighting", "good_no_glare", "Poor vs Good quality"),
        ("poor_lighting", "different_face", "Poor quality vs Different person"),
    ]
    
    print_separator("Face Comparison Tests")
    
    successful_tests = 0
    total_tests = len(test_scenarios)
    
    for i, (img1_key, img2_key, description) in enumerate(test_scenarios, 1):
        img1_url = TEST_IMAGES.get(img1_key)
        img2_url = TEST_IMAGES.get(img2_key)
        
        if not img1_url or not img2_url:
            print(f"\n❌ Test {i}: Skipped - Missing image URLs")
            continue
        
        print(f"\n📋 Test {i}/{total_tests}: {description}")
        print(f"   Image 1: {img1_key}")
        print(f"   Image 2: {img2_key}")
        
        result = tester.compare_faces(img1_url, img2_url, threshold=0.6)
        
        if result['success'] and result['data'].get('success'):
            successful_tests += 1
        
        print_test_result(result, description)
        
        # Small delay between tests
        time.sleep(1)
    
    # Summary
    print_separator("Test Summary")
    print(f"✅ Successful Tests: {successful_tests}/{total_tests}")
    print(f"📊 Success Rate: {(successful_tests/total_tests)*100:.1f}%")
    
    if successful_tests == total_tests:
        print("🎉 All tests passed!")
    elif successful_tests > total_tests * 0.7:
        print("✨ Most tests passed - system appears to be working well")
    else:
        print("⚠️  Many tests failed - please check system configuration")

def run_single_test(img1_key: str, img2_key: str, threshold: float = 0.6):
    """Run a single comparison test."""
    tester = FaceMatchTester()
    
    img1_url = TEST_IMAGES.get(img1_key)
    img2_url = TEST_IMAGES.get(img2_key)
    
    if not img1_url or not img2_url:
        print(f"❌ Error: Invalid image keys. Available: {list(TEST_IMAGES.keys())}")
        return
    
    print(f"🔍 Comparing {img1_key} vs {img2_key}")
    print(f"   Threshold: {threshold}")
    
    result = tester.compare_faces(img1_url, img2_url, threshold)
    print_test_result(result, f"{img1_key} vs {img2_key}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Face Recognition API Test Script")
    parser.add_argument("--api-url", default="http://localhost:8000", 
                       help="API base URL")
    parser.add_argument("--single", nargs=2, metavar=("IMG1", "IMG2"),
                       help="Run single test with two image keys")
    parser.add_argument("--threshold", type=float, default=0.6,
                       help="Similarity threshold (0.0-1.0)")
    parser.add_argument("--list-images", action="store_true",
                       help="List available test images")
    
    args = parser.parse_args()
    
    if args.list_images:
        print("Available test images:")
        for key, url in TEST_IMAGES.items():
            print(f"  {key}: {url}")
        exit(0)
    
    # Update tester with custom URL
    if args.api_url != "http://localhost:8000":
        FaceMatchTester.__init__ = lambda self, base_url=args.api_url: setattr(self, 'base_url', base_url) or setattr(self, 'session', requests.Session())
    
    if args.single:
        run_single_test(args.single[0], args.single[1], args.threshold)
    else:
        run_comprehensive_tests() 