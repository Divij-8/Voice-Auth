"""
Integration Test Script - Verifies Frontend & Backend Communication
Tests all API endpoints and displays detailed diagnostics
"""

import requests
import json
import sys
from pathlib import Path

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header(text):
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}{text}{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✓{RESET} {text}")

def print_error(text):
    print(f"{RED}✗{RESET} {text}")

def print_warning(text):
    print(f"{YELLOW}⚠{RESET} {text}")

def print_info(text):
    print(f"{BLUE}ℹ{RESET} {text}")

def test_backend_health():
    """Test if backend is running"""
    print_header("Backend Health Check")
    
    try:
        print_info("Checking if FastAPI backend is running on http://localhost:8000...")
        response = requests.get('http://localhost:8000/', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_success("Backend is online!")
            print(f"  Status: {data.get('status')}")
            print(f"  Message: {data.get('message')}")
            print(f"  Enrolled: {data.get('enrolled')}")
            print(f"  Cross-Encoder: {data.get('cross_encoder')}")
            return True
        else:
            print_error(f"Backend returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to backend on http://localhost:8000")
        print_warning("Make sure to run: python main.py")
        return False
    except Exception as e:
        print_error(f"Backend health check failed: {str(e)}")
        return False

def test_api_status():
    """Test /api/status endpoint"""
    print_header("API Status Endpoint")
    
    try:
        print_info("Testing GET /api/status...")
        response = requests.get('http://localhost:8000/api/status', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_success("Status endpoint working!")
            print(f"\n{json.dumps(data, indent=2)}")
            
            enrolled = data.get('enrolled', False)
            cross_encoder = data.get('stats', {}).get('cross_encoder_enabled', False)
            
            if enrolled:
                print_success(f"System is enrolled with {data['stats']['enrollment_count']} samples")
            else:
                print_warning("System not enrolled - add training data to 'training_data/' folder")
            
            if cross_encoder:
                print_success("Cross-Encoder enhancement is enabled")
            else:
                print_warning("Cross-Encoder enhancement is disabled")
            
            return True
        else:
            print_error(f"Status endpoint returned {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Status endpoint test failed: {str(e)}")
        return False

def test_frontend_availability():
    """Test if frontend is accessible"""
    print_header("Frontend Availability Check")
    
    try:
        print_info("Checking if frontend server is running on http://localhost:3000...")
        response = requests.get('http://localhost:3000/', timeout=5)
        
        if response.status_code == 200:
            print_success("Frontend is accessible!")
            if 'Voice Guardian' in response.text:
                print_success("index.html is being served correctly")
            return True
        else:
            print_error(f"Frontend returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to frontend on http://localhost:3000")
        print_warning("Make sure to run: python frontend_server.py")
        return False
    except Exception as e:
        print_error(f"Frontend check failed: {str(e)}")
        return False

def test_cors_headers():
    """Test if CORS headers are properly configured"""
    print_header("CORS Configuration Check")
    
    try:
        print_info("Checking CORS headers on backend...")
        response = requests.options('http://localhost:8000/api/status', timeout=5)
        
        headers = response.headers
        cors_origin = headers.get('Access-Control-Allow-Origin')
        cors_methods = headers.get('Access-Control-Allow-Methods')
        cors_headers = headers.get('Access-Control-Allow-Headers')
        
        print_success("CORS headers found:")
        print(f"  Allow-Origin: {cors_origin}")
        print(f"  Allow-Methods: {cors_methods}")
        print(f"  Allow-Headers: {cors_headers}")
        
        if cors_origin:
            print_success("CORS is properly configured for cross-origin requests")
            return True
        else:
            print_warning("CORS headers not found - might cause issues")
            return False
    except Exception as e:
        print_warning(f"CORS check failed: {str(e)}")
        return False

def check_training_data():
    """Check if training data exists"""
    print_header("Training Data Check")
    
    training_dir = Path("training_data")
    
    if not training_dir.exists():
        print_warning("'training_data' folder does not exist")
        print_info("Create it and add voice samples:")
        print("  mkdir -p training_data")
        print("  cp your_voice_samples.wav training_data/")
        return False
    
    audio_files = list(training_dir.glob("*.wav")) + \
                  list(training_dir.glob("*.mp3")) + \
                  list(training_dir.glob("*.flac"))
    
    if not audio_files:
        print_warning(f"'training_data' folder exists but is empty")
        print_info("Add 10-20 voice samples for best results")
        return False
    
    print_success(f"Found {len(audio_files)} audio files in training_data/:")
    for f in audio_files[:5]:
        print(f"  - {f.name}")
    if len(audio_files) > 5:
        print(f"  ... and {len(audio_files) - 5} more")
    
    return True

def check_models():
    """Check if pre-trained models exist"""
    print_header("Pre-trained Models Check")
    
    models_dir = Path("pretrained_models/ecapa-tdnn")
    
    if not models_dir.exists():
        print_warning("Pre-trained models directory not found")
        print_info("Models will be downloaded on first run (requires internet)")
        return True
    
    required_files = [
        "classifier.ckpt",
        "embedding_model.ckpt",
        "hyperparams.yaml",
        "label_encoder.ckpt"
    ]
    
    found_files = [f for f in required_files if (models_dir / f).exists()]
    
    if len(found_files) == len(required_files):
        print_success(f"All pre-trained models found ({len(found_files)}/{len(required_files)})")
        return True
    else:
        print_warning(f"Only {len(found_files)}/{len(required_files)} model files found")
        print_info("Missing files will be downloaded on startup")
        return True

def print_summary(results):
    """Print test summary"""
    print_header("Integration Test Summary")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    print(f"Results: {BOLD}{GREEN}{passed_tests}/{total_tests}{RESET} tests passed\n")
    
    for test_name, passed in results.items():
        status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
        print(f"  {status}  {test_name}")
    
    print(f"\n{BOLD}Overall Status:{RESET}")
    
    if passed_tests == total_tests:
        print(f"{GREEN}✓ All systems operational! Ready to use.{RESET}\n")
        print("Next steps:")
        print("1. Open http://localhost:3000 in your browser")
        print("2. Click the microphone button to record your voice")
        print("3. System will verify and display results\n")
        return True
    elif passed_tests >= total_tests - 1:
        print(f"{YELLOW}⚠ Most systems working, some warnings to address{RESET}\n")
        return True
    else:
        print(f"{RED}✗ Critical issues found - cannot proceed{RESET}\n")
        print("Troubleshooting:")
        print("1. Ensure both servers are running:")
        print("   - Backend: python main.py")
        print("   - Frontend: python frontend_server.py")
        print("2. Check that ports 3000 and 8000 are available")
        print("3. Verify training data exists: ls -la training_data/\n")
        return False

def main():
    print(f"\n{BOLD}{BLUE}")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  VOICE GUARDIAN - INTEGRATION TEST".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    print(f"{RESET}\n")
    
    results = {}
    
    # Run tests
    results["Backend Health"] = test_backend_health()
    
    if results["Backend Health"]:
        results["API Status"] = test_api_status()
        results["CORS Configuration"] = test_cors_headers()
    else:
        print_warning("Skipping backend API tests - backend not running")
        results["API Status"] = False
        results["CORS Configuration"] = False
    
    results["Frontend Available"] = test_frontend_availability()
    results["Training Data"] = check_training_data()
    results["Pre-trained Models"] = check_models()
    
    # Print summary
    success = print_summary(results)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
