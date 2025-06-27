#!/usr/bin/env python3
"""
Startup script for MARV Content Moderation Engine
"""

import os
import sys
import subprocess
import time
import webbrowser
import requests
from pathlib import Path

def print_banner():
    """Print startup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║    🧠 MARV Content Moderation Engine                        ║
    ║    Production-ready multi-layer content moderation system   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    try:
        import fastapi
        import uvicorn
        import sqlalchemy
        print("✅ Core dependencies: OK")
    except ImportError as e:
        print(f"❌ Missing core dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    try:
        import transformers
        print("✅ ML dependencies: OK")
    except ImportError as e:
        print(f"⚠️ ML dependencies missing: {e}")
        print("Please activate your virtual environment and run: pip install -r requirements.txt")
        return False
    
    return True

def get_python_executable():
    """Get the correct Python executable (prefer venv)"""
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return sys.executable
    
    # Try to find venv Python
    venv_python = Path("venv/Scripts/python.exe")  # Windows
    if not venv_python.exists():
        venv_python = Path("venv/bin/python")  # Unix
    
    if venv_python.exists():
        return str(venv_python)
    
    return sys.executable

def setup_database():
    """Setup database if needed"""
    print("🗄️ Setting up database...")
    
    try:
        python_exe = get_python_executable()
        result = subprocess.run([
            python_exe, "scripts/setup_database.py", "--test"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Database setup: OK")
            return True
        else:
            print(f"❌ Database setup failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Database setup error: {e}")
        return False

def start_backend():
    """Start the FastAPI backend"""
    print("🚀 Starting backend server...")
    
    try:
        # Start uvicorn server with correct Python executable
        python_exe = get_python_executable()
        process = subprocess.Popen([
            python_exe, "-m", "uvicorn", 
            "app.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
        
        # Wait a moment for server to start
        time.sleep(5)  # Increased wait time for ML models to load
        
        # Check if server is running
        try:
            # First try the health endpoint
            response = requests.get("http://localhost:8000/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                status = health_data.get("status", "unknown")
                
                if status in ["healthy", "degraded"]:
                    print("✅ Backend server: Running on http://localhost:8000")
                    print(f"   Status: {status}")
                    return process
                else:
                    print(f"❌ Backend server: Unhealthy status '{status}'")
                    return None
            else:
                print(f"❌ Backend server: HTTP {response.status_code}")
                return None
                
        except requests.exceptions.RequestException:
            # Fallback: try the root endpoint
            try:
                response = requests.get("http://localhost:8000/", timeout=5)
                if response.status_code == 200:
                    print("✅ Backend server: Running on http://localhost:8000")
                    print("   (Health endpoint unavailable, but server is responding)")
                    return process
                else:
                    print(f"❌ Backend server: HTTP {response.status_code}")
                    return None
            except requests.exceptions.RequestException:
                print("❌ Backend server: Not responding")
                return None
            
    except Exception as e:
        print(f"❌ Backend server error: {e}")
        return None

def open_frontend():
    """Open the frontend in browser"""
    print("🌐 Opening frontend...")
    
    frontend_path = Path("frontend/index.html")
    if frontend_path.exists():
        try:
            webbrowser.open(f"file://{frontend_path.absolute()}")
            print("✅ Frontend opened in browser")
        except Exception as e:
            print(f"❌ Could not open frontend: {e}")
            print(f"Please open manually: {frontend_path.absolute()}")
    else:
        print("❌ Frontend file not found")

def run_tests():
    """Run system tests"""
    print("🧪 Running system tests...")
    
    try:
        python_exe = get_python_executable()
        result = subprocess.run([
            python_exe, "tests/test_moderation.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ System tests: PASSED")
        else:
            print(f"❌ System tests: FAILED")
            print(result.stderr)
    except Exception as e:
        print(f"❌ Test execution error: {e}")

def main():
    """Main startup function"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup database
    if not setup_database():
        print("⚠️ Continuing without database setup...")
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("❌ Failed to start backend server")
        sys.exit(1)
    
    # Open frontend
    open_frontend()
    
    # Run tests (optional)
    run_tests()
    
    print("\n🎉 MARV Content Moderation Engine is ready!")
    print("\n📋 Available endpoints:")
    print("   - API Documentation: http://localhost:8000/docs")
    print("   - Health Check: http://localhost:8000/health")
    print("   - Frontend: frontend/index.html")
    print("\n🛑 Press Ctrl+C to stop the server")
    
    try:
        # Keep the server running
        backend_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        backend_process.terminate()
        backend_process.wait()
        print("✅ Server stopped")

if __name__ == "__main__":
    main() 