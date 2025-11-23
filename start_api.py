#!/usr/bin/env python3
"""
Quick start script for Bank OCR API
"""

import subprocess
import sys
import time
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'tensorflow', 'opencv-python', 'numpy', 'pillow'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_missing_packages(packages):
    """Install missing packages"""
    if packages:
        print(f"📦 Installing missing packages: {', '.join(packages)}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
            print("✅ Packages installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install packages: {e}")
            return False
    return True

def check_models():
    """Check if OCR models are available"""
    models_dir = Path("models")
    required_models = [
        "complete_ocr_character_model.h5",
        "complete_ocr_system_metadata.json"
    ]
    
    missing_models = []
    for model in required_models:
        if not (models_dir / model).exists():
            missing_models.append(model)
    
    return missing_models

def main():
    print("🚀 Bank OCR API Startup")
    print("=" * 50)
    
    # Check requirements
    print("1. Checking Python packages...")
    missing_packages = check_requirements()
    
    if missing_packages:
        print(f"⚠ Missing packages: {', '.join(missing_packages)}")
        if not install_missing_packages(missing_packages):
            print("❌ Failed to install required packages. Please run:")
            print("   pip install -r requirements.txt")
            return False
    else:
        print("✅ All required packages are installed")
    
    # Check models
    print("\n2. Checking OCR models...")
    missing_models = check_models()
    
    if missing_models:
        print(f"⚠ Missing models: {', '.join(missing_models)}")
        print("📝 To train models, run the Jupyter notebooks first:")
        print("   jupyter notebook")
        print("   Open: notebooks/complete_bank_ocr_system.ipynb")
        print("\n🔄 Starting API in demo mode with mock data...")
    else:
        print("✅ OCR models found")
    
    print("\n3. Starting FastAPI server...")
    print("🌐 API will be available at:")
    print("   • Main endpoint: http://localhost:8000/upload-document/")
    print("   • Documentation: http://localhost:8000/docs")
    print("   • Health check: http://localhost:8000/health")
    print("\n📱 Angular frontend should point to: http://localhost:8000")
    print("⏹  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the API server
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "bank_ocr_api:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("\n💡 Try running manually:")
        print("   python bank_ocr_api.py")

if __name__ == "__main__":
    main()