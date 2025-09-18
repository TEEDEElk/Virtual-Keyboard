#!/usr/bin/env python3
"""
Setup script for Virtual Keyboard project
Checks dependencies and provides setup instructions
"""

import sys
import subprocess
import os

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version OK: {sys.version}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'cv2', 'mediapipe', 'numpy', 'pyautogui', 
        'tensorflow', 'gtts', 'pygame', 'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    return missing_packages

def check_camera():
    """Check if camera is available"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera is available")
            cap.release()
            return True
        else:
            print("⚠️  Camera not detected or in use")
            return False
    except ImportError:
        print("⚠️  Cannot check camera (OpenCV not installed)")
        return False

def main():
    print("🎯 Virtual Keyboard Setup Check")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    print("\n📦 Checking Dependencies:")
    print("-" * 30)
    missing = check_dependencies()
    
    print("\n📷 Checking Camera:")
    print("-" * 30)
    camera_ok = check_camera()
    
    print("\n" + "=" * 50)
    
    if not missing and camera_ok:
        print("🎉 Setup Complete! You can run:")
        print("   python3 main.py")
    elif not missing:
        print("⚠️  Dependencies OK, but camera issues detected")
        print("   You can still try the demo:")
        print("   python3 demo.py")
    else:
        print("❌ Missing dependencies. Install with:")
        print("   pip install -r requirements.txt")
        print("\n   Or install individual packages:")
        pip_packages = {
            'cv2': 'opencv-python',
            'mediapipe': 'mediapipe',
            'pyautogui': 'pyautogui',
            'tensorflow': 'tensorflow',
            'gtts': 'gtts',
            'pygame': 'pygame',
            'sklearn': 'scikit-learn'
        }
        for pkg in missing:
            pip_name = pip_packages.get(pkg, pkg)
            print(f"   pip install {pip_name}")
    
    print("\n📚 Documentation available in:")
    print("   README.md - Quick start guide")
    print("   DOCUMENTATION.md - Detailed documentation")
    print("   demo.py - Interactive demo (no camera needed)")
    
    print("\n🎮 Try the demo anytime with:")
    print("   python3 demo.py")

if __name__ == "__main__":
    main()