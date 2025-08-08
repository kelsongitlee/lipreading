#!/usr/bin/env python3
"""
Test script to verify the Visual Speech Recognition setup
Run this after installation to check if everything is working correctly.
"""

import sys
import importlib
import subprocess
from pathlib import Path

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name}: {e}")
        return False

def test_command(command, description):
    """Test if a command can be executed"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ {description}")
            return True
        else:
            print(f"❌ {description}: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description}: {e}")
        return False

def main():
    print("🧪 Testing Visual Speech Recognition Setup")
    print("=" * 50)
    
    # Test Python version
    print(f"🐍 Python version: {sys.version}")
    
    # Test core dependencies
    print("\n📦 Testing core dependencies:")
    core_deps = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("torchaudio", "TorchAudio"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("skimage", "scikit-image"),
        ("av", "PyAV"),
        ("hydra", "Hydra"),
        ("mediapipe", "MediaPipe"),
    ]
    
    core_success = 0
    for module, name in core_deps:
        if test_import(module, name):
            core_success += 1
    
    # Test optional dependencies
    print("\n🔧 Testing optional dependencies:")
    optional_deps = [
        ("ibug.face_detection", "iBug Face Detection"),
        ("ibug.face_alignment", "iBug Face Alignment"),
        ("ffmpeg", "FFmpeg Python"),
        ("tqdm", "tqdm"),
        ("PIL", "Pillow"),
        ("matplotlib", "Matplotlib"),
    ]
    
    optional_success = 0
    for module, name in optional_deps:
        if test_import(module, name):
            optional_success += 1
    
    # Test project structure
    print("\n📁 Testing project structure:")
    required_files = [
        "infer.py",
        "eval.py", 
        "crop_mouth.py",
        "configs/LRS3_V_WER19.1.ini",
        "pipelines/pipeline.py",
        "pipelines/model.py",
        "requirements.txt"
    ]
    
    structure_success = 0
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
            structure_success += 1
        else:
            print(f"❌ {file_path} (missing)")
    
    # Test GPU availability
    print("\n🖥️ Testing GPU availability:")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_count} device(s)")
            print(f"   Device 0: {gpu_name}")
        else:
            print("⚠️ GPU not available (will use CPU)")
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
    
    # Test FFmpeg
    print("\n🎬 Testing FFmpeg:")
    test_command("ffmpeg -version", "FFmpeg installation")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SETUP SUMMARY:")
    print(f"   Core dependencies: {core_success}/{len(core_deps)} ✅")
    print(f"   Optional dependencies: {optional_success}/{len(optional_deps)} ✅")
    print(f"   Project structure: {structure_success}/{len(required_files)} ✅")
    
    if core_success == len(core_deps) and structure_success == len(required_files):
        print("\n🎉 SETUP IS READY!")
        print("\nNext steps:")
        print("1. Download pre-trained models (see README.md)")
        print("2. Place models in benchmarks/ directories")
        print("3. Upload your video files")
        print("4. Run: python infer.py config_filename=configs/LRS3_V_WER19.1.ini data_filename=your_video.mp4")
    else:
        print("\n⚠️ SETUP NEEDS ATTENTION!")
        print("Please check the failed items above and reinstall missing dependencies.")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
