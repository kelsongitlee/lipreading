#!/usr/bin/env python3
"""
Google Colab Setup Script for Visual Speech Recognition
This script automates the setup process for the lip reading project in Google Colab.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and print output"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print("Error output:", e.stderr)
        return False

def setup_colab_environment():
    """Set up the complete environment in Google Colab"""
    
    print("🚀 Setting up Visual Speech Recognition in Google Colab")
    print("This will take a few minutes...")
    
    # Step 1: Clone the repository
    if not run_command("git clone https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages", 
                      "Cloning the repository"):
        return False
    
    # Step 2: Change to project directory
    os.chdir("Visual_Speech_Recognition_for_Multiple_Languages")
    
    # Step 3: Install PyTorch with CUDA support (Colab has GPU)
    if not run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118", 
                      "Installing PyTorch with CUDA support"):
        return False
    
    # Step 4: Install other dependencies
    if not run_command("pip install -r requirements.txt", 
                      "Installing project dependencies"):
        return False
    
    # Step 5: Install additional dependencies that might be missing
    additional_packages = [
        "mediapipe>=0.10.0",
        "ibug-face-detection>=0.1.0", 
        "ibug-face-alignment>=0.1.0",
        "ffmpeg-python>=0.2.0",
        "tqdm>=4.60.0"
    ]
    
    for package in additional_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"Warning: Failed to install {package}")
    
    # Step 6: Install ffmpeg
    if not run_command("apt-get update && apt-get install -y ffmpeg", 
                      "Installing ffmpeg"):
        print("Warning: Failed to install ffmpeg via apt-get")
    
    # Step 7: Create necessary directories
    directories = [
        "benchmarks/LRS3/models",
        "benchmarks/LRS3/language_models", 
        "benchmarks/LRS3/labels"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Download a pre-trained model (see model zoo in README)")
    print("2. Place the model files in the appropriate directories")
    print("3. Upload your video files")
    print("4. Run inference using: python infer.py config_filename=configs/LRS3_V_WER19.1.ini data_filename=your_video.mp4")
    
    return True

def download_sample_model():
    """Download a sample model for testing"""
    print("\n📥 Downloading sample LRS3 model...")
    
    # This is a sample command - you'll need to replace with actual download links
    # The actual download links are in the README.md file
    print("Please download models manually from the links in README.md")
    print("For LRS3 English model, use: LRS3_V_WER19.1.ini")
    
    return True

if __name__ == "__main__":
    if setup_colab_environment():
        print("\n✅ Environment setup completed!")
        download_sample_model()
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
