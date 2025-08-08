# 🚀 Google Colab Setup Guide for Visual Speech Recognition

This guide will help you set up and use the lip reading project in Google Colab with your own videos.

## 📋 Prerequisites

1. **Google Account** - You need a Google account to access Colab
2. **Test Videos** - Prepare video files with people speaking clearly
3. **Good Internet Connection** - For downloading models and dependencies

## 🎯 Step-by-Step Setup

### Step 1: Open Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Sign in with your Google account
3. Create a new notebook or open an existing one

### Step 2: Enable GPU (Important!)

1. Go to **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Set **Runtime type** to **Python 3**
4. Click **Save**

### Step 3: Clone and Setup the Project

Copy and paste this code into a Colab cell:

```python
# Clone the repository
!git clone https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages
%cd Visual_Speech_Recognition_for_Multiple_Languages

# Install PyTorch with CUDA support
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
!pip install -r requirements.txt

# Install additional dependencies
!pip install mediapipe>=0.10.0 ibug-face-detection>=0.1.0 ibug-face-alignment>=0.1.0 ffmpeg-python>=0.2.0 tqdm>=4.60.0

# Install ffmpeg
!apt-get update && apt-get install -y ffmpeg

# Create necessary directories
!mkdir -p benchmarks/LRS3/models benchmarks/LRS3/language_models benchmarks/LRS3/labels

print("✅ Environment setup completed!")
```

### Step 4: Download Pre-trained Models

You need to download model files manually from the provided links. Here are the recommended models:

#### For English (Best Performance):
- **Model**: LRS3 Visual-only (19.1% WER)
- **Download Link**: [Google Drive](http://bit.ly/40EAtyX) or [Baidu Drive](https://bit.ly/3ZjbrV5) (key: dqsy)
- **Language Model**: [Google Drive](http://bit.ly/3FE4XsV) or [Baidu Drive](http://bit.ly/3yRI5SY) (key: t9ep)

#### For Other Languages:
- **Chinese**: CMLR_V_WER8.0.ini - [Download](https://bit.ly/3fR8RkU)
- **Spanish**: CMUMOSEAS_V_ES_WER44.5.ini - [Download](https://bit.ly/34MjWBW)
- **French**: CMUMOSEAS_V_FR_WER58.6.ini - [Download](https://bit.ly/3Ik6owb)
- **Portuguese**: CMUMOSEAS_V_PT_WER51.4.ini - [Download](https://bit.ly/3HjXCgo)

#### How to Download and Upload:

1. **Download the model files** from the links above
2. **Extract the ZIP files** on your computer
3. **Upload to Colab** using the file upload feature:

```python
from google.colab import files
import os

# Upload model files
print("📁 Please upload the model files...")
uploaded = files.upload()

# Extract and move files to correct locations
# (You'll need to manually move files to the correct directories)
```

### Step 5: Upload Your Video

```python
from google.colab import files

print("📹 Please upload your video file...")
uploaded = files.upload()

# Get the uploaded filename
video_filename = list(uploaded.keys())[0]
print(f"✅ Uploaded: {video_filename}")

# Move to project directory
!mv "{video_filename}" ./
```

### Step 6: Run Lip Reading Inference

```python
# For English (LRS3 model)
!python infer.py config_filename=configs/LRS3_V_WER19.1.ini data_filename="{video_filename}" detector=mediapipe

# For Chinese
# !python infer.py config_filename=configs/CMLR_V_WER8.0.ini data_filename="{video_filename}" detector=mediapipe

# For Spanish
# !python infer.py config_filename=configs/CMUMOSEAS_V_ES_WER44.5.ini data_filename="{video_filename}" detector=mediapipe
```

## 🎯 Quick Start Commands

Here's a complete setup you can copy-paste into Colab:

```python
# Complete setup and inference in one go
!git clone https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages
%cd Visual_Speech_Recognition_for_Multiple_Languages

# Install dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -r requirements.txt
!pip install mediapipe>=0.10.0 ibug-face-detection>=0.1.0 ibug-face-alignment>=0.1.0 ffmpeg-python>=0.2.0 tqdm>=4.60.0
!apt-get update && apt-get install -y ffmpeg

# Create directories
!mkdir -p benchmarks/LRS3/models benchmarks/LRS3/language_models benchmarks/LRS3/labels

print("✅ Setup completed! Now upload your models and videos.")
```

## 🔧 Configuration Options

### Face Detection Methods:
- **MediaPipe** (faster): `detector=mediapipe`
- **RetinaFace** (more accurate): `detector=retinaface`

### GPU/CPU Usage:
- **GPU** (default): `gpu_idx=0`
- **CPU**: `gpu_idx=-1`

### Example Commands:

```python
# Use RetinaFace with CPU
!python infer.py config_filename=configs/LRS3_V_WER19.1.ini data_filename="video.mp4" detector=retinaface gpu_idx=-1

# Use MediaPipe with GPU (default)
!python infer.py config_filename=configs/LRS3_V_WER19.1.ini data_filename="video.mp4" detector=mediapipe
```

## 🎬 Video Requirements

For best results, your videos should have:

- ✅ **Clear face visibility** - Person's face should be clearly visible
- ✅ **Good lighting** - Well-lit environment
- ✅ **Stable camera** - Minimal camera movement
- ✅ **Clear speech** - Person should speak clearly
- ✅ **Supported format** - MP4, AVI, MOV, etc.

## 🚨 Troubleshooting

### Common Issues:

1. **"No module named 'torch'"**
   - Re-run the setup commands
   - Make sure PyTorch installation completed

2. **"Model file not found"**
   - Check that model files are in correct directories
   - Verify file paths in config files

3. **"Face not detected"**
   - Try switching between MediaPipe and RetinaFace
   - Check video quality and lighting
   - Ensure face is clearly visible

4. **"CUDA out of memory"**
   - Use CPU: add `gpu_idx=-1`
   - Restart Colab runtime
   - Try shorter video clips

5. **Poor transcription quality**
   - Check video quality
   - Try different models
   - Ensure clear lip movements

## 📚 Additional Resources

- [Original Repository](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages)
- [Official Tutorial](https://colab.research.google.com/drive/1jfb6e4xxhXHbmQf-nncdLno1u0b4j614)
- [Research Paper](https://arxiv.org/abs/2202.13084)

## 🎉 Success!

Once you see output like:
```
hyp: [transcribed text from your video]
```

You've successfully performed lip reading! The model has converted the visual speech in your video into text.

---

**Note**: This project is for research and benchmarking purposes only. Please respect the license terms and use responsibly.
