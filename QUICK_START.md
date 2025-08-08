# 🚀 Quick Start Guide - Visual Speech Recognition

## 🎯 What You Need to Know

**Input**: Video files with people speaking  
**Output**: Text transcription of what they're saying (lip reading)  
**Supported Languages**: English, Chinese, Spanish, French, Portuguese, Italian  

## 🎬 What You Need to Prepare

1. **Test Videos**: MP4/AVI/MOV files with clear face visibility
2. **Google Account**: For Google Colab (free GPU access)
3. **Internet Connection**: For downloading models and dependencies

## 🚀 Fastest Way to Get Started

### Step 1: Open Google Colab
- Go to [colab.research.google.com](https://colab.research.google.com)
- Sign in with your Google account
- Create a new notebook

### Step 2: Enable GPU
- Click **Runtime** → **Change runtime type**
- Set **Hardware accelerator** to **GPU**
- Click **Save**

### Step 3: Copy-Paste This Code
```python
# Complete setup in one go
!git clone https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages
%cd Visual_Speech_Recognition_for_Multiple_Languages

# Install everything
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -r requirements.txt
!pip install mediapipe>=0.10.0 ibug-face-detection>=0.1.0 ibug-face-alignment>=0.1.0 ffmpeg-python>=0.2.0 tqdm>=4.60.0
!apt-get update && apt-get install -y ffmpeg

# Create directories
!mkdir -p benchmarks/LRS3/models benchmarks/LRS3/language_models benchmarks/LRS3/labels

print("✅ Setup completed!")
```

### Step 4: Download Models
**For English (Best Performance)**:
- Download from: [Google Drive](http://bit.ly/40EAtyX) or [Baidu Drive](https://bit.ly/3ZjbrV5) (key: dqsy)
- Extract and upload to Colab
- Place in `benchmarks/LRS3/models/` directory

### Step 5: Upload Your Video
```python
from google.colab import files
uploaded = files.upload()
video_filename = list(uploaded.keys())[0]
!mv "{video_filename}" ./
```

### Step 6: Run Lip Reading!
```python
!python infer.py config_filename=configs/LRS3_V_WER19.1.ini data_filename="{video_filename}" detector=mediapipe
```

## 🎯 Expected Output
```
hyp: [transcribed text from your video]
```

## 🔧 Different Languages

| Language | Config File | Download Link |
|----------|-------------|---------------|
| English | `LRS3_V_WER19.1.ini` | [Download](http://bit.ly/40EAtyX) |
| Chinese | `CMLR_V_WER8.0.ini` | [Download](https://bit.ly/3fR8RkU) |
| Spanish | `CMUMOSEAS_V_ES_WER44.5.ini` | [Download](https://bit.ly/34MjWBW) |
| French | `CMUMOSEAS_V_FR_WER58.6.ini` | [Download](https://bit.ly/3Ik6owb) |

## 🎬 Video Requirements

- ✅ **Clear face** - Person's face should be visible
- ✅ **Good lighting** - Well-lit environment  
- ✅ **Stable camera** - Minimal movement
- ✅ **Clear speech** - Person speaks clearly
- ✅ **Supported format** - MP4, AVI, MOV, etc.

## 🚨 Common Issues

1. **"No module named 'torch'"** → Re-run setup commands
2. **"Model file not found"** → Check model download and placement
3. **"Face not detected"** → Try `detector=retinaface` instead
4. **"CUDA out of memory"** → Add `gpu_idx=-1` to use CPU

## 📚 More Resources

- [Detailed Setup Guide](COLAB_SETUP_GUIDE.md)
- [Original Repository](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages)
- [Official Tutorial](https://colab.research.google.com/drive/1jfb6e4xxhXHbmQf-nncdLno1u0b4j614)

## 🎉 Success!

You're now ready to perform lip reading on your videos! The model will convert visual speech into text transcriptions.

---

**Note**: This project is for research purposes. Please respect the license terms.
