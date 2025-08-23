# 🚀 Lip Reading Setup - Google Colab

**Simple setup for lip reading in Google Colab with just 4 text files!**

## 📋 What You Need

1. **Google Colab** (free GPU access)
2. **Google Drive** with your model files
3. **These 4 text files** (copy-paste into Colab cells)

## 🎯 How to Use

### **Step 1: Open Google Colab**
- Go to [colab.research.google.com](https://colab.research.google.com)
- Sign in with your Google account
- Create a new notebook

### **Step 2: Enable GPU**
- Click **Runtime** → **Change runtime type**
- Set **Hardware accelerator** to **GPU**
- Click **Save**

### **Step 3: Run the Cells**

#### **Cell 1: Setup Environment**
Copy-paste the content of `colab_cell_1_setup.txt`
- Clones repository
- Installs PyTorch with CUDA
- Installs dependencies

#### **Cell 2: Download Models**
Copy-paste the content of `colab_cell_2_download.txt`
- Downloads models from your Google Drive
- Sets up correct directory structure

#### **Cell 3: Test Video Inference**
Copy-paste the content of `colab_cell_3_test_inference.txt`
- Tests with a video sample
- Verifies everything works

#### **Cell 4: Real-time Webcam**
Copy-paste the content of `colab_cell_4_realtime_webcam.txt`
- Starts webcam interface
- Press SPACE to record, SPACE again to process

#### **Cell 5: Web Server (Optional)**
Copy-paste the content of `colab_cell_5_web_server.txt`
- Launches Flask web server with ngrok
- Creates public URL accessible from anywhere
- Provides web interface with 2 tabs:
  - 📁 Video Upload: Upload and process video files
  - 📹 Live Webcam: Real-time webcam lip reading

## 🔑 Your Google Drive IDs

Make sure these match your actual Google Drive folders:
- **Main Model**: `1JkOPriFqzoHdZZl2XEsaNMnos9hxq0mO`
- **Language Model**: `1pl7-S77Bo6Tb5Sprl-rbjJCSTGCWxkMo`
- **Video Samples**: `1gTbqLsCbuj87MT-_wviKjfMdl58l6ECZ`

## 📁 Expected Structure

After running, you should have:
```
/content/lipreading/
├── benchmarks/LRS3/
│   ├── models/LRS3_V_WER19.1/
│   │   ├── model.pth
│   │   └── model.json
│   └── language_models/lm_en_subword/
│       ├── model.pth
│       └── model.json
├── video_samples/
│   └── video_muted1.mp4
├── configs/
│   └── LRS3_V_WER19.1.ini
└── web/                          # Web server files
    ├── app.py                    # Flask application
    ├── templates/
    │   └── index.html           # Web interface
    └── static/
        └── app.js               # JavaScript functionality
```

## 🎮 Controls

- **SPACE**: Start/stop recording
- **Webcam**: Shows live preview
- **Red Circle**: Ready to record
- **Green Circle**: Recording active

## 🚨 Troubleshooting

If downloads fail:
1. Check Google Drive folder permissions (must be public)
2. Manually download and upload files
3. Ensure correct directory structure

## 🎉 That's It!

**Just copy-paste each text file into the corresponding Colab cell and run!** 

No complex setup, no alternative methods, no redundant code - just **one simple, working solution**. 🎯
