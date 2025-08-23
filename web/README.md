# Lip Reading Web Application

A professional Flask-based web interface for real-time lip reading using deep learning models.

## 🌐 Features

- **Video Upload**: Upload MP4/AVI/MOV files for lip reading analysis
- **Real-time Webcam**: Live lip reading from webcam with face detection
- **Professional UI**: Clean, responsive web interface
- **Public Access**: Automatic ngrok tunnel for external access
- **GPU Acceleration**: Optimized for Google Colab GPU instances

## 🚀 Quick Start (Google Colab)

1. Open `colab/setup.ipynb` in Google Colab
2. Run all cells in order:
   - Setup repository and dependencies
   - Download models from Google Drive
   - Start web server
3. Access your public URL (automatically generated)

## 📁 Project Structure

```
web/
├── app.py                  # Main Flask application
├── routes/                 # API endpoints
│   ├── upload.py          # Video upload handling
│   └── webcam.py          # Webcam processing
├── templates/             # HTML templates
│   ├── base.html         # Base template
│   └── index.html        # Main interface
├── static/               # Frontend assets
│   ├── css/style.css    # Styling
│   └── js/app.js        # JavaScript
└── utils/               # Backend utilities
    ├── model_loader.py  # Model initialization
    ├── face_detector.py # MediaPipe integration
    └── video_processor.py # Video enhancement
```

## 🔧 Local Development

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-web.txt

# Run the server
cd web/
python app.py
```

## 🎯 API Endpoints

- `GET /` - Main web interface
- `POST /api/upload/video` - Process uploaded video
- `POST /api/webcam/start-session` - Start webcam session
- `POST /api/webcam/process-frame` - Process webcam frame
- `POST /api/webcam/process-session` - Process complete session

## 💡 Technical Details

- **Backend**: Flask with Blueprint architecture
- **Frontend**: Vanilla JavaScript with HTML5 video API
- **Models**: Pre-trained lip reading models (LRS3_V_WER19.1)
- **Face Detection**: MediaPipe for real-time detection
- **Video Processing**: OpenCV with quality enhancement
- **Deployment**: ngrok tunnel for public access

## 🛠️ Customization

### Adding New Features
1. Create new route in `routes/`
2. Add corresponding template in `templates/`
3. Register blueprint in `app.py`

### Modifying UI
- Edit CSS in `static/css/style.css`
- Update JavaScript in `static/js/app.js`
- Modify HTML templates in `templates/`

### Enhancing Models
- Update utilities in `utils/`
- Modify model loading in `model_loader.py`
- Adjust processing in `video_processor.py`
