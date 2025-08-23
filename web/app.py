from flask import Flask, render_template, request, jsonify
import os
import tempfile
import cv2
import numpy as np
import base64
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for webcam
video_frames = []
recording = False
pipeline = None

# Function to set pipeline from external source
def set_pipeline(model_pipeline):
    global pipeline
    pipeline = model_pipeline
    print(f"✅ Pipeline set in Flask app: {pipeline is not None}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            # Save uploaded file temporarily
            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(temp_path)
            
            # Process video (this will be implemented when model is loaded)
            if pipeline:
                try:
                    result = pipeline(temp_path)
                    # Clean up
                    os.remove(temp_path)
                    return jsonify({'success': True, 'result': result})
                except Exception as e:
                    os.remove(temp_path)
                    return jsonify({'error': f'Processing failed: {str(e)}'}), 500
            else:
                os.remove(temp_path)
                return jsonify({'error': 'Model not loaded'}), 500
                
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/webcam/start', methods=['POST'])
def start_recording():
    global recording, video_frames
    recording = True
    video_frames = []
    return jsonify({'success': True, 'status': 'Recording started'})

@app.route('/webcam/stop', methods=['POST'])
def stop_recording():
    global recording, video_frames
    recording = False
    
    if len(video_frames) < 50:  # At least 2 seconds
        return jsonify({'error': 'Recording too short, need at least 2 seconds'}), 400
    
    try:
        # Create temporary video file
        output_path = tempfile.mktemp(suffix='.mp4')
        height, width = video_frames[0].shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 25, (width, height), False)
        
        for frame in video_frames:
            out.write(frame)
        out.release()
        
        # Process with pipeline
        if pipeline:
            result = pipeline(output_path)
            # Clean up
            os.remove(output_path)
            return jsonify({'success': True, 'result': result})
        else:
            os.remove(output_path)
            return jsonify({'error': 'Model not loaded'}), 500
            
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/webcam/frame', methods=['POST'])
def process_frame():
    global video_frames, recording
    if not recording:
        return jsonify({'success': False, 'message': 'Not recording'})
    
    try:
        # Get base64 frame data
        data = request.json
        frame_data = data.get('frame', '').split(',')[1]  # Remove data:image/jpeg;base64,
        
        # Decode frame
        image_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        video_frames.append(frame_gray)
        return jsonify({'success': True, 'frames_count': len(video_frames)})
        
    except Exception as e:
        return jsonify({'error': f'Frame processing failed: {str(e)}'}), 500

@app.route('/status')
def status():
    return jsonify({
        'model_loaded': pipeline is not None,
        'pipeline_type': str(type(pipeline)) if pipeline else 'None',
        'recording': recording,
        'frames_count': len(video_frames)
    })

@app.route('/debug')
def debug():
    return jsonify({
        'pipeline_exists': pipeline is not None,
        'pipeline_type': str(type(pipeline)) if pipeline else 'None',
        'pipeline_has_call': hasattr(pipeline, '__call__') if pipeline else False,
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER'])
    })

@app.route('/set_pipeline', methods=['POST'])
def set_pipeline_route():
    global pipeline
    try:
        # This is a simple way to check if pipeline is working
        # In production, you'd want proper authentication
        pipeline = "TEST_PIPELINE"  # Just for testing
        return jsonify({'success': True, 'message': 'Pipeline set to test mode'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
