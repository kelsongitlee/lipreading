"""
Video upload route handlers
"""
import os
import tempfile
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from web.utils.video_processor import VideoProcessor

upload_bp = Blueprint('upload', __name__, url_prefix='/api/upload')

@upload_bp.route('/video', methods=['POST'])
def process_video():
    """Handle video upload and processing"""
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'})
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Validate file type
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({'success': False, 'error': 'Unsupported file format'})
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"📁 Processing uploaded video: {filename}")
        
        # Process the video
        processor = VideoProcessor(current_app.model_loader.pipeline)
        result = processor.process_video_file(filepath)
        
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)
        
        if result:
            clean_result, is_filtered = processor.filter_repetitive_result(result)
            if clean_result and not is_filtered:
                return jsonify({'success': True, 'result': clean_result})
            else:
                return jsonify({'success': True, 'result': 'No clear speech detected (filtered noise)'})
        else:
            return jsonify({'success': True, 'result': 'No speech detected'})
            
    except Exception as e:
        print(f"❌ Video processing error: {e}")
        return jsonify({'success': False, 'error': str(e)})
