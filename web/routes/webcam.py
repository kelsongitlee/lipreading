"""
Webcam processing route handlers
"""
import base64
import tempfile
import cv2
import numpy as np
from flask import Blueprint, request, jsonify, current_app
from web.utils.video_processor import VideoProcessor

webcam_bp = Blueprint('webcam', __name__, url_prefix='/api/webcam')

# Global webcam session state
webcam_sessions = {}

def get_session_id():
    """Get session ID from request (simple implementation)"""
    return request.remote_addr  # Use IP as session ID for simplicity

@webcam_bp.route('/start-session', methods=['POST'])
def start_session():
    """Start a new webcam recording session"""
    session_id = get_session_id()
    
    webcam_sessions[session_id] = {
        'recording': False,
        'frames': [],
        'face_detected': False,
        'speaking_detected': False,
        'prev_landmarks': None,
        'landmark_history': []
    }
    
    return jsonify({'success': True, 'message': 'Session started'})

@webcam_bp.route('/toggle-recording', methods=['POST'])
def toggle_recording():
    """Toggle recording state"""
    session_id = get_session_id()
    
    if session_id not in webcam_sessions:
        return jsonify({'success': False, 'error': 'No active session'})
    
    session = webcam_sessions[session_id]
    session['recording'] = not session['recording']
    
    if session['recording']:
        # Clear previous session data
        session['frames'] = []
        session['face_detected'] = False
        session['speaking_detected'] = False
        session['prev_landmarks'] = None
        session['landmark_history'] = []
        print("🟢 Recording started!")
        return jsonify({'success': True, 'recording': True, 'message': 'Recording started'})
    else:
        print("🔴 Recording stopped!")
        return jsonify({'success': True, 'recording': False, 'message': 'Recording stopped'})

@webcam_bp.route('/process-frame', methods=['POST'])
def process_frame():
    """Process incoming webcam frame"""
    try:
        session_id = get_session_id()
        
        if session_id not in webcam_sessions:
            return jsonify({'success': False, 'error': 'No active session'})
        
        session = webcam_sessions[session_id]
        
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({'success': False, 'error': 'No frame data provided'})
        
        # Only process if recording
        if not session['recording']:
            return jsonify({
                'success': True, 
                'face_detected': False, 
                'speaking_detected': False,
                'recording': False
            })
        
        # Decode frame
        image_bytes = base64.b64decode(data['frame'])
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face and speaking
        current_app.face_detector.detect_face_and_speaking(frame_rgb, session)
        
        # Store frame for later processing
        session['frames'].append(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY))
        
        return jsonify({
            'success': True,
            'face_detected': session['face_detected'],
            'speaking_detected': session['speaking_detected'],
            'recording': session['recording'],
            'frame_count': len(session['frames'])
        })
        
    except Exception as e:
        print(f"❌ Frame processing error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@webcam_bp.route('/process-session', methods=['POST'])
def process_session():
    """Process the complete webcam recording session"""
    try:
        session_id = get_session_id()
        
        if session_id not in webcam_sessions:
            return jsonify({'success': False, 'error': 'No active session'})
        
        session = webcam_sessions[session_id]
        
        if len(session['frames']) < 30:  # Need at least 1 second at 30fps
            return jsonify({'success': True, 'result': 'Recording too short - need at least 2 seconds'})
        
        print(f"🎬 Processing webcam session with {len(session['frames'])} frames")
        
        # Process frames
        processor = VideoProcessor(current_app.model_loader.pipeline)
        result = processor.process_frame_sequence(session['frames'])
        
        # Clear session frames
        session['frames'] = []
        session['recording'] = False
        session['face_detected'] = False
        session['speaking_detected'] = False
        session['prev_landmarks'] = None
        session['landmark_history'] = []
        
        if result:
            clean_result, is_filtered = processor.filter_repetitive_result(result)
            if clean_result and not is_filtered:
                return jsonify({'success': True, 'result': clean_result})
            else:
                return jsonify({'success': True, 'result': 'No clear speech detected (filtered repetitive output)'})
        else:
            return jsonify({'success': True, 'result': 'No speech detected'})
            
    except Exception as e:
        print(f"❌ Webcam session processing error: {e}")
        return jsonify({'success': False, 'error': str(e)})
