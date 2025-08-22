"""
Face detection and speaking detection utilities
"""
import numpy as np
import mediapipe as mp

class FaceDetector:
    """Handles face detection and speaking detection using MediaPipe"""
    
    def __init__(self):
        self.face_mesh = None
        self._initialize_mediapipe()
    
    def _initialize_mediapipe(self):
        """Initialize MediaPipe face mesh"""
        try:
            print("🔧 Initializing MediaPipe face detection...")
            
            mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            print("✅ MediaPipe initialized!")
            
        except Exception as e:
            print(f"❌ MediaPipe initialization error: {e}")
            raise
    
    def detect_face_and_speaking(self, frame_rgb, session):
        """
        Detect face and speaking from frame and update session state
        
        Args:
            frame_rgb: RGB frame from webcam
            session: Session dictionary to update with detection results
        """
        try:
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                session['face_detected'] = True
                
                landmarks = results.multi_face_landmarks[0]
                
                # Get mouth landmarks (using MediaPipe face mesh indices)
                mouth_indices = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
                mouth_landmarks = []
                
                for idx in mouth_indices:
                    if idx < len(landmarks.landmark):
                        landmark = landmarks.landmark[idx]
                        mouth_landmarks.append([landmark.x, landmark.y])
                
                mouth_landmarks = np.array(mouth_landmarks)
                
                # Calculate movement for speaking detection
                if session['prev_landmarks'] is not None:
                    movement = np.mean(np.linalg.norm(mouth_landmarks - session['prev_landmarks'], axis=1))
                    session['landmark_history'].append(movement)
                    
                    # Keep last 10 measurements
                    if len(session['landmark_history']) > 10:
                        session['landmark_history'].pop(0)
                    
                    avg_movement = np.mean(session['landmark_history'])
                    session['speaking_detected'] = avg_movement > 0.003
                
                session['prev_landmarks'] = mouth_landmarks
            else:
                session['face_detected'] = False
                session['speaking_detected'] = False
                
        except Exception as e:
            print(f"❌ Face detection error: {e}")
            session['face_detected'] = False
            session['speaking_detected'] = False
    
    def is_ready(self):
        """Check if face detector is ready"""
        return self.face_mesh is not None
