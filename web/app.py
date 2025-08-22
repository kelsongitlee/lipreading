"""
Main Flask Application for Lip Reading Web Interface
"""
import os
import sys
from flask import Flask

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from web.utils.model_loader import ModelLoader
from web.utils.face_detector import FaceDetector
from web.routes import upload_bp, webcam_bp

def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)
    
    # Configuration
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
    app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
    app.config['SECRET_KEY'] = 'lip-reading-secret-key'
    
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize global components
    print("🚀 Initializing Lip Reading Models...")
    model_loader = ModelLoader()
    face_detector = FaceDetector()
    
    # Store in app context
    app.model_loader = model_loader
    app.face_detector = face_detector
    
    # Register blueprints
    app.register_blueprint(upload_bp)
    app.register_blueprint(webcam_bp)
    
    # Main route
    @app.route('/')
    def index():
        from flask import render_template
        return render_template('index.html')
    
    return app

def setup_ngrok(port=5000):
    """Setup ngrok tunnel with authentication"""
    import subprocess
    import time
    
    print("🔧 Setting up ngrok...")
    
    # Install ngrok if not exists
    if not os.path.exists('./ngrok'):
        os.system("wget -q https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz")
        os.system("tar -xzf ngrok-v3-stable-linux-amd64.tgz")
        os.system("chmod +x ngrok")
    
    # Set auth token
    auth_token = "30xzV3aCADfX3hfNJflBDvKZgnZ_3i1LLR3zGnUxZzSjcG1a3"
    os.system(f"./ngrok config add-authtoken {auth_token}")
    
    # Start ngrok tunnel
    print(f"🚀 Starting ngrok tunnel on port {port}...")
    process = subprocess.Popen(
        ["./ngrok", "http", str(port), "--log=stdout"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for tunnel to start
    time.sleep(3)
    
    # Get public URL
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:4040/api/tunnels"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        import json
        tunnels = json.loads(result.stdout)
        
        if 'tunnels' in tunnels and len(tunnels['tunnels']) > 0:
            public_url = tunnels['tunnels'][0]['public_url']
            print(f"🌐 Public URL: {public_url}")
            return public_url
    except Exception as e:
        print(f"⚠️ Could not get ngrok URL: {e}")
    
    return None

if __name__ == '__main__':
    try:
        # Create Flask app
        app = create_app()
        
        # Setup ngrok
        public_url = setup_ngrok(5000)
        
        print("\n" + "="*60)
        print("🎬 LIP READING WEB SERVER READY!")
        print("="*60)
        print(f"🌐 Access your website at: {public_url or 'Check ngrok output above'}")
        print("📁 Video Upload: Upload MP4/AVI files for processing")
        print("📹 Webcam: Real-time lip reading from your camera")
        print("⚡ Models loaded and ready for processing")
        print("="*60)
        
        # Start Flask server
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except Exception as e:
        print(f"❌ Server startup error: {e}")
        import traceback
        traceback.print_exc()
