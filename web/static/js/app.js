// Global variables
let uploadedFile = null;
let webcamStream = null;
let isRecording = false;
let recordingStartTime = null;
let frameInterval = null;
let sessionStarted = false;

// Tab switching
function switchTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
    
    // Show selected tab
    document.getElementById(tabName + '-tab').classList.add('active');
    event.target.classList.add('active');
    
    // Initialize webcam when switching to webcam tab
    if (tabName === 'webcam' && !webcamStream) {
        initializeWebcam();
    }
}

// Video Upload Functions
function handleVideoUpload(input) {
    const file = input.files[0];
    if (file) {
        uploadedFile = file;
        document.getElementById('processBtn').disabled = false;
        document.getElementById('uploadStatus').className = 'status ready';
        document.getElementById('uploadStatus').textContent = `Selected: ${file.name}`;
        document.querySelector('.upload-area div').textContent = `Selected: ${file.name}`;
    }
}

function processVideo() {
    if (!uploadedFile) return;
    
    const formData = new FormData();
    formData.append('video', uploadedFile);
    
    document.getElementById('uploadStatus').className = 'status processing';
    document.getElementById('uploadStatus').innerHTML = '<div class="loading"></div> Processing video...';
    document.getElementById('uploadResult').textContent = 'Processing...';
    document.getElementById('processBtn').disabled = true;
    
    fetch('/api/upload/video', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            document.getElementById('uploadStatus').className = 'status success';
            document.getElementById('uploadStatus').textContent = 'Processing complete!';
            document.getElementById('uploadResult').textContent = data.result || 'No speech detected';
        } else {
            document.getElementById('uploadStatus').className = 'status error';
            document.getElementById('uploadStatus').textContent = 'Error: ' + data.error;
            document.getElementById('uploadResult').textContent = 'Processing failed';
        }
        document.getElementById('processBtn').disabled = false;
    })
    .catch(error => {
        document.getElementById('uploadStatus').className = 'status error';
        document.getElementById('uploadStatus').textContent = 'Network error';
        document.getElementById('uploadResult').textContent = 'Failed to process video';
        document.getElementById('processBtn').disabled = false;
    });
}

// Webcam Functions
async function initializeWebcam() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        document.getElementById('webcamVideo').srcObject = webcamStream;
        
        // Start webcam session
        await startWebcamSession();
        
        document.getElementById('webcamStatus').className = 'status ready';
        document.getElementById('webcamStatus').textContent = 'Webcam ready! Press "Start Recording" to begin.';
    } catch (error) {
        document.getElementById('webcamStatus').className = 'status error';
        document.getElementById('webcamStatus').textContent = 'Unable to access webcam: ' + error.message;
    }
}

async function startWebcamSession() {
    try {
        const response = await fetch('/api/webcam/start-session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await response.json();
        sessionStarted = data.success;
    } catch (error) {
        console.error('Failed to start webcam session:', error);
    }
}

function toggleRecording() {
    if (!sessionStarted) {
        console.error('Webcam session not started');
        return;
    }
    
    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
}

async function startRecording() {
    try {
        const response = await fetch('/api/webcam/toggle-recording', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await response.json();
        
        if (data.success && data.recording) {
            isRecording = true;
            recordingStartTime = Date.now();
            
            document.getElementById('startStopBtn').textContent = 'Stop Recording';
            document.getElementById('processCurrentBtn').disabled = false;
            document.getElementById('recordIndicator').className = 'indicator-dot recording';
            document.getElementById('recordStatus').textContent = 'Recording';
            document.getElementById('webcamStatus').className = 'status recording';
            document.getElementById('webcamStatus').textContent = 'Recording... Speak to camera, then stop to process.';
            
            // Start sending frames
            frameInterval = setInterval(captureAndSendFrame, 100); // 10 FPS
        }
    } catch (error) {
        console.error('Failed to start recording:', error);
    }
}

async function stopRecording() {
    try {
        const response = await fetch('/api/webcam/toggle-recording', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await response.json();
        
        if (data.success && !data.recording) {
            isRecording = false;
            
            document.getElementById('startStopBtn').textContent = 'Start Recording';
            document.getElementById('processCurrentBtn').disabled = true;
            document.getElementById('recordIndicator').className = 'indicator-dot';
            document.getElementById('recordStatus').textContent = 'Ready';
            document.getElementById('faceIndicator').className = 'indicator-dot';
            document.getElementById('speakIndicator').className = 'indicator-dot';
            
            // Stop sending frames
            if (frameInterval) {
                clearInterval(frameInterval);
                frameInterval = null;
            }
            
            // Process the recorded session
            processWebcamSession();
        }
    } catch (error) {
        console.error('Failed to stop recording:', error);
    }
}

function captureAndSendFrame() {
    if (!isRecording || !webcamStream) return;
    
    const video = document.getElementById('webcamVideo');
    const canvas = document.getElementById('webcamCanvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    
    // Convert to base64
    const imageData = canvas.toDataURL('image/jpeg', 0.95);
    const base64Data = imageData.split(',')[1];
    
    // Send frame to server
    fetch('/api/webcam/process-frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame: base64Data })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Update indicators
            document.getElementById('faceIndicator').className = 
                'indicator-dot' + (data.face_detected ? ' face' : '');
            document.getElementById('speakIndicator').className = 
                'indicator-dot' + (data.speaking_detected ? ' speaking' : '');
            
            // Update recording duration
            const duration = (Date.now() - recordingStartTime) / 1000;
            document.getElementById('webcamStatus').textContent = 
                `Recording... (${duration.toFixed(1)}s) Speak to camera, then stop to process.`;
        }
    })
    .catch(error => console.error('Frame processing error:', error));
}

function processCurrentBuffer() {
    processWebcamSession();
}

function processWebcamSession() {
    document.getElementById('webcamStatus').className = 'status processing';
    document.getElementById('webcamStatus').innerHTML = '<div class="loading"></div> Processing recorded session...';
    document.getElementById('webcamResult').textContent = 'Processing...';
    
    fetch('/api/webcam/process-session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            document.getElementById('webcamStatus').className = 'status success';
            document.getElementById('webcamStatus').textContent = 'Processing complete!';
            document.getElementById('webcamResult').textContent = data.result || 'No speech detected';
        } else {
            document.getElementById('webcamStatus').className = 'status error';
            document.getElementById('webcamStatus').textContent = 'Error: ' + data.error;
            document.getElementById('webcamResult').textContent = 'Processing failed';
        }
    })
    .catch(error => {
        document.getElementById('webcamStatus').className = 'status error';
        document.getElementById('webcamStatus').textContent = 'Network error';
        document.getElementById('webcamResult').textContent = 'Failed to process session';
    });
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('Lip Reading Web App loaded');
});
