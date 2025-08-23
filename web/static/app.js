// Global variables
let stream = null;
let isRecording = false;
let frameInterval = null;

// Tab switching
function switchTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    // Remove active class from all tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected tab content
    document.getElementById(tabName).classList.add('active');
    
    // Add active class to selected tab
    event.target.classList.add('active');
    
    // Initialize webcam if switching to webcam tab
    if (tabName === 'webcam') {
        initWebcam();
    }
}

// Initialize webcam
async function initWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: 640, 
                height: 480 
            } 
        });
        
        const video = document.getElementById('webcam');
        video.srcObject = stream;
        
        console.log('Webcam initialized successfully');
    } catch (err) {
        console.error('Error accessing webcam:', err);
        document.getElementById('recordingStatus').textContent = 'Error: Cannot access webcam';
    }
}

// Toggle recording
function toggleRecording() {
    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
}

// Start recording
function startRecording() {
    isRecording = true;
    const btn = document.getElementById('recordBtn');
    const status = document.getElementById('recordingStatus');
    
    btn.textContent = 'Stop Recording';
    btn.classList.add('recording');
    status.textContent = 'Recording... Speak to the camera';
    
    // Start capturing frames
    frameInterval = setInterval(captureFrame, 40); // 25 FPS
    
    console.log('Recording started');
}

// Stop recording
async function stopRecording() {
    isRecording = false;
    const btn = document.getElementById('recordBtn');
    const status = document.getElementById('recordingStatus');
    
    btn.textContent = 'Start Recording';
    btn.classList.remove('recording');
    status.textContent = 'Processing...';
    
    // Stop capturing frames
    if (frameInterval) {
        clearInterval(frameInterval);
        frameInterval = null;
    }
    
    // Process the recorded frames
    await processWebcamRecording();
    
    status.textContent = 'Ready to record again';
}

// Capture frame from webcam
function captureFrame() {
    if (!isRecording || !stream) return;
    
    const video = document.getElementById('webcam');
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    
    // Convert to base64 and send to server
    const frameData = canvas.toDataURL('image/jpeg', 0.95);
    
    fetch('/webcam/frame', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ frame: frameData })
    }).catch(err => console.error('Error sending frame:', err));
}

// Process webcam recording
async function processWebcamRecording() {
    try {
        const response = await fetch('/webcam/stop', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            showResult('webcamResult', `🗣️ Result: ${data.result}`, 'success');
        } else {
            showResult('webcamResult', `❌ Error: ${data.error}`, 'error');
        }
    } catch (err) {
        showResult('webcamResult', `❌ Error: ${err.message}`, 'error');
    }
}

// File upload handling
document.getElementById('videoFile').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('fileInfo').style.display = 'block';
    }
});

// Drag and drop handling
const uploadArea = document.getElementById('uploadArea');

uploadArea.addEventListener('dragover', function(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', function() {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', function(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        document.getElementById('videoFile').files = files;
        document.getElementById('fileName').textContent = files[0].name;
        document.getElementById('fileInfo').style.display = 'block';
    }
});

// Process uploaded video
async function processVideo() {
    const fileInput = document.getElementById('videoFile');
    const file = fileInput.files[0];
    
    if (!file) {
        showResult('uploadResult', '❌ Please select a video file first', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('video', file);
    
    showResult('uploadResult', '🔄 Processing video...', 'status');
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            showResult('uploadResult', `🗣️ Result: ${data.result}`, 'success');
        } else {
            showResult('uploadResult', `❌ Error: ${data.error}`, 'error');
        }
    } catch (err) {
        showResult('uploadResult', `❌ Error: ${err.message}`, 'error');
    }
}

// Show result message
function showResult(elementId, message, type) {
    const element = document.getElementById(elementId);
    element.innerHTML = `<div class="result ${type}">${message}</div>`;
}

// Keyboard shortcuts
document.addEventListener('keydown', function(event) {
    if (event.code === 'Space' && document.getElementById('webcam').classList.contains('active')) {
        event.preventDefault();
        toggleRecording();
    }
});

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Initialize webcam if starting on webcam tab
    if (document.getElementById('webcam').classList.contains('active')) {
        initWebcam();
    }
});
