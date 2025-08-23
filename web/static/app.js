// Simple webcam functionality
let webcamStream = null;
let isRecording = false;
let recordingStartTime = null;

// Tab switching
function switchTab(tabName) {
    // Hide all content
    document.getElementById('upload').style.display = 'none';
    document.getElementById('webcam').style.display = 'none';
    
    // Show selected content
    document.getElementById(tabName).style.display = 'block';
    
    // Initialize webcam if needed
    if (tabName === 'webcam') {
        initWebcam();
    }
}

// Initialize webcam
async function initWebcam() {
    try {
        const video = document.getElementById('webcam-video');
        webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = webcamStream;
        console.log('Webcam ready');
    } catch (error) {
        console.error('Webcam error:', error);
        alert('Cannot access webcam');
    }
}

// Start recording
function startRecording() {
    if (!webcamStream) {
        alert('Webcam not ready');
        return;
    }
    
    isRecording = true;
    recordingStartTime = Date.now();
    
    // Update UI
    document.getElementById('start-btn').disabled = true;
    document.getElementById('stop-btn').disabled = false;
    document.getElementById('recording-status').textContent = '🟢';
    document.getElementById('recording-text').textContent = 'Recording';
    
    console.log('Recording started');
}

// Stop recording
function stopRecording() {
    if (!isRecording) return;
    
    isRecording = false;
    const duration = ((Date.now() - recordingStartTime) / 1000).toFixed(1);
    
    // Update UI
    document.getElementById('start-btn').disabled = false;
    document.getElementById('stop-btn').disabled = true;
    document.getElementById('recording-status').textContent = '🔴';
    document.getElementById('recording-text').textContent = 'Ready';
    
    // Show results
    document.getElementById('results-section').style.display = 'block';
    document.getElementById('result-content').innerHTML = `
        <strong>Recording Complete!</strong><br>
        Duration: ${duration} seconds<br>
        <br>
        <strong>Sample Result:</strong><br>
        "Hello, this is a test recording for lip reading."
    `;
    
    console.log('Recording stopped');
}

// Reset
function resetWebcam() {
    isRecording = false;
    document.getElementById('start-btn').disabled = false;
    document.getElementById('stop-btn').disabled = true;
    document.getElementById('recording-status').textContent = '🔴';
    document.getElementById('recording-text').textContent = 'Ready';
    document.getElementById('results-section').style.display = 'none';
}

// Keyboard shortcuts
document.addEventListener('keydown', (event) => {
    if (event.code === 'Space') {
        event.preventDefault();
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    }
});

// File upload
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('fileInfo').style.display = 'block';
    }
}

function processVideo() {
    const fileInput = document.getElementById('videoFile');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a file first');
        return;
    }
    
    document.getElementById('uploadResult').innerHTML = 'Processing video...';
    
    const formData = new FormData();
    formData.append('video', file);
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            document.getElementById('uploadResult').innerHTML = `Result: ${data.result}`;
        } else {
            document.getElementById('uploadResult').innerHTML = `Error: ${data.error}`;
        }
    })
    .catch(error => {
        document.getElementById('uploadResult').innerHTML = `Error: ${error.message}`;
    });
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Add event listeners
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const resetBtn = document.getElementById('reset-btn');
    
    if (startBtn) startBtn.addEventListener('click', startRecording);
    if (stopBtn) stopBtn.addEventListener('click', stopRecording);
    if (resetBtn) resetBtn.addEventListener('click', resetWebcam);
    
    // File input
    const fileInput = document.getElementById('videoFile');
    if (fileInput) fileInput.addEventListener('change', handleFileSelect);
    
    // Start with upload tab
    switchTab('upload');
});
