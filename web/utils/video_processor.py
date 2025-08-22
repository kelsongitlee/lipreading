"""
Video processing utilities for lip reading
"""
import os
import tempfile
import cv2
import numpy as np

class VideoProcessor:
    """Handles video processing for lip reading"""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
    
    def process_video_file(self, video_path):
        """
        Process uploaded video file with enhancement
        
        Args:
            video_path: Path to the video file
            
        Returns:
            str: Lip reading result or None
        """
        try:
            # Apply enhanced preprocessing
            temp_processed = tempfile.mktemp(suffix='.mp4')
            
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_processed, fourcc, fps, (width, height), False)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Convert to grayscale and enhance
                enhanced_frame = self._enhance_frame(frame)
                out.write(enhanced_frame)
            
            cap.release()
            out.release()
            
            # Process with lip reading model
            result = self.pipeline(temp_processed)
            
            # Clean up
            if os.path.exists(temp_processed):
                os.remove(temp_processed)
                
            return result
            
        except Exception as e:
            print(f"❌ Video processing error: {e}")
            return None
    
    def process_frame_sequence(self, frames):
        """
        Process a sequence of frames from webcam
        
        Args:
            frames: List of grayscale frames
            
        Returns:
            str: Lip reading result or None
        """
        try:
            if len(frames) < 30:  # Need minimum frames
                return None
            
            # Create temporary video file
            output_path = tempfile.mktemp(suffix='.mp4')
            height, width = frames[0].shape
            
            # Use 25 FPS for processing
            fps = 25
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), False)
            
            # Write enhanced frames
            for frame in frames:
                enhanced_frame = self._enhance_grayscale_frame(frame)
                out.write(enhanced_frame)
            
            out.release()
            
            # Process with lip reading model
            result = self.pipeline(output_path)
            
            # Clean up
            if os.path.exists(output_path):
                os.remove(output_path)
            
            return result
            
        except Exception as e:
            print(f"❌ Frame sequence processing error: {e}")
            return None
    
    def _enhance_frame(self, frame):
        """
        Enhance color frame for better lip reading accuracy
        
        Args:
            frame: BGR color frame
            
        Returns:
            Enhanced grayscale frame
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization
        enhanced = cv2.equalizeHist(gray)
        
        # Apply sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Slight blur to reduce noise
        final_frame = cv2.GaussianBlur(sharpened, (3, 3), 0.5)
        
        return final_frame
    
    def _enhance_grayscale_frame(self, frame):
        """
        Enhance grayscale frame for better lip reading accuracy
        
        Args:
            frame: Grayscale frame
            
        Returns:
            Enhanced grayscale frame
        """
        # Apply histogram equalization
        enhanced = cv2.equalizeHist(frame)
        
        # Apply sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Slight blur to reduce noise
        final_frame = cv2.GaussianBlur(sharpened, (3, 3), 0.5)
        
        return final_frame
    
    def filter_repetitive_result(self, text, threshold=0.8):
        """
        Filter out repetitive results
        
        Args:
            text: Input text to filter
            threshold: Repetition threshold (0.0 to 1.0)
            
        Returns:
            tuple: (filtered_text or None, is_filtered boolean)
        """
        if not text or not text.strip():
            return None, True
            
        words = text.strip().split()
        if len(words) == 0:
            return None, True
            
        # Check for repetition
        unique_words = set(words)
        if len(unique_words) <= 1 and len(words) > 1:
            return None, True  # All same word
            
        most_common_word = max(set(words), key=words.count)
        repetition_ratio = words.count(most_common_word) / len(words)
        
        if repetition_ratio >= threshold:
            return None, True  # Too repetitive
            
        return text.strip().upper(), False
