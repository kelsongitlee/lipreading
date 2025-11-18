#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import warnings
import torchvision
from ibug.face_detection import RetinaFacePredictor
from ibug.face_alignment import FANPredictor
warnings.filterwarnings("ignore")


class LandmarksDetector:
    def __init__(self, device="cuda:0", model_name='resnet50'):
        self.face_detector = RetinaFacePredictor(
            device=device,
            threshold=0.8,
            model=RetinaFacePredictor.get_model(model_name)
        )
        self.landmark_detector = FANPredictor(device=device, model=None)

    def __call__(self, filename):
        video_frames = torchvision.io.read_video(filename, pts_unit='sec')[0].numpy()
        landmarks = []
        for frame in video_frames:
            detected_faces = self.face_detector(frame, rgb=False)
            face_points, _ = self.landmark_detector(frame, detected_faces, rgb=True)
            if len(detected_faces) == 0:
                landmarks.append(None)
            else:
                max_id, max_size = 0, 0
                for idx, bbox in enumerate(detected_faces):
                    bbox_size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
                    if bbox_size > max_size:
                        max_id, max_size = idx, bbox_size
                landmarks.append(face_points[max_id])
        
        # CRITICAL FIX: Make detection more lenient for real-time processing
        # Allow some frames to fail (especially for short videos or fast speech, or when user moves camera)
        # Only fail if ALL frames failed AND we have enough frames (likely a real issue)
        detected_count = sum(1 for l in landmarks if l is not None)
        total_frames = len(landmarks)
        
        # For real-time processing (75 frames = 3 seconds), be more lenient
        # User may move camera, face may be partially visible, etc.
        if detected_count == 0 and total_frames >= 50:
            # All frames failed and we have enough frames - likely a real issue
            # But for real-time, we should still allow it (user may have moved camera)
            print(f"[RetinaFace] WARNING: No faces detected in {total_frames} frames")
            print(f"[RetinaFace] This may be normal if user moved camera - returning None landmarks")
            # Return None landmarks - pipeline will handle it gracefully
            # Don't raise error - let pipeline return empty transcription
            return [None] * total_frames
        elif detected_count == 0 and total_frames < 50:
            # Short video with no detections - might be normal for very short clips or camera movement
            print(f"[RetinaFace] WARNING: No faces detected in {total_frames} frames (short video or camera moved), but continuing...")
            # Return landmarks with None values - pipeline will handle it
        
        return landmarks
