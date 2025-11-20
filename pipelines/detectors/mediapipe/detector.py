#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import warnings
import torchvision
import mediapipe as mp
import os
import cv2
import numpy as np


class LandmarksDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.short_range_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0)
        self.full_range_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1)

    def __call__(self, filename):
        print(f"[MediaPipe __call__] Processing video: {filename}")
        video_frames = torchvision.io.read_video(filename, pts_unit='sec')[0].numpy()
        print(f"[MediaPipe __call__] Loaded {len(video_frames)} frames")
        
        landmarks = self.detect(video_frames, self.full_range_detector)
        if all(element is None for element in landmarks):
            print(f"[MediaPipe __call__] Full range detector found no faces, trying short range...")
            landmarks = self.detect(video_frames, self.short_range_detector)
            
            # Check if we detected any faces at all
            detected_count = sum(1 for l in landmarks if l is not None)
            total_frames = len(landmarks)
            
            if detected_count == 0:
                # No faces detected - return list of None values (like RetinaFace does)
                print(f"[MediaPipe __call__] WARNING: No faces detected in {total_frames} frames, returning None list")
                return landmarks  # Return list of None values, don't raise error
            elif detected_count < total_frames * 0.3:
                # Less than 30% of frames have faces - likely poor quality
                print(f"[MediaPipe __call__] WARNING: Only {detected_count}/{total_frames} frames detected ({detected_count/total_frames*100:.1f}%)")
        
        print(f"[MediaPipe __call__] Returning landmarks list")
        return landmarks

    def detect(self, video_frames, detector):
        landmarks = []
        detected_frames = 0
        for frame_idx, frame in enumerate(video_frames):
            results = detector.process(frame)
            if not results.detections:
                landmarks.append(None)
                continue
            detected_frames += 1
            face_points = []
            for idx, detected_faces in enumerate(results.detections):
                max_id, max_size = 0, 0
                bboxC = detected_faces.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bbox_size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
                if bbox_size > max_size:
                    max_id, max_size = idx, bbox_size
                lmx = [
                    [int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(0).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(0).value].y * ih)],
                    [int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(1).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(1).value].y * ih)],
                    [int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(2).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(2).value].y * ih)],
                    [int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(3).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(3).value].y * ih)],
                    ]
                face_points.append(lmx)
            landmarks.append(np.array(face_points[max_id]))
        
        # Debug output
        print(f"[MediaPipe] Detected faces in {detected_frames}/{len(video_frames)} frames ({detected_frames/len(video_frames)*100:.1f}%)")
        
        return landmarks

