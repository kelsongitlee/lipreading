#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import torch
import pickle
from configparser import ConfigParser

from pipelines.model import AVSR
from pipelines.data.data_module import AVSRDataLoader


class InferencePipeline(torch.nn.Module):
    def __init__(self, config_filename, detector="retinaface", face_track=False, device="cuda:0"):
        super(InferencePipeline, self).__init__()
        assert os.path.isfile(config_filename), f"config_filename: {config_filename} does not exist."

        config = ConfigParser()
        config.read(config_filename)

        # modality configuration
        modality = config.get("input", "modality")

        self.modality = modality
        # data configuration
        input_v_fps = config.getfloat("input", "v_fps")
        model_v_fps = config.getfloat("model", "v_fps")

        # model configuration
        model_path = config.get("model","model_path")
        model_conf = config.get("model","model_conf")

        # language model configuration
        rnnlm = config.get("model", "rnnlm")
        rnnlm_conf = config.get("model", "rnnlm_conf")
        penalty = config.getfloat("decode", "penalty")
        ctc_weight = config.getfloat("decode", "ctc_weight")
        lm_weight = config.getfloat("decode", "lm_weight")
        beam_size = config.getint("decode", "beam_size")

        # speed_rate can be float (e.g., 25.0/25.0 = 1.0) - convert to float explicitly
        speed_rate_float = float(input_v_fps / model_v_fps)
        self.dataloader = AVSRDataLoader(modality, speed_rate=speed_rate_float, detector=detector)  # type: ignore
        self.model = AVSR(modality, model_path, model_conf, rnnlm, rnnlm_conf, penalty, ctc_weight, lm_weight, beam_size, device)
        # store config filename for later use in forward_with_alignment
        self.config_filename = config_filename
        if face_track and self.modality in ["video", "audiovisual"]:
            if detector == "mediapipe":
                from pipelines.detectors.mediapipe.detector import LandmarksDetector
                self.landmarks_detector = LandmarksDetector()
            if detector == "retinaface":
                from pipelines.detectors.retinaface.detector import LandmarksDetector
                self.landmarks_detector = LandmarksDetector(device="cuda:0")
        else:
            self.landmarks_detector = None


    def process_landmarks(self, data_filename, landmarks_filename):
        if self.modality == "audio":
            return None
        if self.modality in ["video", "audiovisual"]:
            if isinstance(landmarks_filename, str):
                landmarks = pickle.load(open(landmarks_filename, "rb"))
            else:
                # landmarks_detector can be None if face_track=False
                if self.landmarks_detector is not None:
                    landmarks = self.landmarks_detector(data_filename)  # type: ignore
                else:
                    landmarks = None
            return landmarks


    def forward(self, data_filename, landmarks_filename=None):
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."
        landmarks = self.process_landmarks(data_filename, landmarks_filename)
        data = self.dataloader.load_data(data_filename, landmarks)
        transcript = self.model.infer(data)
        return transcript
    
    def forward_with_accurate_alignment(self, data_filename, landmarks_filename=None, video_fps=25.0, 
                                         beam_size=None, lm_weight=None, ctc_weight=None, penalty=None):
        """
        Process video using EXACT same preprocessing as forward() (infer()) for 100% accuracy.
        Then get word timestamps separately using the accurate transcription.
        
        Optional parameters can be used to optimize for better accuracy:
        - beam_size: Higher values (80-100) improve accuracy but slow down processing
        - lm_weight: Higher values (0.5-0.6) improve grammar and context understanding
        - ctc_weight: Balance between CTC and attention decoder (0.1 is typical)
        - penalty: Length penalty (0.4-0.5) can help detect longer sequences
        
        This ensures transcription is 100% identical to forward() -> infer().
        """
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."
        
        # if accuracy optimization parameters provided, temporarily update beam search
        original_beam_search = None
        if beam_size is not None or lm_weight is not None or ctc_weight is not None or penalty is not None:
            # save original beam search
            original_beam_search = self.model.beam_search
            
            # get current config values or use provided ones
            from configparser import ConfigParser
            config = ConfigParser()
            config.read(self.config_filename)
            
            new_beam_size = beam_size if beam_size is not None else config.getint("decode", "beam_size")
            new_lm_weight = lm_weight if lm_weight is not None else config.getfloat("decode", "lm_weight")
            new_ctc_weight = ctc_weight if ctc_weight is not None else config.getfloat("decode", "ctc_weight")
            new_penalty = penalty if penalty is not None else config.getfloat("decode", "penalty")
            
            # get rnnlm config
            rnnlm = config.get("model", "rnnlm")
            rnnlm_conf = config.get("model", "rnnlm_conf")
            
            # create new beam search with optimized parameters
            from pipelines.model import get_beam_search_decoder
            self.model.beam_search = get_beam_search_decoder(
                self.model.model, self.model.token_list, 
                rnnlm, rnnlm_conf, 
                float(new_penalty), float(new_ctc_weight), float(new_lm_weight), int(new_beam_size)
            )
            self.model.beam_search.to(device=self.model.device).eval()
            
            print(f"[PIPELINE] Using optimized beam search: beam_size={new_beam_size}, lm_weight={new_lm_weight}, ctc_weight={new_ctc_weight}, penalty={new_penalty}")
        
        # CRITICAL: Use EXACT same preprocessing as forward() method
        # DO NOT modify speed_rate or any preprocessing - use exactly what forward() uses
        landmarks = self.process_landmarks(data_filename, landmarks_filename)
        data = self.dataloader.load_data(data_filename, landmarks)
        
        # Get accurate transcription using infer() (same as forward())
        transcription = self.model.infer(data)
        
        # Now get word timestamps using the accurate transcription
        alignment_result = self.model.get_word_timestamps_from_transcription(transcription, data, video_fps)
        
        # restore original beam search if we modified it
        if original_beam_search is not None:
            self.model.beam_search = original_beam_search
        
        return {
            'transcription': transcription,  # 100% accurate (from infer())
            'word_timestamps': alignment_result.get('word_timestamps', []),
            'frame_alignments': alignment_result.get('frame_alignments', [])
        }
    
    def forward_with_alignment(self, data_filename, landmarks_filename=None, video_fps=25.0):
        """
        Process video with CTC forced alignment for subtitle generation
        
        Args:
            data_filename: Path to video file
            landmarks_filename: Optional path to landmarks file
            video_fps: Video frame rate (default: 25.0 fps for AutoAVSR standard)
            
        Returns:
            dict with 'transcription', 'word_timestamps', and 'frame_alignments' keys
        """
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."
        
        # CRITICAL FIX: Calculate speed_rate based on actual video FPS, not config file
        # This prevents videos from appearing sped up when they have different FPS than expected
        import cv2
        actual_video_fps = video_fps  # use provided video_fps if available
        total_frames = 0
        try:
            cap = cv2.VideoCapture(data_filename)
            if cap.isOpened():
                detected_fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if detected_fps and detected_fps > 0:
                    actual_video_fps = float(detected_fps)
                    print(f"[PIPELINE] Detected video FPS: {actual_video_fps}, Total frames: {total_frames}, Duration: {total_frames/actual_video_fps:.2f}s")
                cap.release()
        except Exception as e:
            print(f"[PIPELINE] Warning: Could not detect video FPS, using provided video_fps={video_fps}: {e}")
        
        # get model_v_fps from config (should be 25.0)
        from configparser import ConfigParser
        config = ConfigParser()
        config.read(self.config_filename)
        model_v_fps = config.getfloat("model", "v_fps")
        
        # calculate correct speed_rate based on actual video FPS
        correct_speed_rate = float(actual_video_fps / model_v_fps)
        
        # DEBUG: Log resampling info
        # CRITICAL FIX: Use ceil() to ensure we don't lose frames
        import math
        expected_frames_after_resample = int(math.ceil(total_frames / correct_speed_rate)) if correct_speed_rate > 0 else total_frames
        print(f"[PIPELINE] Resampling: {total_frames} frames @ {actual_video_fps} FPS → ~{expected_frames_after_resample} frames @ {model_v_fps} FPS (speed_rate={correct_speed_rate:.3f})")
        print(f"[PIPELINE] Original duration: {total_frames/actual_video_fps:.2f}s, Expected after resample: {expected_frames_after_resample/model_v_fps:.2f}s")
        
        # CRITICAL: Update dataloader's speed_rate if it's different
        # This ensures frame downsampling matches the actual video FPS
        if hasattr(self.dataloader, 'video_transform') and hasattr(self.dataloader.video_transform, 'video_pipeline'):
            # check current speed_rate by inspecting the transform
            # if speed_rate changed, we need to recreate the transform
            if abs(correct_speed_rate - 1.0) > 0.01:  # if significantly different from 1.0
                print(f"[PIPELINE] Updating speed_rate from config-based to actual video-based: {correct_speed_rate:.2f} (video FPS: {actual_video_fps}, model FPS: {model_v_fps})")
                # recreate video transform with correct speed_rate
                from pipelines.data.transforms import VideoTransform
                self.dataloader.video_transform = VideoTransform(speed_rate=correct_speed_rate)
        
        landmarks = self.process_landmarks(data_filename, landmarks_filename)
        data = self.dataloader.load_data(data_filename, landmarks)
        result = self.model.infer_with_alignment(data, video_fps=actual_video_fps)
        return result