#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torchaudio
import torchvision


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class VideoTransform:
    def __init__(self, speed_rate):
        # CRITICAL FIX: Ensure we include ALL frames including the last one
        # Use ceil() to round up, ensuring we don't lose the last frame
        def resample_frames(x):
            if speed_rate == 1:
                return x
            num_frames = x.shape[0]
            target_frames = int(torch.ceil(torch.tensor(num_frames / speed_rate)).item())
            # CRITICAL: Always include the last frame (x.shape[0]-1)
            # Create indices that evenly sample from 0 to last_frame, including both endpoints
            if target_frames <= 1:
                return x[0:1]  # At least one frame
            # Use linspace that includes both start (0) and end (num_frames-1)
            indices = torch.linspace(0, num_frames - 1, target_frames, dtype=torch.int64)
            # Ensure last frame is always included
            indices[-1] = num_frames - 1
            return torch.index_select(x, dim=0, index=indices)
        
        self.video_pipeline = torch.nn.Sequential(
            FunctionalModule(lambda x: x.unsqueeze(-1)),
            FunctionalModule(resample_frames),
            FunctionalModule(lambda x: x.permute(3, 0, 1, 2)),
            FunctionalModule(lambda x: x / 255.),
            torchvision.transforms.CenterCrop(88),
            torchvision.transforms.Normalize(0.421, 0.165),
        )

    def __call__(self, sample):
        return self.video_pipeline(sample)


class AudioTransform:
    def __init__(self):
        self.audio_pipeline = torch.nn.Sequential(
            FunctionalModule(lambda x: torch.nn.functional.layer_norm(x, x.shape, eps=0)),
            FunctionalModule(lambda x: x.transpose(0, 1)),
        )

    def __call__(self, sample):
        return self.audio_pipeline(sample)
