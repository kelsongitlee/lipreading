# Project Overview

## Three-Layer Architecture

### Layer 1: Official Libraries
**What**: PyTorch, ESPnet core  
**Purpose**: Base deep learning framework  
**Status**: No modifications needed

### Layer 2: Original Repository
**Source**: [mpc001/Visual_Speech_Recognition_for_Multiple_Languages](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages)  
**What**: Pre-trained models, inference pipeline, custom ESPnet modifications  
**Purpose**: Visual speech recognition foundation  
**Status**: Mostly unchanged, some modifications in CTC

### Layer 3: Your Customizations
**What**: Subtitle generation, dynamic FPS detection, real-time optimizations  
**Purpose**: Add timestamp generation and improve video handling  
**Status**: Your code additions

---

## Core Components

### 1. Video Processing
- Face detection (MediaPipe/RetinaFace)
- Mouth region cropping
- Frame downsampling (speed_rate)

### 2. Feature Extraction
- Transformer encoder converts frames → features
- Output: (T, D) tensor where T=time steps, D=feature dimension

### 3. Decoding
- Beam search generates text from features
- CTC provides frame-to-token alignment

### 4. Timestamp Generation
- CTC forced alignment maps frames to tokens
- Token grouping creates word boundaries
- Frame indices converted to timestamps

---

## Key Files

| File | Purpose | Modified? |
|------|---------|-----------|
| `pipelines/pipeline.py` | Main inference orchestration | ✅ Yes (FPS detection) |
| `pipelines/model.py` | Model wrapper, timestamp generation | ✅ Yes (alignment logic) |
| `espnet/nets/pytorch_backend/ctc.py` | CTC alignment implementation | ✅ Yes (2D/3D handling) |
| `espnet/nets/beam_search.py` | Beam search decoding | ❌ No |
| `espnet/nets/pytorch_backend/e2e_asr_transformer.py` | Video encoder | ❌ No |

---

## Data Flow

```
Video File
  ↓
Face Detection & Cropping
  ↓
Frame Downsampling (speed_rate)
  ↓
Transformer Encoder → Features (T, D)
  ↓
Beam Search → Transcription + Token IDs
  ↓
CTC Forced Alignment → Frame-to-Token Mapping
  ↓
Word Grouping → Word Timestamps
  ↓
Subtitle File (VTT)
```

---

## Next Steps

- Understand [CTC](02_CTC.md) for alignment
- Learn [Beam Search](03_BeamSearch.md) for decoding
- See [Processing Flow](07_ProcessingFlow.md) for complete pipeline

