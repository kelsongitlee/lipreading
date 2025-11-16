# Transformer Encoder

## What It Does

Transformer encoder converts video frames into feature vectors that represent the visual speech information.

**Input**: Video frames (sequence of images)  
**Output**: Feature tensor (T, D) where:
- T = time steps (number of frames after processing)
- D = feature dimension (typically 512-1024)

---

## Why We Need It

### Feature Extraction
- Raw video frames are too high-dimensional (e.g., 96×96×3 = 27,648 values per frame)
- Need compact representation that captures speech-relevant information
- Transformer encoder learns to extract lip movement patterns

### Downstream Processing
- Beam search and CTC work on features, not raw pixels
- Features encode semantic information (phonemes, words)
- Enables efficient decoding

---

## Architecture

### Pipeline
```
Video Frames (B, T, H, W, C)
  ↓
ResNet/Conv3D Backbone
  ↓
Transformer Encoder Layers
  ↓
Encoder Features (T, D)
```

### Components

#### 1. Backbone (ResNet/Conv3D)
**Purpose**: Initial feature extraction from frames  
**Output**: Lower-dimensional frame representations

#### 2. Transformer Encoder
**Purpose**: Temporal modeling with self-attention  
**Key mechanism**: Self-attention allows each frame to attend to all other frames

**Layers**:
- Multi-head self-attention
- Feed-forward network
- Layer normalization
- Residual connections

---

## Key Method: `encode()`

**Location**: `espnet/nets/pytorch_backend/e2e_asr_transformer.py`

**Signature**:
```python
def encode(self, x):
    """
    Args:
        x: Video frames (B, T, H, W, C) or processed features
    
    Returns:
        enc_feats: Encoder features (T, D)
    """
```

**What it does**:
1. Processes frames through backbone
2. Applies transformer encoder layers
3. Returns feature tensor ready for decoding

---

## Model Variants

### Video-Only Model
**File**: `espnet/nets/pytorch_backend/e2e_asr_transformer.py`  
**Input**: Video frames only  
**Use**: Visual speech recognition

### Audio-Visual Model
**File**: `espnet/nets/pytorch_backend/e2e_asr_transformer_av.py`  
**Input**: Video frames + audio  
**Use**: Combined audio-visual recognition (more accurate)

---

## Self-Attention Mechanism

### What It Does
Allows each frame to "look at" all other frames to understand context.

**Example**:
- Frame 10 (mouth opening) can attend to:
  - Frame 9 (preceding context)
  - Frame 11 (following context)
  - Frame 5 (distant context if relevant)

### Why Important
- Lip reading requires temporal context
- Phonemes span multiple frames
- Self-attention captures long-range dependencies

---

## Feature Dimensions

### Typical Values
- **T (time steps)**: 50-100 frames (depends on video length and downsampling)
- **D (feature dimension)**: 512 or 1024

### Example
For a 3-second video at 25fps:
- Input: 75 frames
- After speed_rate downsampling: ~30-40 frames
- Encoder output: (30-40, 512) tensor

---

## Usage in Your Code

**File**: `pipelines/model.py:infer_with_alignment()`

**Flow**:
```python
# 1. Load and preprocess video
data = self.dataloader.load_data(video_path, landmarks)

# 2. Encode video to features
enc_feats = self.model.encode(data)  # (T, D)

# 3. Use features for decoding
result = self.beam_search(enc_feats)

# 4. Use features for alignment
aligned_frames = self.model.ctc.forced_align(enc_feats, token_ids)
```

---

## Important Notes

### Sequential Processing
- Transformer processes frames sequentially (self-attention dependencies)
- Cannot parallelize across time steps
- GPU acceleration helps with matrix operations within each step

### Pre-trained Weights
- Model weights are pre-trained on large datasets (LRS3, etc.)
- Do not modify architecture without retraining
- Only load weights, never modify them

### Feature Quality
- Better features → better transcription
- Depends on:
  - Face detection quality
  - Mouth cropping accuracy
  - Frame downsampling (speed_rate)

---

## Related Concepts

- **Beam Search**: Decodes encoder features to text
- **CTC**: Aligns encoder features to tokens
- **Speed Rate**: Affects number of frames (T) fed to encoder
- **Processing Flow**: See how encoder fits into the pipeline

---

## Further Reading

- **Paper**: "Attention Is All You Need" (Vaswani et al., 2017)
- **Focus**: Understand self-attention mechanism and encoder architecture

