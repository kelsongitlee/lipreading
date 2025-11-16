# Complete Processing Flow

## End-to-End Pipeline

From video upload to subtitle file generation.

---

## Step-by-Step Flow

### 1. Video Upload
**Input**: Video file (MP4, AVI, etc.)  
**Location**: User uploads via web interface

---

### 2. Pipeline Initialization
**File**: `pipelines/pipeline.py:InferencePipeline.__init__()`

**Actions**:
- Load model config (`configs/*.ini`)
- Initialize face detector (MediaPipe/RetinaFace)
- Create AVSR model wrapper
- Set up data loader

**Output**: Initialized pipeline ready for inference

---

### 3. FPS Detection ✅ **Your addition**
**File**: `pipelines/pipeline.py:forward_with_alignment()`  
**Lines**: 97-130

**Actions**:
```python
# Detect actual video FPS
cap = cv2.VideoCapture(video_path)
actual_fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate speed_rate
speed_rate = actual_fps / 25.0

# Update VideoTransform
self.dataloader.video_transform = VideoTransform(speed_rate=speed_rate)
```

**Output**: Correct speed_rate for frame downsampling

---

### 4. Video Loading
**File**: `pipelines/data/data_module.py:AVSRDataLoader.load_data()`

**Actions**:
- Read video file: `torchvision.io.read_video()`
- Extract all frames: (T, H, W, C) array
- Load landmarks file (if provided)

**Output**: Raw video frames + landmarks

---

### 5. Face Detection & Cropping
**File**: `pipelines/detectors/*/detector.py`

**Actions**:
- Detect faces in each frame
- Extract 68 facial landmarks
- Crop mouth region (landmarks 48-68)
- Apply affine transformation for alignment
- Output: 96×96 grayscale mouth patches

**Output**: (T, 96, 96, 1) mouth sequence

---

### 6. Frame Downsampling
**File**: `pipelines/data/transforms.py:VideoTransform`

**Actions**:
- Apply speed_rate to sample frames
- Example: 60fps → take every 2.4th frame → 25fps
- Normalize pixel values

**Output**: Downsampled frames matching model FPS

---

### 7. Feature Encoding
**File**: `espnet/nets/pytorch_backend/e2e_asr_transformer.py:encode()`

**Actions**:
- Process frames through ResNet/Conv3D backbone
- Apply Transformer encoder layers
- Self-attention across frames

**Output**: Encoder features (T, D)
- T = number of frames (after downsampling)
- D = feature dimension (512-1024)

---

### 8. Beam Search Decoding
**File**: `espnet/nets/batch_beam_search.py:BatchBeamSearch`

**Actions**:
- Initialize with empty sequence
- For each step:
  - Generate token candidates
  - Score using CTC + decoder + LM
  - Keep top-K (beam_size) hypotheses
- Return best hypothesis

**Output**:
- Transcription text
- Token sequence: [token1, token2, ..., tokenN]
- Token IDs for alignment

---

### 9. CTC Forced Alignment ✅ **Your addition**
**File**: `espnet/nets/pytorch_backend/ctc.py:forced_align()`

**Actions**:
```python
# Input: enc_feats (T, D), token_ids (L,)
# Compute frame-level token probabilities
log_probs = log_softmax(enc_feats)  # (T, vocab_size)

# Use Viterbi algorithm to find best alignment
aligned_frames = viterbi_align(log_probs, token_ids)
```

**Output**: Frame-to-token mapping
- Example: [5, 5, 5, 12, 12, 8, 8, 8, 3, 7, 7]
- Each frame assigned to a token

---

### 10. Word Grouping ✅ **Your addition**
**File**: `pipelines/model.py:infer_with_alignment()`  
**Lines**: 187-244

**Actions**:
```python
# SentencePiece tokens have ▁ prefix for word boundaries
# Group consecutive tokens with same token ID
words = []
current_word = []
for token_id, frame_idx in zip(tokens, aligned_frames):
    if token_has_word_prefix(token_id):  # ▁ prefix
        if current_word:
            words.append(current_word)
        current_word = [frame_idx]
    else:
        current_word.append(frame_idx)
```

**Output**: Word-level frame groups
- Each word has list of frame indices

---

### 11. Timestamp Conversion ✅ **Your addition**
**File**: `pipelines/model.py:infer_with_alignment()`  
**Lines**: 200-244

**Actions**:
```python
word_timestamps = []
for word_frames in words:
    start_frame = min(word_frames)
    end_frame = max(word_frames)
    word_timestamps.append({
        'word': token_to_word(word_tokens),
        'start': start_frame / video_fps,  # Convert to seconds
        'end': end_frame / video_fps
    })
```

**Output**: Word timestamps
```python
[
    {'word': 'hello', 'start': 0.0, 'end': 0.4},
    {'word': 'world', 'start': 0.5, 'end': 0.9},
    ...
]
```

---

### 12. Subtitle Generation
**File**: `backend/app/api/transcription.py:generate_subtitle_file()`

**Actions**:
- Format word timestamps as VTT file
- Group words into subtitle lines (readability)
- Add timing information

**Output**: WebVTT subtitle file

---

## Data Transformations

### Dimensions at Each Stage

```
1. Video File
   → (T_raw, H, W, C)  # e.g., (180, 720, 1280, 3) at 60fps

2. After Face Detection
   → (T_raw, 96, 96, 1)  # Mouth crops

3. After Downsampling (speed_rate=2.4)
   → (T_down, 96, 96, 1)  # e.g., (75, 96, 96, 1) at 25fps

4. After Encoding
   → (T_down, D)  # e.g., (75, 512)

5. After Beam Search
   → Token sequence: [5, 12, 8, 3, 7]  # L tokens

6. After CTC Alignment
   → Frame-to-token: [5, 5, 5, 12, 12, 8, 8, 8, 3, 7, 7]  # T_down values

7. After Word Grouping
   → Word timestamps: [{word, start, end}, ...]
```

---

## Key Decision Points

### 1. Detector Choice
- **MediaPipe**: Faster, CPU-based
- **RetinaFace**: More accurate, GPU-based

### 2. Beam Size
- **35-60**: High accuracy, slower
- **5-10**: Fast, lower accuracy

### 3. Language Model
- **Enabled (lm_weight=0.4)**: Better grammar
- **Disabled (lm_weight=0.0)**: Faster

### 4. Speed Rate
- **Auto-detected**: ✅ Your implementation
- **Manual**: Can cause issues if wrong

---

## Performance Characteristics

### Time Breakdown (approximate)
- Video loading: 5-10%
- Face detection: 20-30%
- Encoding: 30-40%
- Beam search: 20-30%
- CTC alignment: 5-10%
- Word grouping: <1%

### Bottlenecks
1. **Face detection**: Sequential per-frame processing
2. **Encoding**: Transformer self-attention (sequential)
3. **Beam search**: Sequential token generation

### Optimization Opportunities
- ✅ GPU acceleration (already used)
- ✅ Batch processing (for multiple videos)
- ❌ Parallel frame processing (limited by architecture)

---

## Error Handling

### Common Failures

1. **Video FPS detection fails**
   - Fallback: Use provided video_fps parameter
   - Log warning

2. **Face detection fails**
   - Skip frame or use interpolation
   - May reduce accuracy

3. **CTC alignment fails**
   - Fallback: Simple word-level distribution
   - Timestamps less accurate

---

## Related Concepts

- **CTC**: Step 9 - Frame-to-token alignment
- **Beam Search**: Step 8 - Text generation
- **Transformer Encoder**: Step 7 - Feature extraction
- **Speed Rate**: Step 3 & 6 - FPS handling
- **Pipelines**: Overall architecture

---

## Debugging Tips

### Add Logging
```python
print(f"[FLOW] Step X: Input shape {input.shape}, Output shape {output.shape}")
```

### Check Intermediate Results
- Save encoder features: `np.save('enc_feats.npy', enc_feats)`
- Save aligned frames: `print(aligned_frames)`
- Visualize word timestamps: `print(word_timestamps)`

### Verify Dimensions
- Ensure T matches between encoding and alignment
- Check token sequence length matches word count

