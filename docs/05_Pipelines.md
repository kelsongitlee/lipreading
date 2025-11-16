# Pipeline Architecture

## Overview

Pipelines organize the inference process from video input to subtitle output.

**Main entry point**: `pipelines/pipeline.py:InferencePipeline`

---

## Core Components

### 1. InferencePipeline
**File**: `pipelines/pipeline.py`  
**Purpose**: High-level orchestration

**Key methods**:
- `forward()`: Basic transcription (no timestamps)
- `forward_with_alignment()`: Transcription + word timestamps ✅ **Your addition**

---

### 2. AVSR Model
**File**: `pipelines/model.py`  
**Purpose**: Model wrapper and inference logic

**Key methods**:
- `infer()`: Basic transcription
- `infer_with_alignment()`: Transcription + timestamps ✅ **Your addition**

---

### 3. AVSRDataLoader
**File**: `pipelines/data/data_module.py`  
**Purpose**: Video/audio loading and preprocessing

**Key methods**:
- `load_video()`: Reads video file
- `load_data()`: Processes video with landmarks

---

### 4. VideoTransform
**File**: `pipelines/data/transforms.py`  
**Purpose**: Frame downsampling based on speed_rate

**Key functionality**:
- Downsamples frames to match model FPS (25.0)
- Example: 60fps video → take every 2.4th frame → 25fps

---

### 5. Face Detectors
**Location**: `pipelines/detectors/`  
**Purpose**: Face detection and landmark extraction

**Options**:
- **MediaPipe**: CPU-based, faster
- **RetinaFace**: GPU-based, more accurate

---

## Pipeline Flow

### Basic Transcription (`forward()`)
```
Video File
  ↓
AVSRDataLoader.load_data()
  - Face detection
  - Mouth cropping
  - Frame downsampling
  ↓
AVSR.infer()
  - Encode → features
  - Beam search → transcription
  ↓
Text Output
```

### With Alignment (`forward_with_alignment()`) ✅ **Your addition**
```
Video File
  ↓
FPS Detection ✅ **Your addition**
  - Detect actual video FPS
  - Calculate speed_rate
  - Update VideoTransform
  ↓
AVSRDataLoader.load_data()
  - Face detection
  - Mouth cropping
  - Frame downsampling (with correct speed_rate)
  ↓
AVSR.infer_with_alignment() ✅ **Your addition**
  - Encode → features
  - Beam search → transcription + token IDs
  - CTC forced alignment → frame-to-token mapping
  - Word grouping → word timestamps
  ↓
{transcription, word_timestamps, frame_alignments}
```

---

## Your Customizations

### 1. FPS Detection
**File**: `pipelines/pipeline.py:forward_with_alignment()`  
**Lines**: 97-130

**What it does**:
```python
# Detect actual video FPS
cap = cv2.VideoCapture(video_path)
actual_fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate correct speed_rate
speed_rate = actual_fps / model_fps  # e.g., 60/25 = 2.4

# Update VideoTransform
self.dataloader.video_transform = VideoTransform(speed_rate=speed_rate)
```

**Why needed**: Prevents video from appearing sped up when FPS doesn't match model expectation.

---

### 2. Timestamp Generation
**File**: `pipelines/model.py:infer_with_alignment()`  
**Lines**: 71-289

**What it does**:
1. Encodes video → features
2. Beam search → transcription + token IDs
3. CTC forced alignment → frame-to-token mapping
4. Groups tokens by ▁ prefix → words
5. Converts frames to timestamps

**Key logic**:
```python
# CTC alignment
aligned_frames = self.model.ctc.forced_align(enc_feats, token_ids)

# Group tokens into words (SentencePiece format)
# Token with ▁ prefix = word start
words = []
current_word = []
for token_id, frame_idx in zip(tokens, aligned_frames):
    if token_has_word_prefix(token_id):
        if current_word:
            words.append(current_word)
        current_word = [frame_idx]
    else:
        current_word.append(frame_idx)

# Convert to timestamps
word_timestamps = []
for word_frames in words:
    start_frame = min(word_frames)
    end_frame = max(word_frames)
    word_timestamps.append({
        'word': token_to_word(word_tokens),
        'start': start_frame / video_fps,
        'end': end_frame / video_fps
    })
```

---

## Data Flow Details

### Video Loading
1. **File reading**: `torchvision.io.read_video()`
2. **Frame extraction**: All frames loaded into memory
3. **Format**: NumPy array (T, H, W, C)

### Face Detection
1. **Detector choice**: MediaPipe or RetinaFace
2. **Landmark extraction**: 68 facial landmarks
3. **Mouth region**: Landmarks 48-68 (mouth area)

### Mouth Cropping
1. **Affine transformation**: Aligns face to reference
2. **Crop**: Extracts 96×96 mouth region
3. **Output**: (T, 96, 96, 1) grayscale sequence

### Frame Downsampling
1. **Speed rate calculation**: `actual_fps / model_fps`
2. **Sampling**: Takes every Nth frame
3. **Output**: Reduced frame count matching model FPS

---

## Configuration

### Model Config
**File**: `configs/*.ini`

**Sections**:
- `[input]`: Modality, FPS
- `[model]`: Model paths, architecture
- `[decode]`: Beam search parameters

### Runtime Config
**Set in code**:
- `detector`: "mediapipe" or "retinaface"
- `face_track`: Enable face tracking
- `device`: "cuda:0" or "cpu"

---

## Error Handling

### Common Issues

1. **Video FPS mismatch**
   - **Symptom**: Video appears sped up
   - **Fix**: Your FPS detection handles this automatically

2. **Face detection failure**
   - **Symptom**: No landmarks detected
   - **Fix**: Check video quality, try different detector

3. **Frame alignment errors**
   - **Symptom**: Incorrect timestamps
   - **Fix**: Verify speed_rate calculation, check CTC output

---

## Related Concepts

- **Speed Rate**: Critical for correct frame downsampling
- **CTC**: Provides frame-to-token alignment
- **Beam Search**: Generates transcription
- **Processing Flow**: Complete end-to-end flow

---

## File Locations

| Component | File | Modified? |
|-----------|------|-----------|
| Main pipeline | `pipelines/pipeline.py` | ✅ Yes |
| Model wrapper | `pipelines/model.py` | ✅ Yes |
| Data loading | `pipelines/data/data_module.py` | ❌ No |
| Transforms | `pipelines/data/transforms.py` | ❌ No |
| Detectors | `pipelines/detectors/*/` | ❌ No |

