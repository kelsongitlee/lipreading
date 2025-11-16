# Key Parameters

## Overview

Parameters that control accuracy, speed, and behavior of the lip reading system.

---

## Decoding Parameters

### `beam_size`
**What**: Number of hypotheses maintained during beam search  
**Type**: Integer  
**Range**: 5-60  
**Default**: 35

**Impact**:
- **Higher (35-60)**: More accurate, slower
  - Explores more transcription possibilities
  - Better for video upload, practice mode
  - 2-3x slower than lower values
- **Lower (5-10)**: Faster, less accurate
  - Fewer paths explored
  - Better for real-time transcription
  - 5-10% accuracy drop

**Location**: `configs/*.ini` → `[decode]` → `beam_size`

**When to adjust**:
- Increase for better accuracy (non-real-time)
- Decrease for faster processing (real-time)

---

### `ctc_weight`
**What**: Balance between CTC and decoder scores  
**Type**: Float  
**Range**: 0.0 to 1.0  
**Default**: 0.1

**Impact**:
- **0.1** (current): Mostly decoder, CTC for alignment
- **1.0**: Pure CTC (no decoder)
- **0.0**: Pure decoder (no CTC alignment)

**Location**: `configs/*.ini` → `[decode]` → `ctc_weight`

**When to adjust**:
- Usually keep at 0.1
- Increase if decoder produces poor results
- Decrease if CTC alignment is problematic

---

### `lm_weight`
**What**: Language model influence on scoring  
**Type**: Float  
**Range**: 0.0 to 1.0  
**Default**: 0.4 (with LM), 0.0 (real-time)

**Impact**:
- **0.4** (current): Strong LM guidance
  - Better grammar, context-aware
  - Requires LM file to be loaded
  - Adds processing overhead
- **0.0**: No LM (CTC + decoder only)
  - Faster, less grammar-aware
  - Used for real-time mode

**Location**: `configs/*.ini` → `[decode]` → `lm_weight`

**When to adjust**:
- Use 0.4 when LM is available (better quality)
- Use 0.0 for maximum speed (real-time)

---

### `penalty`
**What**: Length penalty to prevent short outputs  
**Type**: Float  
**Range**: 0.0 to 1.0  
**Default**: 0.3

**Impact**:
- **0.3** (current): Moderate penalty
  - Prefers longer, more complete transcriptions
- **0.0**: No penalty
  - May produce shorter outputs
- **Higher**: Stronger preference for longer outputs

**Location**: `configs/*.ini` → `[decode]` → `penalty`

**When to adjust**:
- Usually keep at 0.3
- Increase if outputs are too short
- Decrease if outputs are too verbose

---

## Video Processing Parameters

### `speed_rate`
**What**: Frame downsampling ratio  
**Type**: Float  
**Formula**: `actual_video_fps / model_fps`  
**Default**: Auto-detected ✅ **Your addition**

**Impact**:
- **Correct value**: Proper frame alignment, good accuracy
- **Wrong value**: Video sped up, misalignment, poor accuracy

**Location**: 
- Calculated in `pipelines/pipeline.py:forward_with_alignment()`
- Applied in `pipelines/data/transforms.py:VideoTransform`

**When to adjust**:
- Automatically handled by FPS detection
- Manual override only if detection fails

---

### `v_fps` (Model FPS)
**What**: Expected video frame rate for model  
**Type**: Float  
**Default**: 25.0

**Impact**:
- Model was trained on 25.0 fps videos
- Input must match for accurate transcription

**Location**: `configs/*.ini` → `[model]` → `v_fps`

**When to adjust**:
- **Never** (unless using different model)
- Changing requires model retraining

---

## Face Detection Parameters

### `detector`
**What**: Face detection algorithm  
**Type**: String  
**Options**: "mediapipe" or "retinaface"  
**Default**: "retinaface"

**Impact**:
- **MediaPipe**: Faster, CPU-based, less accurate
- **RetinaFace**: Slower, GPU-based, more accurate

**Location**: Set in `pipelines/pipeline.py:InferencePipeline.__init__()`

**When to adjust**:
- Use MediaPipe for speed (CPU-only systems)
- Use RetinaFace for accuracy (GPU available)

---

### `face_track`
**What**: Enable face tracking  
**Type**: Boolean  
**Default**: False

**Impact**:
- **True**: Tracks face across frames (smoother)
- **False**: Detects face independently per frame

**Location**: Set in `pipelines/pipeline.py:InferencePipeline.__init__()`

**When to adjust**:
- Enable for videos with face movement
- Disable for static faces (faster)

---

## Model Parameters

### `modality`
**What**: Input type  
**Type**: String  
**Options**: "video", "audio", "audiovisual"  
**Default**: "video"

**Impact**:
- **video**: Visual-only (lip reading)
- **audio**: Audio-only (not used in this project)
- **audiovisual**: Combined (more accurate, requires audio)

**Location**: `configs/*.ini` → `[input]` → `modality`

**When to adjust**:
- Use "video" for visual-only
- Use "audiovisual" if audio is available

---

## Device Parameters

### `device`
**What**: Computation device  
**Type**: String  
**Options**: "cuda:0", "cpu"  
**Default**: "cuda:0"

**Impact**:
- **cuda:0**: GPU acceleration (10-50x faster)
- **cpu**: CPU-only (slower, no GPU required)

**Location**: Set in `pipelines/pipeline.py:InferencePipeline.__init__()`

**When to adjust**:
- Use GPU if available (recommended)
- Use CPU if no GPU (much slower)

---

## Parameter Interaction

### Accuracy vs Speed Trade-off

**High Accuracy**:
- `beam_size=60`
- `lm_weight=0.4`
- `detector="retinaface"`
- **Result**: Best quality, slowest

**Balanced**:
- `beam_size=35`
- `lm_weight=0.4`
- `detector="retinaface"`
- **Result**: Good quality, reasonable speed (current default)

**High Speed**:
- `beam_size=5`
- `lm_weight=0.0`
- `detector="mediapipe"`
- **Result**: Fastest, lower accuracy

---

## Configuration Files

### Location
`configs/*.ini`

### Format
```ini
[input]
modality = video
v_fps = 25.0

[model]
model_path = benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth
model_conf = benchmarks/LRS3/models/LRS3_V_WER19.1/model.json
v_fps = 25.0
rnnlm = benchmarks/LRS3/language_models/lm_en_subword/model.pth
rnnlm_conf = benchmarks/LRS3/language_models/lm_en_subword/model.json

[decode]
beam_size = 35
ctc_weight = 0.1
lm_weight = 0.4
penalty = 0.3
```

---

## Dynamic Parameter Adjustment

### Runtime Changes
Some parameters can be adjusted without restarting:

- ✅ `beam_size`: Can be changed per video
- ✅ `lm_weight`: Can be disabled for speed
- ❌ `detector`: Requires pipeline reinitialization
- ❌ `device`: Requires model reload

### Your Implementation
FPS detection automatically adjusts `speed_rate` per video ✅

---

## Recommended Settings

### Video Upload (Practice Mode)
```ini
beam_size = 35
lm_weight = 0.4
detector = retinaface
```

### Real-Time Transcription
```ini
beam_size = 5
lm_weight = 0.0
detector = mediapipe
```

### Maximum Accuracy
```ini
beam_size = 60
lm_weight = 0.4
detector = retinaface
```

---

## Related Concepts

- **Beam Search**: Uses `beam_size`, `ctc_weight`, `lm_weight`, `penalty`
- **Speed Rate**: Affected by `v_fps` and actual video FPS
- **Processing Flow**: See how parameters affect the pipeline

