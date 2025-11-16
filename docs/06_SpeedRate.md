# Speed Rate & FPS Handling

## What It Does

Speed rate controls frame downsampling to match the model's expected frame rate (25.0 fps).

**Formula**: `speed_rate = actual_video_fps / model_fps`

**Example**: 60fps video → `speed_rate = 60/25 = 2.4` → take every 2.4th frame

---

## Why We Need It

### Model Expectation
- Model was trained on 25.0 fps videos
- Input must match training FPS for accurate transcription
- Wrong FPS causes frame misalignment

### Problem Without It
- **60fps video with speed_rate=1.0**: Model sees 60 frames instead of 25
  - Video appears sped up
  - Frame-to-token alignment incorrect
  - Poor transcription accuracy

- **30fps video with speed_rate=1.0**: Model sees 30 frames instead of 25
  - Slight speedup
  - Minor alignment issues

---

## How It Works

### Frame Downsampling
```
Original: [frame0, frame1, frame2, frame3, frame4, ...]  (60fps)
Speed rate: 2.4
Downsampled: [frame0, frame2, frame5, frame7, ...]  (25fps)
```

**Algorithm** (in `VideoTransform`):
```python
for i in range(num_frames):
    target_frame = int(i * speed_rate)
    if target_frame < len(frames):
        sampled_frames.append(frames[target_frame])
```

---

## Your Implementation

### Automatic FPS Detection
**File**: `pipelines/pipeline.py:forward_with_alignment()`  
**Lines**: 97-130

**Code**:
```python
# Detect actual video FPS
import cv2
cap = cv2.VideoCapture(data_filename)
if cap.isOpened():
    detected_fps = cap.get(cv2.CAP_PROP_FPS)
    if detected_fps and detected_fps > 0:
        actual_video_fps = float(detected_fps)
        print(f"[PIPELINE] Detected video FPS: {actual_video_fps}")
cap.release()

# Get model FPS from config
from configparser import ConfigParser
config = ConfigParser()
config.read(self.config_filename)
model_v_fps = config.getfloat("model", "v_fps")  # 25.0

# Calculate correct speed_rate
correct_speed_rate = float(actual_video_fps / model_v_fps)

# Update VideoTransform if needed
if abs(correct_speed_rate - 1.0) > 0.01:
    from pipelines.data.transforms import VideoTransform
    self.dataloader.video_transform = VideoTransform(speed_rate=correct_speed_rate)
```

---

## Common FPS Values

| Video FPS | Speed Rate | Frames Sampled |
|-----------|------------|----------------|
| 25.0 | 1.0 | All frames |
| 30.0 | 1.2 | Every 1.2nd frame |
| 60.0 | 2.4 | Every 2.4th frame |
| 24.0 | 0.96 | Almost all frames |

---

## Impact on Processing

### Frame Count
- **Before**: 75 frames (3 seconds at 25fps)
- **After (60fps video)**: ~31 frames (3 seconds at 25fps after downsampling)

### Processing Time
- Fewer frames → faster encoding
- But downsampling adds overhead
- Net effect: Slightly faster for high FPS videos

### Accuracy
- **Correct speed_rate**: Proper alignment, good accuracy
- **Wrong speed_rate**: Misalignment, poor accuracy

---

## Where It's Used

### 1. VideoTransform
**File**: `pipelines/data/transforms.py`  
**Purpose**: Applies frame downsampling

### 2. Pipeline Initialization
**File**: `pipelines/pipeline.py:__init__()`  
**Purpose**: Sets initial speed_rate from config

### 3. Dynamic Adjustment
**File**: `pipelines/pipeline.py:forward_with_alignment()`  
**Purpose**: Updates speed_rate based on actual video FPS ✅ **Your addition**

---

## Debugging

### Check Logs
Look for: `[PIPELINE] Detected video FPS: X`

**Expected**:
- Should match video's actual FPS
- Speed rate should be calculated correctly

### Common Issues

1. **FPS detection fails**
   - **Symptom**: Uses default FPS, wrong speed_rate
   - **Fix**: Check video file format, ensure OpenCV can read it

2. **Speed rate not updated**
   - **Symptom**: VideoTransform still uses old speed_rate
   - **Fix**: Verify `forward_with_alignment()` is called, not `forward()`

3. **Config FPS mismatch**
   - **Symptom**: Model FPS in config doesn't match actual model
   - **Fix**: Check `configs/*.ini` → `[model]` → `v_fps` should be 25.0

---

## Important Notes

### One-Time Detection
- FPS is detected once per video
- Assumes constant FPS throughout video
- Variable FPS videos may have issues

### Model FPS
- Always 25.0 for AutoAVSR models
- Don't change this without retraining
- Checked from config file, not hardcoded

### Speed Rate Precision
- Float values are fine (e.g., 2.4)
- VideoTransform handles fractional frame indices
- Rounding happens during sampling

---

## Related Concepts

- **Pipelines**: Where FPS detection is implemented
- **VideoTransform**: Applies the downsampling
- **Processing Flow**: See how speed_rate affects the pipeline

---

## Further Reading

- **Video Processing**: Understanding frame rates and temporal sampling
- **OpenCV**: `cv2.VideoCapture.get(cv2.CAP_PROP_FPS)` documentation

