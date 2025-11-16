# Complete List of Adjustable Parameters

## Overview

This document lists **ALL** parameters that can be adjusted to affect **speed** and **accuracy** in this project, organized by configuration file sections and code locations.

---

## Configuration File Parameters (`configs/*.ini`)

### Section: `[input]`

#### `modality`
- **Type**: String
- **Options**: `"video"`, `"audio"`, `"audiovisual"`
- **Default**: `"video"`
- **Location**: `configs/*.ini` → `[input]` → `modality`
- **Code Location**: `pipelines/pipeline.py:25`

**Impact on Accuracy**:
- `"video"`: Visual-only (standard lip reading)
- `"audiovisual"`: More accurate (requires audio track)
- `"audio"`: Not used in this project

**Impact on Speed**:
- `"video"`: Standard speed
- `"audiovisual"`: Slightly slower (processes both video and audio)

**When to Adjust**:
- Use `"audiovisual"` if video has audio track (better accuracy)
- Use `"video"` for visual-only videos

---

#### `v_fps`
- **Type**: Float
- **Default**: `25.0`
- **Location**: `configs/*.ini` → `[input]` → `v_fps`
- **Code Location**: `pipelines/pipeline.py:29`

**Impact on Accuracy**:
- Must match actual video FPS for correct frame alignment
- Wrong value causes misalignment and poor accuracy

**Impact on Speed**:
- No direct impact (used for speed_rate calculation)

**When to Adjust**:
- Usually keep at 25.0
- Only change if you know your videos are consistently different FPS

**Note**: Your code auto-detects actual FPS in `forward_with_alignment()`, so this is less critical now.

---

### Section: `[model]`

#### `model_path`
- **Type**: String (file path)
- **Example**: `"benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth"`
- **Location**: `configs/*.ini` → `[model]` → `model_path`
- **Code Location**: `pipelines/pipeline.py:33`

**Impact on Accuracy**:
- Different models have different accuracy
- LRS3_V_WER19.1: 19.1% WER (Word Error Rate)
- LRS3_V_WER32.3: 32.3% WER (less accurate)
- GRID_V_WER1.2: 1.2% WER (very accurate, but smaller dataset)

**Impact on Speed**:
- Minimal (all models similar size)
- Loading time may vary slightly

**When to Adjust**:
- Use higher accuracy model for better results
- Use model trained on your target dataset if available

---

#### `model_conf`
- **Type**: String (file path)
- **Example**: `"benchmarks/LRS3/models/LRS3_V_WER19.1/model.json"`
- **Location**: `configs/*.ini` → `[model]` → `model_conf`
- **Code Location**: `pipelines/pipeline.py:34`

**Impact**: Defines model architecture (usually don't change)

---

#### `v_fps` (Model FPS)
- **Type**: Float
- **Default**: `25.0`
- **Location**: `configs/*.ini` → `[model]` → `v_fps`
- **Code Location**: `pipelines/pipeline.py:30`

**Impact on Accuracy**:
- Model was trained on this FPS
- Must match for accurate transcription
- **DO NOT CHANGE** unless using different model

**Impact on Speed**:
- Used to calculate speed_rate (affects frame downsampling)

**When to Adjust**:
- **NEVER** (unless using different model)
- Changing requires model retraining

---

#### `rnnlm` (Language Model Path)
- **Type**: String (file path) or empty
- **Example**: `"benchmarks/LRS3/language_models/lm_en_subword/model.pth"`
- **Location**: `configs/*.ini` → `[model]` → `rnnlm`
- **Code Location**: `pipelines/pipeline.py:37`

**Impact on Accuracy**:
- **With LM**: Better grammar, context-aware, 5-10% accuracy improvement
- **Without LM**: Less grammar-aware, faster

**Impact on Speed**:
- **With LM**: Slower (adds 10-20% processing time)
- **Without LM**: Faster (no LM loading/scoring overhead)

**When to Adjust**:
- Set to empty string `""` to disable LM (faster, less accurate)
- Set to LM path to enable (slower, more accurate)

---

#### `rnnlm_conf` (Language Model Config)
- **Type**: String (file path) or empty
- **Example**: `"benchmarks/LRS3/language_models/lm_en_subword/model.json"`
- **Location**: `configs/*.ini` → `[model]` → `rnnlm_conf`
- **Code Location**: `pipelines/pipeline.py:38`

**Impact**: Defines LM architecture (must match rnnlm)

---

### Section: `[decode]`

#### `beam_size` ⭐ **MOST IMPORTANT FOR SPEED/ACCURACY**
- **Type**: Integer
- **Range**: `5` to `60`
- **Default**: `35`
- **Location**: `configs/*.ini` → `[decode]` → `beam_size`
- **Code Location**: `pipelines/pipeline.py:42`, `pipelines/model.py:404`

**Impact on Accuracy**:
- **Higher (35-60)**: More accurate
  - Explores more transcription possibilities
  - 5-10% accuracy improvement over lower values
- **Lower (5-10)**: Less accurate
  - Fewer paths explored
  - 5-10% accuracy drop

**Impact on Speed**:
- **Higher (35-60)**: 2-3x slower
- **Lower (5-10)**: 2-3x faster
- **Linear relationship**: beam_size=60 is ~2x slower than beam_size=30

**Recommended Values**:
- **Maximum Accuracy**: `60`
- **Balanced (Default)**: `35`
- **Real-Time**: `5-10`
- **Fast Processing**: `10-20`

**When to Adjust**:
- Increase for better accuracy (non-real-time)
- Decrease for faster processing (real-time)

---

#### `ctc_weight` ⭐ **AFFECTS ACCURACY**
- **Type**: Float
- **Range**: `0.0` to `1.0`
- **Default**: `0.1`
- **Location**: `configs/*.ini` → `[decode]` → `ctc_weight`
- **Code Location**: `pipelines/pipeline.py:40`, `pipelines/model.py:404`

**Impact on Accuracy**:
- **0.1** (default): Balanced (mostly decoder, CTC for alignment)
- **0.0**: Pure decoder (no CTC alignment, may affect timestamps)
- **1.0**: Pure CTC (no decoder, usually less accurate)
- **0.2-0.3**: More CTC influence (may help with alignment)

**Impact on Speed**:
- Minimal (both CTC and decoder run anyway)
- Slight overhead if ctc_weight=0.0 (skips some CTC calculations)

**Recommended Values**:
- **Standard**: `0.1` (keep default)
- **If alignment issues**: `0.2-0.3`
- **If decoder issues**: `0.0` (CTC only, not recommended)

**When to Adjust**:
- Usually keep at 0.1
- Increase if decoder produces poor results
- Decrease if CTC alignment is problematic

---

#### `lm_weight` ⭐ **AFFECTS ACCURACY AND SPEED**
- **Type**: Float
- **Range**: `0.0` to `1.0`
- **Default**: `0.4` (with LM), `0.0` (without LM)
- **Location**: `configs/*.ini` → `[decode]` → `lm_weight`
- **Code Location**: `pipelines/pipeline.py:41`, `pipelines/model.py:404`

**Impact on Accuracy**:
- **0.4** (with LM): Better grammar, context-aware, 5-10% accuracy improvement
- **0.0** (no LM): Less grammar-aware, faster
- **0.6-0.8**: Very strong LM influence (may over-correct)

**Impact on Speed**:
- **0.4**: Slower (adds 10-20% processing time)
- **0.0**: Faster (no LM scoring overhead)

**Recommended Values**:
- **Maximum Accuracy**: `0.4-0.6` (with LM)
- **Balanced**: `0.4` (with LM)
- **Real-Time**: `0.0` (no LM)
- **Fast Processing**: `0.0-0.2`

**When to Adjust**:
- Use 0.4 when LM is available (better quality)
- Use 0.0 for maximum speed (real-time)
- Increase to 0.6 if grammar is important

---

#### `penalty` ⭐ **AFFECTS OUTPUT LENGTH**
- **Type**: Float
- **Range**: `0.0` to `1.0`
- **Default**: `0.3`
- **Location**: `configs/*.ini` → `[decode]` → `penalty`
- **Code Location**: `pipelines/pipeline.py:39`, `pipelines/model.py:404`

**Impact on Accuracy**:
- **0.3** (default): Prefers longer, more complete transcriptions
- **0.0**: No penalty (may produce shorter outputs)
- **0.5-0.7**: Strong preference for longer outputs (may be verbose)

**Impact on Speed**:
- Minimal (just affects scoring, not computation)

**Recommended Values**:
- **Standard**: `0.3` (keep default)
- **If outputs too short**: `0.4-0.5`
- **If outputs too verbose**: `0.1-0.2`

**When to Adjust**:
- Usually keep at 0.3
- Increase if outputs are too short
- Decrease if outputs are too verbose

---

## Code-Level Parameters (Set in Python Code)

### `detector` ⭐ **AFFECTS SPEED AND ACCURACY**
- **Type**: String
- **Options**: `"mediapipe"`, `"retinaface"`
- **Default**: `"retinaface"`
- **Location**: `pipelines/pipeline.py:17` (InferencePipeline.__init__)

**Impact on Accuracy**:
- **"retinaface"**: More accurate face detection (GPU-based)
- **"mediapipe"**: Less accurate (CPU-based, faster)

**Impact on Speed**:
- **"retinaface"**: Slower (GPU-based, 20-30% of total time)
- **"mediapipe"**: Faster (CPU-based, 10-15% of total time)
- **Speed difference**: MediaPipe is ~2x faster for face detection

**Recommended Values**:
- **Maximum Accuracy**: `"retinaface"` (GPU available)
- **Balanced**: `"retinaface"` (GPU available)
- **Real-Time**: `"mediapipe"` (faster)
- **CPU-Only Systems**: `"mediapipe"` (only option)

**When to Adjust**:
- Use MediaPipe for speed (CPU-only systems)
- Use RetinaFace for accuracy (GPU available)

---

### `face_track`
- **Type**: Boolean
- **Options**: `True`, `False`
- **Default**: `False`
- **Location**: `pipelines/pipeline.py:17` (InferencePipeline.__init__)

**Impact on Accuracy**:
- **True**: Smoother face tracking across frames (better for moving faces)
- **False**: Independent detection per frame (standard)

**Impact on Speed**:
- **True**: Slightly slower (tracking overhead)
- **False**: Faster (standard)

**Recommended Values**:
- **Standard**: `False`
- **Moving Faces**: `True`

**When to Adjust**:
- Enable for videos with significant face movement
- Disable for static faces (faster)

---

### `device` ⭐ **AFFECTS SPEED**
- **Type**: String
- **Options**: `"cuda:0"`, `"cpu"`, `"cuda:1"` (if multiple GPUs)
- **Default**: `"cuda:0"`
- **Location**: `pipelines/pipeline.py:17` (InferencePipeline.__init__)

**Impact on Accuracy**:
- No impact (same model, different hardware)

**Impact on Speed**:
- **"cuda:0"**: GPU acceleration (10-50x faster)
- **"cpu"**: CPU-only (much slower, 10-50x slower)
- **Speed difference**: GPU is essential for reasonable speed

**Recommended Values**:
- **Standard**: `"cuda:0"` (GPU available)
- **CPU-Only**: `"cpu"` (no GPU, very slow)

**When to Adjust**:
- Use GPU if available (recommended)
- Use CPU if no GPU (much slower)

---

### `video_fps` (Runtime Parameter)
- **Type**: Float
- **Default**: `25.0`
- **Location**: `pipelines/pipeline.py:83` (forward_with_alignment parameter)

**Impact on Accuracy**:
- Must match actual video FPS for correct timestamps
- Wrong value causes incorrect word timestamps

**Impact on Speed**:
- No direct impact (used for timestamp calculation)

**When to Adjust**:
- Usually auto-detected (your code handles this)
- Manual override only if detection fails

**Note**: Your code auto-detects this in `forward_with_alignment()`, so usually not needed.

---

## Derived Parameters (Calculated, Not Directly Set)

### `speed_rate` ⭐ **AFFECTS ACCURACY**
- **Type**: Float
- **Formula**: `actual_video_fps / model_v_fps`
- **Location**: Calculated in `pipelines/pipeline.py:119`, applied in `pipelines/data/transforms.py`

**Impact on Accuracy**:
- **Correct value**: Proper frame alignment, good accuracy
- **Wrong value**: Video sped up, misalignment, poor accuracy
- **Critical**: Wrong speed_rate can cause 20-30% accuracy drop

**Impact on Speed**:
- No direct impact (affects frame count, not processing speed per frame)

**When to Adjust**:
- **Automatically handled** by your FPS detection code
- Manual override only if detection fails

**Note**: Your code automatically calculates this in `forward_with_alignment()`.

---

## Parameter Interaction Matrix

### Speed vs Accuracy Trade-offs

| Parameter | Higher Value | Lower Value |
|-----------|--------------|-------------|
| `beam_size` | More accurate, slower | Less accurate, faster |
| `lm_weight` | More accurate, slower | Less accurate, faster |
| `detector` | More accurate (retinaface), slower | Less accurate (mediapipe), faster |
| `ctc_weight` | More CTC influence | More decoder influence |
| `penalty` | Longer outputs | Shorter outputs |

---

## Recommended Configurations

### Configuration 1: Maximum Accuracy
```ini
[input]
modality = audiovisual  # if audio available, else "video"
v_fps = 25.0

[model]
model_path = benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth
model_conf = benchmarks/LRS3/models/LRS3_V_WER19.1/model.json
v_fps = 25.0
rnnlm = benchmarks/LRS3/language_models/lm_en_subword/model.pth
rnnlm_conf = benchmarks/LRS3/language_models/lm_en_subword/model.json

[decode]
beam_size = 60
ctc_weight = 0.1
lm_weight = 0.4
penalty = 0.3
```

**Python Code**:
```python
pipeline = InferencePipeline(
    config_filename="configs/LRS3_V_WER19.1.ini",
    detector="retinaface",  # More accurate
    face_track=True,        # For moving faces
    device="cuda:0"         # GPU
)
```

**Result**: Best accuracy, slowest (3-5 seconds per video)

---

### Configuration 2: Balanced (Default)
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

**Python Code**:
```python
pipeline = InferencePipeline(
    config_filename="configs/LRS3_V_WER19.1.ini",
    detector="retinaface",  # Balanced
    face_track=False,       # Standard
    device="cuda:0"         # GPU
)
```

**Result**: Good accuracy, reasonable speed (~3 seconds per video)

---

### Configuration 3: Real-Time / Fast Processing
```ini
[input]
modality = video
v_fps = 25.0

[model]
model_path = benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth
model_conf = benchmarks/LRS3/models/LRS3_V_WER19.1/model.json
v_fps = 25.0
rnnlm =   # Empty = no language model
rnnlm_conf =   # Empty

[decode]
beam_size = 5
ctc_weight = 0.1
lm_weight = 0.0  # No LM for speed
penalty = 0.3
```

**Python Code**:
```python
pipeline = InferencePipeline(
    config_filename="configs/LRS3_V_WER19.1.ini",
    detector="mediapipe",   # Faster
    face_track=False,       # Standard
    device="cuda:0"         # GPU
)
```

**Result**: Fastest, lower accuracy (~1 second per video, 5-10% accuracy drop)

---

### Configuration 4: CPU-Only System
```ini
[input]
modality = video
v_fps = 25.0

[model]
model_path = benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth
model_conf = benchmarks/LRS3/models/LRS3_V_WER19.1/model.json
v_fps = 25.0
rnnlm =   # Empty = no language model
rnnlm_conf =   # Empty

[decode]
beam_size = 10  # Lower for CPU
ctc_weight = 0.1
lm_weight = 0.0  # No LM for speed
penalty = 0.3
```

**Python Code**:
```python
pipeline = InferencePipeline(
    config_filename="configs/LRS3_V_WER19.1.ini",
    detector="mediapipe",   # CPU-based
    face_track=False,       # Standard
    device="cpu"            # CPU-only
)
```

**Result**: Works on CPU, very slow (30-60 seconds per video)

---

## Parameter Priority (Most Impact to Least)

### For Accuracy
1. **`beam_size`** - Biggest impact (5-10% accuracy difference)
2. **`lm_weight`** - Significant impact (5-10% accuracy difference)
3. **`detector`** - Moderate impact (2-5% accuracy difference)
4. **`speed_rate`** - Critical if wrong (20-30% accuracy drop)
5. **`ctc_weight`** - Minor impact (1-2% accuracy difference)
6. **`penalty`** - Affects output length, not accuracy

### For Speed
1. **`device`** - Biggest impact (10-50x speed difference)
2. **`beam_size`** - Significant impact (2-3x speed difference)
3. **`detector`** - Moderate impact (2x speed difference for face detection)
4. **`lm_weight`** - Moderate impact (10-20% speed difference)
5. **`face_track`** - Minor impact (5-10% speed difference)

---

## Quick Reference Table

| Parameter | File Section | Type | Default | Affects Speed? | Affects Accuracy? | Priority |
|-----------|--------------|------|---------|----------------|-------------------|----------|
| `beam_size` | `[decode]` | int | 35 | ✅ High | ✅ High | ⭐⭐⭐ |
| `lm_weight` | `[decode]` | float | 0.4 | ✅ Medium | ✅ High | ⭐⭐⭐ |
| `detector` | Code | string | "retinaface" | ✅ High | ✅ Medium | ⭐⭐ |
| `device` | Code | string | "cuda:0" | ✅ Very High | ❌ No | ⭐⭐⭐ |
| `ctc_weight` | `[decode]` | float | 0.1 | ❌ No | ✅ Low | ⭐ |
| `penalty` | `[decode]` | float | 0.3 | ❌ No | ❌ No* | ⭐ |
| `speed_rate` | Calculated | float | auto | ❌ No | ✅ Critical | ⭐⭐⭐ |
| `face_track` | Code | bool | False | ✅ Low | ✅ Low | ⭐ |
| `modality` | `[input]` | string | "video" | ✅ Low | ✅ Medium | ⭐⭐ |

*Affects output length, not accuracy

---

## Summary

### Top 3 Parameters for Accuracy
1. **`beam_size`** - Increase to 60 for maximum accuracy
2. **`lm_weight`** - Set to 0.4 (with LM) for better grammar
3. **`speed_rate`** - Must be correct (auto-detected by your code)

### Top 3 Parameters for Speed
1. **`device`** - Use "cuda:0" (GPU) for 10-50x speedup
2. **`beam_size`** - Decrease to 5-10 for 2-3x speedup
3. **`detector`** - Use "mediapipe" for 2x faster face detection

### Parameters You Should NOT Change
- `v_fps` in `[model]` section (model training FPS)
- `model_path` / `model_conf` (unless using different model)
- `speed_rate` (auto-calculated, don't override)

---

## Related Documents

- [Parameters](08_Parameters.md) - Detailed parameter explanations
- [Processing Flow](07_ProcessingFlow.md) - How parameters affect pipeline
- [Parallel Computing Benefits](11_ParallelComputingBenefits.md) - Speed optimization strategies

