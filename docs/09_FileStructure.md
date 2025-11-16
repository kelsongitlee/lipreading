# File Structure & Purposes

## Overview

Organization of files and folders in the lip reading project.

---

## Root Directory

```
lipreading/
├── benchmarks/          # Pre-trained models (gitignored)
├── configs/            # Model configuration files
├── doc/                # Documentation images
├── espnet/             # Modified ESPnet framework
├── hydra_configs/      # Hydra configuration (if used)
├── pipelines/          # Main inference code
├── tools/              # Utility scripts
├── web/                # Web interface
├── .gitignore          # Git ignore rules
├── LICENSE             # License file
├── README.md           # Project readme
├── requirements.txt    # Python dependencies
└── requirements-web.txt # Web dependencies
```

---

## `/pipelines/` - Main Inference Code

### Purpose
Your primary code for video processing and inference.

### Structure
```
pipelines/
├── data/
│   ├── data_module.py      # Video/audio loading
│   ├── transforms.py        # Frame downsampling
│   └── noise/              # Noise samples
├── detectors/
│   ├── mediapipe/          # MediaPipe face detector
│   │   ├── detector.py
│   │   └── video_process.py
│   └── retinaface/         # RetinaFace detector
│       ├── detector.py
│       └── video_process.py
├── metrics/
│   └── measures.py         # Evaluation metrics
├── model.py                # Model wrapper ✅ YOUR CODE
├── pipeline.py             # Main pipeline ✅ YOUR CODE
└── tokens/
    └── unigram5000_units.txt  # Token vocabulary
```

### Key Files

#### `pipelines/pipeline.py`
**Purpose**: High-level inference orchestration  
**Modified**: ✅ Yes (FPS detection)  
**Key classes**:
- `InferencePipeline`: Main entry point
  - `forward()`: Basic transcription
  - `forward_with_alignment()`: Transcription + timestamps ✅

#### `pipelines/model.py`
**Purpose**: Model wrapper and inference logic  
**Modified**: ✅ Yes (timestamp generation)  
**Key classes**:
- `AVSR`: Audio-Visual Speech Recognition model
  - `infer()`: Basic transcription
  - `infer_with_alignment()`: Transcription + timestamps ✅

#### `pipelines/data/data_module.py`
**Purpose**: Video/audio loading and preprocessing  
**Modified**: ❌ No  
**Key classes**:
- `AVSRDataLoader`: Loads and transforms video/audio

#### `pipelines/data/transforms.py`
**Purpose**: Video frame downsampling  
**Modified**: ❌ No  
**Key classes**:
- `VideoTransform`: Downsamples frames based on speed_rate

---

## `/espnet/` - Modified ESPnet Framework

### Purpose
Custom modifications to ESPnet for visual speech recognition.

### Structure
```
espnet/
├── asr/
│   └── asr_utils.py        # ASR utilities
├── nets/
│   ├── pytorch_backend/
│   │   ├── backbones/      # ResNet, Conv3D extractors
│   │   ├── ctc.py          # CTC implementation ✅ MODIFIED
│   │   ├── e2e_asr_transformer.py      # Video encoder
│   │   ├── e2e_asr_transformer_av.py   # Audio-visual encoder
│   │   ├── lm/             # Language models
│   │   └── transformer/    # Transformer components
│   ├── beam_search.py      # Beam search algorithm
│   ├── batch_beam_search.py # Batch beam search
│   └── ctc_prefix_score.py # CTC prefix scoring
└── utils/                  # Utility functions
```

### Key Files

#### `espnet/nets/pytorch_backend/ctc.py`
**Purpose**: CTC loss and forced alignment  
**Modified**: ✅ Yes (2D/3D shape handling)  
**Key methods**:
- `log_softmax()`: Frame-level probabilities ✅ MODIFIED
- `forced_align()`: Frame-to-token alignment ✅ MODIFIED

#### `espnet/nets/pytorch_backend/e2e_asr_transformer.py`
**Purpose**: Video-only Transformer model  
**Modified**: ❌ No  
**Key methods**:
- `encode()`: Video frames → encoder features

#### `espnet/nets/pytorch_backend/e2e_asr_transformer_av.py`
**Purpose**: Audio-visual Transformer model  
**Modified**: ❌ No  
**Key methods**:
- `encode()`: Video + audio → encoder features

#### `espnet/nets/beam_search.py`
**Purpose**: Beam search decoding algorithm  
**Modified**: ❌ No  
**Key classes**:
- `BeamSearch`: Main beam search implementation

#### `espnet/nets/batch_beam_search.py`
**Purpose**: Batch beam search (faster)  
**Modified**: ❌ No  
**Key classes**:
- `BatchBeamSearch`: Batch processing version

---

## `/configs/` - Model Configuration

### Purpose
INI files specifying model paths and hyperparameters.

### Structure
```
configs/
├── LRS3_V_WER19.1.ini      # LRS3 visual model config
├── LRS3_V_WER32.3.ini      # Alternative LRS3 config
├── GRID_V_WER1.2.ini       # GRID dataset config
└── ...                     # Other dataset configs
```

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

## `/benchmarks/` - Pre-trained Models

### Purpose
Downloaded model weights and language models (gitignored).

### Structure
```
benchmarks/
├── LRS3/
│   ├── models/
│   │   └── LRS3_V_WER19.1/    # Visual model weights
│   └── language_models/
│       └── lm_en_subword/      # English language model
├── GRID/
│   └── models/                 # GRID model weights
└── ...                         # Other datasets
```

### Note
- Models are large (100-200 MB each)
- Not tracked in git (see `.gitignore`)
- Download separately using links in README

---

## `/web/` - Web Interface

### Purpose
Flask web server for video upload and transcription.

### Structure
```
web/
├── app.py              # Flask application
├── static/
│   └── app.js         # Frontend JavaScript
└── templates/
    └── index.html     # Frontend HTML
```

---

## `/tools/` - Utility Scripts

### Purpose
Helper scripts for data processing and evaluation.

---

## Key File Locations

| Component | File | Modified? |
|-----------|------|-----------|
| Main pipeline | `pipelines/pipeline.py` | ✅ Yes |
| Model wrapper | `pipelines/model.py` | ✅ Yes |
| CTC alignment | `espnet/nets/pytorch_backend/ctc.py` | ✅ Yes |
| Beam search | `espnet/nets/beam_search.py` | ❌ No |
| Video encoder | `espnet/nets/pytorch_backend/e2e_asr_transformer.py` | ❌ No |
| Data loading | `pipelines/data/data_module.py` | ❌ No |
| Transforms | `pipelines/data/transforms.py` | ❌ No |
| Face detectors | `pipelines/detectors/*/` | ❌ No |

---

## File Modification Status

### ✅ Your Customizations
1. `pipelines/pipeline.py`: FPS detection, speed_rate adjustment
2. `pipelines/model.py`: Timestamp generation, word grouping
3. `espnet/nets/pytorch_backend/ctc.py`: 2D/3D shape handling

### ❌ Original Repo (Unchanged)
- All other files in `espnet/`
- Face detectors
- Data loaders
- Transforms

---

## Related Concepts

- **Pipelines**: See how files work together
- **Processing Flow**: See file usage in the pipeline
- **Parameters**: See where parameters are configured

