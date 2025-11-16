# CTC (Connectionist Temporal Classification)

## What It Does

CTC aligns video frames to text tokens **without requiring frame-level labels**.

**Problem it solves**: Video has many frames (e.g., 75 frames), but transcription has few words (e.g., 5 words). How do we map frames to words?

**Solution**: CTC finds the best alignment path through frames that produces the correct text.

---

## Why We Need It

### For Timestamp Generation
- Model outputs frame-level probabilities for each token
- CTC finds which frame corresponds to which token
- Enables word-level timestamp generation for subtitles

### Without CTC
- No way to know when each word occurs in the video
- Can only generate transcription without timing information

---

## How It Works

### 1. Input
- **Encoder features**: (T, D) tensor
  - T = number of frames (e.g., 75)
  - D = feature dimension (e.g., 512)
- **Token sequence**: List of token IDs (e.g., [5, 12, 8, 3, 7])

### 2. Process
```
For each frame:
  1. Calculate probability distribution over all tokens
  2. Use dynamic programming to find best alignment path
  3. Map frames to tokens (allowing blanks/repeats)
```

### 3. Output
- **Frame-to-token mapping**: List showing which token each frame aligns to
- Example: `[5, 5, 5, 12, 12, 8, 8, 8, 3, 7, 7]`
  - Frames 0-2 → Token 5
  - Frames 3-4 → Token 12
  - etc.

---

## Key Method: `forced_align()`

**Location**: `espnet/nets/pytorch_backend/ctc.py`

**Signature**:
```python
def forced_align(self, h, y, blank_id=0):
    """
    Args:
        h: Encoder features (T, D) or (B, T, D)
        y: Token sequence (L,) where L = length
        blank_id: Blank token ID (usually 0)
    
    Returns:
        aligned_frames: List of token IDs, one per frame
    """
```

**What it does**:
1. Computes log probabilities for each frame: `log_softmax(h)`
2. Uses Viterbi algorithm to find best alignment
3. Returns frame-level token assignments

---

## Your Modifications

### Problem
Original code only handled 3D input `(B, T, D)` with batch dimension, but encoder returns 2D `(T, D)`.

### Solution
**File**: `espnet/nets/pytorch_backend/ctc.py`

**Changes**:
- `log_softmax()`: Handles both 2D and 3D inputs
- `forced_align()`: Adds batch dimension if needed, removes after processing

**Code location**:
- Lines 160-166: Shape handling in `log_softmax()`
- Lines 186-260: Shape handling in `forced_align()`

---

## Usage in Your Code

**File**: `pipelines/model.py:infer_with_alignment()`

**Flow**:
```python
# 1. Get encoder features
enc_feats = self.model.encode(data)  # (T, D)

# 2. Get token IDs from beam search
token_ids = beam_search_result['token_ids']  # [5, 12, 8, ...]

# 3. Perform CTC forced alignment
aligned_frames = self.model.ctc.forced_align(
    enc_feats, 
    np.array(token_ids),
    blank_id=0
)  # [5, 5, 5, 12, 12, 8, ...]

# 4. Group consecutive tokens into words
# 5. Convert frame indices to timestamps
```

---

## Important Notes

### Blank Tokens
- CTC uses blank tokens (ID=0) to handle:
  - Silence between words
  - Frame repetitions
- Blanks are filtered out when grouping tokens into words

### Token Repetition
- Same token can align to multiple consecutive frames
- This is normal and expected
- Word grouping logic handles this

### Frame-to-Timestamp Conversion
```python
timestamp = frame_index / video_fps
# Example: frame 25 at 25fps = 1.0 second
```

---

## Related Concepts

- **Beam Search**: Generates the token sequence that CTC aligns
- **SentencePiece**: Tokenization format (▁ prefix indicates word start)
- **Processing Flow**: See how CTC fits into the complete pipeline

---

## Further Reading

- **Paper**: "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks" (Graves et al., 2006)
- **Focus**: Understand the dynamic programming alignment algorithm

