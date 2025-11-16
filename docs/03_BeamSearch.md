# Beam Search

## What It Does

Beam search decodes encoder features into text by exploring multiple candidate sequences and keeping the best ones.

**Problem it solves**: There are exponentially many possible transcriptions. How do we find the best one efficiently?

**Solution**: Maintain top-K hypotheses at each step, prune the rest.

---

## Why We Need It

### For Transcription Generation
- Encoder outputs features, not text
- Need to convert features → tokens → words
- Beam search finds the most likely token sequence

### Alternative: Greedy Decoding
- Only considers best token at each step
- Faster but less accurate
- Beam search explores multiple paths, finds better results

---

## How It Works

### 1. Initialization
- Start with empty sequence
- Score: 0.0

### 2. Expansion
```
For each step:
  1. Take top-K (beam_size) current hypotheses
  2. For each hypothesis, generate next token candidates
  3. Score each candidate using:
     - CTC score (frame alignment)
     - Decoder score (context)
     - Language model score (grammar)
     - Length penalty
  4. Keep top-K best candidates
  5. Repeat until end-of-sequence token
```

### 3. Output
- Best hypothesis (highest total score)
- Token sequence: [token1, token2, ..., tokenN]

---

## Key Parameters

### `beam_size`
**What**: Number of hypotheses to maintain  
**Impact**:
- **Higher (35-60)**: More accurate, slower
  - Explores more possibilities
  - Better for video upload, practice mode
- **Lower (5-10)**: Faster, less accurate
  - Fewer paths explored
  - Better for real-time transcription

**Location**: `configs/*.ini` or dynamically set

**Current setting**: 35 (balanced)

---

### `ctc_weight`
**What**: How much to trust CTC vs decoder  
**Range**: 0.0 to 1.0  
**Impact**:
- **0.1** (current): Mostly decoder, CTC for alignment
- **1.0**: Pure CTC (no decoder)
- **0.0**: Pure decoder (no CTC alignment)

**Location**: `configs/*.ini`

**Current setting**: 0.1

---

### `lm_weight`
**What**: Language model influence  
**Range**: 0.0 to 1.0  
**Impact**:
- **0.4** (current): Strong LM guidance
  - Better grammar, context-aware
  - Requires LM file to be loaded
- **0.0**: No LM (CTC + decoder only)
  - Faster, less grammar-aware
  - Used for real-time mode

**Location**: `configs/*.ini`

**Current setting**: 0.4 (with LM), 0.0 (real-time)

---

### `penalty`
**What**: Length penalty to prevent short outputs  
**Range**: Typically 0.0 to 1.0  
**Impact**:
- **0.3** (current): Moderate penalty
  - Prefers longer, more complete transcriptions
- **0.0**: No penalty
  - May produce shorter outputs

**Location**: `configs/*.ini`

**Current setting**: 0.3

---

## Scoring Formula

```
Total Score = 
  (1 - ctc_weight) × decoder_score +
  ctc_weight × ctc_score +
  lm_weight × lm_score +
  penalty × length_bonus
```

**Example** (current settings):
```
Total Score = 
  0.9 × decoder_score +
  0.1 × ctc_score +
  0.4 × lm_score +
  0.3 × length_bonus
```

---

## Implementation

### Main Algorithm
**File**: `espnet/nets/beam_search.py`

**Key class**: `BeamSearch`
- Maintains beam of hypotheses
- Expands and scores candidates
- Returns best hypothesis

### Batch Processing
**File**: `espnet/nets/batch_beam_search.py`

**Key class**: `BatchBeamSearch`
- Processes multiple sequences in parallel
- Faster for batch inference
- Used in your code

---

## Usage in Your Code

**File**: `pipelines/model.py:AVSR.__init__()`

**Creation**:
```python
self.beam_search = get_beam_search_decoder(
    model=self.model,
    token_list=self.token_list,
    rnnlm=rnnlm,
    rnnlm_conf=rnnlm_conf,
    penalty=penalty,
    ctc_weight=ctc_weight,
    lm_weight=lm_weight,
    beam_size=beam_size
)
```

**Usage**:
```python
# In infer_with_alignment()
result = self.beam_search(enc_feats)
transcription = result['yseq']  # Token sequence
token_ids = result['token_ids']  # For CTC alignment
```

---

## Accuracy vs Speed Trade-off

### High Accuracy (beam_size=60)
- **Pros**: Best transcription quality
- **Cons**: 3-5x slower
- **Use**: Video upload, practice mode

### Balanced (beam_size=35)
- **Pros**: Good accuracy, reasonable speed
- **Cons**: Not real-time
- **Use**: Current default

### High Speed (beam_size=5)
- **Pros**: Near real-time
- **Cons**: 5-10% accuracy drop
- **Use**: Real-time transcription

---

## Important Notes

### Sequential Processing
- Beam search processes tokens sequentially
- Cannot be parallelized (each step depends on previous)
- GPU acceleration helps with matrix operations, not sequence

### Language Model
- Optional but recommended
- Improves grammar and context
- Adds loading and scoring overhead
- Disable for maximum speed

### Token Sequence
- Output is token IDs, not words
- Need to convert using token_list
- SentencePiece tokens have ▁ prefix for word boundaries

---

## Related Concepts

- **CTC**: Provides alignment scores for beam search
- **Transformer Encoder**: Generates features that beam search decodes
- **Processing Flow**: See how beam search fits into the pipeline

---

## Further Reading

- **Paper**: "A Tutorial on Beam Search" (Lowerre, 1976)
- **Focus**: Understand the pruning and expansion algorithm

