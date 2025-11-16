# Lip Reading Model Documentation

## Quick Navigation

### Core Concepts
1. [Overview](01_Overview.md) - Project structure and three-layer architecture
2. [CTC](02_CTC.md) - Connectionist Temporal Classification for alignment
3. [Beam Search](03_BeamSearch.md) - Decoding algorithm for transcription
4. [Transformer Encoder](04_TransformerEncoder.md) - Video to feature conversion
5. [Speed Rate](06_SpeedRate.md) - FPS handling and frame downsampling

### Architecture & Flow
6. [Pipelines](05_Pipelines.md) - Pipeline architecture and components
7. [Processing Flow](07_ProcessingFlow.md) - Complete video-to-subtitle flow

### Implementation Details
8. [Parameters](08_Parameters.md) - Key parameters and their impact
9. [File Structure](09_FileStructure.md) - Folder and file purposes

---

## Reading Order

**For beginners:**
1. Start with [Overview](01_Overview.md)
2. Read [CTC](02_CTC.md) and [Beam Search](03_BeamSearch.md)
3. Understand [Processing Flow](07_ProcessingFlow.md)
4. Reference [Parameters](08_Parameters.md) when needed

**For developers:**
1. [Overview](01_Overview.md) - Quick context
2. [Pipelines](05_Pipelines.md) - Your code structure
3. [File Structure](09_FileStructure.md) - Where everything is
4. Reference specific concepts as needed

---

## Key Takeaways

- **CTC**: Aligns frames to text tokens (used for timestamps)
- **Beam Search**: Decodes features to text (accuracy vs speed trade-off)
- **Speed Rate**: Critical for correct frame downsampling (auto-detected)
- **Transformer**: Converts video frames to feature vectors

