# Gujarati Dialect Detection (Text & Speech)

## Task
Binary dialect classification: Standard vs Kathiawadi Gujarati.

## Motivation
Low-resource Indian dialects are under-studied, especially in multimodal settings. This project explores the fusion of linguistic (text) and acoustic (speech) signals to improve dialect identification in noisy, real-world environments.

## Dataset
- **Text:** ~1500 samples collected from YouTube comments & transcripts.
- **Audio:** 75 in-the-wild clips (approx. 5–10s each).
- **Labels:** Semi-automated acquisition with partial manual verification.
- **Note:** Dataset is not released publicly due to licensing restrictions.

## Models
**Text:**
- TF-IDF + SVM (Baseline)
- MuRIL (Multilingual Representations for Indian Languages) - Fine-tuned

**Audio:**
- MFCC + SVM (Baseline)
- Wav2Vec2-XLSR-53 (Meta) - Fine-tuned on Gujarati speech

## Experiments
1. **Text-only:** Evaluation of MuRIL on short, informal text.
2. **Audio-only:** Evaluation of Wav2Vec2 on noisy, in-the-wild audio segments.
3. **Fusion:** Preliminary exploration of decision-level fusion strategies.

## Results
*Summary of preliminary findings (Metrics to be added)*

## Limitations
- Small audio dataset size relative to text.
- Noisy labels inherent in semi-automated data collection.
- Independent (unpaired) text and audio samples used for training.

## Reproducibility
- Seed fixed for experimentation: `42`
- Codebase includes training scripts, data augmentation pipelines, and inference logic.

## Demo (Optional)
A minimal Streamlit interface (`src/app.py`) is included to demonstrate the inference pipeline.
