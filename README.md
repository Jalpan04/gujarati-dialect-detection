# Gujarati Dialect Detection (Text & Speech)

## Task
Binary dialect classification: **Standard** vs **Kathiawadi** Gujarati.

## Motivation
Low-resource Indian dialects are under-studied, especially in multimodal settings. Standard NLP tools often fail on dialectal variations present in social media text and in-the-wild speech. This project explores the effectiveness of Transformer-based models (MuRIL and Wav2Vec2) for this task.

## Dataset
- **Text**: ~1500 samples collected from YouTube comments & transcripts.
- **Audio**: ~75 in-the-wild clips (5–10s duration), totaling ~8 minutes of speech.
- **Labels**: Semi-automated keyword-based extraction with partial manual verification.
- *Note: Data is not released publicly due to licensing restrictions.*

## Models
### Text
- **TF-IDF + SVM**: Baseline approach.
- **MuRIL (fine-tuned)**: Large-scale pretrained model optimized for Indian languages.

### Audio
- **MFCC + SVM**: Baseline acoustic feature approach.
- **Wav2Vec2-XLSR (fine-tuned)**: Self-supervised model for cross-lingual speech representation.

## Experiments
- **Text-only**: Comparing BERT-based representations vs statistical baselines.
- **Audio-only**: Assessing Wav2Vec2's robustness to background noise.
- **Decision-level fusion**: Combining predictions to handle modality-specific ambiguity.

## Results
(Preliminary Results)
| Modality | Model | Accuracy |
| :--- | :--- | :--- |
| Text | MuRIL | ~95% |
| Audio | Wav2Vec2 | ~85% |

## Limitations
- **Small Audio Dataset**: Limited diversity in speakers and recording conditions.
- **Noisy Labels**: Weak supervision from video metadata leads to occasional mislabeling.
- **Unpaired Data**: Text and audio samples are independent; no synchronized specific transcripts.

## Reproducibility
- Seeds and configurations are defined in `src/config.py` (if applicable) or training scripts.
- Training scripts: `src/train_muril.py`, `src/models/train_wav2vec2.py`.

## Demo (Optional)
A Streamlit-based UI is included for demonstration.
Run `python -m streamlit run src/app.py`
