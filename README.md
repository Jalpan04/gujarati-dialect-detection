# Gujarati Dialect Detection

This repository contains a system for detecting Gujarati dialects using a multimodal approach:
- **Text model**: MuRIL (Google) for Gujarati text classification.
- **Audio model**: Wav2Vec2‑XLSR fine‑tuned on cleaned Gujarati speech.

## Quick Start
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the Streamlit UI:
   ```
   python -m streamlit run src/app.py
   ```
3. Run inference directly:
   ```python
   from inference import GujaratiDialectPredictor
   predictor = GujaratiDialectPredictor()
   result = predictor.predict(text="તમારો ટેક્સ્ટ અહીં", audio_path="path/to/audio.wav")
   print(result)
   ```

## Project Structure
- `src/` – source code (models, inference, UI)
- `models/` – trained model checkpoints
- `data/` – scripts for data collection and preprocessing
- `requirements.txt` – Python dependencies

## License
MIT License.
