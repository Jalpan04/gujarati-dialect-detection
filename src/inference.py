
import torch
import librosa
import numpy as np
import os
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from data.normalize import normalize_gujarati_text

class GujaratiDialectPredictor:
    def __init__(self, text_model_path="models/muril_finetuned/final", 
                 audio_model_path="models/wav2vec2_gujarati/final"):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading System on {self.device}...")
        self.labels = ['Standard', 'Kathiawadi'] # 0=Standard, 1=Kathiawadi

        # 1. Load Text Pipeline (MuRIL)
        print(f"Loading Text Model from {text_model_path}...")
        try:
            self.text_tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
            # Try loading local model, else fallback (though in this project we rely on local)
            self.text_model = AutoModelForSequenceClassification.from_pretrained(text_model_path).to(self.device)
            self.text_model.eval()
            self.text_enabled = True
        except Exception as e:
            print(f"Warning: Failed to load Text Model ({e}).")
            self.text_enabled = False

        # 2. Load Audio Model (Wav2Vec2)
        print(f"Loading Audio Model from {audio_model_path}...")
        try:
            self.audio_extractor = Wav2Vec2FeatureExtractor.from_pretrained(audio_model_path)
            self.audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(audio_model_path).to(self.device)
            self.audio_model.eval()
            self.audio_enabled = True
        except Exception as e:
            print(f"Warning: Failed to load Audio Model ({e}).")
            self.audio_enabled = False

    def predict_text(self, text):
        if not self.text_enabled:
            return None
            
        clean_text = normalize_gujarati_text(text)
        
        # Tokenize (MuRIL expects standard BERT-like inputs)
        inputs = self.text_tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=128, padding="max_length").to(self.device)
        
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            
        return probs

    def predict_audio(self, audio_path):
        if not self.audio_enabled:
            return None
            
        # Load and Resample (Wav2Vec2 mandates 16kHz)
        speech, sr = librosa.load(audio_path, sr=16000)
        
        # Normalize volume
        speech = librosa.util.normalize(speech)
        
        # Feature Extraction
        inputs = self.audio_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.audio_model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        return probs

    def predict(self, text=None, audio_path=None):
        text_probs = None
        audio_probs = None
        
        if text:
            text_probs = self.predict_text(text)
            
        if audio_path:
            audio_probs = self.predict_audio(audio_path)
            
        # Fusion Logic (Simple Average for now)
        if text_probs is not None and audio_probs is not None:
            # Check dimensions match
            if len(text_probs) != len(audio_probs):
                # Fallback if label counts differ (rare)
                final_probs = text_probs 
            else:
                final_probs = (text_probs + audio_probs) / 2
                
            idx = np.argmax(final_probs)
            return {
                'final_prediction': self.labels[idx],
                'confidence': float(final_probs[idx]),
                'model_used': 'Fusion (Text + Audio)',
                'details': {l: float(p) for l, p in zip(self.labels, final_probs)}
            }
            
        elif text_probs is not None:
             idx = np.argmax(text_probs)
             return {
                'final_prediction': self.labels[idx],
                'confidence': float(text_probs[idx]),
                'model_used': 'MuRIL (Text Only)',
                'details': {l: float(p) for l, p in zip(self.labels, text_probs)}
             }

        elif audio_probs is not None:
             idx = np.argmax(audio_probs)
             return {
                'final_prediction': self.labels[idx],
                'confidence': float(audio_probs[idx]),
                'model_used': 'Wav2Vec2 (Audio Only)',
                'details': {l: float(p) for l, p in zip(self.labels, audio_probs)}
             }
        
        return {"error": "No valid input provided or models failed to load."}

if __name__ == "__main__":
    # Quick Test
    predictor = GujaratiDialectPredictor()
    print("Test Text:", predictor.predict(text="કેમ છો?"))
