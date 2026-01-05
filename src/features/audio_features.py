import librosa
import numpy as np
import os

class AudioPreprocessor:
    def __init__(self, sample_rate=16000, duration=10):
        self.sr = sample_rate
        self.duration = duration
        self.target_len = sample_rate * duration

    def load_audio(self, file_path):
        """Loads and pads/trims audio to fixed length."""
        try:
            # Load with resampling
            y, _ = librosa.load(file_path, sr=self.sr, duration=self.duration)
            
            # Pad or Trim
            if len(y) < self.target_len:
                padding = self.target_len - len(y)
                y = np.pad(y, (0, padding), 'constant')
            else:
                y = y[:self.target_len]
                
            # Normalize
            y = librosa.util.normalize(y)
            return y
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def extract_mel_spectrogram(self, y):
        """
        Extracts Mel Spectrogram.
        Output Shape: (n_mels, time_steps)
        """
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=self.sr, 
                n_fft=2048, 
                hop_length=512, 
                n_mels=128
            )
            # Convert to Log scale (dB)
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            return mel_db
        except Exception as e:
            print(f"Error extracting Mel Spec: {e}")
            return None

    def extract_mfcc(self, y):
        """
        Extracts MFCCs.
        Output Shape: (n_mfcc, time_steps)
        """
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=40)
            return mfccs
        except Exception as e:
            print(f"Error extracting MFCC: {e}")
            return None

if __name__ == "__main__":
    # Create a dummy wav file for testing
    import soundfile as sf
    dummy_wav = "test_audio.wav"
    sr = 16000
    # Generate 3 seconds of white noise
    dummy_data = np.random.uniform(-1, 1, size=sr*3)
    sf.write(dummy_wav, dummy_data, sr)
    
    print("Testing Audio Preprocessor...")
    processor = AudioPreprocessor(sample_rate=16000, duration=5)
    
    # Test Load
    y = processor.load_audio(dummy_wav)
    print(f"Loaded Audio Shape: {y.shape} (Expected {16000*5})")
    
    # Test Mel
    mel = processor.extract_mel_spectrogram(y)
    print(f"Mel Spectrogram Shape: {mel.shape} (Expected (128, ...))")
    
    # Test MFCC
    mfcc = processor.extract_mfcc(y)
    print(f"MFCC Shape: {mfcc.shape} (Expected (40, ...))")
    
    # Cleanup
    os.remove(dummy_wav)
