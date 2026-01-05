
import numpy as np
import librosa
import random

class AudioAugmentor:
    """
    Applies random augmentations to audio arrays.
    Techniques:
    1. White Noise injection
    2. Time Stretching (Speed up/slow down)
    3. Pitch Shifting
    4. Time Masking (silence random chunks)
    """
    
    def __init__(self, sr=16000):
        self.sr = sr
        
    def add_noise(self, data, noise_factor=0.005):
        noise = np.random.randn(len(data))
        augmented_data = data + noise_factor * noise
        # Cast back to same type
        augmented_data = augmented_data.astype(type(data[0]))
        return augmented_data

    def time_stretch(self, data, rate=None):
        if rate is None:
            # Random rate between 0.8 and 1.2
            rate = np.random.uniform(0.8, 1.2)
        return librosa.effects.time_stretch(data, rate=rate)

    def pitch_shift(self, data, steps=None):
        if steps is None:
            # Random steps between -2 and 2 semitones
            steps = np.random.randint(-2, 2)
        return librosa.effects.pitch_shift(data, sr=self.sr, n_steps=steps)
        
    def random_gain(self, data):
        gain = np.random.uniform(0.8, 1.2)
        return data * gain

    def augment(self, data):
        """Apply a random combination of augmentations"""
        if np.random.random() < 0.5:
            data = self.add_noise(data)
            
        if np.random.random() < 0.3:
            data = self.time_stretch(data)
            
        if np.random.random() < 0.3:
            data = self.pitch_shift(data)
            
        if np.random.random() < 0.5:
            data = self.random_gain(data)
            
        return data
