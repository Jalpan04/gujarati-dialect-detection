import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from features.audio_features import AudioPreprocessor
from models.audio_crnn import GujaratiDialectCRNN

# Implement a Dataset Class for Real Audio
class RealAudioDataset(Dataset):
    def __init__(self, audio_dir, labels_map):
        self.files = []
        self.labels = []
        self.preprocessor = AudioPreprocessor()
        
        # Scan directory: dataset/audio/raw/{dialect_id}.wav ??
        # Or parse filename: {dialect}_{id}.wav
        # Current Miner format: {dialect}_{id}.wav in dataset/audio/raw/
        
        all_wavs = glob.glob(os.path.join(audio_dir, "*.wav"))
        for f in all_wavs:
            filename = os.path.basename(f)
            # Simple heuristic since filename starts with dialect
            # e.g. saurashtra_VideoID.wav
            for dialect, label_idx in labels_map.items():
                if filename.startswith(dialect):
                    self.files.append(f)
                    self.labels.append(label_idx)
                    break
        
        print(f"Found {len(self.files)} Audio files.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]
        
        # Preprocess
        wav = self.preprocessor.load_audio(path)
        if wav is None:
            # Handle error gracefully? Return zero tensor for now
            return torch.zeros(1, 128, 313), torch.tensor(label)
            
        mel = self.preprocessor.extract_mel_spectrogram(wav)
        # (128, T) -> (1, 128, T)
        mel_tensor = torch.tensor(mel).float().unsqueeze(0)
        
        return mel_tensor, torch.tensor(label)

def train_audio_model():
    print("--- Training Audio CRNN ---")
    
    # Config
    EPOCHS = 50
    BATCH_SIZE = 8
    LR = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Prepare Data
    # 1. Prepare Data
    # MERGED MAP: Saurashtra & Kathiawadi -> Kathiawadi (1). Standard -> Standard (0).
    # Others ignored for now or mapped if data existed.
    LABELS_MAP = {
        'standard': 0, 
        'kathiawadi': 1,
        'saurashtra': 1  # Merge into Kathiawadi
    }
    
    dataset = RealAudioDataset("D:/gujarati_project_data/dataset/audio/clean", LABELS_MAP)
    if len(dataset) == 0:
        print("No data found. Skipping Audio Training.")
        return
        
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Model
    # Now strictly 2 classes
    model = GujaratiDialectCRNN(num_classes=2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for mels, labels in loader:
            mels, labels = mels.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(mels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.4f}")
        
    # 4. Save
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/audio_crnn.pth")
    print("Audio Model Saved.")

if __name__ == "__main__":
    train_audio_model()
    # Add train_fusion_model() here later
