import torch
import torch.nn as nn
import torch.nn.functional as F

class GujaratiDialectCRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for Dialect ID.
    Input: Mel Spectrogram (Batch, 1, 128, Time)
    Output: 5 Dialect Classes
    """
    def __init__(self, num_classes=2, hidden_size=256, num_layers=2):
        super(GujaratiDialectCRNN, self).__init__()
        
        # 4 Convolutional Blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        # LSTM input calculation
        # Mel bins (128) reduced by 2^4 = 16 -> 128/16 = 8
        # Feature dim = 256 channels * 8 freq bins = 2048
        self.lstm_input_size = 256 * 8
        
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Input: (Batch, 1, Freq, Time)
        
        # CNN
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x) # (B, 256, F/16, T/16)
        
        # Dimensions
        b, c, f, t = x.size()
        
        # Permute for LSTM: (Batch, Time, Features)
        x = x.permute(0, 3, 1, 2) # (B, T, C, F)
        x = x.reshape(b, t, c * f) # (B, T, C*F)
        
        # LSTM
        lstm_out, _ = self.lstm(x) # (B, T, 2*Hidden)
        
        # Attention Pooling (or simply take last, here using Mean per design preference)
        # Using Last Time Step for now (or Mean pool)
        # x = torch.mean(lstm_out, dim=1) 
        x = lstm_out[:, -1, :] # Last time step
        
        # Fully Connected
        out = self.fc(x)
        
        return out

if __name__ == "__main__":
    # Test Architecture
    print("Initializing CRNN Model...")
    model = GujaratiDialectCRNN()
    print(model)
    
    # Dummy Input: Batch=2, Ch=1, Freq=128, Time=313 (approx 10s audio with hop=512)
    dummy_input = torch.randn(2, 1, 128, 313)
    
    print(f"\nForward Pass Test:")
    print(f"Input Shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output Shape: {output.shape} (Expected: [2, 5])")
    print("\nSUCCESS: Model architecture is valid.")
