import torch
import torch.nn as nn
import numpy as np
from .base_model import BaseModel

class PriceModel(BaseModel):
    def __init__(self, sequence_length: int = 24, num_classes: int = 5):  # 24 points for 1 year of 15-day intervals
        super().__init__()
        
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        
        # CNN layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Calculate the size after CNN layers
        cnn_output_size = 64 * (sequence_length // 8)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_classes)  # Changed to num_classes
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, sequence_length)
        x = x.unsqueeze(1)  # Add channel dimension
        
        # CNN feature extraction
        features = self.conv_layers(x)
        
        # Reshape for LSTM
        features = features.permute(0, 2, 1)  # (batch_size, seq_len, features)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Classification
        return self.classifier(last_output)
    
    def preprocess_price_data(self, price_data: dict) -> torch.Tensor:
        """Preprocess price data for the model."""
        if not price_data or 'prices' not in price_data:
            return torch.zeros((1, self.sequence_length))
            
        prices = np.array(price_data['prices'])
        
        # Normalize prices
        prices = (prices - prices.mean()) / (prices.std() + 1e-8)
        
        # Pad or truncate to sequence_length
        if len(prices) < self.sequence_length:
            prices = np.pad(prices, (0, self.sequence_length - len(prices)))
        else:
            prices = prices[-self.sequence_length:]
            
        return torch.tensor(prices, dtype=torch.float32).unsqueeze(0)
    
    def predict_price_data(self, price_data: dict) -> torch.Tensor:
        """Predict based on price data."""
        self.eval()
        with torch.no_grad():
            features = self.preprocess_price_data(price_data)
            return self.forward(features) 