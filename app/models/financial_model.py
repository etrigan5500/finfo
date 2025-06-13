import torch
import torch.nn as nn
import numpy as np
from .base_model import BaseModel

class FinancialModel(BaseModel):
    def __init__(self, input_size: int = 10, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.classifier(features)
    
    def preprocess_financial_data(self, financial_data: dict) -> torch.Tensor:
        """Preprocess financial data for the model."""
        # Extract and normalize features
        features = []
        
        # Revenue growth
        if 'revenue' in financial_data and len(financial_data['revenue']) >= 2:
            revenue_growth = (financial_data['revenue'][0] - financial_data['revenue'][1]) / financial_data['revenue'][1]
            features.append(revenue_growth)
        else:
            features.append(0.0)
            
        # Net income growth
        if 'net_income' in financial_data and len(financial_data['net_income']) >= 2:
            net_income_growth = (financial_data['net_income'][0] - financial_data['net_income'][1]) / abs(financial_data['net_income'][1])
            features.append(net_income_growth)
        else:
            features.append(0.0)
            
        # EPS growth
        if 'eps' in financial_data and len(financial_data['eps']) >= 2:
            eps_growth = (financial_data['eps'][0] - financial_data['eps'][1]) / abs(financial_data['eps'][1])
            features.append(eps_growth)
        else:
            features.append(0.0)
            
        # Add other financial metrics
        metrics = ['pe_ratio', 'market_cap', 'dividend_yield']
        for metric in metrics:
            if metric in financial_data and financial_data[metric] is not None:
                features.append(float(financial_data[metric]))
            else:
                features.append(0.0)
                
        # Convert to tensor and normalize
        features = torch.tensor(features, dtype=torch.float32)
        features = (features - features.mean()) / (features.std() + 1e-8)
        
        return features.unsqueeze(0)  # Add batch dimension
    
    def predict_financial_data(self, financial_data: dict) -> torch.Tensor:
        """Predict based on financial data."""
        self.eval()
        with torch.no_grad():
            features = self.preprocess_financial_data(financial_data)
            return self.forward(features) 