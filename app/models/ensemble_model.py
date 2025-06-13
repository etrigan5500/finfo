import torch
import torch.nn as nn
from .base_model import BaseModel
from .news_model import NewsModel
from .financial_model import FinancialModel
from .price_model import PriceModel

class EnsembleModel(BaseModel):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        
        # Initialize individual models
        self.news_model = NewsModel(num_classes=num_classes)
        self.financial_model = FinancialModel(num_classes=num_classes)
        self.price_model = PriceModel(num_classes=num_classes)
        
        # Ensemble classifier - takes concatenated features from all models
        self.ensemble = nn.Sequential(
            nn.Linear(num_classes * 3, 32),  # 5*3 = 15 input features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_classes)  # Final 5-class output
        )
        
    def forward(self, news_input_ids, news_attention_mask, financial_data, price_data):
        # Get predictions from individual models
        news_pred = self.news_model.forward(news_input_ids, news_attention_mask)
        financial_pred = self.financial_model.predict_financial_data(financial_data)
        price_pred = self.price_model.predict_price_data(price_data)
        
        # Combine predictions
        combined = torch.cat([news_pred, financial_pred, price_pred], dim=1)
        
        # Final prediction
        return self.ensemble(combined)
    
    def predict(self, news_texts: list, financial_data: dict, price_data: dict) -> dict:
        """Make a prediction using all three models."""
        self.eval()
        with torch.no_grad():
            # Preprocess news data
            news_input = self.news_model.preprocess_text(news_texts)
            
            # Get predictions
            prediction_logits = self.forward(
                news_input['input_ids'], 
                news_input['attention_mask'], 
                financial_data, 
                price_data
            )
            
            # Convert to probabilities
            probabilities = torch.softmax(prediction_logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]
            
            # Get individual model predictions for analysis
            news_pred = self.news_model.predict_text(news_texts)
            financial_pred = self.financial_model.predict_financial_data(financial_data)
            price_pred = self.price_model.predict_price_data(price_data)
            
            # Convert individual predictions to class probabilities
            news_probs = torch.softmax(news_pred, dim=1)
            financial_probs = torch.softmax(financial_pred, dim=1)
            price_probs = torch.softmax(price_pred, dim=1)
            
            return {
                'prediction_class': predicted_class.item(),
                'confidence': confidence.item(),
                'class_probabilities': probabilities.squeeze().tolist(),
                'individual_predictions': {
                    'news': news_probs.squeeze().tolist(),
                    'financial': financial_probs.squeeze().tolist(),
                    'price': price_probs.squeeze().tolist()
                }
            }
    
    def save(self, path: str):
        """Save all models."""
        torch.save({
            'ensemble_state_dict': self.ensemble.state_dict(),
            'news_model_state_dict': self.news_model.state_dict(),
            'financial_model_state_dict': self.financial_model.state_dict(),
            'price_model_state_dict': self.price_model.state_dict(),
            'num_classes': self.num_classes
        }, path)
    
    def load(self, path: str):
        """Load all models."""
        checkpoint = torch.load(path)
        self.ensemble.load_state_dict(checkpoint['ensemble_state_dict'])
        self.news_model.load_state_dict(checkpoint['news_model_state_dict'])
        self.financial_model.load_state_dict(checkpoint['financial_model_state_dict'])
        self.price_model.load_state_dict(checkpoint['price_model_state_dict']) 