import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from .base_model import BaseModel

class NewsModel(BaseModel):
    def __init__(self, model_name: str = 'distilbert-base-uncased', num_classes: int = 5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.num_classes = num_classes
        
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier(pooled_output)
    
    def preprocess_text(self, texts: list) -> dict:
        """Preprocess text for the model."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
    
    def predict_text(self, texts: list) -> torch.Tensor:
        """Predict sentiment for a list of texts."""
        self.eval()
        with torch.no_grad():
            inputs = self.preprocess_text(texts)
            return self.forward(inputs['input_ids'], inputs['attention_mask']) 