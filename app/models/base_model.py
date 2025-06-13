import torch
import torch.nn as nn
from typing import Dict, Any

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def save(self, path: str):
        torch.save({
            'model_state_dict': self.state_dict(),
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_type': self.__class__.__name__,
            'parameters': sum(p.numel() for p in self.parameters())
        } 