from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Stock Trend Predictor",
    description="API for predicting stock price trends based on news, financial data, and historical prices",
    version="1.0.0"
)

class StockPredictionRequest(BaseModel):
    company_name: str
    ticker_symbol: Optional[str] = None

class StockPredictionResponse(BaseModel):
    prediction_class: int  # 0-4 for the 5 classes
    prediction_label: str  # Human readable label
    confidence: float
    class_probabilities: dict  # Probabilities for each class
    news_summary: str
    financial_metrics: dict
    price_trend: dict

# Class mapping
CLASS_LABELS = {
    0: "📉 Strong Decrease",
    1: "🔻 Moderate Decrease", 
    2: "➡️ Stable",
    3: "🔺 Moderate Increase",
    4: "📈 Strong Increase"
}

CLASS_DESCRIPTIONS = {
    0: "Expected return < -2%",
    1: "Expected return -2% to -0.5%",
    2: "Expected return -0.5% to +0.5%",
    3: "Expected return +0.5% to +2%",
    4: "Expected return > +2%"
}

@app.get("/")
async def root():
    return {"message": "Welcome to Stock Trend Predictor API"}

@app.get("/classes")
async def get_classes():
    """Get information about the 5 prediction classes."""
    return {
        "classes": {
            str(k): {
                "label": v,
                "description": CLASS_DESCRIPTIONS[k]
            } for k, v in CLASS_LABELS.items()
        }
    }

@app.post("/predict", response_model=StockPredictionResponse)
async def predict_stock_trend(request: StockPredictionRequest):
    try:
        # TODO: Implement the prediction pipeline
        # 1. Collect news data
        # 2. Get financial data
        # 3. Get price data
        # 4. Run predictions through models
        # 5. Combine results
        
        # Placeholder response with 5-class prediction
        import random
        prediction_class = random.randint(0, 4)
        
        # Mock probabilities that sum to 1
        import numpy as np
        probs = np.random.dirichlet(np.ones(5))  # Generate random probabilities that sum to 1
        class_probabilities = {CLASS_LABELS[i]: float(probs[i]) for i in range(5)}
        
        return StockPredictionResponse(
            prediction_class=prediction_class,
            prediction_label=CLASS_LABELS[prediction_class],
            confidence=float(max(probs)),
            class_probabilities=class_probabilities,
            news_summary="Sample news summary",
            financial_metrics={"revenue": 1000000},
            price_trend={"trend": "upward"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 