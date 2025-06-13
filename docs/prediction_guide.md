# Stock Prediction Usage Guide

This guide covers how to use trained models to get stock predictions for any company.

## 🎯 Quick Start

Get predictions in 3 ways:
1. **REST API** (recommended for applications)
2. **Python script** (direct model usage)
3. **Interactive notebook** (exploration and testing)

## 🚀 Method 1: REST API (Recommended)

### Step 1: Setup Environment

```bash
# Clone repository (if not already done)
git clone https://github.com/your-repo/stock-predictor
cd stock-predictor

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Place Trained Models

Put your trained model files in the correct location:

```
training/models/
├── ensemble_model.pth      # Main model (required)
├── news_model.pth         # Individual models (optional)
├── financial_model.pth    
└── price_model.pth        
```

### Step 3: Configure Environment

Create `.env` file (optional for enhanced features):

```bash
# Optional - for real-time news data
GOOGLE_API_KEY=your_google_api_key
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id

# Optional - for enhanced analysis
GEMINI_API_KEY=your_gemini_api_key
```

### Step 4: Start API Server

```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 5: Make Predictions

**Using curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"company_name": "Apple Inc", "ticker_symbol": "AAPL"}'
```

**Using Python requests:**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"company_name": "Apple Inc", "ticker_symbol": "AAPL"}
)

prediction = response.json()
print(f"Prediction: {prediction['prediction_label']}")
print(f"Confidence: {prediction['confidence']:.2%}")
print(f"Class probabilities: {prediction['class_probabilities']}")
```

**Response format:**
```json
{
  "prediction_class": 3,
  "prediction_label": "🔺 Moderate Increase",
  "confidence": 0.75,
  "class_probabilities": {
    "0": 0.05,
    "1": 0.10,
    "2": 0.10,
    "3": 0.75,
    "4": 0.00
  },
  "news_summary": "Recent news about Apple...",
  "financial_metrics": {...},
  "price_trend": {...}
}
```

## 💻 Method 2: Direct Python Usage

### Step 1: Load Models

```python
import torch
from app.models.ensemble_model import EnsembleModel
from app.models.news_model import NewsModel
from transformers import AutoTokenizer
import yfinance as yf
import numpy as np

# Load the ensemble model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EnsembleModel(num_classes=5)
model.load_state_dict(torch.load('training/models/ensemble_model.pth', map_location=device)['model_state_dict'])
model.eval()

# Load tokenizer for news processing
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
```

### Step 2: Prepare Data

```python
def prepare_prediction_data(ticker, company_name):
    """Prepare data for a single prediction."""
    
    # 1. Get price data
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y", interval="1wk")
    price_data = hist['Close'].resample('15D').last().dropna()[-24:].tolist()
    
    # Normalize price data
    price_tensor = torch.tensor(price_data, dtype=torch.float32)
    price_tensor = (price_tensor - price_tensor.mean()) / (price_tensor.std() + 1e-8)
    
    # 2. Get financial data (simplified example)
    info = stock.info
    financial_metrics = [
        0.05,  # Revenue growth (example)
        0.03,  # Net income growth
        0.04,  # EPS growth
        info.get('trailingPE', 15) / 100 if info.get('trailingPE') else 0.15,
        0.02,  # Debt-to-equity
        0.06   # ROE
    ]
    financial_tensor = torch.tensor(financial_metrics, dtype=torch.float32)
    financial_tensor = (financial_tensor - financial_tensor.mean()) / (financial_tensor.std() + 1e-8)
    
    # 3. Get news data (example with synthetic news)
    news_text = f"{company_name} reports steady growth with positive market outlook."
    news_inputs = tokenizer(
        news_text,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    return {
        'price_data': price_tensor.unsqueeze(0),
        'financial_data': financial_tensor.unsqueeze(0),
        'news_input_ids': news_inputs['input_ids'],
        'news_attention_mask': news_inputs['attention_mask']
    }
```

### Step 3: Make Predictions

```python
def predict_stock(ticker, company_name):
    """Get stock prediction for a company."""
    
    # Prepare data
    data = prepare_prediction_data(ticker, company_name)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(
            data['news_input_ids'],
            data['news_attention_mask'], 
            data['financial_data'],
            data['price_data']
        )
        
        # Get probabilities and prediction
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Class labels
    class_labels = {
        0: "📉 Strong Decrease",
        1: "🔻 Moderate Decrease", 
        2: "➡️ Stable",
        3: "🔺 Moderate Increase",
        4: "📈 Strong Increase"
    }
    
    return {
        'ticker': ticker,
        'company': company_name,
        'predicted_class': predicted_class,
        'prediction_label': class_labels[predicted_class],
        'confidence': confidence,
        'all_probabilities': probabilities[0].tolist()
    }

# Example usage
result = predict_stock('AAPL', 'Apple Inc')
print(f"Company: {result['company']}")
print(f"Prediction: {result['prediction_label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 📓 Method 3: Interactive Jupyter Notebook

### Create Prediction Notebook

```python
# Create a new notebook: prediction_demo.ipynb

# Cell 1: Setup
import torch
from app.models.ensemble_model import EnsembleModel
from transformers import AutoTokenizer
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Cell 2: Load Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EnsembleModel(num_classes=5)
checkpoint = torch.load('training/models/ensemble_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Cell 3: Interactive Prediction Function
def interactive_predict(ticker):
    """Interactive prediction with visualization."""
    
    # Get company info
    stock = yf.Ticker(ticker)
    info = stock.info
    company_name = info.get('longName', ticker)
    
    print(f"Analyzing: {company_name} ({ticker})")
    print("-" * 50)
    
    # Get and plot price data
    hist = stock.history(period="1y")
    plt.figure(figsize=(12, 6))
    plt.plot(hist.index, hist['Close'], label='Close Price')
    plt.title(f'{company_name} - 1 Year Price History')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Make prediction (using simplified data preparation)
    result = predict_stock(ticker, company_name)
    
    # Display results
    print(f"\n🎯 Prediction: {result['prediction_label']}")
    print(f"📊 Confidence: {result['confidence']:.1%}")
    
    # Show probability distribution
    classes = ['Strong Decrease', 'Moderate Decrease', 'Stable', 'Moderate Increase', 'Strong Increase']
    probs = result['all_probabilities']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, probs, color=['red', 'orange', 'gray', 'lightgreen', 'green'])
    plt.title('Prediction Probability Distribution')
    plt.ylabel('Probability')
    plt.xticks(rotation=45)
    
    # Highlight predicted class
    bars[result['predicted_class']].set_color('blue')
    plt.tight_layout()
    plt.show()
    
    return result

# Cell 4: Test Multiple Companies
companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

results = []
for ticker in companies:
    try:
        result = interactive_predict(ticker)
        results.append(result)
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")

# Display summary
summary_df = pd.DataFrame([
    {
        'Company': r['company'],
        'Ticker': r['ticker'],
        'Prediction': r['prediction_label'],
        'Confidence': f"{r['confidence']:.1%}"
    }
    for r in results
])

print("\n📈 Prediction Summary:")
print(summary_df.to_string(index=False))
```

## 🔧 Advanced Usage

### Batch Predictions

```python
def batch_predict(tickers):
    """Predict multiple stocks at once."""
    results = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            company_name = info.get('longName', ticker)
            
            result = predict_stock(ticker, company_name)
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    
    return results

# Example usage
tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
predictions = batch_predict(tech_stocks)

# Sort by confidence
predictions.sort(key=lambda x: x['confidence'], reverse=True)

for pred in predictions:
    print(f"{pred['ticker']}: {pred['prediction_label']} ({pred['confidence']:.1%})")
```

### Custom News Integration

```python
def predict_with_custom_news(ticker, news_articles):
    """Make prediction with custom news articles."""
    
    # Combine news articles
    combined_news = " ".join(news_articles)
    
    # Get other data as usual
    stock = yf.Ticker(ticker)
    # ... (price and financial data preparation)
    
    # Use custom news
    news_inputs = tokenizer(
        combined_news,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # Make prediction with custom news
    # ... (prediction logic)
    
    return result

# Example usage
custom_news = [
    "Apple reports record Q4 earnings beating analyst expectations",
    "iPhone sales surge in emerging markets driving revenue growth",
    "Apple announces major AI initiatives for upcoming products"
]

result = predict_with_custom_news('AAPL', custom_news)
```

## 📊 Understanding Predictions

### Class Definitions

| Class | Label | Return Range | Interpretation |
|-------|-------|-------------|----------------|
| 0 | 📉 Strong Decrease | < -2% | Significant decline expected |
| 1 | 🔻 Moderate Decrease | -2% to -0.5% | Mild decline expected |
| 2 | ➡️ Stable | -0.5% to +0.5% | Little change expected |
| 3 | 🔺 Moderate Increase | +0.5% to +2% | Mild growth expected |
| 4 | 📈 Strong Increase | > +2% | Significant growth expected |

### Confidence Interpretation

- **> 80%**: Very confident prediction
- **60-80%**: Confident prediction  
- **40-60%**: Moderate confidence
- **< 40%**: Low confidence (consider other factors)

## 🛠️ Troubleshooting

### Model Loading Issues

```python
# If model files are missing
print("Available model files:")
import os
model_dir = 'training/models/'
if os.path.exists(model_dir):
    for file in os.listdir(model_dir):
        if file.endswith('.pth'):
            print(f"  - {file}")
else:
    print("Models directory not found. Please train models first.")
```

### Data Collection Issues

```python
# Test data availability
def test_data_availability(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        info = stock.info
        
        print(f"✅ {ticker}: {len(hist)} price points, company info available: {bool(info)}")
        return True
    except Exception as e:
        print(f"❌ {ticker}: {e}")
        return False

# Test multiple tickers
test_tickers = ['AAPL', 'MSFT', 'INVALID']
for ticker in test_tickers:
    test_data_availability(ticker)
```

### Performance Issues

- Use `torch.jit.script()` for faster inference
- Batch multiple predictions together
- Cache yfinance data for repeated queries
- Use CPU if GPU memory is limited

## 🎯 Production Deployment

For production use:

1. **Use the REST API** for scalability
2. **Add authentication** for security
3. **Implement caching** for performance
4. **Add monitoring** for reliability
5. **Use Docker** for deployment

See the main README for deployment instructions. 