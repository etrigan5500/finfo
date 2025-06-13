# Stock Trend Predictor

A machine learning application that predicts stock price trends using ensemble models trained on news sentiment, financial metrics, and historical price data.

## 🎯 What It Does

Predicts stock movements in **5 classes** with confidence scores:
- 📉 **Strong Decrease** (< -2%)
- 🔻 **Moderate Decrease** (-2% to -0.5%)
- ➡️ **Stable** (-0.5% to +0.5%)
- 🔺 **Moderate Increase** (+0.5% to +2%)
- 📈 **Strong Increase** (> +2%)

## 🏗️ Architecture

**3 Specialized Models + Ensemble:**
1. **News Model** - DistilBERT sentiment analysis on financial news
2. **Financial Model** - Neural network on company fundamentals  
3. **Price Model** - CNN-LSTM hybrid on historical price trends
4. **Ensemble Model** - Combines all three for final prediction

## 📁 Project Structure

```
stock_predictor/
├── 📖 docs/                    # Documentation guides
│   ├── training_guide.md      # Complete training setup
│   ├── prediction_guide.md    # Using trained models
│   └── google_search_setup.md # API setup guide
├── 🚀 app/                     # Production API
│   ├── api/                   # FastAPI endpoints
│   ├── models/                # Model implementations
│   └── utils/                 # Helper functions
├── 📊 training/               # Model training
│   ├── notebooks/             # Google Colab training
│   └── models/                # Saved model weights
├── 🔧 data_collection/        # Data gathering
│   ├── news/                  # News article collection
│   ├── financial/             # Financial data
│   └── price/                 # Stock price data
└── ⚙️ config/                  # Configuration files
```

## 🚀 Quick Start

### Option 1: Use Pre-trained Models
```bash
# Clone and setup
git clone https://github.com/your-repo/stock-predictor
cd stock-predictor
pip install -r requirements.txt

# Add your trained models to training/models/
# Start API server
uvicorn app.api.main:app --reload

# Get prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"company_name": "Apple Inc", "ticker_symbol": "AAPL"}'
```

### Option 2: Train Your Own Models
```bash
# 1. Setup Google Custom Search API (optional, for real news)
# 2. Open training/notebooks/train_models.ipynb in Google Colab
# 3. Run all cells to train models
# 4. Download trained models
# 5. Use for predictions
```

## 📚 Documentation

| Guide | Purpose | When to Use |
|-------|---------|-------------|
| **[Training Guide](docs/training_guide.md)** | Complete model training setup | Want to train your own models |
| **[Prediction Guide](docs/prediction_guide.md)** | Using trained models | Want to get stock predictions |
| **[Google Search Setup](docs/google_search_setup.md)** | API configuration | Need real news data for training |

## 🌟 Key Features

### 💰 Cost-Effective Data Collection
- **Google Custom Search**: 100 FREE requests/day (vs SerpAPI's 100/month)
- **yfinance**: FREE stock price data
- **Synthetic news fallback**: Works without API keys

### 🎯 Efficient Training Options
- **efficient_6mo**: 6 months data, quick training
- **balanced_1y**: 1 year data, recommended balance
- **detailed_6mo**: 6 months daily data, high resolution  
- **original_2y**: 2 years data, maximum coverage

### 🔧 Flexible Deployment
- **REST API**: Production-ready FastAPI server
- **Direct Python**: Import and use models directly
- **Jupyter Notebooks**: Interactive exploration
- **Google Colab**: Free GPU training

## 📊 Training Data Sources

1. **News Articles** (Google Custom Search JSON API)
   - Financial news and stock predictions
   - Semantic clustering for diversity
   - 40 API calls per training run (20 companies × 2 queries)

2. **Financial Metrics** (yfinance)
   - Revenue, earnings, P/E ratios
   - Company fundamentals
   - FREE unlimited access

3. **Price History** (yfinance)  
   - 6 months to 2 years of data
   - Multiple time intervals (daily/weekly)
   - FREE unlimited access

## 🎯 Use Cases

- **Investment Research**: Get AI-powered stock predictions
- **Portfolio Management**: Batch analyze multiple stocks
- **Trading Signals**: Integrate with trading systems via API
- **Educational**: Learn ML applied to finance
- **Research**: Experiment with ensemble methods

## 🔧 Requirements

**For Training:**
- Python 3.8+
- 4GB+ RAM (8GB+ recommended)
- Google Colab (free GPU) or local GPU

**For Predictions:**
- Python 3.8+
- 2GB+ RAM
- Trained model files

## 🚀 Getting Started

1. **Training Models**: Follow the **[Training Guide](docs/training_guide.md)**
2. **Using Models**: Follow the **[Prediction Guide](docs/prediction_guide.md)**
3. **API Setup**: See **[Google Search Setup](docs/google_search_setup.md)**

## 🎉 Benefits Over Alternatives

| Feature | This Project | Traditional Approaches |
|---------|-------------|----------------------|
| **Data Cost** | FREE (Google Custom Search) | Expensive (Bloomberg, Reuters APIs) |
| **Training** | Google Colab (free GPU) | Requires expensive cloud resources |
| **Models** | 3 specialized + ensemble | Single model approaches |
| **News Processing** | Semantic clustering | Manual curation |
| **Deployment** | Ready-to-use API | Custom infrastructure needed |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Stock predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with qualified financial advisors and do your own research before making investment choices. 