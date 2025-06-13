# Stock Prediction Model Training Guide

This guide covers everything needed to train the stock prediction models, including Google Colab setup and prerequisites.

## 🎯 Overview

Train a 5-class stock prediction system using:
- **News sentiment analysis** (DistilBERT)
- **Financial metrics analysis** (Neural network)  
- **Price trend analysis** (CNN-LSTM hybrid)
- **Ensemble combination** for final prediction

**Prediction Classes:**
- 📉 Strong Decrease (< -2%)
- 🔻 Moderate Decrease (-2% to -0.5%)
- ➡️ Stable (-0.5% to +0.5%)
- 🔺 Moderate Increase (+0.5% to +2%)
- 📈 Strong Increase (> +2%)

## 📋 Prerequisites

### 1. Google Custom Search API Setup (Optional but Recommended)

**Why needed?** Real news data for training (vs synthetic data)

**Setup Steps:**
1. **Google Cloud Console:**
   - Go to [console.cloud.google.com](https://console.cloud.google.com/)
   - Create/select project
   - Enable "Custom Search API" in APIs & Services
   - Create API key in Credentials

2. **Custom Search Engine:**
   - Go to [cse.google.com](https://cse.google.com/cse/)
   - Create search engine with `*` (entire web) or specific news sites
   - Copy the Search Engine ID (cx parameter)

**Cost:** FREE - 100 requests/day (enough for 2.5 training runs daily)

### 2. Hardware Requirements

**Minimum:**
- 4GB RAM
- 2GB storage  
- CPU training (slow but works)

**Recommended:**
- 8GB+ RAM
- 5GB storage
- GPU (Google Colab provides free GPU access)

## 🚀 Google Colab Training (Recommended)

### Step 1: Open Training Notebook

1. Go to [Google Colab](https://colab.research.google.com/)
2. **Option A - Direct Upload:**
   - Download `training/notebooks/train_models.ipynb`
   - Upload to Colab via "File" → "Upload notebook"

3. **Option B - GitHub Integration:**
   - In Colab: "File" → "Open notebook" → "GitHub"
   - Enter repository URL: `https://github.com/your-repo/stock-predictor`
   - Select `training/notebooks/train_models.ipynb`

### Step 2: Enable GPU (Free Tier)

```python
# In Colab: Runtime → Change runtime type → GPU → Save
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
```

### Step 3: Install Dependencies

The notebook automatically installs required packages:

```python
!pip install -q torch transformers pandas numpy scikit-learn yfinance sentence-transformers python-dotenv beautifulsoup4 requests
```

### Step 4: Configure API Keys (Optional)

For real news data, set up your API credentials:

```python
# Set your Google Custom Search API credentials
GOOGLE_API_KEY = "your_google_api_key_here"  # or None for synthetic data
GOOGLE_SEARCH_ENGINE_ID = "your_search_engine_id_here"  # or None for synthetic data
```

**Without API keys:** Uses synthetic news data (still effective for training)

### Step 5: Choose Training Configuration

Select data collection strategy:

```python
# Choose configuration:
SELECTED_CONFIG = 'balanced_1y'  # Recommended

# Available options:
# 'efficient_6mo'  - 6 months, weekly data (fast)
# 'balanced_1y'    - 1 year, weekly data (recommended)  
# 'detailed_6mo'   - 6 months, daily data (detailed)
# 'original_2y'    - 2 years, daily data (comprehensive)
```

### Step 6: Run Training

Execute all cells in sequence. The process includes:

1. **Data Collection** (~5-10 minutes)
   - Stock prices from yfinance
   - News articles (real or synthetic)
   - Financial metrics

2. **Model Training** (~10-20 minutes on GPU)
   - Individual model training
   - Ensemble combination
   - Validation and checkpointing

3. **Model Saving**
   - Trained models saved to `/content/models/`
   - Download for local use

## 💻 Local Training

### Step 1: Environment Setup

```bash
# Clone repository
git clone https://github.com/your-repo/stock-predictor
cd stock-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment

Create `.env` file:

```bash
# Optional - for real news data
GOOGLE_API_KEY=your_google_api_key
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id

# Optional - for other features
GEMINI_API_KEY=your_gemini_api_key
```

### Step 3: Run Training

```bash
# Start Jupyter notebook
jupyter notebook training/notebooks/train_models.ipynb

# Or convert to Python script and run
jupyter nbconvert --to script training/notebooks/train_models.ipynb
python train_models.py
```

## 📊 Training Configurations Explained

| Config | Data Period | Interval | Sequence Length | Use Case |
|--------|-------------|----------|-----------------|----------|
| `efficient_6mo` | 6 months | Weekly | 12 | Quick experiments |
| `balanced_1y` | 1 year | Weekly | 24 | **Recommended** |
| `detailed_6mo` | 6 months | Daily | 12 | High resolution |
| `original_2y` | 2 years | Daily | 24 | Maximum data |

**Recommendation:** Use `balanced_1y` for best balance of performance and training time.

## 🔧 Troubleshooting

### Common Issues

**Out of Memory:**
- Reduce batch size from 16 to 8 or 4
- Use `efficient_6mo` configuration
- In Colab: Runtime → Factory reset runtime

**Slow Training:**
- Ensure GPU is enabled in Colab
- Use smaller configurations for testing
- Consider reducing `num_samples_per_ticker`

**API Quota Exceeded:**
- Google Custom Search: Wait for daily reset
- Use synthetic news data temporarily
- Reduce number of companies in training

**CUDA Errors:**
- Restart runtime in Colab
- Ensure PyTorch with CUDA support is installed
- Fallback to CPU training (set device='cpu')

### Performance Optimization

**Speed up data collection:**
```python
# Reduce companies for testing
tech_companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Only 5 companies

# Reduce samples per company
num_samples_per_ticker = 20  # Instead of 50
```

**Faster training:**
```python
# Reduce epochs
num_epochs = 3  # Instead of 5

# Increase batch size (if memory allows)
batch_size = 32  # Instead of 16
```

## 📁 Output Files

After training, you'll have:

```
models/
├── ensemble_model.pth      # Main ensemble model
├── news_model.pth         # News sentiment model
├── financial_model.pth    # Financial metrics model
├── price_model.pth        # Price trend model
└── best_ensemble_model.pth # Best checkpoint
```

## 📤 Downloading from Google Colab

```python
# Download trained models
from google.colab import files

# Download all models
files.download('ensemble_model.pth')
files.download('news_model.pth') 
files.download('financial_model.pth')
files.download('price_model.pth')

# Or zip and download all at once
!zip -r models.zip *.pth
files.download('models.zip')
```

## 🎯 Next Steps

After training:
1. **Download models** from Colab (if used)
2. **Set up prediction environment** (see Prediction Guide)
3. **Test predictions** on new companies
4. **Deploy API** for production use

## 📚 Additional Resources

- [Google Colab Guide](https://colab.research.google.com/notebooks/intro.ipynb)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Google Custom Search Setup](./google_search_setup.md) 