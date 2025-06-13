import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys - Updated to use Google Custom Search JSON API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')  # Google Custom Search JSON API key
GOOGLE_SEARCH_ENGINE_ID = os.getenv('GOOGLE_SEARCH_ENGINE_ID')  # Custom Search Engine ID (cx)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Model paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'training', 'models')
PRICE_MODEL_PATH = os.path.join(MODEL_DIR, 'price_model.pth')
NEWS_MODEL_PATH = os.path.join(MODEL_DIR, 'news_model.pth')
FINANCIAL_MODEL_PATH = os.path.join(MODEL_DIR, 'financial_model.pth')
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, 'ensemble_model.pth')

# Data collection settings - Updated for Google Custom Search
NEWS_SEARCH_QUERIES = [
    "{company} financial news",
    "{company} stock prediction news"  # Reduced from 3 to 2 queries for efficiency
]
NUM_NEWS_RESULTS = 20  # 10 per query (API limit) = 20 total
NUM_NEWS_CLUSTERS = 3

# Financial data settings
FINANCIAL_METRICS = [
    'revenue',
    'net_income',
    'eps',
    'total_assets',
    'total_liabilities',
    'operating_cash_flow',
    'free_cash_flow',
    'pe_ratio',
    'market_cap',
    'dividend_yield'
]

# Price data settings
PRICE_HISTORY_DAYS = 365
PRICE_INTERVAL_DAYS = 15

# Model settings
NEWS_MODEL_NAME = 'distilbert-base-uncased'
PRICE_SEQUENCE_LENGTH = 24  # 24 points for 1 year of 15-day intervals
FINANCIAL_INPUT_SIZE = len(FINANCIAL_METRICS)
NUM_CLASSES = 5  # 5-class prediction system

# Class definitions
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

# Return thresholds for classification
RETURN_THRESHOLDS = {
    'strong_decrease': -0.02,  # < -2%
    'moderate_decrease': -0.005,  # -2% to -0.5%
    'stable_upper': 0.005,  # -0.5% to +0.5%
    'moderate_increase': 0.02,  # +0.5% to +2%
    # Strong increase: > +2%
}

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000

# Google Custom Search API settings
GOOGLE_SEARCH_BASE_URL = "https://www.googleapis.com/customsearch/v1"
GOOGLE_SEARCH_SAFE = "active"
GOOGLE_SEARCH_DATE_RESTRICT = "m1"  # Last month for relevancy
GOOGLE_SEARCH_FIELDS = "items(title,snippet,link,displayLink)"

# News site filters for better results
NEWS_SITE_FILTERS = [
    "site:reuters.com",
    "site:bloomberg.com", 
    "site:cnbc.com",
    "site:marketwatch.com",
    "site:finance.yahoo.com"
]

# API settings
API_WORKERS = 4 