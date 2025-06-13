import yfinance as yf
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta

def get_ticker_from_company_name(company_name: str) -> Optional[str]:
    """Get ticker symbol from company name using yfinance."""
    try:
        # Search for the company
        search_results = yf.Ticker(company_name)
        if search_results.info:
            return search_results.info.get('symbol')
    except:
        pass
    return None

def classify_return(return_pct: float) -> int:
    """Classify return percentage into 5 categories."""
    if return_pct < -0.02:  # < -2%
        return 0  # Strong Decrease
    elif return_pct < -0.005:  # -2% to -0.5%
        return 1  # Moderate Decrease
    elif return_pct <= 0.005:  # -0.5% to +0.5%
        return 2  # Stable
    elif return_pct <= 0.02:  # +0.5% to +2%
        return 3  # Moderate Increase
    else:  # > +2%
        return 4  # Strong Increase

def get_class_label(class_id: int) -> str:
    """Get human-readable label for class ID."""
    labels = {
        0: "📉 Strong Decrease",
        1: "🔻 Moderate Decrease", 
        2: "➡️ Stable",
        3: "🔺 Moderate Increase",
        4: "📈 Strong Increase"
    }
    return labels.get(class_id, "Unknown")

def get_class_description(class_id: int) -> str:
    """Get description for class ID."""
    descriptions = {
        0: "Expected return < -2%",
        1: "Expected return -2% to -0.5%",
        2: "Expected return -0.5% to +0.5%",
        3: "Expected return +0.5% to +2%",
        4: "Expected return > +2%"
    }
    return descriptions.get(class_id, "Unknown")

def calculate_technical_indicators(price_data: pd.DataFrame) -> Dict:
    """Calculate technical indicators from price data."""
    # Calculate moving averages
    price_data['SMA_20'] = price_data['Close'].rolling(window=20).mean()
    price_data['SMA_50'] = price_data['Close'].rolling(window=50).mean()
    
    # Calculate RSI
    delta = price_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    price_data['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    exp1 = price_data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = price_data['Close'].ewm(span=26, adjust=False).mean()
    price_data['MACD'] = exp1 - exp2
    price_data['Signal_Line'] = price_data['MACD'].ewm(span=9, adjust=False).mean()
    
    return {
        'sma_20': price_data['SMA_20'].iloc[-1],
        'sma_50': price_data['SMA_50'].iloc[-1],
        'rsi': price_data['RSI'].iloc[-1],
        'macd': price_data['MACD'].iloc[-1],
        'signal_line': price_data['Signal_Line'].iloc[-1]
    }

def format_financial_metrics(metrics: Dict) -> Dict:
    """Format financial metrics for display."""
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if abs(value) >= 1e9:
                formatted[key] = f"${value/1e9:.2f}B"
            elif abs(value) >= 1e6:
                formatted[key] = f"${value/1e6:.2f}M"
            elif abs(value) >= 1e3:
                formatted[key] = f"${value/1e3:.2f}K"
            else:
                formatted[key] = f"${value:.2f}"
        else:
            formatted[key] = value
    return formatted

def calculate_prediction_confidence(probabilities: List[float]) -> float:
    """Calculate overall confidence score from class probabilities."""
    # Confidence is the maximum probability
    return max(probabilities)

def interpret_prediction_result(prediction_result: Dict) -> Dict:
    """Interpret and enhance prediction result with additional insights."""
    class_id = prediction_result['prediction_class']
    probabilities = prediction_result['class_probabilities']
    
    # Calculate confidence metrics
    confidence = max(probabilities)
    uncertainty = 1 - confidence
    
    # Determine if prediction is strong or weak
    strength = "Strong" if confidence > 0.7 else "Moderate" if confidence > 0.5 else "Weak"
    
    # Calculate expected return range
    expected_ranges = {
        0: "< -2%",
        1: "-2% to -0.5%", 
        2: "-0.5% to +0.5%",
        3: "+0.5% to +2%",
        4: "> +2%"
    }
    
    return {
        'predicted_class': class_id,
        'predicted_label': get_class_label(class_id),
        'expected_return_range': expected_ranges[class_id],
        'confidence': confidence,
        'uncertainty': uncertainty,
        'prediction_strength': strength,
        'all_probabilities': {
            get_class_label(i): prob for i, prob in enumerate(probabilities)
        }
    }

def get_date_range(days: int) -> tuple:
    """Get start and end dates for data collection."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d') 