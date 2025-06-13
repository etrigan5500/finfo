import yfinance as yf
import pandas as pd
from typing import Dict, List
from datetime import datetime, timedelta

class FinancialCollector:
    def __init__(self):
        pass
    
    def get_financial_data(self, ticker: str) -> Dict:
        """Get financial data for a company."""
        try:
            stock = yf.Ticker(ticker)
            
            # Get quarterly financials
            quarterly_financials = stock.quarterly_financials
            quarterly_balance_sheet = stock.quarterly_balance_sheet
            quarterly_cashflow = stock.quarterly_cashflow
            
            # Get key metrics
            info = stock.info
            
            # Extract relevant metrics
            financial_metrics = {
                'revenue': self._extract_metric(quarterly_financials, 'Total Revenue'),
                'net_income': self._extract_metric(quarterly_financials, 'Net Income'),
                'eps': self._extract_metric(quarterly_financials, 'Basic EPS'),
                'total_assets': self._extract_metric(quarterly_balance_sheet, 'Total Assets'),
                'total_liabilities': self._extract_metric(quarterly_balance_sheet, 'Total Liabilities'),
                'operating_cash_flow': self._extract_metric(quarterly_cashflow, 'Operating Cash Flow'),
                'free_cash_flow': self._extract_metric(quarterly_cashflow, 'Free Cash Flow'),
                'pe_ratio': info.get('trailingPE', None),
                'market_cap': info.get('marketCap', None),
                'dividend_yield': info.get('dividendYield', None)
            }
            
            return financial_metrics
            
        except Exception as e:
            print(f"Error getting financial data for {ticker}: {str(e)}")
            return {}
    
    def _extract_metric(self, df: pd.DataFrame, metric_name: str) -> List[float]:
        """Extract a specific metric from the financial data."""
        try:
            if metric_name in df.index:
                return df.loc[metric_name].tolist()[:4]  # Last 4 quarters
            return []
        except:
            return []
    
    def get_stock_price_data(self, ticker: str, period: str = "1y", interval: str = "15d") -> Dict:
        """Get historical stock price data."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            
            # Calculate additional metrics
            hist['Returns'] = hist['Close'].pct_change()
            hist['Volatility'] = hist['Returns'].rolling(window=4).std()
            
            return {
                'dates': hist.index.strftime('%Y-%m-%d').tolist(),
                'prices': hist['Close'].tolist(),
                'volumes': hist['Volume'].tolist(),
                'returns': hist['Returns'].tolist(),
                'volatility': hist['Volatility'].tolist()
            }
            
        except Exception as e:
            print(f"Error getting stock price data for {ticker}: {str(e)}")
            return {}
    
    def collect_all_financial_data(self, ticker: str) -> Dict:
        """Collect all financial data for a company."""
        financial_data = self.get_financial_data(ticker)
        price_data = self.get_stock_price_data(ticker)
        
        return {
            'financial_metrics': financial_data,
            'price_data': price_data
        } 