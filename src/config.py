# config.py
import os
from pathlib import Path
import streamlit as st

# API Configuration
def get_api_key():
    """Get API key from environment or Streamlit secrets"""
    # Try environment variable first
    api_key = os.getenv('FINNHUB_API_KEY')
    
    # Try Streamlit secrets if available
    if not api_key:
        try:
            api_key = st.secrets.get("api_keys", {}).get("finnhub")
        except:
            pass
    
    if not api_key:
        raise ValueError("FINNHUB_API_KEY not found in environment or secrets")
    
    return api_key

# File paths
PROJECT_ROOT = Path(__file__).parent.parent if __file__ else Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

TREND_FILE = DATA_DIR / "trend_df.csv"
LOG_FILE = DATA_DIR / "trend_log.csv"
SECTOR_LOG_FILE = DATA_DIR / "sector_momentum_log.csv"

# App settings
REFRESH_INTERVAL = 300  # 5 minutes
TOP_TICKERS = 30
TOP_CRYPTOS = 5

# Stock tickers (top 30 S&P 500)
TOP_SP500_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'JPM', 'JNJ',
    'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'BAC', 'ADBE', 'CMCSA',
    'NFLX', 'XOM', 'VZ', 'T', 'PFE', 'INTC', 'CVX', 'WMT', 'CSCO', 'ABT'
]

# Crypto symbols
CRYPTO_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT']

# Sector mapping
SECTOR_MAPPING = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 
    'AMZN': 'Consumer Discretionary', 'NVDA': 'Technology', 'META': 'Technology',
    'TSLA': 'Consumer Discretionary', 'BRK.B': 'Financials', 'JPM': 'Financials',
    'JNJ': 'Healthcare', 'V': 'Financials', 'PG': 'Consumer Staples',
    'UNH': 'Healthcare', 'HD': 'Consumer Discretionary', 'MA': 'Financials',
    'DIS': 'Consumer Discretionary', 'PYPL': 'Financials', 'BAC': 'Financials',
    'ADBE': 'Technology', 'CMCSA': 'Communication Services', 'NFLX': 'Communication Services',
    'XOM': 'Energy', 'VZ': 'Communication Services', 'T': 'Communication Services',
    'PFE': 'Healthcare', 'INTC': 'Technology', 'CVX': 'Energy',
    'WMT': 'Consumer Staples', 'CSCO': 'Technology', 'ABT': 'Healthcare'
}

# Finnhub API endpoints
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Data validation
MIN_VALID_PRICE = 0.01
MAX_VALID_PRICE = 1000000
MAX_MOMENTUM_PERCENT = 50  # Cap extreme momentum values