# src/market_analyzer.py
import finnhub
import pandas as pd
from datetime import datetime
import yfinance as yf

from src.config import (
    TOP_SP500_TICKERS, CRYPTO_SYMBOLS, SECTOR_MAPPING,
    TREND_FILE, LOG_FILE, SECTOR_LOG_FILE
)
from src.utils import rate_limited_retry, safe_api_call, save_data_safely, logger

class MarketAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = finnhub.Client(api_key=api_key)
        
    @rate_limited_retry(max_retries=3, delay=1)
    def get_stock_quote(self, ticker):
        """Get stock quote with error handling"""
        return self.client.quote(ticker)
    
    def calculate_momentum(self, current_price, previous_close):
        """Calculate price momentum"""
        if previous_close and previous_close > 0:
            return ((current_price - previous_close) / previous_close) * 100
        return 0
    
    def analyze_stocks(self):
        """Analyze stock trends and sectors"""
        results = []
        
        for ticker in TOP_SP500_TICKERS:
            try:
                quote = self.get_stock_quote(ticker)
                
                if quote and 'c' in quote and 'pc' in quote:
                    momentum = self.calculate_momentum(quote['c'], quote['pc'])
                    
                    results.append({
                        'Ticker': ticker,
                        'Current_Price': quote['c'],
                        'Previous_Close': quote['pc'],
                        'Momentum_%': round(momentum, 2),
                        'Sector': SECTOR_MAPPING.get(ticker, 'Unknown'),
                        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    
            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {str(e)}")
                continue
        
        df = pd.DataFrame(results)
        
        # Calculate trend scores
        if not df.empty:
            df['Trend_Score'] = df['Momentum_%'].apply(self.calculate_trend_score)
            df = df.sort_values('Trend_Score', ascending=False)
        
        return df
    
    def calculate_trend_score(self, momentum):
        """Convert momentum to trend score"""
        if momentum > 2:
            return min(100, 50 + momentum * 10)
        elif momentum < -2:
            return max(0, 50 + momentum * 10)
        else:
            return 50
    
    def analyze_sectors(self, stock_df):
        """Aggregate sector-level momentum"""
        if stock_df.empty:
            return pd.DataFrame()
        
        sector_analysis = stock_df.groupby('Sector').agg({
            'Momentum_%': 'mean',
            'Trend_Score': 'mean',
            'Ticker': 'count'
        }).round(2)
        
        sector_analysis.columns = ['Avg_Momentum_%', 'Avg_Trend_Score', 'Stock_Count']
        sector_analysis['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return sector_analysis.reset_index()
    
    def analyze_crypto(self):
        """Analyze crypto trends"""
        results = []
        
        for symbol in CRYPTO_SYMBOLS:
            try:
                # Using yfinance for crypto data as fallback
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                
                if not hist.empty and len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    previous_close = hist['Close'].iloc[-2]
                    momentum = self.calculate_momentum(current_price, previous_close)
                    
                    results.append({
                        'Symbol': symbol,
                        'Current_Price': round(current_price, 2),
                        'Previous_Close': round(previous_close, 2),
                        'Momentum_%': round(momentum, 2),
                        'Trend_Score': self.calculate_trend_score(momentum),
                        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    
            except Exception as e:
                logger.error(f"Error analyzing crypto {symbol}: {str(e)}")
                continue
        
        return pd.DataFrame(results)
    
    def run_full_analysis(self):
        """Run complete market analysis"""
        logger.info("Starting market analysis...")
        
        # Analyze stocks
        stock_df = self.analyze_stocks()
        if not stock_df.empty:
            save_data_safely(stock_df, TREND_FILE)
            logger.info(f"Analyzed {len(stock_df)} stocks")
        
        # Analyze sectors
        sector_df = self.analyze_sectors(stock_df)
        if not sector_df.empty:
            # Append to sector log
            try:
                existing_log = pd.read_csv(SECTOR_LOG_FILE)
                sector_log = pd.concat([existing_log, sector_df], ignore_index=True)
            except:
                sector_log = sector_df
            
            save_data_safely(sector_log, SECTOR_LOG_FILE)
            logger.info(f"Analyzed {len(sector_df)} sectors")
        
        # Analyze crypto
        crypto_df = self.analyze_crypto()
        logger.info(f"Analyzed {len(crypto_df)} cryptocurrencies")
        
        return {
            'stocks': stock_df,
            'sectors': sector_df,
            'crypto': crypto_df
        }