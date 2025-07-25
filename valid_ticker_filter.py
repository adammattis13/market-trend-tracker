# valid_ticker_filter.py
import pandas as pd
import finnhub
import time
import logging
from datetime import datetime
import sys
from pathlib import Path

# Import configurations from src directory
from src.config import (
    get_api_key, TOP_SP500_TICKERS, TREND_FILE, LOG_FILE,
    SECTOR_MAPPING, MIN_VALID_PRICE, MAX_MOMENTUM_PERCENT
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrendAnalyzer:
    def __init__(self):
        """Initialize the trend analyzer with API client"""
        try:
            self.api_key = get_api_key()
            self.client = finnhub.Client(api_key=self.api_key)
            logger.info("Finnhub client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Finnhub client: {str(e)}")
            raise
    
    def get_quote_with_retry(self, ticker, max_retries=3):
        """Get stock quote with retry logic"""
        for attempt in range(max_retries):
            try:
                quote = self.client.quote(ticker)
                if quote and 'c' in quote and quote['c'] > MIN_VALID_PRICE:
                    return quote
                else:
                    logger.warning(f"Invalid quote data for {ticker}")
                    return None
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Retry {attempt + 1} for {ticker} after {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to get quote for {ticker}: {str(e)}")
                    return None
    
    def calculate_momentum(self, current_price, previous_close):
        """Calculate price momentum with validation"""
        if previous_close and previous_close > 0:
            momentum = ((current_price - previous_close) / previous_close) * 100
            # Cap extreme values
            momentum = max(-MAX_MOMENTUM_PERCENT, min(MAX_MOMENTUM_PERCENT, momentum))
            return momentum
        return 0
    
    def calculate_trend_score(self, momentum):
        """Convert momentum to trend score (0-100)"""
        # Normalize momentum to 0-100 score
        # -50% momentum = 0 score, 0% = 50 score, +50% = 100 score
        normalized = (momentum + MAX_MOMENTUM_PERCENT) / (2 * MAX_MOMENTUM_PERCENT)
        return int(normalized * 100)
    
    def analyze_ticker(self, ticker):
        """Analyze a single ticker"""
        quote = self.get_quote_with_retry(ticker)
        
        if not quote:
            return None
        
        try:
            current_price = quote.get('c', 0)
            previous_close = quote.get('pc', 0)
            volume = quote.get('v', 0)
            
            if current_price <= 0 or previous_close <= 0:
                return None
            
            momentum = self.calculate_momentum(current_price, previous_close)
            trend_score = self.calculate_trend_score(momentum)
            
            return {
                'Ticker': ticker,
                'Sector': SECTOR_MAPPING.get(ticker, 'Unknown'),
                'Current_Price': round(current_price, 2),
                'Previous_Close': round(previous_close, 2),
                'Volume': volume,
                'Momentum_%': round(momentum, 2),
                'Trend_Score': trend_score,
                'Signal': self.get_signal(momentum),
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {str(e)}")
            return None
    
    def get_signal(self, momentum):
        """Generate trading signal based on momentum"""
        if momentum > 3:
            return 'STRONG BUY'
        elif momentum > 1:
            return 'BUY'
        elif momentum < -3:
            return 'STRONG SELL'
        elif momentum < -1:
            return 'SELL'
        else:
            return 'HOLD'
    
    def save_data_safely(self, df, filepath):
        """Save DataFrame with error handling"""
        backup_path = None
        try:
            # Create backup if file exists
            if filepath.exists():
                backup_path = filepath.with_suffix('.backup')
                if backup_path.exists():
                    backup_path.unlink()
                filepath.rename(backup_path)
            
            # Save new data
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {len(df)} records to {filepath}")
            
            # Remove backup if save successful
            if backup_path and backup_path.exists():
                backup_path.unlink()
                
        except Exception as e:
            logger.error(f"Failed to save data: {str(e)}")
            # Restore backup if it exists
            if backup_path and backup_path.exists():
                backup_path.rename(filepath)
            raise
    
    def run_analysis(self):
        """Run the full trend analysis"""
        logger.info(f"Starting trend analysis for {len(TOP_SP500_TICKERS)} tickers...")
        
        results = []
        
        # Analyze each ticker with progress reporting
        for i, ticker in enumerate(TOP_SP500_TICKERS):
            if i > 0 and i % 5 == 0:
                logger.info(f"Progress: {i}/{len(TOP_SP500_TICKERS)} tickers analyzed")
                time.sleep(0.5)  # Rate limiting
            
            ticker_data = self.analyze_ticker(ticker)
            if ticker_data:
                results.append(ticker_data)
        
        # Create DataFrame and sort by trend score
        df = pd.DataFrame(results)
        
        if df.empty:
            logger.error("No valid data collected")
            return pd.DataFrame()
        
        # Sort by trend score
        df = df.sort_values('Trend_Score', ascending=False)
        
        # Save to file
        self.save_data_safely(df, TREND_FILE)
        
        # Log summary statistics
        logger.info(f"\nAnalysis Complete!")
        logger.info(f"Total tickers analyzed: {len(df)}")
        logger.info(f"Average momentum: {df['Momentum_%'].mean():.2f}%")
        logger.info(f"Top performer: {df.iloc[0]['Ticker']} ({df.iloc[0]['Momentum_%']:.2f}%)")
        logger.info(f"Worst performer: {df.iloc[-1]['Ticker']} ({df.iloc[-1]['Momentum_%']:.2f}%)")
        
        # Save sector summary
        self.save_sector_summary(df)
        
        return df
    
    def save_sector_summary(self, df):
        """Calculate and save sector-level summary"""
        try:
            sector_summary = df.groupby('Sector').agg({
                'Momentum_%': ['mean', 'count'],
                'Trend_Score': 'mean'
            }).round(2)
            
            sector_summary.columns = ['Avg_Momentum_%', 'Stock_Count', 'Avg_Trend_Score']
            sector_summary['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Append to log file
            if LOG_FILE.exists():
                try:
                    existing_log = pd.read_csv(LOG_FILE)
                    # Keep only last 100 entries per sector
                    existing_log = existing_log.groupby('Sector').tail(99)
                except Exception as e:
                    logger.warning(f"Could not read existing log: {e}")
                    existing_log = pd.DataFrame()
            else:
                existing_log = pd.DataFrame()
            
            # Append new data
            new_log = pd.concat([existing_log, sector_summary.reset_index()], ignore_index=True)
            new_log.to_csv(LOG_FILE, index=False)
            
            logger.info(f"Sector summary saved to {LOG_FILE}")
            
        except Exception as e:
            logger.error(f"Failed to save sector summary: {str(e)}")

def main():
    """Main execution function"""
    try:
        analyzer = TrendAnalyzer()
        df = analyzer.run_analysis()
        
        if not df.empty:
            print("\n" + "="*60)
            print("TOP 10 TRENDING STOCKS")
            print("="*60)
            print(df[['Ticker', 'Sector', 'Current_Price', 'Momentum_%', 'Signal']].head(10).to_string(index=False))
            print("="*60)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()