#!/usr/bin/env python3
"""
Market Trend Tracker - Core Trend Analyzer
==========================================

This module analyzes stock market trends for the top S&P 500 companies,
calculating momentum indicators and trend scores.

Author: Adam Mattis (Enhanced Version)
Date: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_manager import DatabaseManager
from api_client import FinnhubClient, DataValidator
from alert_system import MarketAlertSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trend_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """
    Main class for analyzing market trends and momentum.
    
    This class handles:
    - Fetching real-time market data
    - Calculating technical indicators
    - Scoring trends and momentum
    - Persisting data to database
    - Generating alerts
    """
    
    # Top 30 S&P 500 tickers by market cap (as of 2024)
    TOP_SP500_TICKERS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
        'JPM', 'JNJ', 'V', 'UNH', 'PG', 'XOM', 'MA', 'HD', 'CVX', 'MRK',
        'ABBV', 'PFE', 'COST', 'WMT', 'DIS', 'BAC', 'CRM', 'NFLX', 'ADBE',
        'TMO', 'ACN', 'LLY'
    ]
    
    # Sector mappings for S&P 500 companies
    SECTOR_MAPPING = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
        'AMZN': 'Consumer Discretionary', 'NVDA': 'Technology', 'META': 'Technology',
        'TSLA': 'Consumer Discretionary', 'BRK.B': 'Financials', 'JPM': 'Financials',
        'JNJ': 'Healthcare', 'V': 'Financials', 'UNH': 'Healthcare',
        'PG': 'Consumer Staples', 'XOM': 'Energy', 'MA': 'Financials',
        'HD': 'Consumer Discretionary', 'CVX': 'Energy', 'MRK': 'Healthcare',
        'ABBV': 'Healthcare', 'PFE': 'Healthcare', 'COST': 'Consumer Staples',
        'WMT': 'Consumer Staples', 'DIS': 'Communication Services',
        'BAC': 'Financials', 'CRM': 'Technology', 'NFLX': 'Communication Services',
        'ADBE': 'Technology', 'TMO': 'Healthcare', 'ACN': 'Technology',
        'LLY': 'Healthcare'
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the trend analyzer.
        
        Args:
            api_key: Finnhub API key (optional, can use environment variable)
        """
        self.client = FinnhubClient(api_key)
        self.db = DatabaseManager()
        self.alert_system = MarketAlertSystem(self.db)
        self.validator = DataValidator()
        
        # Check API health
        if not self.client.health_check():
            logger.warning("API health check failed - will use cached data if available")
    
    def calculate_momentum(self, current_price: float, previous_price: float, 
                          volume_ratio: float = 1.0) -> float:
        """
        Calculate momentum score for a stock.
        
        Momentum is calculated using price change percentage with volume weighting.
        
        Args:
            current_price: Current stock price
            previous_price: Previous closing price
            volume_ratio: Current volume / average volume
            
        Returns:
            Momentum score (-100 to 100)
        """
        if previous_price <= 0:
            return 0.0
        
        # Basic price momentum
        price_momentum = ((current_price - previous_price) / previous_price) * 100
        
        # Apply volume weighting (capped at 2x to avoid extreme values)
        volume_weight = min(volume_ratio, 2.0) if volume_ratio > 0 else 1.0
        
        # Final momentum score
        momentum = price_momentum * volume_weight
        
        # Cap momentum at reasonable bounds
        return max(min(momentum, 100.0), -100.0)
    
    def calculate_trend_score(self, momentum: float, price_change: float, 
                            volume_ratio: float, volatility: float = 0.0) -> float:
        """
        Calculate overall trend score for a stock.
        
        Combines multiple factors into a single trend score.
        
        Args:
            momentum: Momentum score
            price_change: Price change percentage
            volume_ratio: Volume ratio
            volatility: Price volatility (optional)
            
        Returns:
            Trend score (0-100)
        """
        # Weight factors
        momentum_weight = 0.4
        price_weight = 0.3
        volume_weight = 0.2
        volatility_weight = 0.1
        
        # Normalize inputs to 0-100 scale
        norm_momentum = (momentum + 100) / 2  # Convert from -100,100 to 0,100
        norm_price = (price_change + 10) * 5  # Convert from -10,10 to 0,100
        norm_volume = min(volume_ratio * 50, 100)  # Convert volume ratio to 0,100
        norm_volatility = max(0, 100 - volatility * 100)  # Lower volatility = higher score
        
        # Calculate weighted score
        trend_score = (
            norm_momentum * momentum_weight +
            norm_price * price_weight +
            norm_volume * volume_weight +
            norm_volatility * volatility_weight
        )
        
        return max(min(trend_score, 100.0), 0.0)
    
    def fetch_ticker_data(self, ticker: str) -> Optional[Dict]:
        """
        Fetch and process data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with processed ticker data or None if fetch failed
        """
        try:
            # Get quote data
            quote = self.client.get_quote(ticker)
            if not quote or not self.validator.validate_quote(quote):
                logger.warning(f"Invalid quote data for {ticker}")
                return None
            
            # Clean and validate data
            clean_data = self.validator.validate_ticker_data(quote)
            
            # Calculate indicators
            momentum = self.calculate_momentum(
                clean_data['c'],  # current price
                clean_data['pc'],  # previous close
                clean_data.get('v', 1.0) / max(clean_data.get('avgVolume', 1.0), 1.0)
            )
            
            trend_score = self.calculate_trend_score(
                momentum,
                clean_data['change_percent'],
                clean_data.get('v', 1.0) / max(clean_data.get('avgVolume', 1.0), 1.0)
            )
            
            return {
                'ticker': ticker,
                'sector': self.SECTOR_MAPPING.get(ticker, 'Unknown'),
                'price': clean_data['c'],
                'price_change': clean_data['change_percent'],
                'momentum': round(momentum, 2),
                'trend_score': round(trend_score, 2),
                'volume': clean_data.get('v', 0),
                'volume_ratio': round(clean_data.get('v', 1.0) / max(clean_data.get('avgVolume', 1.0), 1.0), 2),
                'high': clean_data['h'],
                'low': clean_data['l'],
                'open': clean_data['o']
            }
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def analyze_trends(self, tickers: List[str] = None) -> pd.DataFrame:
        """
        Analyze trends for multiple tickers.
        
        Args:
            tickers: List of tickers to analyze (default: TOP_SP500_TICKERS)
            
        Returns:
            DataFrame with trend analysis results
        """
        if tickers is None:
            tickers = self.TOP_SP500_TICKERS
        
        logger.info(f"Analyzing trends for {len(tickers)} tickers...")
        
        results = []
        failed_tickers = []
        
        # Fetch data for all tickers
        for ticker in tickers:
            data = self.fetch_ticker_data(ticker)
            if data:
                results.append(data)
            else:
                failed_tickers.append(ticker)
        
        if failed_tickers:
            logger.warning(f"Failed to fetch data for: {', '.join(failed_tickers)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        if df.empty:
            logger.error("No valid data fetched")
            return pd.DataFrame()
        
        # Sort by trend score
        df = df.sort_values('trend_score', ascending=False)
        
        # Add ranking
        df['rank'] = range(1, len(df) + 1)
        
        # Add timestamp
        df['timestamp'] = datetime.now()
        
        logger.info(f"Successfully analyzed {len(df)} tickers")
        
        return df
    
    def calculate_sector_trends(self, ticker_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sector-level trends from ticker data.
        
        Args:
            ticker_df: DataFrame with ticker trends
            
        Returns:
            DataFrame with sector trends
        """
        if ticker_df.empty:
            return pd.DataFrame()
        
        # Group by sector
        sector_stats = ticker_df.groupby('sector').agg({
            'momentum': ['mean', 'std'],
            'trend_score': 'mean',
            'volume_ratio': 'mean',
            'ticker': 'count'
        }).round(2)
        
        # Flatten column names
        sector_stats.columns = ['avg_momentum', 'momentum_std', 
                               'sector_score', 'avg_volume_ratio', 'stock_count']
        
        # Reset index to make sector a column
        sector_stats = sector_stats.reset_index()
        
        # Sort by sector score
        sector_stats = sector_stats.sort_values('sector_score', ascending=False)
        
        logger.info(f"Calculated trends for {len(sector_stats)} sectors")
        
        return sector_stats
    
    def run_analysis(self, save_to_db: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, List]:
        """
        Run complete trend analysis workflow.
        
        Args:
            save_to_db: Whether to save results to database
            
        Returns:
            Tuple of (ticker_trends, sector_trends, alerts)
        """
        start_time = datetime.now()
        logger.info("Starting trend analysis...")
        
        try:
            # Check market status
            market_status = self.client.get_market_status()
            if not market_status.get('isOpen', True):
                logger.warning("Market is closed - using latest available data")
            
            # Analyze ticker trends
            ticker_trends = self.analyze_trends()
            
            if ticker_trends.empty:
                logger.error("No ticker data available")
                return pd.DataFrame(), pd.DataFrame(), []
            
            # Calculate sector trends
            sector_trends = self.calculate_sector_trends(ticker_trends)
            
            # Generate alerts (mock crypto data for now)
            crypto_df = pd.DataFrame()  # Will be implemented in crypto_analyzer.py
            alerts = self.alert_system.process_all_alerts(
                ticker_trends, sector_trends, crypto_df
            )
            
            # Save to database
            if save_to_db:
                self.db.save_trends(ticker_trends)
                self.db.save_sector_trends(sector_trends)
                logger.info("Data saved to database")
            
            # Log summary
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Analysis completed in {duration:.1f} seconds")
            logger.info(f"Top movers: {ticker_trends.head(5)['ticker'].tolist()}")
            logger.info(f"Generated {len(alerts)} alerts")
            
            return ticker_trends, sector_trends, alerts
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise


def main():
    """Main entry point for the trend analyzer."""
    try:
        # Initialize analyzer
        analyzer = TrendAnalyzer()
        
        # Run analysis
        ticker_trends, sector_trends, alerts = analyzer.run_analysis()
        
        # Display results
        print("\n=== TOP 10 TRENDING STOCKS ===")
        print(ticker_trends[['rank', 'ticker', 'sector', 'trend_score', 
                            'momentum', 'price_change']].head(10))
        
        print("\n=== SECTOR PERFORMANCE ===")
        print(sector_trends[['sector', 'sector_score', 'avg_momentum', 
                           'stock_count']].head())
        
        print(f"\n=== ALERTS ({len(alerts)} total) ===")
        for alert in alerts[:5]:  # Show first 5 alerts
            print(f"[{alert.severity.value}] {alert.message}")
        
        # Optionally save to CSV for backward compatibility
        if '--csv' in sys.argv:
            ticker_trends.to_csv('trend_df.csv', index=False)
            logger.info("Saved results to trend_df.csv")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()