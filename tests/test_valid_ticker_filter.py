#!/usr/bin/env python3
"""
Unit Tests for Market Trend Analyzer
====================================

Tests for the core trend analysis functionality.

Run with: pytest test_valid_ticker_filter.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from valid_ticker_filter import TrendAnalyzer


class TestTrendAnalyzer:
    """Test suite for TrendAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a TrendAnalyzer instance with mocked dependencies."""
        with patch('valid_ticker_filter.FinnhubClient'), \
             patch('valid_ticker_filter.DatabaseManager'), \
             patch('valid_ticker_filter.MarketAlertSystem'):
            analyzer = TrendAnalyzer()
            # Mock the client health check
            analyzer.client.health_check.return_value = True
            return analyzer
    
    @pytest.fixture
    def sample_quote_data(self):
        """Sample quote data from API."""
        return {
            'c': 150.25,   # current price
            'h': 152.00,   # high
            'l': 148.50,   # low
            'o': 149.00,   # open
            'pc': 148.00,  # previous close
            'v': 1000000,  # volume
            'avgVolume': 800000
        }
    
    def test_initialization(self, analyzer):
        """Test proper initialization of TrendAnalyzer."""
        assert analyzer is not None
        assert hasattr(analyzer, 'client')
        assert hasattr(analyzer, 'db')
        assert hasattr(analyzer, 'alert_system')
        assert len(analyzer.TOP_SP500_TICKERS) == 30
    
    def test_calculate_momentum_basic(self, analyzer):
        """Test basic momentum calculation."""
        # Test positive momentum
        momentum = analyzer.calculate_momentum(110, 100, 1.0)
        assert momentum == 10.0
        
        # Test negative momentum
        momentum = analyzer.calculate_momentum(90, 100, 1.0)
        assert momentum == -10.0
        
        # Test with zero previous price
        momentum = analyzer.calculate_momentum(100, 0, 1.0)
        assert momentum == 0.0
    
    def test_calculate_momentum_with_volume(self, analyzer):
        """Test momentum calculation with volume weighting."""
        # High volume should amplify momentum
        momentum = analyzer.calculate_momentum(110, 100, 1.5)
        assert momentum == 15.0  # 10% * 1.5
        
        # Volume capped at 2x
        momentum = analyzer.calculate_momentum(110, 100, 3.0)
        assert momentum == 20.0  # 10% * 2.0 (capped)
        
        # Low volume
        momentum = analyzer.calculate_momentum(110, 100, 0.5)
        assert momentum == 5.0  # 10% * 0.5
    
    def test_calculate_momentum_bounds(self, analyzer):
        """Test momentum bounds (-100 to 100)."""
        # Test upper bound
        momentum = analyzer.calculate_momentum(500, 100, 2.0)
        assert momentum == 100.0  # Capped at 100
        
        # Test lower bound
        momentum = analyzer.calculate_momentum(0, 100, 2.0)
        assert momentum == -100.0  # Capped at -100
    
    def test_calculate_trend_score(self, analyzer):
        """Test trend score calculation."""
        # Neutral case
        score = analyzer.calculate_trend_score(0, 0, 1.0, 0.0)
        assert 40 <= score <= 60  # Should be around middle
        
        # Strong positive trend
        score = analyzer.calculate_trend_score(50, 5, 1.5, 0.1)
        assert score > 70
        
        # Strong negative trend
        score = analyzer.calculate_trend_score(-50, -5, 0.5, 0.5)
        assert score < 30
        
        # Bounds test
        score = analyzer.calculate_trend_score(100, 10, 2.0, 0.0)
        assert 0 <= score <= 100
    
    def test_fetch_ticker_data_success(self, analyzer, sample_quote_data):
        """Test successful ticker data fetch."""
        # Mock the API response
        analyzer.client.get_quote.return_value = sample_quote_data
        analyzer.validator.validate_quote.return_value = True
        analyzer.validator.validate_ticker_data.return_value = {
            'c': 150.25,
            'pc': 148.00,
            'h': 152.00,
            'l': 148.50,
            'o': 149.00,
            'v': 1000000,
            'avgVolume': 800000,
            'change_percent': 1.52
        }
        
        result = analyzer.fetch_ticker_data('AAPL')
        
        assert result is not None
        assert result['ticker'] == 'AAPL'
        assert result['sector'] == 'Technology'
        assert result['price'] == 150.25
        assert 'momentum' in result
        assert 'trend_score' in result
        assert 0 <= result['trend_score'] <= 100
    
    def test_fetch_ticker_data_failure(self, analyzer):
        """Test ticker data fetch with API failure."""
        # Mock API failure
        analyzer.client.get_quote.return_value = None
        
        result = analyzer.fetch_ticker_data('AAPL')
        
        assert result is None
    
    def test_fetch_ticker_data_invalid_quote(self, analyzer):
        """Test ticker data fetch with invalid quote data."""
        # Mock invalid quote
        analyzer.client.get_quote.return_value = {'c': None}  # Missing required fields
        analyzer.validator.validate_quote.return_value = False
        
        result = analyzer.fetch_ticker_data('AAPL')
        
        assert result is None
    
    def test_analyze_trends_success(self, analyzer):
        """Test trend analysis for multiple tickers."""
        # Mock successful fetches
        mock_data = {
            'ticker': 'AAPL',
            'sector': 'Technology',
            'price': 150.25,
            'price_change': 1.52,
            'momentum': 12.5,
            'trend_score': 75.0,
            'volume': 1000000,
            'volume_ratio': 1.25,
            'high': 152.00,
            'low': 148.50,
            'open': 149.00
        }
        
        analyzer.fetch_ticker_data = Mock(return_value=mock_data.copy())
        
        result = analyzer.analyze_trends(['AAPL', 'MSFT', 'GOOGL'])
        
        assert len(result) == 3
        assert 'rank' in result.columns
        assert result['rank'].tolist() == [1, 2, 3]
        assert 'timestamp' in result.columns
    
    def test_analyze_trends_partial_failure(self, analyzer):
        """Test trend analysis with some tickers failing."""
        # Mock mixed results
        def mock_fetch(ticker):
            if ticker == 'AAPL':
                return {
                    'ticker': 'AAPL',
                    'sector': 'Technology',
                    'price': 150.25,
                    'price_change': 1.52,
                    'momentum': 12.5,
                    'trend_score': 75.0,
                    'volume': 1000000,
                    'volume_ratio': 1.25,
                    'high': 152.00,
                    'low': 148.50,
                    'open': 149.00
                }
            return None
        
        analyzer.fetch_ticker_data = Mock(side_effect=mock_fetch)
        
        result = analyzer.analyze_trends(['AAPL', 'INVALID'])
        
        assert len(result) == 1
        assert result.iloc[0]['ticker'] == 'AAPL'
    
    def test_analyze_trends_empty_result(self, analyzer):
        """Test trend analysis with all tickers failing."""
        analyzer.fetch_ticker_data = Mock(return_value=None)
        
        result = analyzer.analyze_trends(['INVALID1', 'INVALID2'])
        
        assert result.empty
    
    def test_calculate_sector_trends(self, analyzer):
        """Test sector trend calculation."""
        # Create sample ticker data
        ticker_data = pd.DataFrame([
            {'ticker': 'AAPL', 'sector': 'Technology', 'momentum': 10, 
             'trend_score': 80, 'volume_ratio': 1.2},
            {'ticker': 'MSFT', 'sector': 'Technology', 'momentum': 15, 
             'trend_score': 85, 'volume_ratio': 1.5},
            {'ticker': 'JPM', 'sector': 'Financials', 'momentum': 5, 
             'trend_score': 60, 'volume_ratio': 0.9},
        ])
        
        result = analyzer.calculate_sector_trends(ticker_data)
        
        assert len(result) == 2  # Two sectors
        assert 'Technology' in result['sector'].values
        assert 'Financials' in result['sector'].values
        
        # Check Technology sector stats
        tech_row = result[result['sector'] == 'Technology'].iloc[0]
        assert tech_row['avg_momentum'] == 12.5
        assert tech_row['sector_score'] == 82.5
        assert tech_row['stock_count'] == 2
    
    def test_calculate_sector_trends_empty(self, analyzer):
        """Test sector trend calculation with empty data."""
        result = analyzer.calculate_sector_trends(pd.DataFrame())
        assert result.empty
    
    @patch('valid_ticker_filter.datetime')
    def test_run_analysis_success(self, mock_datetime, analyzer):
        """Test complete analysis workflow."""
        # Mock datetime
        mock_now = datetime(2024, 1, 15, 10, 30)
        mock_datetime.now.return_value = mock_now
        
        # Mock market status
        analyzer.client.get_market_status.return_value = {'isOpen': True}
        
        # Mock analyze_trends
        ticker_data = pd.DataFrame([
            {'ticker': 'AAPL', 'sector': 'Technology', 'momentum': 10, 
             'trend_score': 80, 'volume_ratio': 1.2}
        ])
        analyzer.analyze_trends = Mock(return_value=ticker_data)
        
        # Mock calculate_sector_trends
        sector_data = pd.DataFrame([
            {'sector': 'Technology', 'sector_score': 80}
        ])
        analyzer.calculate_sector_trends = Mock(return_value=sector_data)
        
        # Mock alert system
        analyzer.alert_system.process_all_alerts.return_value = []
        
        # Run analysis
        ticker_trends, sector_trends, alerts = analyzer.run_analysis()
        
        # Verify results
        assert not ticker_trends.empty
        assert not sector_trends.empty
        assert isinstance(alerts, list)
        
        # Verify database saves were called
        analyzer.db.save_trends.assert_called_once()
        analyzer.db.save_sector_trends.assert_called_once()
    
    def test_run_analysis_market_closed(self, analyzer):
        """Test analysis when market is closed."""
        # Mock market closed
        analyzer.client.get_market_status.return_value = {'isOpen': False}
        
        # Mock successful data fetch
        ticker_data = pd.DataFrame([
            {'ticker': 'AAPL', 'sector': 'Technology', 'momentum': 10, 
             'trend_score': 80, 'volume_ratio': 1.2}
        ])
        analyzer.analyze_trends = Mock(return_value=ticker_data)
        analyzer.calculate_sector_trends = Mock(return_value=pd.DataFrame())
        analyzer.alert_system.process_all_alerts.return_value = []
        
        # Should still work with warning
        ticker_trends, sector_trends, alerts = analyzer.run_analysis()
        
        assert not ticker_trends.empty
    
    def test_run_analysis_failure(self, analyzer):
        """Test analysis with critical failure."""
        # Mock analyze_trends to raise exception
        analyzer.analyze_trends = Mock(side_effect=Exception("API Error"))
        
        # Should raise exception
        with pytest.raises(Exception):
            analyzer.run_analysis()
    
    def test_sector_mapping_completeness(self, analyzer):
        """Test that all tickers have sector mappings."""
        for ticker in analyzer.TOP_SP500_TICKERS:
            assert ticker in analyzer.SECTOR_MAPPING
            assert analyzer.SECTOR_MAPPING[ticker] != 'Unknown'


class TestIntegration:
    """Integration tests for the trend analyzer."""
    
    @pytest.mark.integration
    def test_end_to_end_workflow(self):
        """Test complete workflow with real-like data."""
        with patch('valid_ticker_filter.FinnhubClient') as mock_client, \
             patch('valid_ticker_filter.DatabaseManager') as mock_db, \
             patch('valid_ticker_filter.MarketAlertSystem') as mock_alerts:
            
            # Set up mocks
            mock_client_instance = mock_client.return_value
            mock_client_instance.health_check.return_value = True
            mock_client_instance.get_market_status.return_value = {'isOpen': True}
            
            # Mock quote responses
            def mock_get_quote(ticker):
                base_price = {'AAPL': 150, 'MSFT': 300, 'GOOGL': 140}.get(ticker, 100)
                return {
                    'c': base_price + np.random.uniform(-2, 2),
                    'pc': base_price,
                    'h': base_price + 3,
                    'l': base_price - 3,
                    'o': base_price,
                    'v': 1000000,
                    'avgVolume': 900000
                }
            
            mock_client_instance.get_quote.side_effect = mock_get_quote
            
            # Create analyzer
            analyzer = TrendAnalyzer()
            
            # Mock validator to return True
            analyzer.validator.validate_quote = Mock(return_value=True)
            analyzer.validator.validate_ticker_data = Mock(side_effect=lambda x: x)
            
            # Run analysis
            ticker_trends, sector_trends, alerts = analyzer.run_analysis(save_to_db=False)
            
            # Verify results structure
            assert not ticker_trends.empty
            assert 'ticker' in ticker_trends.columns
            assert 'momentum' in ticker_trends.columns
            assert 'trend_score' in ticker_trends.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])