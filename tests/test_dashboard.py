#!/usr/bin/env python3
"""
Unit Tests for Enhanced Dashboard
=================================

Tests for the Streamlit dashboard components.

Run with: pytest test_dashboard.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import streamlit as st

# Mock streamlit for testing
st.write = Mock()
st.header = Mock()
st.metric = Mock()
st.plotly_chart = Mock()
st.dataframe = Mock()

from dashboard import MarketDashboard


class TestMarketDashboard:
    """Test suite for MarketDashboard class."""
    
    @pytest.fixture
    def dashboard(self):
        """Create a MarketDashboard instance with mocked dependencies."""
        with patch('dashboard.DatabaseManager'), \
             patch('dashboard.FinnhubClient'), \
             patch('dashboard.TrendAnalyzer'), \
             patch('dashboard.MarketAlertSystem'):
            dashboard = MarketDashboard()
            return dashboard
    
    @pytest.fixture
    def sample_trends_data(self):
        """Sample trends data for testing."""
        return pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            'sector': ['Technology', 'Technology', 'Technology', 'Consumer', 'Consumer'],
            'price': [150.25, 300.50, 140.75, 3200.00, 900.25],
            'price_change': [1.5, -0.8, 2.1, -1.2, 5.5],
            'momentum': [12.5, -5.2, 15.3, -8.1, 25.7],
            'trend_score': [85, 45, 90, 40, 95],
            'volume_ratio': [1.2, 0.8, 1.5, 0.9, 2.1],
            'timestamp': datetime.now()
        })
    
    @pytest.fixture
    def sample_alerts_data(self):
        """Sample alerts data for testing."""
        return pd.DataFrame({
            'id': [1, 2, 3],
            'alert_type': ['momentum_surge', 'volume_spike', 'sector_rotation'],
            'severity': ['CRITICAL', 'WARNING', 'INFO'],
            'ticker': ['TSLA', 'AAPL', None],
            'sector': ['Consumer', 'Technology', 'Technology'],
            'message': [
                'TSLA momentum surge: 25.7%',
                'AAPL volume spike: 1.2x',
                'Sector rotation in Technology'
            ],
            'timestamp': datetime.now()
        })
    
    def test_initialization(self, dashboard):
        """Test dashboard initialization."""
        assert dashboard is not None
        assert hasattr(dashboard, 'db')
        assert hasattr(dashboard, 'client')
        assert hasattr(dashboard, 'analyzer')
        assert hasattr(dashboard, 'alert_system')
    
    def test_render_header(self, dashboard):
        """Test header rendering."""
        # Mock the dependencies
        dashboard.client.get_market_status = Mock(return_value={'isOpen': True, 'session': 'market'})
        dashboard.db.get_database_stats = Mock(return_value={'trends_count': 1000})
        dashboard.client.get_stats = Mock(return_value={
            'success_rate': 95.5,
            'cache_stats': {'hit_rate': 75.0}
        })
        
        # Mock streamlit components
        with patch('dashboard.st.markdown'), \
             patch('dashboard.st.columns', return_value=[Mock(), Mock(), Mock(), Mock()]), \
             patch('dashboard.st.metric'):
            dashboard.render_header()
        
        # Verify methods were called
        dashboard.client.get_market_status.assert_called_once()
        dashboard.db.get_database_stats.assert_called_once()
    
    def test_render_alerts_section_with_alerts(self, dashboard, sample_alerts_data):
        """Test alerts section rendering with active alerts."""
        dashboard.db.get_unacknowledged_alerts = Mock(return_value=sample_alerts_data)
        
        with patch('dashboard.st.header'), \
             patch('dashboard.st.markdown'), \
             patch('dashboard.st.button', return_value=False):
            dashboard.render_alerts_section()
        
        dashboard.db.get_unacknowledged_alerts.assert_called_once()
    
    def test_render_alerts_section_no_alerts(self, dashboard):
        """Test alerts section rendering with no alerts."""
        dashboard.db.get_unacknowledged_alerts = Mock(return_value=pd.DataFrame())
        
        with patch('dashboard.st.header'), \
             patch('dashboard.st.info') as mock_info:
            dashboard.render_alerts_section()
        
        mock_info.assert_called_once()
    
    def test_render_top_movers(self, dashboard, sample_trends_data):
        """Test top movers rendering."""
        # Mock data
        gainers = sample_trends_data[sample_trends_data['momentum'] > 0]
        losers = sample_trends_data[sample_trends_data['momentum'] < 0]
        
        dashboard.db.get_top_movers = Mock(side_effect=[gainers, losers])
        
        with patch('dashboard.st.header'), \
             patch('dashboard.st.columns', return_value=[Mock(), Mock()]), \
             patch('dashboard.st.subheader'), \
             patch('dashboard.st.markdown'):
            dashboard.render_top_movers()
        
        # Should be called twice (gainers and losers)
        assert dashboard.db.get_top_movers.call_count == 2
    
    def test_render_mover_card(self, dashboard):
        """Test individual mover card rendering."""
        stock = pd.Series({
            'ticker': 'AAPL',
            'sector': 'Technology',
            'momentum': 12.5,
            'price': 150.25,
            'volume_ratio': 1.2
        })
        
        with patch('dashboard.st.markdown') as mock_markdown:
            dashboard._render_mover_card(stock, is_gainer=True)
        
        # Check that card HTML was rendered
        mock_markdown.assert_called_once()
        call_args = mock_markdown.call_args[0][0]
        assert 'AAPL' in call_args
        assert '12.50%' in call_args
        assert 'positive' in call_args
    
    def test_render_sector_analysis(self, dashboard):
        """Test sector analysis rendering."""
        sector_data = pd.DataFrame({
            'sector': ['Technology', 'Healthcare'],
            'avg_momentum': [10.5, -5.2],
            'sector_score': [85, 45],
            'timestamp': [datetime.now(), datetime.now()]
        })
        
        dashboard.db.get_sector_trends_history = Mock(return_value=sector_data)
        
        with patch('dashboard.st.header'), \
             patch('dashboard.st.tabs', return_value=[Mock(), Mock(), Mock()]):
            dashboard.render_sector_analysis()
        
        dashboard.db.get_sector_trends_history.assert_called_once()
    
    def test_render_trend_analysis(self, dashboard, sample_trends_data):
        """Test trend analysis rendering."""
        dashboard.db.get_latest_trends = Mock(return_value=sample_trends_data)
        
        with patch('dashboard.st.header'), \
             patch('dashboard.st.plotly_chart'), \
             patch('dashboard.st.subheader'), \
             patch('dashboard.st.dataframe'):
            dashboard.render_trend_analysis()
        
        dashboard.db.get_latest_trends.assert_called_once()
    
    def test_render_performance_metrics(self, dashboard):
        """Test performance metrics rendering."""
        # Mock stats
        dashboard.client.get_stats = Mock(return_value={
            'total_api_calls': 1000,
            'success_rate': 95.5,
            'cache_stats': {'hit_rate': 75.0},
            'remaining_calls': 45
        })
        
        dashboard.db.get_database_stats = Mock(return_value={
            'trends_count': 5000,
            'sector_trends_count': 100,
            'alerts_count': 50,
            'db_size_mb': 12.5,
            'unacknowledged_alerts': {'CRITICAL': 2, 'WARNING': 5, 'INFO': 10},
            'date_range': '2024-01-01 to 2024-01-15'
        })
        
        with patch('dashboard.st.header'), \
             patch('dashboard.st.columns', return_value=[Mock(), Mock(), Mock()]), \
             patch('dashboard.st.markdown'), \
             patch('dashboard.st.metric'), \
             patch('dashboard.st.caption'):
            dashboard.render_performance_metrics()
        
        dashboard.client.get_stats.assert_called_once()
        dashboard.db.get_database_stats.assert_called_once()
    
    def test_error_handling(self, dashboard):
        """Test error handling in dashboard."""
        # Mock a database error
        dashboard.db.get_latest_trends = Mock(side_effect=Exception("Database error"))
        
        with patch('dashboard.st.header'), \
             patch('dashboard.st.warning') as mock_warning:
            dashboard.render_trend_analysis()
        
        # Should show warning when no data
        mock_warning.assert_called_once()


class TestDashboardIntegration:
    """Integration tests for dashboard functionality."""
    
    @pytest.mark.integration
    def test_dashboard_startup(self):
        """Test dashboard can start up properly."""
        with patch('dashboard.DatabaseManager'), \
             patch('dashboard.FinnhubClient'), \
             patch('dashboard.TrendAnalyzer'), \
             patch('dashboard.MarketAlertSystem'), \
             patch('streamlit.set_page_config'), \
             patch('streamlit.markdown'):
            
            dashboard = MarketDashboard()
            assert dashboard is not None
    
    @pytest.mark.integration
    def test_data_flow(self):
        """Test data flow through dashboard components."""
        with patch('dashboard.DatabaseManager') as mock_db, \
             patch('dashboard.FinnhubClient') as mock_client, \
             patch('dashboard.TrendAnalyzer') as mock_analyzer, \
             patch('dashboard.MarketAlertSystem') as mock_alerts:
            
            # Set up mock returns
            mock_db_instance = mock_db.return_value
            mock_db_instance.get_latest_trends.return_value = pd.DataFrame({
                'ticker': ['AAPL'],
                'momentum': [10.0]
            })
            
            dashboard = MarketDashboard()
            
            # Verify connections
            assert dashboard.db == mock_db_instance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])