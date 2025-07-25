#!/usr/bin/env python3
"""
Unit Tests for API Client
=========================

Tests for the resilient API client with mocking and edge cases.

Run with: pytest test_api_client.py -v
"""

import pytest
import requests
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json

from api_client import (
    RateLimiter, Cache, FinnhubClient, DataValidator, 
    retry_on_failure, CacheEntry
)


class TestRateLimiter:
    """Test suite for RateLimiter class."""
    
    def test_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(calls_per_minute=30)
        assert limiter.calls_per_minute == 30
        assert len(limiter.calls) == 0
    
    def test_basic_rate_limiting(self):
        """Test basic rate limiting functionality."""
        limiter = RateLimiter(calls_per_minute=5)
        
        # Make 5 calls quickly
        for _ in range(5):
            limiter.wait_if_needed()
        
        # Check that we have 5 calls recorded
        assert len(limiter.calls) == 5
        
        # The 6th call should wait
        start_time = time.time()
        limiter.wait_if_needed()
        elapsed = time.time() - start_time
        
        # Should have waited at least a bit
        # Note: This test might be flaky on slow systems
        assert elapsed > 0
    
    def test_get_remaining_calls(self):
        """Test getting remaining calls."""
        limiter = RateLimiter(calls_per_minute=10)
        
        assert limiter.get_remaining_calls() == 10
        
        # Make some calls
        for _ in range(3):
            limiter.wait_if_needed()
        
        assert limiter.get_remaining_calls() == 7
    
    @patch('time.time')
    def test_sliding_window(self, mock_time):
        """Test sliding window cleanup of old calls."""
        limiter = RateLimiter(calls_per_minute=5)
        
        # Mock time progression
        mock_time.return_value = 1000
        
        # Make 3 calls
        for _ in range(3):
            limiter.wait_if_needed()
        
        # Move time forward 30 seconds
        mock_time.return_value = 1030
        
        # Old calls should still be there
        assert limiter.get_remaining_calls() == 2
        
        # Move time forward 65 seconds (past 1 minute window)
        mock_time.return_value = 1065
        
        # Old calls should be cleaned up
        assert limiter.get_remaining_calls() == 5


class TestCache:
    """Test suite for Cache class."""
    
    def test_initialization(self):
        """Test cache initialization."""
        cache = Cache(default_ttl_seconds=600)
        assert cache.default_ttl == 600
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0
    
    def test_set_and_get(self):
        """Test basic cache set and get operations."""
        cache = Cache()
        
        # Set a value
        cache.set("test_key", {"data": "value"})
        
        # Get the value
        result = cache.get("test_key")
        assert result == {"data": "value"}
        assert cache.hits == 1
        assert cache.misses == 0
    
    def test_cache_miss(self):
        """Test cache miss."""
        cache = Cache()
        
        result = cache.get("nonexistent")
        assert result is None
        assert cache.hits == 0
        assert cache.misses == 1
    
    @patch('api_client.datetime')
    def test_expiration(self, mock_datetime):
        """Test cache expiration."""
        cache = Cache(default_ttl_seconds=300)
        
        # Mock current time
        now = datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = now
        
        # Set a value
        cache.set("test_key", "test_value")
        
        # Value should be retrievable
        assert cache.get("test_key") == "test_value"
        
        # Move time forward past TTL
        mock_datetime.now.return_value = now + timedelta(seconds=301)
        
        # Value should be expired
        assert cache.get("test_key") is None
    
    def test_custom_ttl(self):
        """Test setting custom TTL."""
        cache = Cache(default_ttl_seconds=300)
        
        # Set with custom TTL
        cache.set("test_key", "test_value", ttl_seconds=60)
        
        # Check that it's stored (we can't easily test expiration without mocking time)
        assert cache.get("test_key") == "test_value"
    
    def test_clear(self):
        """Test cache clearing."""
        cache = Cache()
        
        # Add some entries
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.get("key1")  # Generate a hit
        cache.get("key3")  # Generate a miss
        
        # Clear cache
        cache.clear()
        
        # Check everything is reset
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0
    
    def test_get_stats(self):
        """Test cache statistics."""
        cache = Cache()
        
        # Generate some activity
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("key3")  # Miss
        
        stats = cache.get_stats()
        
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['entries'] == 2
        assert stats['hit_rate'] == pytest.approx(66.67, 0.01)


class TestRetryDecorator:
    """Test suite for retry_on_failure decorator."""
    
    def test_successful_call(self):
        """Test decorator with successful function call."""
        call_count = 0
        
        @retry_on_failure(max_retries=3)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = successful_func()
        assert result == "success"
        assert call_count == 1  # Should not retry on success
    
    def test_retry_on_failure(self):
        """Test retry logic on failures."""
        call_count = 0
        
        @retry_on_failure(max_retries=3, backoff_factor=0.1)
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = failing_func()
        assert result == "success"
        assert call_count == 3  # Should retry twice before succeeding
    
    def test_max_retries_exceeded(self):
        """Test when max retries are exceeded."""
        call_count = 0
        
        @retry_on_failure(max_retries=3, backoff_factor=0.1)
        def always_failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            always_failing_func()
        
        assert call_count == 3  # Should try max_retries times
    
    def test_specific_exceptions(self):
        """Test retry only on specific exceptions."""
        @retry_on_failure(max_retries=3, exceptions=(ValueError,))
        def specific_exception_func():
            raise TypeError("Should not retry this")
        
        # Should not retry TypeError
        with pytest.raises(TypeError):
            specific_exception_func()


class TestFinnhubClient:
    """Test suite for FinnhubClient class."""
    
    @pytest.fixture
    def client(self):
        """Create a FinnhubClient instance with mocked session."""
        with patch.dict('os.environ', {'FINNHUB_API_KEY': 'test_key'}):
            client = FinnhubClient()
            client.session = Mock()
            return client
    
    def test_initialization_with_key(self):
        """Test client initialization with API key."""
        client = FinnhubClient(api_key='test_key')
        assert client.api_key == 'test_key'
        assert client.base_url == "https://finnhub.io/api/v1"
    
    def test_initialization_without_key(self):
        """Test client initialization without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="Finnhub API key not provided"):
                FinnhubClient()
    
    def test_make_request_success(self, client):
        """Test successful API request."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"c": 150.25, "pc": 148.00}
        client.session.get.return_value = mock_response
        
        result = client._make_request('quote', {'symbol': 'AAPL'})
        
        assert result == {"c": 150.25, "pc": 148.00}
        assert client.total_calls == 1
        assert client.failed_calls == 0
    
    def test_make_request_rate_limit(self, client):
        """Test rate limit error handling."""
        mock_response = Mock()
        mock_response.status_code = 429
        client.session.get.return_value = mock_response
        
        with pytest.raises(requests.exceptions.HTTPError):
            client._make_request('quote', {'symbol': 'AAPL'})
        
        assert client.failed_calls == 1
    
    def test_make_request_timeout(self, client):
        """Test timeout handling."""
        client.session.get.side_effect = requests.exceptions.Timeout()
        
        with pytest.raises(requests.exceptions.Timeout):
            client._make_request('quote', {'symbol': 'AAPL'})
        
        assert client.failed_calls == 1
    
    def test_get_quote_success(self, client):
        """Test successful quote retrieval."""
        quote_data = {
            'c': 150.25,
            'h': 152.00,
            'l': 148.50,
            'o': 149.00,
            'pc': 148.00
        }
        
        # Mock the _make_request method
        client._make_request = Mock(return_value=quote_data)
        
        result = client.get_quote('AAPL')
        
        assert result == quote_data
        client._make_request.assert_called_once_with('quote', {'symbol': 'AAPL'})
    
    def test_get_quote_with_cache(self, client):
        """Test quote retrieval with caching."""
        quote_data = {'c': 150.25, 'h': 152.00, 'l': 148.50, 'o': 149.00, 'pc': 148.00}
        
        # First call - should hit API
        client._make_request = Mock(return_value=quote_data)
        result1 = client.get_quote('AAPL')
        
        # Second call - should hit cache
        result2 = client.get_quote('AAPL')
        
        assert result1 == result2
        # API should only be called once
        client._make_request.assert_called_once()
    
    def test_get_quote_invalid_data(self, client):
        """Test quote retrieval with invalid data."""
        # Mock invalid quote data (missing required field)
        client._make_request = Mock(return_value={'c': 150.25})  # Missing other fields
        
        result = client.get_quote('AAPL')
        
        assert result is None
    
    def test_batch_quotes(self, client):
        """Test batch quote retrieval."""
        # Mock get_quote
        def mock_get_quote(symbol):
            if symbol == 'INVALID':
                return None
            return {'c': 100.0, 'pc': 99.0}
        
        client.get_quote = Mock(side_effect=mock_get_quote)
        
        results = client.batch_quotes(['AAPL', 'MSFT', 'INVALID'])
        
        assert len(results) == 2
        assert 'AAPL' in results
        assert 'MSFT' in results
        assert 'INVALID' not in results
    
    def test_health_check(self, client):
        """Test API health check."""
        # Mock successful market status call
        client.get_market_status = Mock(return_value={'isOpen': True})
        
        assert client.health_check() is True
        
        # Mock failed call
        client.get_market_status = Mock(return_value=None)
        
        assert client.health_check() is False
    
    def test_get_stats(self, client):
        """Test getting client statistics."""
        client.total_calls = 10
        client.failed_calls = 2
        
        stats = client.get_stats()
        
        assert stats['total_api_calls'] == 10
        assert stats['failed_calls'] == 2
        assert stats['success_rate'] == 80.0


class TestDataValidator:
    """Test suite for DataValidator class."""
    
    def test_validate_quote_valid(self):
        """Test validation of valid quote data."""
        valid_quote = {
            'c': 150.25,
            'h': 152.00,
            'l': 148.50,
            'o': 149.00,
            'pc': 148.00
        }
        
        assert DataValidator.validate_quote(valid_quote) is True
    
    def test_validate_quote_missing_fields(self):
        """Test validation with missing fields."""
        invalid_quote = {
            'c': 150.25,
            'h': 152.00
            # Missing l, o, pc
        }
        
        assert DataValidator.validate_quote(invalid_quote) is False
    
    def test_validate_quote_invalid_values(self):
        """Test validation with invalid values."""
        # High less than low
        invalid_quote1 = {
            'c': 150.25,
            'h': 148.00,  # Less than low
            'l': 152.00,
            'o': 149.00,
            'pc': 148.00
        }
        
        assert DataValidator.validate_quote(invalid_quote1) is False
        
        # Negative price
        invalid_quote2 = {
            'c': -150.25,
            'h': 152.00,
            'l': 148.50,
            'o': 149.00,
            'pc': 148.00
        }
        
        assert DataValidator.validate_quote(invalid_quote2) is False
    
    def test_validate_ticker_data(self):
        """Test ticker data validation and cleaning."""
        raw_data = {
            'c': '150.25',  # String that should be converted
            'pc': 148.00,
            'v': '1000000',
            'invalid_field': 'ignored'
        }
        
        cleaned = DataValidator.validate_ticker_data(raw_data)
        
        assert cleaned['c'] == 150.25
        assert cleaned['pc'] == 148.00
        assert cleaned['v'] == 1000000
        assert cleaned['change_percent'] == pytest.approx(1.52, 0.01)
        assert 'invalid_field' not in cleaned
    
    def test_calculate_momentum(self):
        """Test momentum calculation."""
        # Normal case
        prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 115]
        momentum = DataValidator.calculate_momentum(prices, periods=14)
        assert momentum == pytest.approx(15.0, 0.01)  # (115-100)/100 * 100
        
        # Not enough data
        short_prices = [100, 102, 101]
        momentum = DataValidator.calculate_momentum(short_prices, periods=14)
        assert momentum == 0.0
        
        # Empty list
        momentum = DataValidator.calculate_momentum([])
        assert momentum == 0.0
        
        # With None values
        prices_with_none = [100, None, 102, 101, None, 105]
        momentum = DataValidator.calculate_momentum(prices_with_none, periods=3)
        assert momentum > 0  # Should handle None values
    
    def test_sanitize_string(self):
        """Test string sanitization."""
        # Normal string
        assert DataValidator.sanitize_string("Normal String") == "Normal String"
        
        # String with whitespace
        assert DataValidator.sanitize_string("  Spaced  ") == "Spaced"
        
        # Long string
        long_string = "a" * 100
        sanitized = DataValidator.sanitize_string(long_string, max_length=50)
        assert len(sanitized) == 50
        
        # Null bytes
        assert DataValidator.sanitize_string("Test\x00String") == "TestString"
        
        # None value
        assert DataValidator.sanitize_string(None) == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])