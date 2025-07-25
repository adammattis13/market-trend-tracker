#!/usr/bin/env python3
"""
Resilient API Client for Market Trend Tracker
==============================================

Handles all external API calls with retry logic, caching, and rate limiting.
Primarily interfaces with Finnhub API for market data.

Author: Adam Mattis (Enhanced Version)
Date: 2024
"""

import requests
import time
import json
import os
from functools import wraps
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass
from threading import Lock
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Thread-safe rate limiter to avoid hitting API limits.
    
    Implements a sliding window approach to track API calls
    and enforce rate limits.
    """
    
    def __init__(self, calls_per_minute: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_minute: Maximum API calls allowed per minute
        """
        self.calls_per_minute = calls_per_minute
        self.calls = []
        self.lock = Lock()
        
    def wait_if_needed(self):
        """
        Wait if we're hitting rate limits.
        
        This method blocks until it's safe to make another API call.
        """
        with self.lock:
            now = time.time()
            
            # Remove calls older than 1 minute
            self.calls = [call_time for call_time in self.calls 
                         if now - call_time < 60]
            
            # Check if we're at the limit
            if len(self.calls) >= self.calls_per_minute:
                # Calculate wait time
                oldest_call = self.calls[0]
                wait_time = 60 - (now - oldest_call) + 0.1  # Add small buffer
                
                if wait_time > 0:
                    logger.warning(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                    time.sleep(wait_time)
                    
                    # Clean up again after waiting
                    now = time.time()
                    self.calls = [call_time for call_time in self.calls 
                                 if now - call_time < 60]
            
            # Record this call
            self.calls.append(now)
    
    def get_remaining_calls(self) -> int:
        """Get number of remaining calls in current window."""
        with self.lock:
            now = time.time()
            self.calls = [call_time for call_time in self.calls 
                         if now - call_time < 60]
            return self.calls_per_minute - len(self.calls)


@dataclass
class CacheEntry:
    """Cache entry with value and expiration."""
    value: Any
    expiry: datetime


class Cache:
    """
    Thread-safe in-memory cache with TTL support.
    
    Provides simple caching functionality to reduce API calls
    and improve performance.
    """
    
    def __init__(self, default_ttl_seconds: int = 300):
        """
        Initialize cache.
        
        Args:
            default_ttl_seconds: Default time-to-live for cache entries
        """
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = Lock()
        self.default_ttl = default_ttl_seconds
        self.hits = 0
        self.misses = 0
        
    def _make_key(self, key: str) -> str:
        """Create a normalized cache key."""
        return hashlib.md5(key.encode()).hexdigest()
        
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if not expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self.lock:
            cache_key = self._make_key(key)
            
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                
                if datetime.now() < entry.expiry:
                    self.hits += 1
                    logger.debug(f"Cache hit for key: {key}")
                    return entry.value
                else:
                    # Remove expired entry
                    del self.cache[cache_key]
                    logger.debug(f"Cache expired for key: {key}")
            
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """
        Set value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds (uses default if None)
        """
        with self.lock:
            cache_key = self._make_key(key)
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
            expiry = datetime.now() + timedelta(seconds=ttl)
            
            self.cache[cache_key] = CacheEntry(value=value, expiry=expiry)
            logger.debug(f"Cached value for key: {key} (TTL: {ttl}s)")
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'hits': self.hits,
                'misses': self.misses,
                'entries': len(self.cache),
                'hit_rate': round(hit_rate, 2)
            }


def retry_on_failure(max_retries: int = 3, backoff_factor: float = 2.0,
                    exceptions: tuple = (Exception,)):
    """
    Decorator to retry failed function calls with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for wait time between retries
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries - 1:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts: {str(e)}")
                        raise
                    
                    wait_time = backoff_factor ** attempt
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1} failed, "
                        f"retrying in {wait_time}s: {str(e)}"
                    )
                    time.sleep(wait_time)
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


class FinnhubClient:
    """
    Resilient Finnhub API client with caching and error handling.
    
    Features:
    - Automatic retry with exponential backoff
    - Rate limiting to avoid API limits
    - Response caching to reduce API calls
    - Comprehensive error handling
    - Thread-safe operations
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Finnhub client.
        
        Args:
            api_key: Finnhub API key (can also use FINNHUB_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('FINNHUB_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "Finnhub API key not provided. "
                "Set FINNHUB_API_KEY environment variable or pass api_key parameter."
            )
        
        self.base_url = "https://finnhub.io/api/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'X-Finnhub-Token': self.api_key,
            'User-Agent': 'MarketTrendTracker/2.0'
        })
        
        # Initialize components
        self.rate_limiter = RateLimiter(calls_per_minute=60)
        self.cache = Cache(default_ttl_seconds=300)
        
        # Track API call statistics
        self.total_calls = 0
        self.failed_calls = 0
        
        logger.info("FinnhubClient initialized successfully")
    
    @retry_on_failure(max_retries=3, backoff_factor=2.0,
                     exceptions=(requests.exceptions.RequestException,))
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Make API request with error handling.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            JSON response as dictionary
            
        Raises:
            Various request exceptions on failure
        """
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/{endpoint}"
        self.total_calls += 1
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            
            # Check for HTTP errors
            if response.status_code == 429:
                logger.error("Rate limit exceeded (429)")
                raise requests.exceptions.HTTPError("Rate limit exceeded")
                
            elif response.status_code == 401:
                logger.error("Invalid API key (401)")
                raise requests.exceptions.HTTPError("Invalid API key")
                
            elif response.status_code == 403:
                logger.error("Access forbidden (403) - Check API permissions")
                raise requests.exceptions.HTTPError("Access forbidden")
                
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            # Check for API errors in response
            if isinstance(data, dict) and data.get('error'):
                raise ValueError(f"API error: {data['error']}")
            
            return data
            
        except requests.exceptions.Timeout:
            self.failed_calls += 1
            logger.error(f"Timeout for {endpoint}")
            raise
            
        except requests.exceptions.ConnectionError:
            self.failed_calls += 1
            logger.error(f"Connection error for {endpoint}")
            raise
            
        except requests.exceptions.HTTPError as e:
            self.failed_calls += 1
            logger.error(f"HTTP error for {endpoint}: {e}")
            raise
            
        except json.JSONDecodeError:
            self.failed_calls += 1
            logger.error(f"Invalid JSON response from {endpoint}")
            raise ValueError("Invalid JSON response")
            
        except Exception as e:
            self.failed_calls += 1
            logger.error(f"Unexpected error for {endpoint}: {e}")
            raise
    
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get real-time quote for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Quote data dictionary or None on failure
        """
        cache_key = f"quote_{symbol}"
        
        # Check cache first
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        try:
            data = self._make_request('quote', {'symbol': symbol})
            
            # Validate response has required fields
            if self._validate_quote(data):
                # Cache for 1 minute (real-time data)
                self.cache.set(cache_key, data, ttl_seconds=60)
                return data
            else:
                logger.warning(f"Invalid quote data for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return None
    
    def get_company_profile(self, symbol: str) -> Optional[Dict]:
        """
        Get company profile information.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Company profile data or None on failure
        """
        cache_key = f"profile_{symbol}"
        
        # Check cache first
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        try:
            data = self._make_request('stock/profile2', {'symbol': symbol})
            
            if data:
                # Cache for 1 hour (static data)
                self.cache.set(cache_key, data, ttl_seconds=3600)
                return data
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get profile for {symbol}: {e}")
            return None
    
    def get_crypto_candles(self, symbol: str, resolution: str = 'D',
                          count: int = 30) -> Optional[Dict]:
        """
        Get cryptocurrency candle data.
        
        Args:
            symbol: Crypto symbol (e.g., 'BINANCE:BTCUSDT')
            resolution: Time resolution (1, 5, 15, 30, 60, D, W, M)
            count: Number of candles to fetch
            
        Returns:
            Candle data dictionary or None on failure
        """
        cache_key = f"crypto_{symbol}_{resolution}_{count}"
        
        # Check cache first
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        try:
            # Calculate time range
            to_timestamp = int(time.time())
            
            # Calculate from_timestamp based on resolution and count
            resolution_seconds = {
                '1': 60, '5': 300, '15': 900, '30': 1800,
                '60': 3600, 'D': 86400, 'W': 604800, 'M': 2592000
            }
            
            seconds_per_candle = resolution_seconds.get(resolution, 86400)
            from_timestamp = to_timestamp - (seconds_per_candle * count)
            
            data = self._make_request('crypto/candle', {
                'symbol': symbol,
                'resolution': resolution,
                'from': from_timestamp,
                'to': to_timestamp
            })
            
            if data and data.get('s') == 'ok':
                # Cache for 5 minutes
                self.cache.set(cache_key, data, ttl_seconds=300)
                return data
            else:
                logger.warning(f"No data available for crypto {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get crypto data for {symbol}: {e}")
            return None
    
    def batch_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get quotes for multiple symbols efficiently.
        
        Args:
            symbols: List of stock ticker symbols
            
        Returns:
            Dictionary mapping symbols to quote data
        """
        results = {}
        
        for symbol in symbols:
            # Small delay between requests to be nice to the API
            if len(results) > 0:
                time.sleep(0.1)
            
            quote = self.get_quote(symbol)
            if quote:
                results[symbol] = quote
            else:
                logger.warning(f"No data for {symbol}")
        
        logger.info(f"Fetched quotes for {len(results)}/{len(symbols)} symbols")
        return results
    
    def get_market_status(self) -> Dict:
        """
        Get current market status.
        
        Returns:
            Market status dictionary
        """
        try:
            data = self._make_request('stock/market-status', {'exchange': 'US'})
            return data
        except Exception as e:
            logger.error(f"Failed to get market status: {e}")
            # Return default closed status on error
            return {
                'isOpen': False,
                'session': 'closed',
                'timezone': 'America/New_York'
            }
    
    def health_check(self) -> bool:
        """
        Check if API is accessible and working.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            # Try to get market status as a simple health check
            status = self.get_market_status()
            return status is not None
        except Exception:
            return False
    
    def _validate_quote(self, quote_data: Dict) -> bool:
        """
        Validate quote data has required fields.
        
        Args:
            quote_data: Quote dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['c', 'h', 'l', 'o', 'pc']
        return all(
            field in quote_data and quote_data[field] is not None
            for field in required_fields
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get client statistics.
        
        Returns:
            Dictionary with client stats
        """
        cache_stats = self.cache.get_stats()
        
        return {
            'total_api_calls': self.total_calls,
            'failed_calls': self.failed_calls,
            'success_rate': round(
                (self.total_calls - self.failed_calls) / self.total_calls * 100
                if self.total_calls > 0 else 0, 2
            ),
            'remaining_calls': self.rate_limiter.get_remaining_calls(),
            'cache_stats': cache_stats
        }


class DataValidator:
    """
    Validate and clean market data.
    
    Provides methods to validate and sanitize data from API responses
    to ensure data quality and prevent errors.
    """
    
    @staticmethod
    def validate_quote(quote_data: Dict) -> bool:
        """
        Validate quote data structure.
        
        Args:
            quote_data: Quote dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(quote_data, dict):
            return False
        
        # Required fields for a valid quote
        required_fields = ['c', 'h', 'l', 'o', 'pc']
        
        # Check all required fields exist and are not None
        for field in required_fields:
            if field not in quote_data or quote_data[field] is None:
                return False
            
            # Ensure numeric fields are actually numeric
            try:
                float(quote_data[field])
            except (TypeError, ValueError):
                return False
        
        # Additional validation
        if quote_data['h'] < quote_data['l']:
            logger.warning("High is less than low - invalid quote")
            return False
        
        if quote_data['c'] <= 0 or quote_data['pc'] <= 0:
            logger.warning("Invalid price values - must be positive")
            return False
        
        return True
    
    @staticmethod
    def validate_ticker_data(data: Dict) -> Dict:
        """
        Clean and validate ticker data.
        
        Args:
            data: Raw ticker data
            
        Returns:
            Cleaned and validated data dictionary
        """
        cleaned = {}
        
        # Define numeric fields and their defaults
        numeric_fields = {
            'c': 0.0,      # current price
            'h': 0.0,      # high
            'l': 0.0,      # low
            'o': 0.0,      # open
            'pc': 0.0,     # previous close
            'v': 0,        # volume
            't': 0         # timestamp
        }
        
        # Clean numeric fields
        for field, default in numeric_fields.items():
            if field in data:
                try:
                    if field == 'v' or field == 't':
                        cleaned[field] = int(data[field])
                    else:
                        cleaned[field] = float(data[field])
                except (TypeError, ValueError):
                    logger.warning(f"Invalid value for {field}, using default")
                    cleaned[field] = default
            else:
                cleaned[field] = default
        
        # Calculate additional fields
        if cleaned['pc'] > 0:
            cleaned['change_percent'] = ((cleaned['c'] - cleaned['pc']) / cleaned['pc']) * 100
        else:
            cleaned['change_percent'] = 0.0
        
        # Add average volume if available
        if 'avgVolume' in data:
            try:
                cleaned['avgVolume'] = int(data['avgVolume'])
            except (TypeError, ValueError):
                cleaned['avgVolume'] = cleaned['v']  # Use current volume as fallback
        else:
            cleaned['avgVolume'] = cleaned['v']
        
        return cleaned
    
    @staticmethod
    def calculate_momentum(prices: List[float], periods: int = 14) -> float:
        """
        Calculate momentum indicator safely.
        
        Args:
            prices: List of price values
            periods: Number of periods for momentum calculation
            
        Returns:
            Momentum value as percentage
        """
        if not prices or len(prices) < periods:
            return 0.0
        
        try:
            # Remove any None values
            clean_prices = [p for p in prices if p is not None and p > 0]
            
            if len(clean_prices) < periods:
                return 0.0
            
            recent = clean_prices[-1]
            past = clean_prices[-periods]
            
            if past > 0:
                return ((recent - past) / past) * 100
            
            return 0.0
            
        except (IndexError, TypeError, ValueError) as e:
            logger.error(f"Error calculating momentum: {e}")
            return 0.0
    
    @staticmethod
    def validate_crypto_data(data: Dict) -> Optional[Dict]:
        """
        Validate cryptocurrency data.
        
        Args:
            data: Raw crypto data from API
            
        Returns:
            Validated data or None if invalid
        """
        if not data or data.get('s') != 'ok':
            return None
        
        # Check required arrays exist
        required_arrays = ['c', 'h', 'l', 'o', 'v', 't']
        for field in required_arrays:
            if field not in data or not isinstance(data[field], list):
                return None
        
        # Ensure all arrays have same length
        lengths = [len(data[field]) for field in required_arrays]
        if len(set(lengths)) != 1 or lengths[0] == 0:
            return None
        
        return data
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 50) -> str:
        """
        Sanitize string values for database storage.
        
        Args:
            value: String to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
        """
        if not value:
            return ""
        
        # Convert to string if needed
        value = str(value)
        
        # Remove potentially harmful characters
        value = value.replace('\x00', '')  # Null bytes
        value = value.strip()
        
        # Truncate if too long
        if len(value) > max_length:
            value = value[:max_length]
        
        return value


# Example usage and testing
def test_client():
    """Test the API client functionality."""
    # Initialize client
    client = FinnhubClient()
    
    print("Testing FinnhubClient...")
    print("-" * 50)
    
    # Test health check
    print("Health check:", client.health_check())
    
    # Test getting a quote
    quote = client.get_quote('AAPL')
    if quote:
        print(f"\nAAPL Quote: ${quote.get('c', 'N/A')}")
    
    # Test market status
    status = client.get_market_status()
    print(f"\nMarket Status: {status.get('session', 'unknown')}")
    
    # Show statistics
    stats = client.get_stats()
    print(f"\nClient Statistics:")
    print(f"  Total API calls: {stats['total_api_calls']}")
    print(f"  Success rate: {stats['success_rate']}%")
    print(f"  Cache hit rate: {stats['cache_stats']['hit_rate']}%")


if __name__ == "__main__":
    test_client()