# src/utils.py
import time
import logging
from functools import wraps
import finnhub
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rate_limited_retry(max_retries=3, delay=1):
    """Decorator for API calls with retry logic"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.RequestException, finnhub.exceptions.FinnhubAPIException) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (attempt + 1)
                        logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"API call failed after {max_retries} attempts: {str(e)}")
            raise last_exception
        return wrapper
    return decorator

def safe_api_call(api_function, default_value=None):
    """Wrapper for safe API calls with default fallback"""
    try:
        return api_function()
    except Exception as e:
        logger.error(f"API call failed: {str(e)}")
        return default_value