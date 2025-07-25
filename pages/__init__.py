"""
Pages package for Market Trend Tracker
Multi-page Streamlit application components
"""

__version__ = "1.0.0"
__author__ = "Adam Mattis"

# Import all page modules for easy access
from . import dashboard_page
from . import portfolio_page  
from . import alerts_page
from . import analysis_page

__all__ = [
    "dashboard_page",
    "portfolio_page", 
    "alerts_page",
    "analysis_page"
]