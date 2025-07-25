# src/run_analysis.py
from src.config import get_api_key
from src.market_analyzer import MarketAnalyzer
from src.utils import logger

def main():
    """Run market analysis"""
    try:
        api_key = get_api_key()
        analyzer = MarketAnalyzer(api_key)
        
        results = analyzer.run_full_analysis()
        
        logger.info("Analysis complete!")
        logger.info(f"Stocks analyzed: {len(results['stocks'])}")
        logger.info(f"Sectors analyzed: {len(results['sectors'])}")
        logger.info(f"Cryptos analyzed: {len(results['crypto'])}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()