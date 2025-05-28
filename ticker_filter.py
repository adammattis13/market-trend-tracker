import yfinance as yf

def is_valid_ticker(ticker, lookback_days=7):
    """
    Returns True if the ticker has valid recent price data from yfinance.
    """
    try:
        df = yf.download(ticker, period=f"{lookback_days}d", progress=False)
        return not df.empty and len(df.columns) > 0
    except Exception:
        return False

def filter_valid_tickers(ticker_list, lookback_days=7, max_workers=10):
    """
    Filters out invalid tickers using multithreading.
    """
    from concurrent.futures import ThreadPoolExecutor

    valid = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(lambda t: (t, is_valid_ticker(t, lookback_days)), ticker_list)
        for ticker, is_valid in results:
            if is_valid:
                valid.append(ticker)
    return valid
