import yfinance as yf
from yahoo_fin import stock_info as si
import json
import os
import re

# File paths for local caches
CACHE_FILE = "industry_map.json"
TICKER_CACHE_FILE = "all_tickers.json"

# --------------------------------------------
# 1. Sector > Industry > Ticker Mapping
# --------------------------------------------

def get_sector_industry_map(limit=2000, use_cache=True):
    """
    Builds or loads a nested map of { sector: { industry: [tickers] } } 
    based on NASDAQ-listed tickers.
    """
    if use_cache and os.path.exists(CACHE_FILE):
        print("📦 Loaded industry map from cache.")
        with open(CACHE_FILE, "r") as f:
            return json.load(f)

    all_tickers = si.tickers_nasdaq()
    all_tickers = list(set(all_tickers))[:limit]
    print(f"🔍 Fetching metadata for {len(all_tickers)} tickers...")

    sector_industry_map = {}

    for i, ticker in enumerate(all_tickers, start=1):
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector")
            industry = info.get("industry")
            if sector and industry:
                sector_industry_map.setdefault(sector, {}).setdefault(industry, []).append(ticker)
            if i % 100 == 0:
                print(f"   ...processed {i} tickers")
        except Exception:
            continue  # skip bad or unsupported tickers

    print(f"✅ Found {len(sector_industry_map)} sectors with valid industry data.")
    
    with open(CACHE_FILE, "w") as f:
        json.dump(sector_industry_map, f)

    return sector_industry_map

# --------------------------------------------
# 2. Flat Sector > Tickers Mapping (for sector_analyzer)
# --------------------------------------------

def get_ticker_sector_map(limit=2000):
    """
    Returns a simplified flat dictionary mapping { sector: [tickers] }.
    Useful for sector-level analysis without industry breakdown.
    """
    nested = get_sector_industry_map(limit=limit)
    flat = {}

    for sector, industries in nested.items():
        for tickers in industries.values():
            flat.setdefault(sector, []).extend(tickers)

    print(f"📊 Flattened sector map includes {len(flat)} sectors.")
    return flat

# --------------------------------------------
# 3. Single Ticker Metadata Lookup
# --------------------------------------------

def get_sector_industry_for_ticker(ticker):
    """
    Fetches sector and industry classification for a given ticker.
    """
    try:
        info = yf.Ticker(ticker).info
        return info.get("sector"), info.get("industry")
    except Exception:
        return None, None

# --------------------------------------------
# 4. Clean NASDAQ Ticker List
# --------------------------------------------

def get_all_tickers(use_cache=True, limit=5000):
    """
    Returns a de-duplicated, filtered list of NASDAQ tickers only.
    Removes junk entries like warrants and special purpose entities.
    """
    if use_cache and os.path.exists(TICKER_CACHE_FILE):
        with open(TICKER_CACHE_FILE, "r") as f:
            return json.load(f)[:limit]

    raw = si.tickers_nasdaq()
    unique = list(set(raw))

    clean = [
        t.upper()
        for t in unique
        if re.match(r"^[A-Z]{1,5}$", t.upper()) and t.isalpha()
    ]

    clean.sort()

    with open(TICKER_CACHE_FILE, "w") as f:
        json.dump(clean, f)

    return clean[:limit]