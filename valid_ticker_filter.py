# valid_ticker_filter.py

import pandas as pd
import requests
import os
from typing import List

# ✅ Setup API Key (env first, then Streamlit)
try:
    FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
    if not FINNHUB_API_KEY:
        import streamlit as st
        FINNHUB_API_KEY = st.secrets["api_keys"]["finnhub"]
except Exception as e:
    raise ValueError("Finnhub API key not found in environment or Streamlit secrets.") from e

# ----------------------------------------
# Valid tickers from recent scan
# ----------------------------------------
VALID_TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'V', 'JNJ',
    'UNH', 'PG', 'HD', 'MA', 'XOM', 'CVX', 'MRK', 'ABBV', 'PEP', 'KO',
    'LLY', 'BAC', 'AVGO', 'COST', 'ADBE', 'PFE', 'TMO', 'ORCL', 'ACN', 'CSCO'
]

SECTOR_MAP = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology', 'GOOGL': 'Communication Services', 'AMZN': 'Consumer Discretionary',
    'META': 'Communication Services', 'TSLA': 'Consumer Discretionary', 'JPM': 'Financials', 'V': 'Financials', 'JNJ': 'Health Care',
    'UNH': 'Health Care', 'PG': 'Consumer Staples', 'HD': 'Consumer Discretionary', 'MA': 'Financials', 'XOM': 'Energy',
    'CVX': 'Energy', 'MRK': 'Health Care', 'ABBV': 'Health Care', 'PEP': 'Consumer Staples', 'KO': 'Consumer Staples',
    'LLY': 'Health Care', 'BAC': 'Financials', 'AVGO': 'Technology', 'COST': 'Consumer Staples', 'ADBE': 'Technology',
    'PFE': 'Health Care', 'TMO': 'Health Care', 'ORCL': 'Technology', 'ACN': 'Information Technology', 'CSCO': 'Technology'
}

# ----------------------------------------
# Fetch price data using Finnhub
# ----------------------------------------
def fetch_price_data(tickers: List[str]) -> pd.DataFrame:
    print("📈 Fetching data from Finnhub...")
    prices = {}
    for ticker in tickers:
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={FINNHUB_API_KEY}"
            response = requests.get(url)
            data = response.json()
            current_price = data.get("c")
            previous_close = data.get("pc")
            if current_price and previous_close:
                prices[ticker] = (current_price - previous_close) / previous_close
            else:
                print(f"⚠️ Incomplete data for {ticker}: {data}")
        except Exception as e:
            print(f"❌ Failed to fetch {ticker}: {e}")
    df = pd.DataFrame.from_dict(prices, orient="index", columns=["momentum_score"])
    df.index.name = "ticker"  # ✅ This ensures reset_index will name the column properly
    return df

# ----------------------------------------
# Compute trend metrics
# ----------------------------------------
def compute_trend(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        print("⚠️ No data to compute trend.")
        return pd.DataFrame()
    df["rank"] = df["momentum_score"].rank(ascending=False)
    df["sector"] = df.index.map(SECTOR_MAP.get)
    return df

# ----------------------------------------
# Log and save trends
# ----------------------------------------
def log_trend_to_csv(trend_df: pd.DataFrame, filepath: str = "trend_log.csv"):
    timestamp = pd.Timestamp.now()
    records = [
        {"timestamp": timestamp, "sector": sector, "avg_trend": round(group["momentum_score"].mean(), 4)}
        for sector, group in trend_df.groupby("sector") if sector is not None
    ]
    pd.DataFrame(records).to_csv(filepath, mode="a", index=False, header=not os.path.exists(filepath))

def save_current_trend_snapshot(trend_df: pd.DataFrame, filepath: str = "trend_df.csv"):
    trend_df = trend_df.reset_index()  # 🛠 Converts ticker index into a proper column
    trend_df.to_csv(filepath, index=False)

# ----------------------------------------
# Main loop
# ----------------------------------------
def main():
    price_data = fetch_price_data(VALID_TICKERS)
    trend = compute_trend(price_data)
    if not trend.empty:
        print("\n🔥 Top Ticker Momentum:")
        print(trend.sort_values("momentum_score", ascending=False).head(10))
        log_trend_to_csv(trend)
        save_current_trend_snapshot(trend)
        print(f"\n✅ Trend snapshot saved with {len(trend)} rows.")
    else:
        print("❌ Trend data unavailable. Check ticker list or API key.")

if __name__ == "__main__":
    main()
