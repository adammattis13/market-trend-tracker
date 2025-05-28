import yfinance as yf
from datetime import datetime
import csv
import os
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------------------
# Analyze average trend score by sector
# ------------------------------------------

def fetch_ticker_trend(ticker):
    try:
        data = yf.download(ticker, period="5d", progress=False)

        if data is None or data.empty:
            return ticker, None

        close_prices = None
        if "Close" in data.columns:
            close_prices = data["Close"].dropna().tolist()
        elif ("Close", ticker) in data.columns:
            close_prices = data[("Close", ticker)].dropna().tolist()
        else:
            return ticker, None

        if len(close_prices) < 2:
            return ticker, None

        current = close_prices[-1]
        history = close_prices[:-1]
        avg = sum(history) / len(history)
        trend = (current - avg) / avg if avg else 0
        return ticker, trend

    except Exception:
        return ticker, None

def analyze_sector_trends(sector_map, log_file="trend_log.csv", max_log_entries=5000, max_workers=10):
    print(f"\n📊 Starting trend analysis")
    print(f"🗃️ Sector map contains {len(sector_map)} sectors.")

    if not sector_map:
        print("❌ No sector mapping provided.")
        return {}

    sector_trends = {}

    for sector, tickers in sector_map.items():
        print(f"\n➡️  Sector: {sector} ({len(tickers)} tickers)")
        trends = []
        ticker_scores = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(fetch_ticker_trend, ticker): ticker for ticker in tickers}
            for future in as_completed(future_to_ticker):
                ticker, trend = future.result()
                if trend is not None:
                    trends.append(trend)
                    ticker_scores.append((ticker, trend))
                    print(f"   ✅ {ticker} trend score: {trend:.4f}")
                else:
                    print(f"   ⚠️  Skipping {ticker} — insufficient or bad data")

        if trends:
            avg_trend = sum(trends) / len(trends)
            sorted_tickers = sorted(ticker_scores, key=lambda x: x[1], reverse=True)

            sector_trends[sector] = {
                "avg_trend": avg_trend,
                "count": len(trends),
                "tickers": sorted_tickers
            }

            print(f"   ✅ {sector} avg_trend: {avg_trend:.4f}")
        else:
            print(f"   ⚠️  No valid data in {sector}")

    if sector_trends:
        print(f"\n✅ Completed trend analysis — {len(sector_trends)} sectors processed")
        print(f"🧪 Writing trend data to {log_file}...")
        log_sector_trends(sector_trends, filename=log_file, max_entries=max_log_entries)
        print(f"📝 Logging complete!")
    else:
        print("⚠️ No sector trends to log.")

    return sector_trends

# ------------------------------------------
# Append and manage trend log file
# ------------------------------------------

def log_sector_trends(sector_data, filename="trend_log.csv", max_entries=5000):
    timestamp = datetime.now().isoformat()

    try:
        with open(filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            for sector, data in sector_data.items():
                writer.writerow([timestamp, sector, data["avg_trend"]])
    except Exception as e:
        print(f"❌ Failed to write to {filename}: {e}")
        return

    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename, names=["timestamp", "sector", "avg_trend"])
            if len(df) > max_entries:
                df = df.tail(max_entries)
                df.to_csv(filename, header=False, index=False)
        except Exception as e:
            print(f"⚠️ Error rotating trend log: {e}")
            os.rename(filename, f"{filename}.bak")

# ------------------------------------------
# Entry Point (Testing mode)
# ------------------------------------------

if __name__ == "__main__":
    test_map = defaultdict(list)
    test_map["Technology"] = ["AAPL", "MSFT", "NVDA"]
    analyze_sector_trends(test_map)
