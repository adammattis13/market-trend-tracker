📊 Market Trend Tracker
A real-time dashboard for monitoring stock and crypto momentum using sector-level analysis and trend scoring. Built with Streamlit, powered by Finnhub, and designed to support fast investment decisions.

🚀 Features
✅ Real-time trend scoring for top 30 S&P 500 tickers

📈 Sector-level momentum analysis and logging

💹 Live momentum for top 5 cryptocurrencies

🧭 Sector trend visualizations over time

🧼 Clean, modular, and extensible Python architecture

🔧 Setup Instructions
1. Clone & Setup
git clone https://github.com/yourname/market-trend-tracker.git
cd market-trend-tracker
python3 -m venv venv
source venv/bin/activate

2. Install Dependencies
pip install -r requirements.txt

Or install manually:
pip install streamlit pandas matplotlib seaborn requests altair streamlit-autorefresh

3. API Key Setup
Option A – Environment Variable
export FINNHUB_API_KEY="your_finnhub_api_key"

Option B – Streamlit Secrets File
Create .streamlit/secrets.toml with:
[api_keys]
finnhub = "your_finnhub_api_key"

4. Run the Analyzer
python valid_ticker_filter.py

5. Launch the Dashboard
streamlit run dashboard.py

Open your browser to:

Local: http://localhost:8501

Network: http://<your_local_IP>:8501

🗂 Project Structure
├── dashboard.py # Streamlit dashboard
├── valid_ticker_filter.py # Core trend analyzer
├── sector_analyzer.py # Sector-level aggregation logic
├── crypto_analyzer.py # Crypto trend fetcher
├── trend_df.csv # Output: Current trend snapshot
├── trend_log.csv # Output: Sector momentum over time
└── .streamlit/secrets.toml # (optional) API key config

🧠 Author
Built by Adam Mattis to monitor market momentum.
