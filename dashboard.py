import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from ticker_universe import get_ticker_sector_map
from sector_analyzer import analyze_sector_trends
import requests
import concurrent.futures
import os

# Configure layout
st.set_page_config(page_title="Market Trend Dashboard", layout="wide")

# Optional: Refresh every 60s
if st.toggle("🔄 Auto-refresh every minute", value=True):
    st_autorefresh(interval=60000, key="auto_refresh")

st.title("📈 Market Sector Trend Intelligence")
st.markdown("Track real-time trend signals across sectors. Auto-refresh is enabled by default.")

# Cache sector map to avoid repeated calls
@st.cache_data(ttl=1800)
def get_cached_sector_map(limit=500):
    return get_ticker_sector_map(limit)

sector_map = get_cached_sector_map()

# Use session cache to store trend analysis
@st.cache_data(ttl=300)
def get_cached_sector_trends(sector_map):
    return analyze_sector_trends(sector_map)

sector_data = get_cached_sector_trends(sector_map)

if not sector_data:
    st.warning("No trend data available.")
    st.stop()

# Convert to DataFrame
df = pd.DataFrame([
    {"Sector": sector, "Average Trend Score": data["avg_trend"], "Ticker Count": data["count"]}
    for sector, data in sector_data.items()
])

# Highlight significant movers
threshold = 0.02
movers = df[abs(df["Average Trend Score"]) >= threshold]

st.subheader("🚀 Significant Sector Movers")
st.dataframe(movers.sort_values(by="Average Trend Score", ascending=False), use_container_width=True)

# Heatmap
st.subheader("🔥 Sector Trend Heatmap")
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(df.set_index("Sector")[["Average Trend Score"]].T, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
plt.title("Sector Trend Scores")
st.pyplot(fig)

# Trend line chart
st.subheader("📊 Sector Trend History")
try:
    trend_log = pd.read_csv("trend_log.csv", names=["timestamp", "sector", "avg_trend"], parse_dates=["timestamp"])
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    for sector in df["Sector"]:
        sector_trend = trend_log[trend_log["sector"] == sector]
        ax2.plot(sector_trend["timestamp"], sector_trend["avg_trend"], label=sector)
    ax2.set_title("Trend Score Over Time")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Average Trend Score")
    ax2.legend()
    st.pyplot(fig2)
except Exception as e:
    st.warning(f"Could not load trend history: {e}")

# Real-time news sentiment overlay
st.subheader("📰 Real-time News Sentiment Overlays")
try:
    api_key = os.getenv("MARKETAUX_API_KEY")
    if not api_key:
        raise ValueError("Missing API key: please set MARKETAUX_API_KEY environment variable.")

    sentiment_api_url = f"https://api.marketaux.com/v1/news/all?api_token={api_key}&language=en&filter_entities=true"
    response = requests.get(sentiment_api_url)
    if response.status_code == 200:
        news_items = response.json().get("data", [])[:10]
        for article in news_items:
            entity_name = article['entities'][0]['name'] if article.get('entities') else 'General'
            st.markdown(f"**{entity_name}**: [{article['title']}]({article['url']})")
    else:
        st.warning("⚠️ Unable to fetch news sentiment.")
except Exception as e:
    st.warning(f"⚠️ News sentiment error: {e}")

# Future features
st.subheader("🛠️ Upcoming Features")
st.markdown("- Email/Slack alerts for high-momentum sectors")
st.markdown("- Ticker-level breakout dashboards")
st.markdown("- Personalized portfolio trend summaries")
