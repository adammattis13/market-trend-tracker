# dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from sector_analyzer import analyze_sectors
from crypto_analyzer import fetch_crypto_momentum

st.set_page_config(layout="wide")
st.title("📊 Market Trend Dashboard")

# --- Load Filtered Ticker Data ---
st.header("✅ Valid Tickers with Trend Data")
try:
    trend_df = pd.read_csv("trend_df.csv")
    st.dataframe(trend_df)
except FileNotFoundError:
    st.warning("trend_df.csv not found. Please run valid_ticker_filter.py first.")
    trend_df = pd.DataFrame()

# --- Sector Trend Analysis ---
st.header("📈 Sector Momentum")
try:
    sector_df = analyze_sectors()
    st.bar_chart(data=sector_df, x="sector", y="momentum_score")
except Exception as e:
    st.error(f"Error loading sector data: {e}")

# --- 🚨 Opportunity Radar ---
st.header("🚨 Opportunity Radar")

if not trend_df.empty:
    spike_threshold = 0.03
    dip_threshold = -0.03

    hot = trend_df[trend_df["momentum_score"] > spike_threshold]
    cold = trend_df[trend_df["momentum_score"] < dip_threshold]

    if not hot.empty:
        st.subheader("🔥 Spike Movers (Momentum > 3%)")
        st.dataframe(hot.sort_values("momentum_score", ascending=False).style
                     .highlight_max(axis=0, color="lightgreen"))

    if not cold.empty:
        st.subheader("🧊 Big Dips (Momentum < -3%)")
        st.dataframe(cold.sort_values("momentum_score").style
                     .highlight_min(axis=0, color="salmon"))

    if hot.empty and cold.empty:
        st.info("No extreme movers at the moment. Market appears stable.")
else:
    st.info("Trend data not loaded.")

# --- 🧭 Sector Trend Over Time ---
st.header("🧭 Sector Trend Over Time")
try:
    trend_log = pd.read_csv("trend_log.csv")
    trend_log["timestamp"] = pd.to_datetime(trend_log["timestamp"])

    chart = alt.Chart(trend_log).mark_line().encode(
        x="timestamp:T",
        y="avg_trend:Q",
        color="sector:N",
        tooltip=["timestamp:T", "sector:N", "avg_trend:Q"]
    ).properties(width=1000, height=400)

    st.altair_chart(chart, use_container_width=True)
except Exception as e:
    st.warning(f"Trend log not available or invalid: {e}")

# --- 💹 Crypto Momentum Panel ---
st.header("💹 Crypto Momentum")
try:
    crypto_df = fetch_crypto_momentum()
    st.dataframe(crypto_df)
    st.bar_chart(data=crypto_df, x="crypto", y="momentum_score")
except Exception as e:
    st.error(f"Error fetching crypto data: {e}")

