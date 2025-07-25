# main_app.py
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Market Trend Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Home page content
st.title("📈 Market Trend Tracker")
st.sidebar.success("Select a page above.")

st.markdown("""
## Welcome to Market Trend Tracker!

This comprehensive platform helps you track market trends, analyze stocks with technical indicators, 
manage your portfolio, and set up intelligent alerts.

### 📊 Available Features:

#### 🏠 **Dashboard**
- Real-time market overview with technical analysis
- Live trend scoring for top S&P 500 stocks  
- Sector performance analysis
- Smart alerts with technical triggers

#### 📈 **Analysis**  
- Advanced technical analysis dashboard
- Interactive candlestick and line charts
- 15+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Stock screener and correlation analysis

#### 🚨 **Alerts**
- Create custom price and technical alerts
- Real-time alert monitoring
- Alert history and performance tracking
- Notification preferences management

#### 💼 **Portfolio**
- Track your holdings and performance
- Portfolio vs S&P 500 comparison
- Asset allocation analysis
- Quick buy/sell actions

### 🚀 Getting Started

👈 **Select a page from the sidebar** to begin exploring the features!

### 📊 System Status

""")

# Status indicators
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("API Status", "🟢 Connected", "Finnhub API")

with col2:
    st.metric("Data Status", "🟢 Live", "Real-time updates")

with col3:
    st.metric("Last Update", "Just now", "Auto-refresh active")

st.markdown("---")
st.markdown("*Built by Adam Mattis*")