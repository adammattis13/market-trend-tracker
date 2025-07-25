"""
Dashboard Page - Main market overview
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

def show():
    """Main dashboard page"""
    
    st.header("🏠 Market Dashboard")
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("S&P 500", "4,783.45", "+23.45 (0.49%)", delta_color="normal")
    with col2:
        st.metric("NASDAQ", "15,234.67", "-45.23 (-0.30%)", delta_color="inverse")
    with col3:
        st.metric("Market Cap", "$45.2T", "+1.2%", delta_color="normal")
    with col4:
        st.metric("VIX", "18.45", "-2.1%", delta_color="inverse")
    
    # Main content area
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Market overview chart
        st.subheader("📈 Market Overview")
        
        # Create sample data for demonstration
        sample_data = create_sample_market_data()
        
        fig = px.line(
            sample_data, 
            x='Date', 
            y='Price', 
            color='Symbol',
            title="Top Stocks Performance (Last 30 Days)"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top movers section
        st.subheader("🚀 Top Movers")
        
        # Create tabs for different categories
        tab1, tab2, tab3 = st.tabs(["📈 Gainers", "📉 Losers", "📊 Most Active"])
        
        with tab1:
            display_top_movers("gainers")
        with tab2:
            display_top_movers("losers")
        with tab3:
            display_top_movers("active")
    
    with col_right:
        # Alerts panel
        st.subheader("🚨 Active Alerts")
        display_alerts_panel()
        
        # Sector performance
        st.subheader("🏭 Sector Performance")
        display_sector_performance()
        
        # Market news (placeholder)
        st.subheader("📰 Market News")
        display_market_news()
        
        # Integration note
        with st.expander("🔧 Integration Status"):
            st.info("📊 Currently using sample data")
            st.info("🔄 Real data integration coming in Phase 2")
            if st.button("🔗 Connect Real Data"):
                st.success("Real data integration planned for next phase!")

def create_sample_market_data():
    """Create sample market data for demonstration"""
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    dates = [datetime.now() - timedelta(days=x) for x in range(30, 0, -1)]
    
    data = []
    for symbol in symbols:
        base_price = np.random.uniform(100, 300)
        prices = []
        for i, date in enumerate(dates):
            price = base_price + np.random.normal(0, 5) + (i * 0.5)  # Slight upward trend
            prices.append(price)
            data.append({
                'Date': date,
                'Symbol': symbol,
                'Price': price
            })
    
    return pd.DataFrame(data)

def display_top_movers(category):
    """Display top movers in a category"""
    
    # Sample data
    if category == "gainers":
        movers = [
            {"Symbol": "NVDA", "Price": "$875.45", "Change": "+45.23", "Change%": "+5.45%"},
            {"Symbol": "AAPL", "Price": "$195.67", "Change": "+8.34", "Change%": "+4.46%"},
            {"Symbol": "MSFT", "Price": "$423.12", "Change": "+15.67", "Change%": "+3.85%"},
            {"Symbol": "GOOGL", "Price": "$148.23", "Change": "+6.89", "Change%": "+4.87%"},
            {"Symbol": "AMZN", "Price": "$145.67", "Change": "+5.23", "Change%": "+3.72%"},
        ]
    elif category == "losers":
        movers = [
            {"Symbol": "META", "Price": "$485.23", "Change": "-25.34", "Change%": "-4.97%"},
            {"Symbol": "NFLX", "Price": "$567.89", "Change": "-18.45", "Change%": "-3.15%"},
            {"Symbol": "TSLA", "Price": "$238.45", "Change": "-12.34", "Change%": "-4.93%"},
            {"Symbol": "UBER", "Price": "$67.89", "Change": "-2.45", "Change%": "-3.48%"},
            {"Symbol": "SNAP", "Price": "$12.34", "Change": "-0.89", "Change%": "-6.72%"},
        ]
    else:  # active
        movers = [
            {"Symbol": "SPY", "Price": "$478.34", "Change": "+2.45", "Change%": "+0.51%"},
            {"Symbol": "QQQ", "Price": "$387.12", "Change": "-1.23", "Change%": "-0.32%"},
            {"Symbol": "IWM", "Price": "$198.45", "Change": "+3.67", "Change%": "+1.88%"},
            {"Symbol": "VTI", "Price": "$234.56", "Change": "+1.23", "Change%": "+0.53%"},
            {"Symbol": "VOO", "Price": "$456.78", "Change": "+2.34", "Change%": "+0.51%"},
        ]
    
    # Display as a clean table
    df = pd.DataFrame(movers)
    
    # Add some styling with markdown
    for _, row in df.iterrows():
        col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
        
        with col1:
            st.markdown(f"**{row['Symbol']}**")
        with col2:
            st.markdown(f"{row['Price']}")
        with col3:
            color = "🟢" if row['Change'].startswith('+') else "🔴"
            st.markdown(f"{color} {row['Change']}")
        with col4:
            st.markdown(f"{row['Change%']}")

def display_alerts_panel():
    """Display active alerts"""
    
    alerts = [
        {"Time": "10:32 AM", "Type": "🔴 Critical", "Message": "AAPL broke resistance at $195"},
        {"Time": "10:15 AM", "Type": "🟡 Warning", "Message": "High volume spike in TSLA"},
        {"Time": "09:45 AM", "Type": "🟢 Info", "Message": "Market opened above previous close"},
        {"Time": "09:30 AM", "Type": "🟡 Warning", "Message": "NVDA approaching overbought levels"},
        {"Time": "09:15 AM", "Type": "🟢 Info", "Message": "Tech sector showing strength"},
    ]
    
    for alert in alerts:
        with st.container():
            st.markdown(f"""
            <div style="
                padding: 0.5rem;
                margin: 0.25rem 0;
                border-left: 3px solid #1f77b4;
                background-color: #f8f9fa;
                border-radius: 5px;
                font-size: 0.85rem;
            ">
                <small style="color: #666;">{alert['Time']}</small><br>
                <strong>{alert['Type']}</strong><br>
                {alert['Message']}
            </div>
            """, unsafe_allow_html=True)

def display_sector_performance():
    """Display sector performance chart"""
    
    sectors = {
        'Technology': 2.45,
        'Healthcare': 1.23,
        'Financial': -0.87,
        'Energy': -1.45,
        'Consumer': 0.67,
        'Industrial': 1.89,
        'Materials': -0.23,
        'Utilities': 0.45
    }
    
    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=list(sectors.values()),
        y=list(sectors.keys()),
        orientation='h',
        marker_color=['green' if x > 0 else 'red' for x in sectors.values()],
        text=[f"{x:+.1f}%" for x in sectors.values()],
        textposition='inside'
    ))
    
    fig.update_layout(
        title="Sector Performance (%)",
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_market_news():
    """Display market news (placeholder)"""
    
    news = [
        "📈 Fed signals potential rate cuts in Q2",
        "💼 Tech earnings beat expectations",
        "⚡ Oil prices surge on supply concerns", 
        "💵 Dollar strengthens against major currencies",
        "🏭 Manufacturing data shows resilience",
        "📊 Jobs report exceeds forecasts"
    ]
    
    for item in news:
        st.markdown(f"• {item}")
        
    # Add refresh button
    if st.button("🔄 Refresh News", use_container_width=True):
        st.success("News refreshed!")