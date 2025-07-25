"""
Portfolio Page - Portfolio tracking and management
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

def show():
    """Portfolio management page"""
    
    st.header("💼 Portfolio Management")
    
    # Portfolio summary
    display_portfolio_summary()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Portfolio performance chart
        st.subheader("📈 Portfolio Performance")
        display_portfolio_chart()
        
        # Holdings table
        st.subheader("📊 Current Holdings")
        display_holdings_table()
    
    with col2:
        # Portfolio actions
        st.subheader("⚡ Quick Actions")
        display_portfolio_actions()
        
        # Allocation pie chart
        st.subheader("🥧 Asset Allocation")
        display_allocation_chart()

def display_portfolio_summary():
    """Display portfolio summary metrics"""
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Value", 
            "$125,430.52", 
            "+$2,340.15 (1.9%)",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "Day P&L", 
            "+$1,245.67", 
            "+0.99%",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            "Total P&L", 
            "+$15,430.52", 
            "+14.0%",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            "Cash", 
            "$8,234.45", 
            "-$500.00",
            delta_color="inverse"
        )
    
    with col5:
        st.metric(
            "Positions", 
            "12", 
            "+1",
            delta_color="normal"
        )

def display_portfolio_chart():
    """Display portfolio performance over time"""
    
    # Generate sample portfolio data
    dates = [datetime.now() - timedelta(days=x) for x in range(90, 0, -1)]
    portfolio_values = []
    sp500_values = []
    
    base_portfolio = 110000
    base_sp500 = 110000
    
    for i, date in enumerate(dates):
        # Portfolio with some outperformance
        portfolio_change = np.random.normal(0.001, 0.02) + 0.0001  # Slight outperformance
        portfolio_value = base_portfolio * (1 + portfolio_change) ** i
        portfolio_values.append(portfolio_value)
        
        # S&P 500 benchmark
        sp500_change = np.random.normal(0.0008, 0.015)
        sp500_value = base_sp500 * (1 + sp500_change) ** i
        sp500_values.append(sp500_value)
    
    df = pd.DataFrame({
        'Date': dates,
        'Portfolio': portfolio_values,
        'S&P 500': sp500_values
    })
    
    fig = px.line(
        df, 
        x='Date', 
        y=['Portfolio', 'S&P 500'],
        title="Portfolio vs S&P 500 (90 Days)"
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    portfolio_return = ((portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]) * 100
    sp500_return = ((sp500_values[-1] - sp500_values[0]) / sp500_values[0]) * 100
    alpha = portfolio_return - sp500_return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Portfolio Return (90d)", f"{portfolio_return:.2f}%")
    with col2:
        st.metric("S&P 500 Return (90d)", f"{sp500_return:.2f}%") 
    with col3:
        st.metric("Alpha", f"{alpha:.2f}%", delta_color="normal" if alpha > 0 else "inverse")

def display_holdings_table():
    """Display current holdings table"""
    
    holdings_data = [
        {
            "Symbol": "AAPL",
            "Company": "Apple Inc.",
            "Shares": 50,
            "Avg Cost": "$185.42",
            "Current Price": "$195.67",
            "Market Value": "$9,783.50",
            "P&L": "+$512.50",
            "P&L%": "+5.5%",
            "Weight": "7.8%"
        },
        {
            "Symbol": "MSFT", 
            "Company": "Microsoft Corp.",
            "Shares": 25,
            "Avg Cost": "$410.23",
            "Current Price": "$423.12",
            "Market Value": "$10,578.00",
            "P&L": "+$322.25",
            "P&L%": "+3.1%",
            "Weight": "8.4%"
        },
        {
            "Symbol": "GOOGL",
            "Company": "Alphabet Inc.",
            "Shares": 30,
            "Avg Cost": "$142.67",
            "Current Price": "$148.23",
            "Market Value": "$4,446.90",
            "P&L": "+$166.80",
            "P&L%": "+3.9%",
            "Weight": "3.5%"
        },
        {
            "Symbol": "TSLA",
            "Company": "Tesla Inc.",
            "Shares": 15,
            "Avg Cost": "$245.67",
            "Current Price": "$238.45",
            "Market Value": "$3,576.75",
            "P&L": "-$108.30",
            "P&L%": "-2.9%",
            "Weight": "2.9%"
        },
        {
            "Symbol": "NVDA",
            "Company": "NVIDIA Corp.",
            "Shares": 8,
            "Avg Cost": "$820.45",
            "Current Price": "$875.23",
            "Market Value": "$7,001.84",
            "P&L": "+$438.24",
            "P&L%": "+6.7%",
            "Weight": "5.6%"
        }
    ]
    
    df = pd.DataFrame(holdings_data)
    
    # Style the dataframe
    def style_pnl(val):
        if val.startswith('-'):
            return 'color: red'
        elif val.startswith('+'):
            return 'color: green'
        return ''
    
    styled_df = df.style.applymap(style_pnl, subset=['P&L', 'P&L%'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Add/Edit position buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("➕ Add Position", use_container_width=True):
            st.info("Add Position dialog would open here")
    with col2:
        if st.button("✏️ Edit Position", use_container_width=True):
            st.info("Edit Position dialog would open here")
    with col3:
        if st.button("📊 Analyze", use_container_width=True):
            st.info("Position analysis would open here")

def display_portfolio_actions():
    """Display portfolio quick actions"""
    
    st.markdown("**🔍 Quick Search**")
    search_symbol = st.text_input("Enter symbol", placeholder="e.g., AAPL")
    
    if search_symbol:
        st.success(f"Found: {search_symbol.upper()}")
        col1, col2 = st.columns(2)
        with col1:
            st.button("📈 Buy", use_container_width=True)
        with col2:
            st.button("📉 Sell", use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("**💰 Cash Management**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💵 Add Cash", use_container_width=True):
            st.info("Cash deposit dialog")
    with col2:
        if st.button("💸 Withdraw", use_container_width=True):
            st.info("Cash withdrawal dialog")
    
    st.markdown("---")
    
    st.markdown("**📋 Portfolio Tools**")
    if st.button("📊 Rebalance", use_container_width=True):
        st.info("Portfolio rebalancing tool")
    
    if st.button("📈 Backtest", use_container_width=True):
        st.info("Backtesting interface")
    
    if st.button("📄 Generate Report", use_container_width=True):
        st.info("Portfolio report generation")

def display_allocation_chart():
    """Display asset allocation pie chart"""
    
    allocation_data = {
        'Technology': 45.2,
        'Healthcare': 15.8,
        'Financial': 12.3,
        'Consumer': 10.1,
        'Energy': 8.7,
        'Cash': 7.9
    }
    
    fig = px.pie(
        values=list(allocation_data.values()),
        names=list(allocation_data.keys()),
        title="Asset Allocation"
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=300)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Target vs Actual comparison
    st.markdown("**🎯 Allocation vs Target**")
    
    target_data = {
        'Technology': {'actual': 45.2, 'target': 40.0},
        'Healthcare': {'actual': 15.8, 'target': 20.0},
        'Financial': {'actual': 12.3, 'target': 15.0},
        'Consumer': {'actual': 10.1, 'target': 10.0},
        'Energy': {'actual': 8.7, 'target': 10.0},
        'Cash': {'actual': 7.9, 'target': 5.0}
    }
    
    for sector, data in target_data.items():
        diff = data['actual'] - data['target']
        delta_color = "normal" if abs(diff) < 2 else "inverse"
        st.metric(
            sector,
            f"{data['actual']:.1f}%",
            f"{diff:+.1f}% vs target",
            delta_color=delta_color
        )

# This is important for Streamlit multi-page apps
if __name__ == "__main__":
    show()
    
# This ensures it works in multi-page app
    show()