# pages/dashboard_page.py
"""
Enhanced Dashboard Page - Main market overview with technical analysis
Restored version with all features to match the quality of other pages
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from root
from valid_ticker_filter import TrendAnalyzer
from src.config import get_api_key, TREND_FILE, LOG_FILE, REFRESH_INTERVAL

# Import technical analysis modules
try:
    from technical_indicators import TechnicalIndicators, TechnicalAnalysis, create_sample_ohlcv_data
    TECHNICAL_AVAILABLE = True
except ImportError as e:
    st.error(f"Technical Analysis Import Error: {e}")
    st.info("Make sure technical_indicators.py is in your project root")
    TECHNICAL_AVAILABLE = False

def show():
    """Enhanced dashboard page with technical analysis"""
    
    st.header("🏠 Enhanced Market Dashboard")
    
    if not TECHNICAL_AVAILABLE:
        st.warning("⚠️ Technical indicators not available. Please check technical_indicators.py")
        # Fall back to basic dashboard
        show_basic_dashboard()
        return
    
    # Top-level controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### 📊 Real-time Market Overview with Technical Analysis")
    
    with col2:
        refresh_rate = st.selectbox("Refresh Rate", ["30s", "1m", "5m", "15m"], index=1)
    
    with col3:
        if st.button("🔄 Refresh Data", use_container_width=True):
            try:
                analyzer = TrendAnalyzer()
                analyzer.run_analysis()
                st.success("Data refreshed!")
                st.rerun()
            except Exception as e:
                st.error(f"Refresh failed: {str(e)}")
    
    # Market summary metrics
    display_market_summary()
    
    # Main content area
    main_tab, technical_tab, signals_tab = st.tabs(["📈 Market Overview", "🔧 Technical Analysis", "🎯 Trading Signals"])
    
    with main_tab:
        display_main_overview()
    
    with technical_tab:
        display_technical_analysis()
    
    with signals_tab:
        display_trading_signals()

def show_basic_dashboard():
    """Fallback basic dashboard if technical indicators not available"""
    # Initialize session state
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()
    
    # Header with refresh
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.subheader("Real-Time Market Trends")
    
    with col2:
        st.metric("Last Update", st.session_state.last_update.strftime("%H:%M:%S"))
    
    with col3:
        if st.button("🔄 Refresh"):
            try:
                analyzer = TrendAnalyzer()
                analyzer.run_analysis()
                st.session_state.last_update = datetime.now()
                st.success("Data refreshed!")
                st.rerun()
            except Exception as e:
                st.error(f"Refresh failed: {str(e)}")
    
    # Load and display basic data
    try:
        if TREND_FILE.exists():
            df = pd.read_csv(TREND_FILE)
            if not df.empty:
                display_basic_metrics(df)
                display_basic_table(df)
        else:
            st.warning("No trend data found. Running initial analysis...")
            analyzer = TrendAnalyzer()
            df = analyzer.run_analysis()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

def display_market_summary():
    """Display enhanced market summary with technical indicators"""
    
    st.markdown("### 📊 Market Summary")
    
    # Top row - Major indices
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("S&P 500", "4,783.45", "+23.45 (0.49%)", delta_color="normal")
    
    with col2:
        st.metric("NASDAQ", "15,234.67", "-45.23 (-0.30%)", delta_color="inverse")
    
    with col3:
        st.metric("Dow Jones", "37,234.89", "+145.67 (0.39%)", delta_color="normal")
    
    with col4:
        st.metric("VIX", "18.45", "-2.1%", delta_color="inverse")
    
    with col5:
        st.metric("USD/EUR", "1.0845", "+0.12%", delta_color="normal")
    
    # Second row - Market indicators
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Market Cap", "$45.2T", "+1.2%", delta_color="normal")
    
    with col2:
        st.metric("Daily Volume", "12.5B", "+5.4%", delta_color="normal")
    
    with col3:
        st.metric("Advancing", "2,847", "+127", delta_color="normal")
    
    with col4:
        st.metric("Declining", "1,892", "-89", delta_color="inverse")
    
    with col5:
        st.metric("Unchanged", "261", "+12", delta_color="off")

def display_main_overview():
    """Display main market overview with enhanced charts"""
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Enhanced market chart with technical indicators
        st.subheader("📈 Market Performance")
        
        # Symbol selector
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            selected_symbol = st.selectbox(
                "Select Symbol",
                ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"],
                index=1
            )
        with col2:
            timeframe = st.selectbox("Timeframe", ["1D", "5D", "1M", "3M", "6M"], index=2)
        with col3:
            chart_type = st.selectbox("Chart Type", ["Candlestick", "Line"], index=0)
        
        # Create enhanced chart
        display_enhanced_price_chart(selected_symbol, timeframe, chart_type)
        
        # Market movers
        st.subheader("🚀 Top Movers with Technical Signals")
        display_enhanced_movers()
    
    with col_right:
        # Technical indicators panel
        st.subheader("🔧 Technical Indicators")
        display_technical_panel(selected_symbol)
        
        # Market alerts
        st.subheader("🚨 Smart Alerts")
        display_smart_alerts()
        
        # Sector performance
        st.subheader("🏭 Sector Performance")
        display_enhanced_sector_performance()

def display_enhanced_price_chart(symbol: str, timeframe: str, chart_type: str):
    """Display enhanced price chart with technical indicators"""
    
    # Create sample data (replace with real data integration)
    days_map = {"1D": 1, "5D": 5, "1M": 30, "3M": 90, "6M": 180}
    days = days_map.get(timeframe, 30)
    
    try:
        sample_data = create_sample_ohlcv_data(symbol, days)
        
        # Chart options
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            show_sma = st.checkbox("Moving Averages", value=True, key=f"sma_{symbol}")
        with col2:
            show_bb = st.checkbox("Bollinger Bands", key=f"bb_{symbol}")
        with col3:
            show_vwap = st.checkbox("VWAP", key=f"vwap_{symbol}")
        with col4:
            show_volume = st.checkbox("Volume", value=True, key=f"vol_{symbol}")
        
        # Create technical analysis
        ta = TechnicalAnalysis(sample_data)
        df_with_indicators = ta.add_all_indicators()
        
        # Create chart based on type
        if chart_type == "Candlestick":
            fig = create_candlestick_chart(df_with_indicators, symbol, show_sma, show_bb, show_vwap, show_volume, chart_type)
        else:
            fig = create_line_chart(df_with_indicators, symbol, show_sma, show_bb, show_vwap, chart_type)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display technical summary
        display_technical_summary_panel(ta, symbol)
        
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")

def create_candlestick_chart(df: pd.DataFrame, symbol: str, show_sma: bool, show_bb: bool, show_vwap: bool, show_volume: bool, chart_type: str = "Candlestick"):
    """Create candlestick chart with technical indicators"""
    
    # Create subplots
    rows = 2 if show_volume else 1
    row_heights = [0.7, 0.3] if show_volume else [1.0]
    
    fig = make_subplots(
        rows=rows, cols=1,
        subplot_titles=[f'{symbol} Price Chart', 'Volume'] if show_volume else [f'{symbol} Price Chart'],
        vertical_spacing=0.05,
        row_heights=row_heights
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#00C851',
            decreasing_line_color='#FF4444'
        ),
        row=1, col=1
    )
    
    # Add technical indicators
    if show_sma and 'SMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', 
                      line=dict(color='orange', width=2)),
            row=1, col=1
        )
        if 'SMA_50' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', 
                          line=dict(color='red', width=2)),
                row=1, col=1
            )
    
    if show_bb and all(col in df.columns for col in ['BB_Upper', 'BB_Lower']):
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', 
                      line=dict(color='gray', width=1), opacity=0.7),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', 
                      line=dict(color='gray', width=1), fill='tonexty', opacity=0.2),
            row=1, col=1
        )
    
    if show_vwap and 'VWAP' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', 
                      line=dict(color='yellow', width=2)),
            row=1, col=1
        )
    
    # Volume chart
    if show_volume and 'volume' in df.columns:
        colors = ['#00C851' if close >= open else '#FF4444' 
                 for close, open in zip(df['close'], df['open'])]
        
        fig.add_trace(
            go.Bar(x=df.index, y=df['volume'], name='Volume', 
                  marker_color=colors, opacity=0.6),
            row=2, col=1
        )
    
    fig.update_layout(
        height=600 if show_volume else 450,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        title=f"{symbol} - {chart_type} Chart"
    )
    
    return fig

def create_line_chart(df: pd.DataFrame, symbol: str, show_sma: bool, show_bb: bool, show_vwap: bool, chart_type: str = "Line"):
    """Create line chart with technical indicators"""
    
    fig = go.Figure()
    
    # Price line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['close'],
            mode='lines',
            name=f'{symbol} Price',
            line=dict(color='#00ff88', width=2)
        )
    )
    
    # Add technical indicators
    if show_sma and 'SMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SMA_20'],
                name='SMA 20',
                line=dict(color='orange', width=2)
            )
        )
        
        if 'SMA_50' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['SMA_50'],
                    name='SMA 50',
                    line=dict(color='red', width=2)
                )
            )
    
    if show_bb and all(col in df.columns for col in ['BB_Upper', 'BB_Lower']):
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', 
                      line=dict(color='gray', width=1), opacity=0.7)
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', 
                      line=dict(color='gray', width=1), fill='tonexty', opacity=0.2)
        )
    
    if show_vwap and 'VWAP' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', 
                      line=dict(color='yellow', width=2))
        )
    
    fig.update_layout(
        title=f"{symbol} - {chart_type} Chart",
        height=450,
        xaxis_title="Date",
        yaxis_title="Price ($)"
    )
    
    return fig

def display_technical_summary_panel(ta: TechnicalAnalysis, symbol: str):
    """Display technical summary panel"""
    
    try:
        signals = ta.get_signals()
        df_with_indicators = ta.add_all_indicators()
        latest = df_with_indicators.iloc[-1]
        
        st.markdown("#### 📊 Technical Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${latest['close']:.2f}")
        
        with col2:
            if 'RSI' in df_with_indicators.columns and not pd.isna(latest['RSI']):
                rsi_value = latest['RSI']
                rsi_delta = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
                st.metric("RSI (14)", f"{rsi_value:.1f}", delta=rsi_delta)
        
        with col3:
            if 'MACD' in df_with_indicators.columns and not pd.isna(latest['MACD']):
                st.metric("MACD", f"{latest['MACD']:.3f}")
        
        with col4:
            if 'ATR' in df_with_indicators.columns and not pd.isna(latest['ATR']):
                st.metric("ATR (14)", f"${latest['ATR']:.2f}")
        
        # Trading signals
        st.markdown("#### 🎯 Current Signals")
        for indicator, signal in signals.items():
            if 'Buy' in signal or 'Bullish' in signal or 'Oversold' in signal:
                st.success(f"🟢 **{indicator}**: {signal}")
            elif 'Sell' in signal or 'Bearish' in signal or 'Overbought' in signal:
                st.error(f"🔴 **{indicator}**: {signal}")
            else:
                st.info(f"🟡 **{indicator}**: {signal}")
                
    except Exception as e:
        st.error(f"Error in technical summary: {str(e)}")

def display_enhanced_movers():
    """Display top movers with technical signals"""
    
    # Load real data if available
    try:
        if TREND_FILE.exists():
            df = pd.read_csv(TREND_FILE)
            
            # Get top gainers and losers
            gainers = df.nlargest(5, 'Momentum_%')
            losers = df.nsmallest(5, 'Momentum_%')
            
            # Add technical signals (simulated for now)
            for idx, row in gainers.iterrows():
                momentum = row['Momentum_%']
                gainers.loc[idx, 'RSI'] = str(round(50 + momentum * 2, 1))
                gainers.loc[idx, 'Signal'] = "🟢 BUY" if momentum > 3 else "🟡 HOLD"
                gainers.loc[idx, 'Volume'] = "High" if momentum > 2 else "Normal"
            
            for idx, row in losers.iterrows():
                momentum = row['Momentum_%']
                losers.loc[idx, 'RSI'] = str(round(50 + momentum * 2, 1))
                losers.loc[idx, 'Signal'] = "🟢 BUY" if momentum < -3 else "🟡 HOLD"
                losers.loc[idx, 'Volume'] = "High" if abs(momentum) > 2 else "Normal"
            
            gainers_data = gainers.to_dict('records')
            losers_data = losers.to_dict('records')
            
        else:
            # Fallback to sample data
            gainers_data = [
                {
                    "Symbol": "NVDA",
                    "Current_Price": 875.45,
                    "Momentum_%": 5.45,
                    "RSI": "42.3",
                    "Signal": "🟢 BUY",
                    "Volume": "High"
                },
                {
                    "Symbol": "AAPL",
                    "Current_Price": 195.67,
                    "Momentum_%": 4.46,
                    "RSI": "58.7",
                    "Signal": "🟡 HOLD",
                    "Volume": "Normal"
                }
            ]
            
            losers_data = [
                {
                    "Symbol": "META",
                    "Current_Price": 485.23,
                    "Momentum_%": -4.97,
                    "RSI": "28.4",
                    "Signal": "🟢 BUY",
                    "Volume": "High"
                },
                {
                    "Symbol": "TSLA",
                    "Current_Price": 238.45,
                    "Momentum_%": -4.93,
                    "RSI": "25.8",
                    "Signal": "🟢 BUY",
                    "Volume": "Very High"
                }
            ]
    except Exception as e:
        st.error(f"Error loading movers data: {str(e)}")
        return
    
    tab1, tab2 = st.tabs(["📈 Top Gainers", "📉 Top Losers"])
    
    with tab1:
        display_enhanced_movers_table(gainers_data, "gainers")
    
    with tab2:
        display_enhanced_movers_table(losers_data, "losers")

def display_enhanced_movers_table(movers_data, category):
    """Display enhanced movers table with technical data"""
    
    for i, mover in enumerate(movers_data):
        with st.container():
            col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1.5, 1, 1, 0.8, 1, 0.8])
            
            with col1:
                st.markdown(f"**{mover.get('Symbol', mover.get('Ticker', 'N/A'))}**")
            
            with col2:
                price = mover.get('Current_Price', 0)
                st.markdown(f"${price:.2f}" if isinstance(price, (int, float)) else str(price))
            
            with col3:
                momentum = mover.get('Momentum_%', 0)
                color = "🟢" if momentum > 0 else "🔴"
                st.markdown(f"{color} {momentum:+.2f}%")
            
            with col4:
                st.markdown(f"{momentum:.2f}%")
            
            with col5:
                rsi = mover.get('RSI', 'N/A')
                if isinstance(rsi, str):
                    rsi_value = float(rsi) if rsi != 'N/A' else 50
                else:
                    rsi_value = float(rsi)
                rsi_color = "🟢" if rsi_value < 30 else "🔴" if rsi_value > 70 else "🟡"
                st.markdown(f"{rsi_color} {rsi}")
            
            with col6:
                st.markdown(f"{mover.get('Signal', '🟡 HOLD')}")
            
            with col7:
                volume = mover.get('Volume', 'Normal')
                volume_color = "🔥" if volume == "Very High" else "📈" if volume == "High" else "📊"
                st.markdown(f"{volume_color}")
        
        if i < len(movers_data) - 1:
            st.markdown("---")

def display_technical_panel(symbol: str):
    """Display technical indicators panel"""
    
    try:
        # Create sample data for technical analysis
        sample_data = create_sample_ohlcv_data(symbol, 50)
        ta = TechnicalAnalysis(sample_data)
        df_with_indicators = ta.add_all_indicators()
        
        # Current technical values
        latest = df_with_indicators.iloc[-1]
        current_price = latest['close']
        
        # Display indicators
        st.markdown("#### 📊 Current Indicators")
        
        # RSI
        if 'RSI' in df_with_indicators.columns and not pd.isna(latest['RSI']):
            rsi = latest['RSI']
            rsi_color = "🟢" if rsi < 30 else "🔴" if rsi > 70 else "🟡"
            st.markdown(f"**RSI (14):** {rsi_color} {rsi:.1f}")
        
        # MACD
        if 'MACD' in df_with_indicators.columns and 'MACD_Signal' in df_with_indicators.columns:
            macd_current = latest['MACD']
            signal_current = latest['MACD_Signal']
            if not pd.isna(macd_current) and not pd.isna(signal_current):
                macd_signal = "🟢 Bullish" if macd_current > signal_current else "🔴 Bearish"
                st.markdown(f"**MACD:** {macd_signal}")
        
        # Bollinger Bands position
        if all(col in df_with_indicators.columns for col in ['BB_Upper', 'BB_Lower']):
            bb_upper = latest['BB_Upper']
            bb_lower = latest['BB_Lower']
            if not pd.isna(bb_upper) and not pd.isna(bb_lower):
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                bb_signal = "🔴 Upper" if bb_position > 0.8 else "🟢 Lower" if bb_position < 0.2 else "🟡 Middle"
                st.markdown(f"**Bollinger:** {bb_signal}")
        
        # Moving averages
        if all(col in df_with_indicators.columns for col in ['SMA_20', 'SMA_50']):
            sma_20 = latest['SMA_20']
            sma_50 = latest['SMA_50']
            if not pd.isna(sma_20) and not pd.isna(sma_50):
                ma_signal = "🟢 Bullish" if current_price > sma_20 > sma_50 else "🔴 Bearish" if current_price < sma_20 < sma_50 else "🟡 Mixed"
                st.markdown(f"**MA Trend:** {ma_signal}")
        
        # Overall signal
        signals = ta.get_signals()
        buy_signals = len([s for s in signals.values() if 'Buy' in s or 'Bullish' in s])
        sell_signals = len([s for s in signals.values() if 'Sell' in s or 'Bearish' in s])
        
        if buy_signals > sell_signals:
            overall = "BUY"
            overall_color = "🟢"
        elif sell_signals > buy_signals:
            overall = "SELL"
            overall_color = "🔴"
        else:
            overall = "NEUTRAL"
            overall_color = "🟡"
            
        st.markdown(f"**Overall:** {overall_color} {overall}")
        
        # Mini technical chart
        st.markdown("#### 📈 Mini Chart")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_with_indicators.index,
                y=df_with_indicators['close'],
                mode='lines',
                name='Price',
                line=dict(color='#00ff88', width=2)
            )
        )
        
        # Add SMA if available
        if 'SMA_20' in df_with_indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators.index,
                    y=df_with_indicators['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange', width=1)
                )
            )
        
        fig.update_layout(
            height=200,
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in technical panel: {str(e)}")

def display_smart_alerts():
    """Display smart alerts with technical triggers"""
    
    alerts = [
        {
            "Time": "10:32 AM",
            "Type": "🔴 Critical",
            "Symbol": "AAPL",
            "Message": "RSI oversold + price below BB lower band",
            "Action": "Strong Buy Signal"
        },
        {
            "Time": "10:15 AM",
            "Type": "🟡 Warning",
            "Symbol": "TSLA",
            "Message": "MACD bullish crossover confirmed",
            "Action": "Consider Long Position"
        },
        {
            "Time": "09:45 AM",
            "Type": "🟢 Info",
            "Symbol": "NVDA",
            "Message": "Breaking above 20-day SMA",
            "Action": "Monitor for Breakout"
        },
        {
            "Time": "09:30 AM",
            "Type": "🔴 Critical",
            "Symbol": "META",
            "Message": "High volume + RSI divergence",
            "Action": "Potential Reversal"
        }
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
                <div style="display: flex; justify-content: space-between;">
                    <small style="color: #666;">{alert['Time']}</small>
                    <strong>{alert['Symbol']}</strong>
                </div>
                <strong>{alert['Type']}</strong><br>
                {alert['Message']}<br>
                <em style="color: #007acc;">{alert['Action']}</em>
            </div>
            """, unsafe_allow_html=True)

def display_enhanced_sector_performance():
    """Display enhanced sector performance with technical analysis"""
    
    # Try to load real sector data
    try:
        if LOG_FILE.exists():
            sector_df = pd.read_csv(LOG_FILE)
            # Get most recent data for each sector
            latest_sectors = sector_df.groupby('Sector').last()
            
            sectors = {}
            for sector, data in latest_sectors.iterrows():
                sectors[sector] = {
                    'return': data.get('Avg_Momentum_%', 0),
                    'trend': '🟢 Bullish' if data.get('Avg_Momentum_%', 0) > 1 else '🔴 Bearish' if data.get('Avg_Momentum_%', 0) < -1 else '🟡 Neutral',
                    'rsi': 50 + data.get('Avg_Momentum_%', 0) * 5  # Simulated RSI
                }
        else:
            # Fallback to sample data
            sectors = {
                'Technology': {'return': 2.45, 'trend': '🟢 Bullish', 'rsi': 58.2},
                'Healthcare': {'return': 1.23, 'trend': '🟡 Neutral', 'rsi': 52.7},
                'Financial': {'return': -0.87, 'trend': '🔴 Bearish', 'rsi': 34.5},
                'Energy': {'return': -1.45, 'trend': '🔴 Bearish', 'rsi': 28.9},
                'Consumer': {'return': 0.67, 'trend': '🟡 Neutral', 'rsi': 48.3},
                'Industrial': {'return': 1.89, 'trend': '🟢 Bullish', 'rsi': 62.1}
            }
    except Exception as e:
        st.error(f"Error loading sector data: {str(e)}")
        sectors = {}
    
    if not sectors:
        st.info("No sector data available")
        return
    
    # Create enhanced sector chart
    fig = go.Figure()
    
    sector_names = list(sectors.keys())
    sector_returns = [data['return'] for data in sectors.values()]
    sector_colors = ['green' if x > 0 else 'red' for x in sector_returns]
    
    fig.add_trace(
        go.Bar(
            x=sector_returns,
            y=sector_names,
            orientation='h',
            marker_color=sector_colors,
            text=[f"{x:+.1f}%" for x in sector_returns],
            textposition='inside'
        )
    )
    
    fig.update_layout(
        title="Sector Performance (%)",
        height=250,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sector details
    st.markdown("#### 🎯 Sector Signals")
    for sector, data in sectors.items():
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**{sector}**")
        with col2:
            st.markdown(f"{data['trend']}")
        with col3:
            rsi_color = "🟢" if data['rsi'] < 30 else "🔴" if data['rsi'] > 70 else "🟡"
            st.markdown(f"{rsi_color} {data['rsi']:.1f}")

def display_technical_analysis():
    """Display dedicated technical analysis tab"""
    
    st.markdown("### 🔧 Advanced Technical Analysis")
    
    # Symbol selector
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        analysis_symbol = st.selectbox(
            "Symbol for Analysis",
            ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"],
            key="tech_analysis_symbol"
        )
    with col2:
        analysis_period = st.selectbox("Period", ["1M", "3M", "6M", "1Y"], index=1)
    with col3:
        if st.button("📊 Analyze", use_container_width=True):
            st.success(f"Analyzing {analysis_symbol}...")
    
    # Create comprehensive technical analysis
    days_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365}
    days = days_map.get(analysis_period, 90)
    
    try:
        sample_data = create_sample_ohlcv_data(analysis_symbol, days)
        ta = TechnicalAnalysis(sample_data)
        df_with_indicators = ta.add_all_indicators()
        
        # Technical indicators chart
        st.subheader("📊 Technical Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            show_rsi_tech = st.checkbox("RSI", value=True, key="tech_rsi")
        with col2:
            show_macd_tech = st.checkbox("MACD", value=True, key="tech_macd")
        with col3:
            show_stoch_tech = st.checkbox("Stochastic", key="tech_stoch")
        with col4:
            show_williams_tech = st.checkbox("Williams %R", key="tech_williams")
        
        # Create technical indicators subplots
        if any([show_rsi_tech, show_macd_tech, show_stoch_tech, show_williams_tech]):
            create_technical_indicators_chart(df_with_indicators, show_rsi_tech, show_macd_tech, show_stoch_tech, show_williams_tech)
        
        # Support/Resistance levels
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Support & Resistance")
            levels = ta.get_support_resistance()
            
            st.metric("Support", f"${levels['support']:.2f}")
            st.metric("Resistance", f"${levels['resistance']:.2f}")
            
            if 'support_levels' in levels:
                st.markdown("**Support Levels:**")
                for i, level in enumerate(levels['support_levels'][-3:], 1):
                    st.write(f"S{i}: ${level:.2f}")
        
        with col2:
            st.markdown("### 📊 Current Signals")
            signals = ta.get_signals()
            
            for indicator, signal in signals.items():
                if 'Buy' in signal or 'Bullish' in signal or 'Oversold' in signal:
                    st.success(f"🟢 **{indicator}**: {signal}")
                elif 'Sell' in signal or 'Bearish' in signal or 'Overbought' in signal:
                    st.error(f"🔴 **{indicator}**: {signal}")
                else:
                    st.info(f"🟡 **{indicator}**: {signal}")
                    
    except Exception as e:
        st.error(f"Error in technical analysis: {str(e)}")

def create_technical_indicators_chart(df: pd.DataFrame, show_rsi: bool, show_macd: bool, show_stoch: bool, show_williams: bool):
    """Create technical indicators chart"""
    
    # Count how many indicators to show
    indicators_count = sum([show_rsi, show_macd, show_stoch, show_williams])
    
    if indicators_count == 0:
        return
    
    # Create subplots
    fig = make_subplots(
        rows=indicators_count, cols=1,
        subplot_titles=[],
        vertical_spacing=0.05
    )
    
    row = 1
    
    # RSI
    if show_rsi and 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
            row=row, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=row, col=1)
        row += 1
    
    # MACD
    if show_macd and all(col in df.columns for col in ['MACD', 'MACD_Signal']):
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='red')),
            row=row, col=1
        )
        row += 1
    
    # Stochastic
    if show_stoch and all(col in df.columns for col in ['Stoch_K', 'Stoch_D']):
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Stoch_K'], name='%K', line=dict(color='blue')),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Stoch_D'], name='%D', line=dict(color='red')),
            row=row, col=1
        )
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=row, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=row, col=1)
        row += 1
    
    # Williams %R
    if show_williams and 'Williams_R' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Williams_R'], name='Williams %R', line=dict(color='orange')),
            row=row, col=1
        )
        fig.add_hline(y=-20, line_dash="dash", line_color="red", row=row, col=1)
        fig.add_hline(y=-80, line_dash="dash", line_color="green", row=row, col=1)
    
    fig.update_layout(height=300 * indicators_count, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

def display_trading_signals():
    """Display trading signals and recommendations"""
    
    st.markdown("### 🎯 Trading Signals & Recommendations")
    
    # Signal strength meter
    st.markdown("#### 📊 Signal Strength")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 🟢 Strong Buy Signals")
        strong_buys = [
            {"Symbol": "META", "Reason": "RSI <30 + MACD Bullish Cross", "Confidence": "85%"},
            {"Symbol": "TSLA", "Reason": "Double Bottom + Volume Spike", "Confidence": "78%"}
        ]
        
        for signal in strong_buys:
            st.markdown(f"**{signal['Symbol']}** - {signal['Confidence']}")
            st.markdown(f"_{signal['Reason']}_")
            st.markdown("---")
    
    with col2:
        st.markdown("##### 🟡 Hold/Watch Signals")
        holds = [
            {"Symbol": "AAPL", "Reason": "Consolidating near SMA", "Confidence": "62%"},
            {"Symbol": "MSFT", "Reason": "Mixed technical signals", "Confidence": "55%"}
        ]
        
        for signal in holds:
            st.markdown(f"**{signal['Symbol']}** - {signal['Confidence']}")
            st.markdown(f"_{signal['Reason']}_")
            st.markdown("---")
    
    with col3:
        st.markdown("##### 🔴 Sell/Avoid Signals")
        sells = [
            {"Symbol": "NFLX", "Reason": "Bearish Divergence + Weak Volume", "Confidence": "73%"}
        ]
        
        for signal in sells:
            st.markdown(f"**{signal['Symbol']}** - {signal['Confidence']}")
            st.markdown(f"_{signal['Reason']}_")
            st.markdown("---")
    
    # Market outlook
    st.markdown("#### 🔮 Market Outlook")
    
    outlook_data = {
        "Short Term (1-7 days)": {"Signal": "🟡 Neutral", "Description": "Mixed technical signals, waiting for breakout direction"},
        "Medium Term (1-4 weeks)": {"Signal": "🟢 Bullish", "Description": "Strong momentum indicators, sector rotation favorable"},
        "Long Term (1-6 months)": {"Signal": "🟢 Bullish", "Description": "Fundamental support, technical trends intact"}
    }
    
    for timeframe, data in outlook_data.items():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"**{timeframe}**")
            st.markdown(f"{data['Signal']}")
        with col2:
            st.markdown(f"{data['Description']}")
        st.markdown("---")
    
    # Risk assessment
    st.markdown("#### ⚖️ Risk Assessment")
    
    risk_metrics = {
        "Market Volatility": {"Level": "Medium", "Color": "🟡", "Value": "VIX: 18.5"},
        "Sector Concentration": {"Level": "Low", "Color": "🟢", "Value": "Diversified"},
        "Technical Risk": {"Level": "Low", "Color": "🟢", "Value": "Support levels holding"},
        "Momentum Risk": {"Level": "Medium", "Color": "🟡", "Value": "Some divergences"}
    }
    
    cols = st.columns(len(risk_metrics))
    for i, (metric, data) in enumerate(risk_metrics.items()):
        with cols[i]:
            st.markdown(f"**{metric}**")
            st.markdown(f"{data['Color']} {data['Level']}")
            st.markdown(f"_{data['Value']}_")

def display_basic_metrics(df):
    """Display basic metrics for fallback dashboard"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_momentum = df['Momentum_%'].mean()
        st.metric("Avg Momentum", f"{avg_momentum:.2f}%", delta=f"{avg_momentum:.2f}%")
    
    with col2:
        positive_stocks = len(df[df['Momentum_%'] > 0])
        st.metric("Stocks Up", f"{positive_stocks}/{len(df)}", delta=f"{(positive_stocks/len(df)*100):.1f}%")
    
    with col3:
        top_performer = df.iloc[0]
        st.metric("Top Performer", top_performer['Ticker'], delta=f"{top_performer['Momentum_%']:.2f}%")
    
    with col4:
        bottom_performer = df.iloc[-1]
        st.metric("Bottom Performer", bottom_performer['Ticker'], delta=f"{bottom_performer['Momentum_%']:.2f}%")

def display_basic_table(df):
    """Display basic table for fallback dashboard"""
    st.subheader("Live Trend Scores")
    
    df_display = df.copy()
    df_display['Signal'] = df_display['Signal'].apply(
        lambda x: f"{'🟢' if 'BUY' in x else '🔴' if 'SELL' in x else '🟡'} {x}"
    )
    
    st.dataframe(
        df_display[['Ticker', 'Sector', 'Current_Price', 'Momentum_%', 'Trend_Score', 'Signal']],
        use_container_width=True,
        height=400
    )

# Call show() once for multi-page app
show()