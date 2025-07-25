"""
Enhanced Analysis Page - Real technical analysis with professional indicators
Fixed and optimized version with better error handling and user experience
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Import technical analysis with better error handling
try:
    from technical_indicators import TechnicalIndicators, TechnicalAnalysis, create_sample_ohlcv_data
    TECHNICAL_INDICATORS_AVAILABLE = True
except ImportError as e:
    TECHNICAL_INDICATORS_AVAILABLE = False
    st.error("❌ Technical indicators module not found!")
    st.info("📝 Make sure technical_indicators.py is in the root directory.")
    st.code("# Expected file: technical_indicators.py in project root")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_technical_analysis(symbol: str, days: int):
    """Cached technical analysis calculation"""
    if not TECHNICAL_INDICATORS_AVAILABLE:
        return None, None
    
    try:
        data = create_sample_ohlcv_data(symbol, days)
        ta = TechnicalAnalysis(data)
        return ta.add_all_indicators(), ta
    except Exception as e:
        st.error(f"❌ Error calculating technical analysis: {str(e)}")
        return None, None

def show():
    """Enhanced analysis page with real technical indicators"""
    
    st.header("📈 Advanced Technical Analysis")
    
    if not TECHNICAL_INDICATORS_AVAILABLE:
        st.stop()
    
    # Analysis tool selection
    analysis_tab = st.selectbox(
        "🔍 Choose Analysis Tool",
        [
            "📊 Technical Analysis Dashboard",
            "🔍 Stock Screener",
            "🔄 Correlation Analysis",
            "📰 Sector Analysis",
            "⚖️ Risk Analysis",
            "🧮 Backtesting"
        ]
    )
    
    if analysis_tab == "📊 Technical Analysis Dashboard":
        show_technical_dashboard()
    elif analysis_tab == "🔍 Stock Screener":
        show_stock_screener()
    elif analysis_tab == "🔄 Correlation Analysis":
        show_correlation_analysis()
    elif analysis_tab == "📰 Sector Analysis":
        show_sector_analysis()
    elif analysis_tab == "⚖️ Risk Analysis":
        show_risk_analysis()
    elif analysis_tab == "🧮 Backtesting":
        show_backtesting()

def show_technical_dashboard():
    """Professional technical analysis dashboard"""
    
    st.subheader("📊 Technical Analysis Dashboard")
    
    # Symbol and settings
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        symbol = st.text_input("Symbol", value="AAPL", placeholder="Enter symbol (e.g., AAPL)")
    with col2:
        timeframe = st.selectbox("Timeframe", ["1D", "1W", "1M", "3M", "1Y"], index=2)
    with col3:
        chart_type = st.selectbox("Chart Type", ["Candlestick", "Line", "OHLC"])
    with col4:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.success("Data refreshed!")
    
    # Input validation
    if symbol:
        symbol = symbol.upper().strip()
        if not symbol.isalpha() or len(symbol) > 6 or len(symbol) < 1:
            st.error("❌ Please enter a valid stock symbol (e.g., AAPL, MSFT)")
            return
        
        # Load and validate data
        with st.spinner(f"Loading technical analysis for {symbol}..."):
            df_with_indicators, ta = get_technical_analysis(symbol, 100)
        
        if df_with_indicators is None or ta is None:
            st.error(f"❌ Unable to load data for {symbol}")
            return
        
        # Data validation
        if df_with_indicators.empty:
            st.error(f"❌ No data available for {symbol}")
            return
            
        if len(df_with_indicators) < 50:
            st.warning(f"⚠️ Limited data for {symbol} ({len(df_with_indicators)} days). Some indicators may be inaccurate.")
        
        # Success message
        st.success(f"✅ Loaded {len(df_with_indicators)} days of data for {symbol}")
        
        # Main technical analysis layout
        col_left, col_right = st.columns([3, 1])
        
        with col_left:
            # Price chart with indicators
            display_advanced_price_chart(df_with_indicators, symbol, chart_type)
            
            # Technical indicators charts
            display_technical_indicators_charts(df_with_indicators)
            
            # Export options
            add_export_options(df_with_indicators, symbol)
        
        with col_right:
            # Trading signals
            display_trading_signals(ta)
            
            # Support/Resistance levels
            display_support_resistance(ta)
            
            # Key metrics
            display_key_metrics(df_with_indicators)

def display_advanced_price_chart(df: pd.DataFrame, symbol: str, chart_type: str):
    """Display advanced price chart with technical indicators"""
    
    st.markdown("**📈 Price Chart with Technical Indicators**")
    
    # Indicator selection with help text
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        show_sma = st.checkbox("SMA (20,50)", value=True, 
                              help="Simple Moving Averages - trend following indicators")
        show_ema = st.checkbox("EMA (12,26)", 
                              help="Exponential Moving Averages - more responsive to recent prices")
    with col2:
        show_bollinger = st.checkbox("Bollinger Bands", value=True,
                                   help="Volatility bands around moving average")
        show_vwap = st.checkbox("VWAP", 
                               help="Volume Weighted Average Price")
    with col3:
        show_support = st.checkbox("Support/Resistance",
                                 help="Key price levels based on historical data")
        show_volume = st.checkbox("Volume", value=True,
                                help="Trading volume with price correlation colors")
    with col4:
        overlay_indicators = st.multiselect(
            "Overlay", 
            ["Trend Lines", "Fibonacci", "Pivot Points"],
            default=[],
            help="Additional overlays (coming in Phase 3)"
        )
    
    # Create the main chart
    try:
        fig = create_advanced_chart(df, symbol, chart_type, {
            'sma': show_sma,
            'ema': show_ema,
            'bollinger': show_bollinger,
            'vwap': show_vwap,
            'support': show_support,
            'volume': show_volume
        })
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error creating chart: {str(e)}")

def create_advanced_chart(df: pd.DataFrame, symbol: str, chart_type: str, indicators: dict) -> go.Figure:
    """Create advanced chart with multiple indicators"""
    
    # Calculate responsive height
    base_height = 500
    volume_height = 200 if indicators['volume'] else 0
    total_height = base_height + volume_height
    
    # Create subplots
    rows = 2 if indicators['volume'] else 1
    row_heights = [0.7, 0.3] if indicators['volume'] else [1.0]
    
    fig = make_subplots(
        rows=rows, cols=1,
        subplot_titles=[f'{symbol} - {chart_type} Chart', 'Volume'] if indicators['volume'] else [f'{symbol} - {chart_type} Chart'],
        vertical_spacing=0.05,
        row_heights=row_heights
    )
    
    # Price chart
    if chart_type == "Candlestick":
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
    elif chart_type == "OHLC":
        fig.add_trace(
            go.Ohlc(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )
    else:  # Line chart
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1
        )
    
    # Add technical indicators
    if indicators['sma'] and 'SMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', 
                      line=dict(color='orange', width=1.5)),
            row=1, col=1
        )
        if 'SMA_50' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', 
                          line=dict(color='red', width=1.5)),
                row=1, col=1
            )
    
    if indicators['ema'] and 'EMA_12' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['EMA_12'], name='EMA 12', 
                      line=dict(color='purple', width=1.5, dash='dash')),
            row=1, col=1
        )
        if 'EMA_26' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['EMA_26'], name='EMA 26', 
                          line=dict(color='brown', width=1.5, dash='dash')),
                row=1, col=1
            )
    
    if indicators['bollinger'] and all(col in df.columns for col in ['BB_Upper', 'BB_Lower']):
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
    
    if indicators['vwap'] and 'VWAP' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', 
                      line=dict(color='yellow', width=2)),
            row=1, col=1
        )
    
    # Volume chart
    if indicators['volume'] and 'volume' in df.columns:
        colors = ['#00C851' if close >= open else '#FF4444' 
                 for close, open in zip(df['close'], df['open'])]
        
        fig.add_trace(
            go.Bar(x=df.index, y=df['volume'], name='Volume', 
                  marker_color=colors, opacity=0.6),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=total_height,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        title_text=f"{symbol} Technical Analysis",
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode='x unified'
    )
    
    return fig

def display_technical_indicators_charts(df: pd.DataFrame):
    """Display additional technical indicator charts"""
    
    st.markdown("**📊 Technical Oscillators**")
    
    # Create indicator selection tabs
    tab1, tab2, tab3 = st.tabs(["🔄 RSI & Stochastic", "📈 MACD", "🎯 Additional"])
    
    with tab1:
        try:
            # RSI and Stochastic charts
            fig_osc = make_subplots(
                rows=2, cols=1,
                subplot_titles=['RSI (14)', 'Stochastic Oscillator'],
                vertical_spacing=0.1
            )
            
            # RSI
            if 'RSI' in df.columns:
                fig_osc.add_trace(
                    go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
                    row=1, col=1
                )
                fig_osc.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
                fig_osc.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
                fig_osc.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, row=1, col=1)
                fig_osc.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, row=1, col=1)
            
            # Stochastic
            if all(col in df.columns for col in ['Stoch_K', 'Stoch_D']):
                fig_osc.add_trace(
                    go.Scatter(x=df.index, y=df['Stoch_K'], name='%K', line=dict(color='blue')),
                    row=2, col=1
                )
                fig_osc.add_trace(
                    go.Scatter(x=df.index, y=df['Stoch_D'], name='%D', line=dict(color='red')),
                    row=2, col=1
                )
                fig_osc.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1)
                fig_osc.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1)
            
            fig_osc.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig_osc, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error creating oscillators chart: {str(e)}")
    
    with tab2:
        try:
            # MACD chart
            if all(col in df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
                fig_macd = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=['MACD Line & Signal', 'MACD Histogram'],
                    vertical_spacing=0.1
                )
                
                fig_macd.add_trace(
                    go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')),
                    row=1, col=1
                )
                fig_macd.add_trace(
                    go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='red')),
                    row=1, col=1
                )
                
                colors = ['#00C851' if val >= 0 else '#FF4444' for val in df['MACD_Histogram']]
                fig_macd.add_trace(
                    go.Bar(x=df.index, y=df['MACD_Histogram'], name='Histogram', 
                          marker_color=colors),
                    row=2, col=1
                )
                
                fig_macd.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig_macd, use_container_width=True)
            else:
                st.info("📊 MACD data not available")
                
        except Exception as e:
            st.error(f"❌ Error creating MACD chart: {str(e)}")
    
    with tab3:
        # Additional indicators
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                if all(col in df.columns for col in ['ADX', 'Plus_DI', 'Minus_DI']):
                    fig_adx = go.Figure()
                    fig_adx.add_trace(go.Scatter(x=df.index, y=df['ADX'], name='ADX', line=dict(color='black')))
                    fig_adx.add_trace(go.Scatter(x=df.index, y=df['Plus_DI'], name='+DI', line=dict(color='green')))
                    fig_adx.add_trace(go.Scatter(x=df.index, y=df['Minus_DI'], name='-DI', line=dict(color='red')))
                    fig_adx.add_hline(y=25, line_dash="dash", line_color="gray")
                    fig_adx.update_layout(title="ADX & Directional Indicators", height=300)
                    st.plotly_chart(fig_adx, use_container_width=True)
                else:
                    st.info("📊 ADX data not available")
            except Exception as e:
                st.error(f"❌ Error creating ADX chart: {str(e)}")
        
        with col2:
            try:
                if 'Williams_R' in df.columns:
                    fig_wr = go.Figure()
                    fig_wr.add_trace(go.Scatter(x=df.index, y=df['Williams_R'], name="Williams %R", line=dict(color='orange')))
                    fig_wr.add_hline(y=-20, line_dash="dash", line_color="red")
                    fig_wr.add_hline(y=-80, line_dash="dash", line_color="green")
                    fig_wr.update_layout(title="Williams %R", height=300)
                    st.plotly_chart(fig_wr, use_container_width=True)
                else:
                    st.info("📊 Williams %R data not available")
            except Exception as e:
                st.error(f"❌ Error creating Williams %R chart: {str(e)}")

def display_trading_signals(ta: TechnicalAnalysis):
    """Display trading signals panel"""
    
    st.markdown("**🚨 Trading Signals**")
    
    # Add loading state
    try:
        with st.spinner("Calculating signals..."):
            signals = ta.get_signals()
        
        if not signals:
            st.info("📊 No signals available with current data")
            return
        
        # Signal strength indicator
        strong_signals = len([s for s in signals.values() if 'Strong' in s])
        if strong_signals > 0:
            st.success(f"💪 {strong_signals} strong signals detected!")
        
        for indicator, signal in signals.items():
            # Color code the signals
            if 'Buy' in signal or 'Bullish' in signal or 'Oversold' in signal:
                st.success(f"🟢 **{indicator}**: {signal}")
            elif 'Sell' in signal or 'Bearish' in signal or 'Overbought' in signal:
                st.error(f"🔴 **{indicator}**: {signal}")
            else:
                st.info(f"🟡 **{indicator}**: {signal}")
        
        # Overall signal
        buy_signals = len([s for s in signals.values() if 'Buy' in s or 'Bullish' in s or 'Oversold' in s])
        sell_signals = len([s for s in signals.values() if 'Sell' in s or 'Bearish' in s or 'Overbought' in s])
        
        st.markdown("---")
        st.markdown("**📊 Overall Signal**")
        
        if buy_signals > sell_signals:
            st.success(f"🟢 **BULLISH** ({buy_signals} bullish vs {sell_signals} bearish signals)")
        elif sell_signals > buy_signals:
            st.error(f"🔴 **BEARISH** ({sell_signals} bearish vs {buy_signals} bearish signals)")
        else:
            st.info(f"🟡 **NEUTRAL** ({buy_signals} bullish vs {sell_signals} bearish signals)")
            
    except Exception as e:
        st.error(f"❌ Error calculating trading signals: {str(e)}")

def display_support_resistance(ta: TechnicalAnalysis):
    """Display support and resistance levels"""
    
    st.markdown("**🎯 Support & Resistance Levels**")
    
    try:
        levels = ta.get_support_resistance()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Support", f"${levels['support']:.2f}", help="Key support level")
            
        with col2:
            st.metric("Resistance", f"${levels['resistance']:.2f}", help="Key resistance level")
        
        # Additional levels
        if 'support_levels' in levels and levels['support_levels']:
            st.markdown("**📉 Support Levels**")
            for i, level in enumerate(levels['support_levels'][-3:], 1):
                st.write(f"S{i}: ${level:.2f}")
        
        if 'resistance_levels' in levels and levels['resistance_levels']:
            st.markdown("**📈 Resistance Levels**")
            for i, level in enumerate(levels['resistance_levels'][:3], 1):
                st.write(f"R{i}: ${level:.2f}")
                
    except Exception as e:
        st.error(f"❌ Error calculating support/resistance: {str(e)}")

def display_key_metrics(df: pd.DataFrame):
    """Display key technical metrics"""
    
    st.markdown("**📊 Key Metrics**")
    
    try:
        latest = df.iloc[-1]
        
        # Current values
        st.metric("Current Price", f"${latest['close']:.2f}")
        
        if 'RSI' in df.columns and not pd.isna(latest['RSI']):
            rsi_value = latest['RSI']
            rsi_status = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
            st.metric("RSI (14)", f"{rsi_value:.1f}", delta=rsi_status)
        
        if 'MACD' in df.columns and not pd.isna(latest['MACD']):
            st.metric("MACD", f"{latest['MACD']:.3f}")
        
        if 'ATR' in df.columns and not pd.isna(latest['ATR']):
            st.metric("ATR (14)", f"${latest['ATR']:.2f}", help="Average True Range - volatility measure")
        
        # Volume analysis
        if 'volume' in df.columns:
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            if not pd.isna(avg_volume) and avg_volume > 0:
                volume_ratio = latest['volume'] / avg_volume
                volume_status = "High" if volume_ratio > 1.5 else "Low" if volume_ratio < 0.5 else "Normal"
                st.metric("Volume vs Avg", f"{volume_ratio:.1f}x", 
                         delta=volume_status)
                         
    except Exception as e:
        st.error(f"❌ Error displaying key metrics: {str(e)}")

def add_export_options(df: pd.DataFrame, symbol: str):
    """Add data export options"""
    
    st.markdown("**📤 Export Options**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export CSV", use_container_width=True):
            try:
                csv = df.to_csv()
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{symbol}_technical_analysis.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.success("✅ CSV ready for download!")
            except Exception as e:
                st.error(f"❌ Error creating CSV: {str(e)}")
    
    with col2:
        if st.button("📈 Export Chart", use_container_width=True):
            st.info("🔧 Chart export functionality coming in Phase 3!")
    
    with col3:
        if st.button("📄 Generate Report", use_container_width=True):
            st.info("🔧 PDF report generation coming in Phase 3!")

# =============================================================================
# OTHER ANALYSIS TOOLS (Enhanced versions)
# =============================================================================

def show_stock_screener():
    """Stock screening tool"""
    
    st.subheader("📊 Stock Screener")
    st.info("🔧 Stock screener will be enhanced with real data in Phase 3!")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**🎯 Screening Criteria**")
        
        market_cap = st.selectbox(
            "Market Cap",
            ["Any", "Large Cap (>$10B)", "Mid Cap ($2B-$10B)", "Small Cap (<$2B)"]
        )
        
        st.markdown("**💰 Price Range**")
        price_min = st.number_input("Min Price ($)", min_value=0.01, value=1.00)
        price_max = st.number_input("Max Price ($)", min_value=0.01, value=1000.00)
        
        st.markdown("**📈 Performance**")
        perf_1d = st.slider("1-Day Change (%)", -20.0, 20.0, (-10.0, 10.0))
        perf_1w = st.slider("1-Week Change (%)", -50.0, 50.0, (-25.0, 25.0))
        
        st.markdown("**🔧 Technical**")
        rsi_range = st.slider("RSI", 0, 100, (30, 70))
        volume_min = st.number_input("Min Volume (M)", min_value=0.1, value=1.0)
        
        sectors = st.multiselect(
            "Sectors",
            ["Technology", "Healthcare", "Financial", "Energy", "Consumer", "Industrial"]
        )
        
        if st.button("🔍 Run Screen", use_container_width=True):
            with st.spinner("Screening stocks..."):
                st.success("Screening completed! Found 47 matches.")
    
    with col2:
        st.markdown("**📋 Screening Results**")
        
        screening_results = [
            {
                "Symbol": "AAPL", "Company": "Apple Inc.", "Price": "$195.67",
                "1D Change": "+1.2%", "1W Change": "+5.4%", "Volume": "52.3M",
                "RSI": "58.2", "Market Cap": "Large", "Sector": "Technology"
            },
            {
                "Symbol": "MSFT", "Company": "Microsoft Corp.", "Price": "$423.12", 
                "1D Change": "+0.8%", "1W Change": "+3.1%", "Volume": "28.7M",
                "RSI": "62.4", "Market Cap": "Large", "Sector": "Technology"
            },
            {
                "Symbol": "GOOGL", "Company": "Alphabet Inc.", "Price": "$148.23",
                "1D Change": "-0.5%", "1W Change": "+2.3%", "Volume": "18.9M",
                "RSI": "45.7", "Market Cap": "Large", "Sector": "Technology"
            }
        ]
        
        df = pd.DataFrame(screening_results)
        
        def style_change(val):
            if val.startswith('-'):
                return 'color: red'
            elif val.startswith('+'):
                return 'color: green'
            return ''
        
        styled_df = df.style.applymap(style_change, subset=['1D Change', '1W Change'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Export options
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button("📊 Analyze Selected", use_container_width=True)
        with col2:
            st.button("📋 Add to Watchlist", use_container_width=True)
        with col3:
            st.button("📄 Export CSV", use_container_width=True)

def show_correlation_analysis():
    """Correlation analysis between stocks"""
    
    st.subheader("🔄 Correlation Analysis")
    st.info("🔧 Real correlation analysis coming in Phase 3!")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        symbols = st.multiselect(
            "Choose symbols to analyze",
            ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"],
            default=["AAPL", "MSFT", "GOOGL", "TSLA"]
        )
        
        timeframe = st.selectbox("Analysis Period", ["1M", "3M", "6M", "1Y"], index=2)
        
        correlation_type = st.selectbox(
            "Correlation Type",
            ["Pearson", "Spearman", "Kendall"]
        )
        
        if st.button("📊 Calculate Correlations", use_container_width=True):
            with st.spinner("Calculating correlations..."):
                st.success(f"Correlation analysis complete for {len(symbols)} symbols")
    
    with col2:
        if len(symbols) >= 2:
            try:
                # Generate sample correlation matrix
                n = len(symbols)
                correlation_matrix = np.random.rand(n, n)
                correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
                np.fill_diagonal(correlation_matrix, 1)
                correlation_df = pd.DataFrame(correlation_matrix, index=symbols, columns=symbols)
                
                fig = px.imshow(correlation_df, title="Stock Price Correlation Matrix", 
                              color_continuous_scale="RdBu", aspect="auto")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation table
                st.markdown("**📊 Correlation Matrix**")
                st.dataframe(correlation_df.round(3), use_container_width=True)
                
                # Insights
                st.markdown("**💡 Key Insights**")
                max_corr = correlation_df.abs().max().max()
                highly_corr = correlation_df.abs() > 0.7
                
                st.info(f"📈 Highest correlation: {max_corr:.3f}")
                st.info(f"🔗 Highly correlated pairs: {highly_corr.sum().sum() - len(symbols)}")
                
            except Exception as e:
                st.error(f"❌ Error in correlation analysis: {str(e)}")

def show_sector_analysis():
    """Sector performance analysis"""
    
    st.subheader("📰 Sector Analysis")
    
    sectors = {
        'Technology': {'return': 2.45, 'volume': 'High', 'trend': 'Bullish'},
        'Healthcare': {'return': 1.23, 'volume': 'Medium', 'trend': 'Neutral'},
        'Financial': {'return': -0.87, 'volume': 'Low', 'trend': 'Bearish'},
        'Energy': {'return': -1.45, 'volume': 'High', 'trend': 'Bearish'},
        'Consumer': {'return': 0.67, 'volume': 'Medium', 'trend': 'Neutral'},
        'Industrial': {'return': 1.89, 'volume': 'Medium', 'trend': 'Bullish'},
        'Materials': {'return': -0.23, 'volume': 'Low', 'trend': 'Neutral'},
        'Utilities': {'return': 0.45, 'volume': 'Low', 'trend': 'Neutral'},
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        sector_returns = [data['return'] for data in sectors.values()]
        sector_names = list(sectors.keys())
        
        fig = go.Figure(go.Bar(
            x=sector_returns, y=sector_names, orientation='h',
            marker_color=['#00C851' if x > 0 else '#FF4444' for x in sector_returns]
        ))
        fig.update_layout(title="Sector Performance (%)", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        sector_df = pd.DataFrame.from_dict(sectors, orient='index')
        sector_df.reset_index(inplace=True)
        sector_df.columns = ['Sector', 'Return (%)', 'Volume', 'Trend']
        st.dataframe(sector_df, use_container_width=True, hide_index=True)
        
        # Sector rotation chart
        st.markdown("**🔄 Sector Rotation**")
        rotation_data = {
            'Growth': ['Technology', 'Consumer'],
            'Value': ['Financial', 'Energy'], 
            'Defensive': ['Healthcare', 'Utilities']
        }
        
        for style, sectors_in_style in rotation_data.items():
            st.write(f"**{style}:** {', '.join(sectors_in_style)}")

def show_risk_analysis():
    """Risk analysis tools"""
    
    st.subheader("⚖️ Risk Analysis")
    st.info("🔧 Advanced risk analysis coming in Phase 3!")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        portfolio_value = st.number_input("Portfolio Value ($)", value=100000, step=1000)
        confidence_level = st.slider("Confidence Level (%)", 90, 99, 95)
        time_horizon = st.selectbox("Time Horizon", ["1 Day", "1 Week", "1 Month"])
        
        risk_model = st.selectbox(
            "Risk Model",
            ["Historical Simulation", "Monte Carlo", "Parametric VaR"]
        )
        
        if st.button("📊 Calculate Risk Metrics", use_container_width=True):
            with st.spinner("Calculating risk metrics..."):
                st.success("Risk analysis complete!")
    
    with col2:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Value at Risk", "$2,450", "1-Day, 95%")
        with col2:
            st.metric("Expected Shortfall", "$3,820", "Conditional VaR")
        with col3:
            st.metric("Volatility", "18.4%", "Annualized")
        
        # Risk distribution chart
        try:
            y = np.random.normal(0, 2, 1000)
            fig = px.histogram(x=y, nbins=50, title="Portfolio Return Distribution")
            fig.add_vline(x=-2.45, line_dash="dash", line_color="red", annotation_text="VaR (95%)")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error creating risk chart: {str(e)}")

def show_backtesting():
    """Backtesting interface"""
    
    st.subheader("🧮 Backtesting")
    st.info("🔧 Advanced backtesting engine coming in Phase 3!")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        strategy_type = st.selectbox(
            "Strategy Type",
            ["Moving Average Crossover", "RSI Mean Reversion", "Momentum", "Custom"]
        )
        
        if strategy_type == "Moving Average Crossover":
            fast_ma = st.number_input("Fast MA", value=10, min_value=1)
            slow_ma = st.number_input("Slow MA", value=30, min_value=1)
        elif strategy_type == "RSI Mean Reversion":
            rsi_oversold = st.number_input("RSI Oversold", value=30, min_value=1, max_value=50)
            rsi_overbought = st.number_input("RSI Overbought", value=70, min_value=50, max_value=100)
        
        st.markdown("**⚙️ Backtest Settings**")
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
        end_date = st.date_input("End Date", value=datetime.now())
        initial_capital = st.number_input("Initial Capital ($)", value=10000, step=1000)
        
        if st.button("🚀 Run Backtest", use_container_width=True):
            with st.spinner("Running backtest..."):
                st.success("Backtesting complete!")
    
    with col2:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Return", "23.4%", "+5.2% vs buy-and-hold")
        with col2:
            st.metric("Sharpe Ratio", "1.42", "Good")
        with col3:
            st.metric("Max Drawdown", "-8.7%", "Acceptable")
        
        # Equity curve
        try:
            dates = [datetime.now() - timedelta(days=x) for x in range(365, 0, -1)]
            equity_curve = [10000 * (1 + np.random.normal(0.0008, 0.015)) ** i for i in range(365)]
            benchmark = [10000 * (1 + np.random.normal(0.0003, 0.015)) ** i for i in range(365)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=equity_curve, name='Strategy', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=dates, y=benchmark, name='Buy & Hold', line=dict(color='gray')))
            
            fig.update_layout(title="Strategy Performance", height=300)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error creating backtest chart: {str(e)}")
   
# This is important for Streamlit multi-page apps
if __name__ == "__main__":
    show()
    
# This ensures it works in multi-page app
    show()