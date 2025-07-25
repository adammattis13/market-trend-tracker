"""
Analysis Page - Advanced market analysis and research tools
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

def show():
    """Advanced analysis page"""
    
    st.header("📈 Advanced Analysis")
    
    # Analysis tool selection
    analysis_tab = st.selectbox(
        "🔍 Choose Analysis Tool",
        [
            "📊 Stock Screener",
            "📈 Technical Analysis", 
            "🔄 Correlation Analysis",
            "📰 Sector Analysis",
            "⚖️ Risk Analysis",
            "🧮 Backtesting"
        ]
    )
    
    if analysis_tab == "📊 Stock Screener":
        show_stock_screener()
    elif analysis_tab == "📈 Technical Analysis":
        show_technical_analysis()
    elif analysis_tab == "🔄 Correlation Analysis":
        show_correlation_analysis()
    elif analysis_tab == "📰 Sector Analysis":
        show_sector_analysis()
    elif analysis_tab == "⚖️ Risk Analysis":
        show_risk_analysis()
    elif analysis_tab == "🧮 Backtesting":
        show_backtesting()

def show_stock_screener():
    """Stock screening tool"""
    
    st.subheader("📊 Stock Screener")
    
    # Screening criteria
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**🎯 Screening Criteria**")
        
        # Market cap
        market_cap = st.selectbox(
            "Market Cap",
            ["Any", "Large Cap (>$10B)", "Mid Cap ($2B-$10B)", "Small Cap (<$2B)"]
        )
        
        # Price range
        st.markdown("**💰 Price Range**")
        price_min = st.number_input("Min Price ($)", min_value=0.01, value=1.00)
        price_max = st.number_input("Max Price ($)", min_value=0.01, value=1000.00)
        
        # Performance filters
        st.markdown("**📈 Performance**")
        perf_1d = st.slider("1-Day Change (%)", -20.0, 20.0, (-10.0, 10.0))
        perf_1w = st.slider("1-Week Change (%)", -50.0, 50.0, (-25.0, 25.0))
        
        # Technical filters
        st.markdown("**🔧 Technical**")
        rsi_range = st.slider("RSI", 0, 100, (30, 70))
        volume_min = st.number_input("Min Volume (M)", min_value=0.1, value=1.0)
        
        # Sector filter
        sectors = st.multiselect(
            "Sectors",
            ["Technology", "Healthcare", "Financial", "Energy", "Consumer", "Industrial"]
        )
        
        if st.button("🔍 Run Screen", use_container_width=True):
            st.success("Screening completed! Found 47 matches.")
    
    with col2:
        # Results table
        st.markdown("**📋 Screening Results**")
        
        # Sample screening results
        screening_results = [
            {
                "Symbol": "AAPL",
                "Company": "Apple Inc.",
                "Price": "$195.67",
                "1D Change": "+1.2%",
                "1W Change": "+5.4%",
                "Volume": "52.3M",
                "RSI": "58.2",
                "Market Cap": "Large",
                "Sector": "Technology"
            },
            {
                "Symbol": "MSFT",
                "Company": "Microsoft Corp.",
                "Price": "$423.12", 
                "1D Change": "+0.8%",
                "1W Change": "+3.1%",
                "Volume": "28.7M",
                "RSI": "62.4",
                "Market Cap": "Large",
                "Sector": "Technology"
            },
            {
                "Symbol": "GOOGL",
                "Company": "Alphabet Inc.",
                "Price": "$148.23",
                "1D Change": "-0.5%",
                "1W Change": "+2.3%",
                "Volume": "18.9M",
                "RSI": "45.7",
                "Market Cap": "Large",
                "Sector": "Technology"
            }
        ]
        
        df = pd.DataFrame(screening_results)
        
        # Style the dataframe
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

def show_technical_analysis():
    """Technical analysis tool"""
    
    st.subheader("📈 Technical Analysis")
    
    # Symbol input
    col1, col2 = st.columns([1, 3])
    
    with col1:
        symbol = st.text_input("Symbol", value="AAPL", placeholder="Enter symbol")
        timeframe = st.selectbox("Timeframe", ["1D", "1W", "1M", "3M", "1Y"])
        
        # Technical indicators
        st.markdown("**📊 Indicators**")
        show_sma = st.checkbox("Simple Moving Average", value=True)
        show_ema = st.checkbox("Exponential Moving Average")
        show_rsi = st.checkbox("RSI", value=True) 
        show_macd = st.checkbox("MACD")
        show_bollinger = st.checkbox("Bollinger Bands")
        show_volume = st.checkbox("Volume", value=True)
    
    with col2:
        if symbol:
            # Generate sample price data
            dates = [datetime.now() - timedelta(days=x) for x in range(90, 0, -1)]
            prices = generate_sample_price_data(len(dates), base_price=195.67)
            volumes = [np.random.uniform(20, 80) for _ in dates]
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=(f'{symbol} Price Chart', 'RSI', 'Volume'),
                vertical_spacing=0.05,
                row_heights=[0.6, 0.2, 0.2]
            )
            
            # Price chart
            fig.add_trace(
                go.Scatter(x=dates, y=prices, name='Price', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Add SMA if selected
            if show_sma:
                sma_20 = pd.Series(prices).rolling(20).mean()
                fig.add_trace(
                    go.Scatter(x=dates, y=sma_20, name='SMA(20)', line=dict(color='orange')),
                    row=1, col=1
                )
            
            # Add EMA if selected
            if show_ema:
                ema_20 = pd.Series(prices).ewm(span=20).mean()
                fig.add_trace(
                    go.Scatter(x=dates, y=ema_20, name='EMA(20)', line=dict(color='red')),
                    row=1, col=1
                )
            
            # RSI
            if show_rsi:
                rsi_values = generate_rsi_data(len(dates))
                fig.add_trace(
                    go.Scatter(x=dates, y=rsi_values, name='RSI', line=dict(color='purple')),
                    row=2, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # Volume
            if show_volume:
                fig.add_trace(
                    go.Bar(x=dates, y=volumes, name='Volume', marker_color='lightblue'),
                    row=3, col=1
                )
            
            fig.update_layout(height=700, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical summary
            st.markdown("**📊 Technical Summary**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${prices[-1]:.2f}", f"{((prices[-1]/prices[-2])-1)*100:+.1f}%")
            with col2:
                st.metric("RSI (14)", f"{rsi_values[-1]:.1f}", "Neutral")
            with col3:
                st.metric("SMA(20)", f"${sma_20.iloc[-1]:.2f}", "Above" if prices[-1] > sma_20.iloc[-1] else "Below")
            with col4:
                st.metric("Volume", f"{volumes[-1]:.1f}M", "Normal")

def show_correlation_analysis():
    """Correlation analysis between stocks"""
    
    st.subheader("🔄 Correlation Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**📋 Select Symbols**")
        
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
            st.success(f"Correlation analysis complete for {len(symbols)} symbols")
    
    with col2:
        if len(symbols) >= 2:
            # Generate sample correlation matrix
            correlation_matrix = generate_correlation_matrix(symbols)
            
            # Correlation heatmap
            fig = px.imshow(
                correlation_matrix,
                title="Stock Price Correlation Matrix",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation table
            st.markdown("**📊 Correlation Matrix**")
            st.dataframe(correlation_matrix.round(3), use_container_width=True)
            
            # Insights
            st.markdown("**💡 Key Insights**")
            max_corr = correlation_matrix.abs().max().max()
            highly_corr = correlation_matrix.abs() > 0.7
            
            st.info(f"📈 Highest correlation: {max_corr:.3f}")
            st.info(f"🔗 Highly correlated pairs: {highly_corr.sum().sum() - len(symbols)}")

def show_sector_analysis():
    """Sector performance analysis"""
    
    st.subheader("📰 Sector Analysis")
    
    # Sector performance data
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
        # Sector performance bar chart
        sector_returns = [data['return'] for data in sectors.values()]
        sector_names = list(sectors.keys())
        
        fig = go.Figure(go.Bar(
            x=sector_returns,
            y=sector_names,
            orientation='h',
            marker_color=['green' if x > 0 else 'red' for x in sector_returns]
        ))
        
        fig.update_layout(
            title="Sector Performance (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sector details table
        sector_df = pd.DataFrame.from_dict(sectors, orient='index')
        sector_df.reset_index(inplace=True)
        sector_df.columns = ['Sector', 'Return (%)', 'Volume', 'Trend']
        
        st.dataframe(sector_df, use_container_width=True, hide_index=True)
        
        # Sector rotation chart (placeholder)
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
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**⚙️ Risk Parameters**")
        
        portfolio_value = st.number_input("Portfolio Value ($)", value=100000, step=1000)
        confidence_level = st.slider("Confidence Level (%)", 90, 99, 95)
        time_horizon = st.selectbox("Time Horizon", ["1 Day", "1 Week", "1 Month"])
        
        risk_model = st.selectbox(
            "Risk Model",
            ["Historical Simulation", "Monte Carlo", "Parametric VaR"]
        )
        
        if st.button("📊 Calculate Risk Metrics", use_container_width=True):
            st.success("Risk analysis complete!")
    
    with col2:
        # Risk metrics
        st.markdown("**📊 Risk Metrics**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Value at Risk", "$2,450", "1-Day, 95%")
        with col2:
            st.metric("Expected Shortfall", "$3,820", "Conditional VaR")
        with col3:
            st.metric("Volatility", "18.4%", "Annualized")
        
        # Risk distribution chart
        x = np.linspace(-10, 10, 1000)
        y = np.random.normal(0, 2, 1000)
        
        fig = px.histogram(x=y, nbins=50, title="Portfolio Return Distribution")
        fig.add_vline(x=-2.45, line_dash="dash", line_color="red", annotation_text="VaR (95%)")
        fig.update_layout(height=300)
        
        st.plotly_chart(fig, use_container_width=True)

def show_backtesting():
    """Backtesting interface"""
    
    st.subheader("🧮 Backtesting")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**📋 Strategy Setup**")
        
        strategy_type = st.selectbox(
            "Strategy Type",
            ["Moving Average Crossover", "RSI Mean Reversion", "Momentum", "Custom"]
        )
        
        # Parameters based on strategy
        if strategy_type == "Moving Average Crossover":
            fast_ma = st.number_input("Fast MA", value=10, min_value=1)
            slow_ma = st.number_input("Slow MA", value=30, min_value=1)
        elif strategy_type == "RSI Mean Reversion":
            rsi_oversold = st.number_input("RSI Oversold", value=30, min_value=1, max_value=50)
            rsi_overbought = st.number_input("RSI Overbought", value=70, min_value=50, max_value=100)
        
        # Backtest parameters
        st.markdown("**⚙️ Backtest Settings**")
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
        end_date = st.date_input("End Date", value=datetime.now())
        initial_capital = st.number_input("Initial Capital ($)", value=10000, step=1000)
        
        if st.button("🚀 Run Backtest", use_container_width=True):
            st.success("Backtesting complete!")
    
    with col2:
        # Backtest results
        st.markdown("**📊 Backtest Results**")
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Return", "23.4%", "+5.2% vs buy-and-hold")
        with col2:
            st.metric("Sharpe Ratio", "1.42", "Good")
        with col3:
            st.metric("Max Drawdown", "-8.7%", "Acceptable")
        
        # Equity curve
        dates = [datetime.now() - timedelta(days=x) for x in range(365, 0, -1)]
        equity_curve = generate_equity_curve(len(dates), 10000)
        benchmark = generate_equity_curve(len(dates), 10000, 0.0003)  # Lower return
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=equity_curve, name='Strategy', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=dates, y=benchmark, name='Buy & Hold', line=dict(color='gray')))
        
        fig.update_layout(title="Strategy Performance", height=300)
        st.plotly_chart(fig, use_container_width=True)

# Helper functions for generating sample data
def generate_sample_price_data(n_points, base_price=100):
    """Generate realistic price data"""
    returns = np.random.normal(0.0005, 0.02, n_points)  # Daily returns
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    return prices

def generate_rsi_data(n_points):
    """Generate RSI-like data"""
    return np.random.uniform(20, 80, n_points)

def generate_correlation_matrix(symbols):
    """Generate sample correlation matrix"""
    n = len(symbols)
    # Generate random correlation matrix
    random_matrix = np.random.randn(n, n)
    correlation_matrix = np.corrcoef(random_matrix)
    
    return pd.DataFrame(correlation_matrix, index=symbols, columns=symbols)

def generate_equity_curve(n_points, initial_value, daily_return=0.0008):
    """Generate equity curve"""
    returns = np.random.normal(daily_return, 0.015, n_points)
    values = [initial_value]
    for ret in returns[1:]:
        values.append(values[-1] * (1 + ret))
    return values