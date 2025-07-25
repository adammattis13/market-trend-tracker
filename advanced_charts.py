"""
Advanced Chart Components
Interactive TradingView-style charts with technical indicators
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import streamlit as st
from technical_indicators import TechnicalIndicators

class AdvancedCharts:
    """
    Advanced charting class for financial data with technical indicators
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with OHLCV data
        
        Args:
            data: DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        """
        self.data = data.copy()
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        self.ti = TechnicalIndicators(self.data)
    
    def create_candlestick_chart(
        self,
        title: str = "Price Chart",
        height: int = 600,
        show_volume: bool = True,
        indicators: List[str] = None
    ) -> go.Figure:
        """
        Create professional candlestick chart with optional indicators
        
        Args:
            title: Chart title
            height: Chart height in pixels
            show_volume: Whether to show volume subplot
            indicators: List of indicators to overlay ['SMA', 'EMA', 'BB', 'VWAP']
        """
        
        if indicators is None:
            indicators = []
        
        # Determine subplot configuration
        rows = 1
        row_heights = [0.7]
        subplot_titles = [title]
        
        if show_volume:
            rows += 1
            row_heights = [0.7, 0.3]
            subplot_titles.append("Volume")
        
        # Create subplots
        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxis=True,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles,
            row_heights=row_heights
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=self.data['Date'],
                open=self.data['Open'],
                high=self.data['High'],
                low=self.data['Low'],
                close=self.data['Close'],
                name="Price",
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444',
                increasing_fillcolor='#00ff88',
                decreasing_fillcolor='#ff4444'
            ),
            row=1, col=1
        )
        
        # Add technical indicators
        if 'SMA' in indicators:
            sma_20 = self.ti.sma(20)
            sma_50 = self.ti.sma(50)
            fig.add_trace(
                go.Scatter(
                    x=self.data['Date'],
                    y=sma_20,
                    name='SMA 20',
                    line=dict(color='orange', width=2)
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=self.data['Date'],
                    y=sma_50,
                    name='SMA 50',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
        
        if 'EMA' in indicators:
            ema_12 = self.ti.ema(12)
            ema_26 = self.ti.ema(26)
            fig.add_trace(
                go.Scatter(
                    x=self.data['Date'],
                    y=ema_12,
                    name='EMA 12',
                    line=dict(color='purple', width=2, dash='dash')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=self.data['Date'],
                    y=ema_26,
                    name='EMA 26',
                    line=dict(color='red', width=2, dash='dash')
                ),
                row=1, col=1
            )
        
        if 'BB' in indicators:
            bb = self.ti.bollinger_bands()
            fig.add_trace(
                go.Scatter(
                    x=self.data['Date'],
                    y=bb['Upper'],
                    name='BB Upper',
                    line=dict(color='gray', width=1),
                    showlegend=False
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=self.data['Date'],
                    y=bb['Lower'],
                    name='BB Lower',
                    line=dict(color='gray', width=1),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)'
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=self.data['Date'],
                    y=bb['Middle'],
                    name='BB Middle',
                    line=dict(color='gray', width=2)
                ),
                row=1, col=1
            )
        
        if 'VWAP' in indicators:
            vwap = self.ti.vwap()
            fig.add_trace(
                go.Scatter(
                    x=self.data['Date'],
                    y=vwap,
                    name='VWAP',
                    line=dict(color='yellow', width=2)
                ),
                row=1, col=1
            )
        
        # Add volume if requested
        if show_volume:
            colors = ['green' if close >= open_price else 'red' 
                     for close, open_price in zip(self.data['Close'], self.data['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=self.data['Date'],
                    y=self.data['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=height,
            template='plotly_dark',
            showlegend=True,
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )
        
        # Update x-axis
        fig.update_xaxes(
            type='date',
            tickformat='%Y-%m-%d',
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)'
        )
        
        # Update y-axis
        fig.update_yaxes(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            title_text="Price ($)",
            row=1, col=1
        )
        
        if show_volume:
            fig.update_yaxes(
                title_text="Volume",
                row=2, col=1
            )
        
        return fig
    
    def create_technical_indicators_chart(
        self,
        indicators: List[str] = ['RSI', 'MACD'],
        height: int = 400
    ) -> go.Figure:
        """
        Create chart with technical indicators in separate subplots
        
        Args:
            indicators: List of indicators ['RSI', 'MACD', 'STOCH', 'CCI']
            height: Chart height
        """
        
        rows = len(indicators)
        if rows == 0:
            return go.Figure()
        
        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxis=True,
            vertical_spacing=0.05,
            subplot_titles=indicators
        )
        
        for i, indicator in enumerate(indicators, 1):
            if indicator == 'RSI':
                rsi = self.ti.rsi()
                fig.add_trace(
                    go.Scatter(
                        x=self.data['Date'],
                        y=rsi,
                        name='RSI',
                        line=dict(color='purple', width=2)
                    ),
                    row=i, col=1
                )
                # Add RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=i, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=i, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="gray", row=i, col=1)
                
            elif indicator == 'MACD':
                macd_data = self.ti.macd()
                fig.add_trace(
                    go.Scatter(
                        x=self.data['Date'],
                        y=macd_data['MACD'],
                        name='MACD',
                        line=dict(color='blue', width=2)
                    ),
                    row=i, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=self.data['Date'],
                        y=macd_data['Signal'],
                        name='Signal',
                        line=dict(color='red', width=2)
                    ),
                    row=i, col=1
                )
                fig.add_trace(
                    go.Bar(
                        x=self.data['Date'],
                        y=macd_data['Histogram'],
                        name='Histogram',
                        marker_color=['green' if x >= 0 else 'red' for x in macd_data['Histogram']],
                        opacity=0.7
                    ),
                    row=i, col=1
                )
                
            elif indicator == 'STOCH':
                stoch = self.ti.stochastic()
                fig.add_trace(
                    go.Scatter(
                        x=self.data['Date'],
                        y=stoch['%K'],
                        name='%K',
                        line=dict(color='blue', width=2)
                    ),
                    row=i, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=self.data['Date'],
                        y=stoch['%D'],
                        name='%D',
                        line=dict(color='red', width=2)
                    ),
                    row=i, col=1
                )
                # Add Stochastic levels
                fig.add_hline(y=80, line_dash="dash", line_color="red", row=i, col=1)
                fig.add_hline(y=20, line_dash="dash", line_color="green", row=i, col=1)
                
            elif indicator == 'CCI':
                cci = self.ti.cci()
                fig.add_trace(
                    go.Scatter(
                        x=self.data['Date'],
                        y=cci,
                        name='CCI',
                        line=dict(color='orange', width=2)
                    ),
                    row=i, col=1
                )
                # Add CCI levels
                fig.add_hline(y=100, line_dash="dash", line_color="red", row=i, col=1)
                fig.add_hline(y=-100, line_dash="dash", line_color="green", row=i, col=1)
        
        # Update layout
        fig.update_layout(
            height=height,
            template='plotly_dark',
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update x-axis
        fig.update_xaxes(
            type='date',
            tickformat='%Y-%m-%d',
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)'
        )
        
        # Update y-axis
        fig.update_yaxes(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)'
        )
        
        return fig
    
    def create_correlation_heatmap(self, symbols: List[str]) -> go.Figure:
        """
        Create correlation heatmap for multiple symbols
        """
        # This would need multiple symbol data - placeholder for now
        correlation_matrix = np.random.rand(len(symbols), len(symbols))
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix, 1.0)
        
        fig = go.Figure(
            data=go.Heatmap(
                z=correlation_matrix,
                x=symbols,
                y=symbols,
                colorscale='RdBu',
                zmid=0,
                text=np.round(correlation_matrix, 2),
                texttemplate='%{text}',
                textfont={"size": 12},
                colorbar=dict(title="Correlation")
            )
        )
        
        fig.update_layout(
            title="Symbol Correlation Matrix",
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    def create_performance_comparison(
        self,
        symbols: List[str],
        normalize: bool = True
    ) -> go.Figure:
        """
        Create performance comparison chart
        """
        fig = go.Figure()
        
        # Generate sample data for multiple symbols
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, symbol in enumerate(symbols):
            # Generate sample performance data
            dates = self.data['Date']
            if normalize:
                # Normalize to 100 at start
                base_value = self.data['Close'].iloc[0]
                performance = (self.data['Close'] / base_value) * 100
            else:
                performance = self.data['Close']
            
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=performance,
                    name=symbol,
                    line=dict(color=colors[i % len(colors)], width=2)
                )
            )
        
        fig.update_layout(
            title="Performance Comparison",
            template='plotly_dark',
            height=400,
            yaxis_title="Performance (%)" if normalize else "Price ($)",
            xaxis_title="Date",
            hovermode='x unified'
        )
        
        return fig

def create_interactive_chart_controls():
    """
    Create Streamlit controls for interactive charts
    """
    st.sidebar.markdown("## 📊 Chart Settings")
    
    # Chart type
    chart_type = st.sidebar.selectbox(
        "Chart Type",
        ["Candlestick", "Line", "Area"],
        index=0
    )
    
    # Time period
    time_period = st.sidebar.selectbox(
        "Time Period",
        ["1D", "5D", "1M", "3M", "6M", "1Y", "2Y", "5Y"],
        index=3
    )
    
    # Technical indicators
    st.sidebar.markdown("### Technical Indicators")
    
    show_ma = st.sidebar.checkbox("Moving Averages", value=True)
    show_bb = st.sidebar.checkbox("Bollinger Bands")
    show_vwap = st.sidebar.checkbox("VWAP")
    
    # Oscillators
    st.sidebar.markdown("### Oscillators")
    
    show_rsi = st.sidebar.checkbox("RSI", value=True)
    show_macd = st.sidebar.checkbox("MACD", value=True)
    show_stoch = st.sidebar.checkbox("Stochastic")
    show_cci = st.sidebar.checkbox("CCI")
    
    # Volume
    show_volume = st.sidebar.checkbox("Volume", value=True)
    
    # Chart height
    chart_height = st.sidebar.slider("Chart Height", 400, 800, 600)
    
    return {
        'chart_type': chart_type,
        'time_period': time_period,
        'indicators': {
            'SMA': show_ma,
            'BB': show_bb,
            'VWAP': show_vwap
        },
        'oscillators': {
            'RSI': show_rsi,
            'MACD': show_macd,
            'STOCH': show_stoch,
            'CCI': show_cci
        },
        'show_volume': show_volume,
        'height': chart_height
    }

def display_technical_summary(ti: TechnicalIndicators, symbol: str):
    """
    Display technical analysis summary
    """
    st.markdown(f"### 📊 Technical Summary - {symbol}")
    
    # Get current values
    current_data = ti.data.iloc[-1]
    current_price = current_data['Close']
    
    # Calculate indicators
    rsi = ti.rsi().iloc[-1]
    macd_data = ti.macd()
    macd_current = macd_data['MACD'].iloc[-1]
    signal_current = macd_data['Signal'].iloc[-1]
    
    bb = ti.bollinger_bands()
    bb_position = (current_price - bb['Lower'].iloc[-1]) / (bb['Upper'].iloc[-1] - bb['Lower'].iloc[-1])
    
    # Get signals
    signals = ti.get_current_signals()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "RSI (14)",
            f"{rsi:.1f}",
            "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
        )
    
    with col2:
        macd_signal = "Bullish" if macd_current > signal_current else "Bearish"
        st.metric(
            "MACD",
            f"{macd_current:.3f}",
            macd_signal
        )
    
    with col3:
        bb_signal = "Lower" if bb_position < 0.2 else "Upper" if bb_position > 0.8 else "Middle"
        st.metric(
            "Bollinger Bands",
            f"{bb_position:.1%}",
            bb_signal
        )
    
    with col4:
        overall_signal = signals.get('Overall', 'NEUTRAL')
        st.metric(
            "Overall Signal",
            overall_signal,
            delta_color="normal" if overall_signal == "BUY" else "inverse" if overall_signal == "SELL" else "off"
        )
    
    # Signal details
    st.markdown("#### 🎯 Detailed Signals")
    signal_cols = st.columns(len(signals) - 1)  # Exclude 'Overall'
    
    for i, (indicator, signal) in enumerate(signals.items()):
        if indicator != 'Overall':
            with signal_cols[i]:
                color = "🟢" if "BUY" in signal else "🔴" if "SELL" in signal else "🟡"
                st.markdown(f"**{indicator}**\n{color} {signal}")

# Example usage
if __name__ == "__main__":
    from technical_indicators import create_sample_data
    
    # Create sample data
    sample_data = create_sample_data('AAPL', 100)
    
    # Create charts
    charts = AdvancedCharts(sample_data)
    
    # Create candlestick chart
    fig = charts.create_candlestick_chart(
        title="AAPL - Advanced Chart",
        indicators=['SMA', 'BB']
    )
    
    print("Advanced charts created successfully!")