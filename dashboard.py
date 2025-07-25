#!/usr/bin/env python3
"""
Enhanced Market Trend Dashboard
===============================

A modern, real-time dashboard for monitoring stock and crypto market trends
with alerts, sector analysis, and interactive visualizations.

Author: Adam Mattis (Enhanced Version)
Date: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
from streamlit_autorefresh import st_autorefresh
import os
import sys
import time
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_manager import DatabaseManager
from api_client import FinnhubClient
from alert_system import MarketAlertSystem
from valid_ticker_filter import TrendAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Market Trend Tracker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Alert styling */
    .alert-critical {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    
    .alert-warning {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    
    .alert-info {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    
    /* Custom header */
    .dashboard-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Trending ticker card */
    .ticker-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .ticker-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Positive/Negative indicators */
    .positive {
        color: #4caf50;
        font-weight: bold;
    }
    
    .negative {
        color: #f44336;
        font-weight: bold;
    }
    
    /* Custom buttons */
    .stButton > button {
        background-color: #2196f3;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #1976d2;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)


class MarketDashboard:
    """Enhanced market trend dashboard with real-time updates and alerts."""
    
    def __init__(self):
        """Initialize dashboard components."""
        self.db = DatabaseManager()
        self.client = FinnhubClient()
        self.analyzer = TrendAnalyzer()
        self.alert_system = MarketAlertSystem(self.db)
        
        # Initialize session state
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
        if 'refresh_interval' not in st.session_state:
            st.session_state.refresh_interval = 60  # seconds
    
    def render_header(self):
        """Render dashboard header with title and status."""
        st.markdown("""
        <div class="dashboard-header">
            <h1>📊 Market Trend Tracker</h1>
            <p>Real-time market momentum analysis and alerts</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Market status and last update
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        
        with col1:
            market_status = self.client.get_market_status()
            status_emoji = "🟢" if market_status.get('isOpen', False) else "🔴"
            st.metric(
                "Market Status",
                f"{status_emoji} {market_status.get('session', 'Closed').title()}"
            )
        
        with col2:
            st.metric(
                "Last Update",
                st.session_state.last_update.strftime("%H:%M:%S")
            )
        
        with col3:
            db_stats = self.db.get_database_stats()
            st.metric(
                "Total Records",
                f"{db_stats.get('trends_count', 0):,}"
            )
        
        with col4:
            api_stats = self.client.get_stats()
            st.metric(
                "API Health",
                f"{api_stats['success_rate']}%",
                f"Cache: {api_stats['cache_stats']['hit_rate']}%"
            )
    
    def render_alerts_section(self):
        """Render alerts section with real-time updates."""
        st.header("🚨 Market Alerts")
        
        # Get unacknowledged alerts
        alerts_df = self.db.get_unacknowledged_alerts(limit=20)
        
        if alerts_df.empty:
            st.info("No active alerts at the moment. Market conditions are stable.")
        else:
            # Group alerts by severity
            critical_alerts = alerts_df[alerts_df['severity'] == 'CRITICAL']
            warning_alerts = alerts_df[alerts_df['severity'] == 'WARNING']
            info_alerts = alerts_df[alerts_df['severity'] == 'INFO']
            
            # Display alerts by severity
            if not critical_alerts.empty:
                st.markdown("### 🔴 Critical Alerts")
                for _, alert in critical_alerts.iterrows():
                    self._render_alert(alert, 'critical')
            
            if not warning_alerts.empty:
                st.markdown("### 🟠 Warning Alerts")
                for _, alert in warning_alerts.iterrows():
                    self._render_alert(alert, 'warning')
            
            if not info_alerts.empty:
                st.markdown("### 🔵 Information")
                for _, alert in info_alerts.iterrows():
                    self._render_alert(alert, 'info')
    
    def _render_alert(self, alert, severity_class):
        """Render individual alert with styling."""
        alert_html = f"""
        <div class="alert-{severity_class}">
            <strong>{alert['alert_type'].replace('_', ' ').title()}</strong><br>
            {alert['message']}<br>
            <small>🕒 {pd.to_datetime(alert['timestamp']).strftime('%H:%M:%S')}</small>
        </div>
        """
        st.markdown(alert_html, unsafe_allow_html=True)
        
        # Add acknowledge button
        if st.button(f"Acknowledge", key=f"ack_{alert['id']}"):
            self.db.acknowledge_alert(alert['id'])
            st.rerun()
    
    def render_top_movers(self):
        """Render top movers section with interactive cards."""
        st.header("🚀 Top Movers")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Top Gainers")
            gainers = self.db.get_top_movers(direction='up', limit=5)
            
            if not gainers.empty:
                for _, stock in gainers.iterrows():
                    self._render_mover_card(stock, is_gainer=True)
            else:
                st.info("No significant gainers at the moment")
        
        with col2:
            st.subheader("📉 Top Losers")
            losers = self.db.get_top_movers(direction='down', limit=5)
            
            if not losers.empty:
                for _, stock in losers.iterrows():
                    self._render_mover_card(stock, is_gainer=False)
            else:
                st.info("No significant losers at the moment")
    
    def _render_mover_card(self, stock, is_gainer):
        """Render individual mover card."""
        momentum_str = f"{stock['momentum']:+.2f}%"
        momentum_class = "positive" if is_gainer else "negative"
        arrow = "↑" if is_gainer else "↓"
        
        card_html = f"""
        <div class="ticker-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>{stock['ticker']}</strong> - {stock['sector']}<br>
                    <small>Price: ${stock['price']:.2f}</small>
                </div>
                <div class="{momentum_class}" style="text-align: right;">
                    {arrow} {momentum_str}<br>
                    <small>Vol: {stock['volume_ratio']:.1f}x</small>
                </div>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
    
    def render_sector_analysis(self):
        """Render sector analysis with charts."""
        st.header("🏢 Sector Analysis")
        
        # Get sector data
        sector_history = self.db.get_sector_trends_history(hours=24)
        
        if sector_history.empty:
            st.warning("No sector data available. Run the analyzer first.")
            return
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Current Performance", "24h Trends", "Heatmap"])
        
        with tab1:
            self._render_sector_performance()
        
        with tab2:
            self._render_sector_trends(sector_history)
        
        with tab3:
            self._render_sector_heatmap()
    
    def _render_sector_performance(self):
        """Render current sector performance."""
        latest_sectors = self.db.get_sector_trends_history(hours=1)
        
        if not latest_sectors.empty:
            # Get most recent data for each sector
            current_sectors = latest_sectors.sort_values('timestamp').groupby('sector').last().reset_index()
            current_sectors = current_sectors.sort_values('sector_score', ascending=False)
            
            # Create horizontal bar chart
            fig = px.bar(
                current_sectors,
                x='avg_momentum',
                y='sector',
                orientation='h',
                color='avg_momentum',
                color_continuous_scale='RdYlGn',
                title='Sector Momentum',
                labels={'avg_momentum': 'Average Momentum (%)', 'sector': 'Sector'}
            )
            
            fig.update_layout(
                height=400,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display sector metrics
            cols = st.columns(len(current_sectors))
            for idx, (_, sector) in enumerate(current_sectors.iterrows()):
                with cols[idx % len(cols)]:
                    delta = f"{sector['avg_momentum']:+.2f}%"
                    st.metric(
                        sector['sector'],
                        f"{sector['sector_score']:.1f}",
                        delta=delta
                    )
    
    def _render_sector_trends(self, sector_history):
        """Render sector trend lines over time."""
        # Create line chart for sector momentum over time
        fig = px.line(
            sector_history,
            x='timestamp',
            y='avg_momentum',
            color='sector',
            title='Sector Momentum Trends (24h)',
            labels={'avg_momentum': 'Average Momentum (%)', 'timestamp': 'Time'}
        )
        
        fig.update_layout(
            height=500,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_sector_heatmap(self):
        """Render sector performance heatmap."""
        # Get latest trends
        trends = self.db.get_latest_trends(limit=100)
        
        if not trends.empty:
            # Create pivot table for heatmap
            heatmap_data = trends.pivot_table(
                values='momentum',
                index='sector',
                columns='ticker',
                aggfunc='mean'
            )
            
            # Create heatmap
            fig = px.imshow(
                heatmap_data,
                labels=dict(x="Ticker", y="Sector", color="Momentum (%)"),
                color_continuous_scale='RdYlGn',
                aspect='auto',
                title='Stock Momentum Heatmap by Sector'
            )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_trend_analysis(self):
        """Render detailed trend analysis."""
        st.header("📊 Trend Analysis")
        
        # Get latest trends
        trends = self.db.get_latest_trends(limit=30)
        
        if trends.empty:
            st.warning("No trend data available. Run the analyzer first.")
            return
        
        # Create interactive scatter plot
        fig = px.scatter(
            trends,
            x='price_change',
            y='momentum',
            size='volume_ratio',
            color='sector',
            hover_data=['ticker', 'price', 'volume'],
            title='Price Change vs Momentum Analysis',
            labels={
                'price_change': 'Price Change (%)',
                'momentum': 'Momentum Score',
                'volume_ratio': 'Volume Ratio'
            }
        )
        
        # Add quadrant lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add quadrant labels
        fig.add_annotation(x=5, y=5, text="Strong Momentum", showarrow=False)
        fig.add_annotation(x=-5, y=5, text="Reversal Potential", showarrow=False)
        fig.add_annotation(x=-5, y=-5, text="Weak Momentum", showarrow=False)
        fig.add_annotation(x=5, y=-5, text="Correction Potential", showarrow=False)
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display trend table with styling
        st.subheader("📋 Detailed Trend Data")
        
        # Format the dataframe for display
        display_df = trends[['ticker', 'sector', 'price', 'price_change', 
                            'momentum', 'trend_score', 'volume_ratio']].copy()
        
        # Apply styling
        def color_negative_red(val):
            color = 'red' if val < 0 else 'green'
            return f'color: {color}'
        
        styled_df = display_df.style.applymap(
            color_negative_red, 
            subset=['price_change', 'momentum']
        ).format({
            'price': '${:.2f}',
            'price_change': '{:+.2f}%',
            'momentum': '{:+.2f}',
            'trend_score': '{:.1f}',
            'volume_ratio': '{:.2f}x'
        })
        
        st.dataframe(styled_df, use_container_width=True, height=400)
    
    def render_performance_metrics(self):
        """Render system performance metrics."""
        st.header("⚡ Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        # API performance
        with col1:
            api_stats = self.client.get_stats()
            st.markdown("### API Performance")
            st.metric("Total Calls", api_stats['total_api_calls'])
            st.metric("Success Rate", f"{api_stats['success_rate']}%")
            st.metric("Cache Hit Rate", f"{api_stats['cache_stats']['hit_rate']}%")
            st.metric("Remaining Calls", api_stats['remaining_calls'])
        
        # Database performance
        with col2:
            db_stats = self.db.get_database_stats()
            st.markdown("### Database Stats")
            st.metric("Total Trends", f"{db_stats.get('trends_count', 0):,}")
            st.metric("Total Sectors", f"{db_stats.get('sector_trends_count', 0):,}")
            st.metric("Total Alerts", f"{db_stats.get('alerts_count', 0):,}")
            st.metric("DB Size", f"{db_stats.get('db_size_mb', 0)} MB")
        
        # Alert statistics
        with col3:
            st.markdown("### Alert Summary")
            unack_alerts = db_stats.get('unacknowledged_alerts', {})
            st.metric("Critical", unack_alerts.get('CRITICAL', 0))
            st.metric("Warnings", unack_alerts.get('WARNING', 0))
            st.metric("Info", unack_alerts.get('INFO', 0))
            
            # Date range
            st.caption(f"Data Range: {db_stats.get('date_range', 'No data')}")
    
    def render_sidebar(self):
        """Render sidebar with controls."""
        with st.sidebar:
            st.header("⚙️ Dashboard Controls")
            
            # Auto-refresh settings
            st.subheader("🔄 Auto Refresh")
            st.session_state.auto_refresh = st.checkbox(
                "Enable Auto Refresh",
                value=st.session_state.auto_refresh
            )
            
            if st.session_state.auto_refresh:
                st.session_state.refresh_interval = st.slider(
                    "Refresh Interval (seconds)",
                    min_value=30,
                    max_value=300,
                    value=st.session_state.refresh_interval,
                    step=30
                )
            
            # Manual refresh button
            if st.button("🔄 Refresh Now", use_container_width=True):
                st.session_state.last_update = datetime.now()
                st.rerun()
            
            # Run analysis button
            st.subheader("📊 Analysis")
            if st.button("▶️ Run Analysis", use_container_width=True):
                with st.spinner("Running analysis..."):
                    try:
                        ticker_trends, sector_trends, alerts = self.analyzer.run_analysis()
                        st.success(f"Analysis complete! Found {len(alerts)} alerts.")
                        st.session_state.last_update = datetime.now()
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
            
            # Database maintenance
            st.subheader("🗄️ Database")
            if st.button("🧹 Clean Old Data", use_container_width=True):
                with st.spinner("Cleaning database..."):
                    deleted = self.db.cleanup_old_data(days_to_keep=30)
                    st.success(f"Cleaned {sum(deleted.values())} records")
            
            if st.button("🔧 Optimize Database", use_container_width=True):
                with st.spinner("Optimizing..."):
                    self.db.vacuum_database()
                    st.success("Database optimized!")
            
            # Display settings
            st.subheader("🎨 Display Settings")
            show_performance = st.checkbox("Show Performance Metrics", value=True)
            show_heatmap = st.checkbox("Show Sector Heatmap", value=True)
            
            return show_performance, show_heatmap
    
    def run(self):
        """Run the dashboard application."""
        # Auto-refresh logic
        if st.session_state.auto_refresh:
            st_autorefresh(
                interval=st.session_state.refresh_interval * 1000,
                limit=None,
                key="auto_refresh_component"
            )
        
        # Render components
        self.render_header()
        
        # Sidebar controls
        show_performance, show_heatmap = self.render_sidebar()
        
        # Main content
        self.render_alerts_section()
        
        st.divider()
        
        self.render_top_movers()
        
        st.divider()
        
        self.render_sector_analysis()
        
        st.divider()
        
        self.render_trend_analysis()
        
        if show_performance:
            st.divider()
            self.render_performance_metrics()
        
        # Footer
        st.divider()
        st.caption(
            f"Market Trend Tracker v2.0 | "
            f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Data provided by Finnhub"
        )


def main():
    """Main entry point for the dashboard."""
    try:
        dashboard = MarketDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Dashboard Error: {str(e)}")
        st.exception(e)
        
        # Provide troubleshooting steps
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            **Common Issues:**
            
            1. **Database not found**: Run `python3 valid_ticker_filter.py` first
            2. **API key error**: Check your FINNHUB_API_KEY environment variable
            3. **Import errors**: Make sure all required files are in the same directory
            4. **No data**: Run the analyzer using the button in the sidebar
            
            **Quick Fixes:**
            ```bash
            # Initialize database
            python3 -c "from db_manager import DatabaseManager; DatabaseManager()"
            
            # Run analyzer
            python3 valid_ticker_filter.py
            ```
            """)


if __name__ == "__main__":
    main()