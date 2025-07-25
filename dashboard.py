# dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime
import time
from streamlit_autorefresh import st_autorefresh

# Import from root level valid_ticker_filter
from valid_ticker_filter import TrendAnalyzer

# Import from src
from src.config import get_api_key, TREND_FILE, LOG_FILE, SECTOR_LOG_FILE, REFRESH_INTERVAL

# Set page config
st.set_page_config(
    page_title="Market Trend Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

def load_trend_data():
    """Load trend data from CSV"""
    try:
        if TREND_FILE.exists():
            return pd.read_csv(TREND_FILE)
        else:
            st.warning("No trend data found. Running analysis...")
            analyzer = TrendAnalyzer()
            return analyzer.run_analysis()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def load_sector_data():
    """Load sector momentum data"""
    try:
        if LOG_FILE.exists():
            return pd.read_csv(LOG_FILE)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading sector data: {str(e)}")
        return pd.DataFrame()

def display_header():
    """Display header with refresh info"""
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.title("📈 Market Trend Tracker")
    
    with col2:
        st.metric("Last Update", st.session_state.last_update.strftime("%H:%M:%S"))
    
    with col3:
        if st.button("🔄 Refresh Now"):
            try:
                analyzer = TrendAnalyzer()
                analyzer.run_analysis()
                st.session_state.last_update = datetime.now()
                st.success("Data refreshed!")
                st.rerun()
            except Exception as e:
                st.error(f"Refresh failed: {str(e)}")

def display_market_overview(df):
    """Display market overview metrics"""
    if df.empty:
        st.warning("No data available")
        return
    
    st.subheader("📊 Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_momentum = df['Momentum_%'].mean()
        st.metric(
            "Average Market Momentum",
            f"{avg_momentum:.2f}%",
            delta=f"{avg_momentum:.2f}%"
        )
    
    with col2:
        positive_stocks = len(df[df['Momentum_%'] > 0])
        st.metric(
            "Stocks Up",
            f"{positive_stocks}/{len(df)}",
            delta=f"{(positive_stocks/len(df)*100):.1f}%"
        )
    
    with col3:
        top_performer = df.iloc[0]
        st.metric(
            "Top Performer",
            top_performer['Ticker'],
            delta=f"{top_performer['Momentum_%']:.2f}%"
        )
    
    with col4:
        bottom_performer = df.iloc[-1]
        st.metric(
            "Bottom Performer",
            bottom_performer['Ticker'],
            delta=f"{bottom_performer['Momentum_%']:.2f}%"
        )

def display_trend_table(df):
    """Display trend table with formatting"""
    st.subheader("📈 Live Trend Scores")
    
    # Add visual indicators
    df_display = df.copy()
    df_display['Signal'] = df_display['Signal'].apply(
        lambda x: f"{'🟢' if 'BUY' in x else '🔴' if 'SELL' in x else '🟡'} {x}"
    )
    
    # Format columns
    column_config = {
        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
        "Sector": st.column_config.TextColumn("Sector", width="medium"),
        "Current_Price": st.column_config.NumberColumn("Price", format="$%.2f", width="small"),
        "Momentum_%": st.column_config.NumberColumn("Momentum %", format="%.2f%%", width="small"),
        "Trend_Score": st.column_config.ProgressColumn("Trend Score", min_value=0, max_value=100, width="medium"),
        "Signal": st.column_config.TextColumn("Signal", width="medium"),
    }
    
    # Display table
    st.dataframe(
        df_display[['Ticker', 'Sector', 'Current_Price', 'Momentum_%', 'Trend_Score', 'Signal']],
        use_container_width=True,
        height=400,
        column_config=column_config,
        hide_index=True
    )

def display_sector_performance(df):
    """Display sector performance chart"""
    st.subheader("🏭 Sector Performance")
    
    if df.empty:
        st.warning("No sector data available")
        return
    
    # Calculate sector averages
    sector_perf = df.groupby('Sector')['Momentum_%'].agg(['mean', 'count']).round(2)
    sector_perf = sector_perf.sort_values('mean', ascending=True)
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if x < 0 else 'green' for x in sector_perf['mean']]
    
    bars = ax.barh(sector_perf.index, sector_perf['mean'], color=colors, alpha=0.7)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.1 if width > 0 else width - 0.1,
                bar.get_y() + bar.get_height()/2,
                f'{width:.2f}%',
                ha='left' if width > 0 else 'right',
                va='center')
    
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Average Momentum %')
    ax.set_title('Sector Performance')
    plt.tight_layout()
    
    st.pyplot(fig)

def display_momentum_chart(df):
    """Display momentum distribution"""
    st.subheader("📊 Momentum Distribution")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1.hist(df['Momentum_%'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Momentum %')
    ax1.set_ylabel('Count')
    ax1.set_title('Momentum Distribution')
    
    # Box plot by sector
    df.boxplot(column='Momentum_%', by='Sector', ax=ax2, rot=45)
    ax2.set_xlabel('Sector')
    ax2.set_ylabel('Momentum %')
    ax2.set_title('Momentum by Sector')
    
    plt.tight_layout()
    st.pyplot(fig)

def display_trend_history():
    """Display historical trend data"""
    st.subheader("📈 Historical Trends")
    
    try:
        if LOG_FILE.exists():
            history_df = pd.read_csv(LOG_FILE)
            if not history_df.empty:
                # Convert timestamp to datetime
                history_df['Timestamp'] = pd.to_datetime(history_df['Timestamp'])
                
                # Plot sector trends over time
                fig, ax = plt.subplots(figsize=(12, 6))
                
                for sector in history_df['Sector'].unique():
                    sector_data = history_df[history_df['Sector'] == sector]
                    ax.plot(sector_data['Timestamp'], 
                           sector_data['Avg_Momentum_%'], 
                           marker='o', 
                           label=sector,
                           alpha=0.7)
                
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax.set_xlabel('Time')
                ax.set_ylabel('Average Momentum %')
                ax.set_title('Sector Momentum Over Time')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("No historical data available yet. Data will accumulate over time.")
    except Exception as e:
        st.error(f"Error displaying history: {str(e)}")

def main():
    """Main dashboard function"""
    # Auto-refresh every 5 minutes
    st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="datarefresh")
    
    # Display header
    display_header()
    
    # Load data
    df = load_trend_data()
    
    if df.empty:
        st.error("No data available. Please check your API key and internet connection.")
        st.info("Make sure FINNHUB_API_KEY is set in your environment variables.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Analysis", "📜 History"])
    
    with tab1:
        # Market overview
        display_market_overview(df)
        
        # Trend table
        display_trend_table(df)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Sector performance
            display_sector_performance(df)
        
        with col2:
            # Top movers
            st.subheader("🚀 Top Movers")
            
            # Top gainers
            st.markdown("**🟢 Top Gainers**")
            top_gainers = df.nlargest(5, 'Momentum_%')[['Ticker', 'Momentum_%', 'Current_Price']]
            st.dataframe(top_gainers, hide_index=True)
            
            st.markdown("**🔴 Top Losers**")
            top_losers = df.nsmallest(5, 'Momentum_%')[['Ticker', 'Momentum_%', 'Current_Price']]
            st.dataframe(top_losers, hide_index=True)
        
        # Momentum distribution
        display_momentum_chart(df)
    
    with tab3:
        # Historical trends
        display_trend_history()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Status
        try:
            api_key = get_api_key()
            st.success("✅ API Connected")
        except:
            st.error("❌ API Key Missing")
        
        # Data info
        st.info(f"Tracking {len(df)} stocks")
        st.info(f"Last update: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Filters
        st.subheader("🔍 Filters")
        
        # Sector filter
        sectors = ['All'] + sorted(df['Sector'].unique().tolist())
        selected_sector = st.selectbox("Select Sector", sectors)
        
        # Signal filter
        signals = ['All'] + sorted(df['Signal'].unique().tolist())
        selected_signal = st.selectbox("Select Signal", signals)
        
        # Apply filters
        if selected_sector != 'All':
            df = df[df['Sector'] == selected_sector]
        
        if selected_signal != 'All':
            df = df[df['Signal'] == selected_signal]

if __name__ == "__main__":
    main()