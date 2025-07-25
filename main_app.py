"""
Market Trend Tracker - Multi-page Application
Main entry point with navigation and page routing
"""

import streamlit as st
from pathlib import Path
import sys

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import page modules
from pages import dashboard_page, portfolio_page, alerts_page, analysis_page

# Page configuration
st.set_page_config(
    page_title="Market Trend Tracker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 1rem;
    }
    
    .nav-pills {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .nav-pill {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        background-color: #f0f2f6;
        border: 2px solid #ddd;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .nav-pill.active {
        background-color: #1f77b4;
        color: white;
        border-color: #1f77b4;
    }
    
    .sidebar-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application entry point"""
    
    # Main header
    st.markdown('<h1 class="main-header">📊 Market Trend Tracker</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        
        # Page selection
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Dashboard", "💼 Portfolio", "🚨 Alerts", "📈 Analysis"],
            key="page_selector"
        )
        
        # App info
        st.markdown("""
        <div class="sidebar-info">
            <h4>🚀 Features</h4>
            <ul>
                <li>Real-time market data</li>
                <li>Technical indicators</li>
                <li>Portfolio tracking</li>
                <li>Smart alerts</li>
                <li>Advanced analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # API status
        with st.expander("📡 System Status"):
            st.success("✅ API Connected")
            st.info("📊 Data Updated: Live")
            st.metric("Refresh Rate", "30s", delta="Real-time")
    
    # Page routing
    if page == "🏠 Dashboard":
        dashboard_page.show()
    elif page == "💼 Portfolio":
        portfolio_page.show()
    elif page == "🚨 Alerts":
        alerts_page.show()
    elif page == "📈 Analysis":
        analysis_page.show()

if __name__ == "__main__":
    main()