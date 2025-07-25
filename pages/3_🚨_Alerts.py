"""
Alerts Page - Alert management and configuration
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import numpy as np

def show():
    """Alerts management page"""
    
    st.header("🚨 Alert Management")
    
    # Alert summary
    display_alert_summary()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Active alerts
        st.subheader("🔔 Active Alerts")
        display_active_alerts()
        
        # Alert history
        st.subheader("📜 Alert History")
        display_alert_history()
    
    with col2:
        # Create new alert
        st.subheader("➕ Create Alert")
        display_create_alert_form()
        
        # Alert settings
        st.subheader("⚙️ Alert Settings")
        display_alert_settings()

def display_alert_summary():
    """Display alert summary metrics"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Alerts", "24", "+3", delta_color="normal")
    
    with col2:
        st.metric("Triggered Today", "8", "+2", delta_color="normal")
    
    with col3:
        st.metric("Critical Alerts", "2", "0", delta_color="normal")
    
    with col4:
        st.metric("Success Rate", "94.2%", "+1.2%", delta_color="normal")

def display_active_alerts():
    """Display currently active alerts"""
    
    active_alerts = [
        {
            "Symbol": "AAPL",
            "Type": "Price Above",
            "Condition": "> $200.00",
            "Current": "$195.67",
            "Status": "⏳ Pending",
            "Priority": "🟡 Medium",
            "Created": "2024-07-20"
        },
        {
            "Symbol": "TSLA", 
            "Type": "Volume Spike",
            "Condition": "> 50M shares",
            "Current": "42.3M",
            "Status": "⏳ Pending",
            "Priority": "🟢 Low",
            "Created": "2024-07-22"
        },
        {
            "Symbol": "NVDA",
            "Type": "RSI Oversold",
            "Condition": "< 30",
            "Current": "32.4",
            "Status": "⏳ Pending", 
            "Priority": "🔴 High",
            "Created": "2024-07-23"
        },
        {
            "Symbol": "MSFT",
            "Type": "Moving Average",
            "Condition": "Cross above 50MA",
            "Current": "Below",
            "Status": "⏳ Pending",
            "Priority": "🟡 Medium",
            "Created": "2024-07-24"
        }
    ]
    
    df = pd.DataFrame(active_alerts)
    
    # Add action buttons column
    for i, alert in enumerate(active_alerts):
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1, 1, 1.5, 1, 1, 1, 1, 1])
        
        with col1:
            st.write(alert["Symbol"])
        with col2:
            st.write(alert["Type"])
        with col3:
            st.write(alert["Condition"])
        with col4:
            st.write(alert["Current"])
        with col5:
            st.write(alert["Status"])
        with col6:
            st.write(alert["Priority"])
        with col7:
            st.write(alert["Created"])
        with col8:
            col_edit, col_delete = st.columns(2)
            with col_edit:
                if st.button("✏️", key=f"edit_{i}", help="Edit alert"):
                    st.info(f"Edit alert for {alert['Symbol']}")
            with col_delete:
                if st.button("🗑️", key=f"delete_{i}", help="Delete alert"):
                    st.success(f"Deleted alert for {alert['Symbol']}")

def display_alert_history():
    """Display alert history with filtering"""
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_filter = st.selectbox(
            "Time Period",
            ["Last 7 days", "Last 30 days", "Last 90 days", "All time"]
        )
    
    with col2:
        status_filter = st.selectbox(
            "Status",
            ["All", "Triggered", "Expired", "Cancelled"]
        )
    
    with col3:
        priority_filter = st.selectbox(
            "Priority",
            ["All", "High", "Medium", "Low"]
        )
    
    # Sample alert history data
    history_data = [
        {
            "Time": "2024-07-25 10:32",
            "Symbol": "AAPL",
            "Type": "Price Above",
            "Condition": "> $190.00",
            "Triggered Value": "$195.67",
            "Status": "✅ Triggered",
            "Priority": "🟡 Medium"
        },
        {
            "Time": "2024-07-25 09:15",
            "Symbol": "TSLA",
            "Type": "Volume Spike", 
            "Condition": "> 40M shares",
            "Triggered Value": "52.3M",
            "Status": "✅ Triggered",
            "Priority": "🟢 Low"
        },
        {
            "Time": "2024-07-24 15:45",
            "Symbol": "GOOGL",
            "Type": "Price Below",
            "Condition": "< $145.00",
            "Triggered Value": "$144.23",
            "Status": "✅ Triggered",
            "Priority": "🔴 High"
        },
        {
            "Time": "2024-07-24 11:22",
            "Symbol": "AMZN",
            "Type": "RSI Overbought",
            "Condition": "> 70",
            "Triggered Value": "72.4",
            "Status": "⏰ Expired",
            "Priority": "🟡 Medium"
        }
    ]
    
    df = pd.DataFrame(history_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Alert frequency chart
    st.markdown("**📊 Alert Frequency (Last 30 Days)**")
    
    # Generate sample frequency data
    dates = [datetime.now() - timedelta(days=x) for x in range(30, 0, -1)]
    triggered_counts = [max(0, int(np.random.poisson(2.5))) for _ in dates]
    
    freq_df = pd.DataFrame({
        'Date': dates,
        'Alerts Triggered': triggered_counts
    })
    
    fig = px.bar(freq_df, x='Date', y='Alerts Triggered', title="Daily Alert Activity")
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)

def display_create_alert_form():
    """Display form to create new alerts"""
    
    with st.form("create_alert_form"):
        # Symbol input
        symbol = st.text_input("Symbol", placeholder="e.g., AAPL")
        
        # Alert type
        alert_type = st.selectbox(
            "Alert Type",
            [
                "Price Above",
                "Price Below", 
                "Price Change %",
                "Volume Spike",
                "RSI Oversold",
                "RSI Overbought",
                "Moving Average Cross",
                "Support/Resistance Break"
            ]
        )
        
        # Condition value
        if alert_type in ["Price Above", "Price Below"]:
            condition = st.number_input("Price ($)", min_value=0.01, value=100.00, step=0.01)
        elif alert_type == "Price Change %":
            condition = st.number_input("Change (%)", value=5.0, step=0.1)
        elif alert_type == "Volume Spike":
            condition = st.number_input("Volume (M shares)", min_value=0.1, value=10.0, step=0.1)
        elif alert_type in ["RSI Oversold", "RSI Overbought"]:
            condition = st.number_input("RSI Level", min_value=0, max_value=100, value=30 if "Oversold" in alert_type else 70)
        else:
            condition = st.text_input("Condition", placeholder="Enter condition details")
        
        # Priority
        priority = st.selectbox("Priority", ["Low", "Medium", "High"])
        
        # Notification settings
        st.markdown("**📱 Notification Settings**")
        col1, col2 = st.columns(2)
        with col1:
            email_notify = st.checkbox("Email", value=True)
            sms_notify = st.checkbox("SMS")
        with col2:
            app_notify = st.checkbox("In-App", value=True)
            webhook_notify = st.checkbox("Webhook")
        
        # Expiration
        expires = st.selectbox(
            "Expires In",
            ["1 hour", "1 day", "1 week", "1 month", "Never"]
        )
        
        # Submit button
        submitted = st.form_submit_button("🚨 Create Alert", use_container_width=True)
        
        if submitted:
            if symbol:
                st.success(f"✅ Alert created for {symbol.upper()}: {alert_type} {condition}")
                st.balloons()
            else:
                st.error("Please enter a symbol")

def display_alert_settings():
    """Display global alert settings"""
    
    st.markdown("**🔔 Notification Preferences**")
    
    # Global notification settings
    email_enabled = st.checkbox("Email Notifications", value=True)
    if email_enabled:
        email = st.text_input("Email Address", value="user@example.com")
    
    sms_enabled = st.checkbox("SMS Notifications") 
    if sms_enabled:
        phone = st.text_input("Phone Number", value="+1 (555) 123-4567")
    
    # Notification frequency
    st.markdown("**⏰ Frequency Settings**")
    
    max_alerts_per_hour = st.slider("Max Alerts per Hour", 1, 50, 10)
    max_alerts_per_day = st.slider("Max Alerts per Day", 1, 200, 50)
    
    # Quiet hours
    st.markdown("**🌙 Quiet Hours**")
    
    quiet_hours_enabled = st.checkbox("Enable Quiet Hours")
    if quiet_hours_enabled:
        col1, col2 = st.columns(2)
        with col1:
            quiet_start = st.time_input("Start Time", value=datetime.strptime("22:00", "%H:%M").time())
        with col2:
            quiet_end = st.time_input("End Time", value=datetime.strptime("08:00", "%H:%M").time())
    
    # Market hours only
    market_hours_only = st.checkbox("Alerts during market hours only", value=True)
    
    # Save settings
    if st.button("💾 Save Settings", use_container_width=True):
        st.success("✅ Settings saved successfully!")
        
# This is important for Streamlit multi-page apps
if __name__ == "__main__":
    show()
    
# This ensures it works in multi-page app
    show()