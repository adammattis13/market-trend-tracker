#!/bin/bash
# run.sh

# Check for API key
if [ -z "$FINNHUB_API_KEY" ]; then
    echo "Error: FINNHUB_API_KEY environment variable not set"
    echo "Please run: export FINNHUB_API_KEY='your_api_key_here'"
    exit 1
fi

# Create directories if they don't exist
mkdir -p data data/backups

# Run the analyzer first to generate data
echo "Running market analysis..."
python valid_ticker_filter.py

# Launch the dashboard
echo "Launching dashboard..."
streamlit run main_app.py