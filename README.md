# 📊 Market Trend Tracker

A real-time financial dashboard for monitoring stock market trends, momentum analysis, and intelligent alerts. Built with Python, Streamlit, and powered by Finnhub API.

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Features

- **Real-Time Market Analysis**: Track top 30 S&P 500 stocks with live updates
- **Intelligent Alert System**: Critical, warning, and info alerts for market movements
- **Sector Analysis**: Monitor sector rotation and performance trends
- **Interactive Visualizations**: Plotly-powered charts and heatmaps
- **Top Movers Dashboard**: Instant view of biggest gainers and losers
- **Performance Metrics**: API health, database stats, and system monitoring
- **Auto-Refresh**: Configurable automatic updates (30-300 seconds)
- **Historical Tracking**: SQLite database for trend analysis

## 📸 Screenshots

### Main Dashboard
- Real-time alerts with severity levels
- Top movers with momentum indicators
- Sector performance analysis
- Interactive trend visualizations

## 🛠️ Technology Stack

- **Backend**: Python 3.9+
- **Frontend**: Streamlit
- **Database**: SQLite
- **API**: Finnhub (market data)
- **Visualization**: Plotly, Altair
- **Testing**: Pytest

## 📋 Prerequisites

- Python 3.9 or higher
- Finnhub API key (free at [finnhub.io](https://finnhub.io))
- 2GB RAM minimum
- Internet connection for real-time data

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/adammattis13/market-trend-tracker.git
cd market-trend-tracker
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
python3 -m pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the root directory:
```bash
FINNHUB_API_KEY=your_api_key_here
```

Or use Streamlit secrets (create `.streamlit/secrets.toml`):
```toml
[api_keys]
finnhub = "your_api_key_here"
```

### 5. Initialize Database
```bash
python3 -c "from db_manager import DatabaseManager; DatabaseManager()"
```

### 6. Run Initial Analysis
```bash
python3 valid_ticker_filter.py
```

## 🚀 Usage

### Start the Dashboard
```bash
python3 -m streamlit run dashboard.py
```

The dashboard will open at `http://localhost:8501`

### Command Line Tools

Run market analysis:
```bash
python3 valid_ticker_filter.py
```

Test API connection:
```bash
python3 api_client.py
```

Clean old data:
```bash
python3 -c "from db_manager import DatabaseManager; db = DatabaseManager(); print(db.cleanup_old_data(30))"
```

## 🧪 Testing

Run all tests:
```bash
python3 -m pytest tests/ -v
```

Run with coverage:
```bash
python3 -m pytest --cov=. --cov-report=html
```

## 📚 Project Structure

```
market-trend-tracker/
├── dashboard.py           # Streamlit dashboard
├── valid_ticker_filter.py # Core trend analysis engine
├── db_manager.py         # Database operations
├── api_client.py         # Finnhub API client
├── alert_system.py       # Alert generation system
├── sector_analyzer.py    # Sector-level analysis
├── crypto_analyzer.py    # Cryptocurrency tracking
├── requirements.txt      # Python dependencies
├── tests/               # Unit tests
│   ├── test_valid_ticker_filter.py
│   ├── test_api_client.py
│   └── test_dashboard.py
├── .env                 # Environment variables (not in git)
├── .gitignore          # Git ignore file
└── README.md           # This file
```

## 🔐 Security

- API keys are stored in environment variables
- Database is local (SQLite) - no external data exposure
- No user data is collected or stored
- All financial data is public market information

## 🚀 Deployment

### Option 1: Streamlit Community Cloud (Recommended)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add your API key in Streamlit secrets
5. Deploy!

### Option 2: Heroku

See `deployment/heroku.md` for detailed instructions.

### Option 3: AWS/GCP/Azure

See `deployment/cloud.md` for cloud deployment guides.

## ⚙️ Configuration

### Modify Alert Thresholds
Edit `alert_system.py`:
```python
self.thresholds = {
    'momentum_surge': 5.0,      # Adjust sensitivity
    'momentum_drop': -5.0,
    'volume_spike': 2.0,
}
```

### Add/Remove Tickers
Edit `valid_ticker_filter.py`:
```python
TOP_SP500_TICKERS = [
    # Add your tickers here
]
```

### Change Refresh Rate
In the dashboard sidebar, adjust the refresh interval slider (30-300 seconds).

## 📈 Performance

- Analysis of 30 tickers: ~2 seconds
- Dashboard refresh: <1 second
- API calls: 60/minute limit
- Database size: ~10MB per month

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Finnhub](https://finnhub.io) for market data API
- [Streamlit](https://streamlit.io) for the amazing dashboard framework
- [Plotly](https://plotly.com) for interactive visualizations

## 📞 Support

- Create an issue for bug reports
- Check existing issues before creating new ones
- For feature requests, use the "enhancement" label

## 🗺️ Roadmap

- [ ] Add cryptocurrency support
- [ ] Email/SMS alerts
- [ ] Machine learning predictions
- [ ] Portfolio tracking
- [ ] News sentiment analysis
- [ ] Mobile app
- [ ] Multi-user support

---

**Built with ❤️ by Adam Mattis**