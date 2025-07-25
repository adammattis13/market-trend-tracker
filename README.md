# 📊 Market Trend Tracker

A real-time dashboard for monitoring stock and crypto momentum using sector-level analysis and trend scoring. Built with Streamlit, powered by Finnhub, and designed to support fast investment decisions.

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Features

✅ **Real-time trend scoring** for top 30 S&P 500 tickers  
📈 **Sector-level momentum analysis** and logging  
💹 **Live momentum** for top 5 cryptocurrencies  
🧭 **Sector trend visualizations** over time  
🧼 **Clean, modular, and extensible** Python architecture  
🔄 **Auto-refresh dashboard** with configurable intervals  
📊 **Interactive charts** with Plotly and Altair  

## 🛠️ Technology Stack

- **Backend**: Python 3.9+
- **Frontend**: Streamlit
- **Data Storage**: CSV files
- **API**: Finnhub (market data)
- **Visualization**: Plotly, Altair, Matplotlib, Seaborn
- **Auto-refresh**: streamlit-autorefresh

## 📋 Prerequisites

- Python 3.9 or higher
- Finnhub API key (free at [finnhub.io](https://finnhub.io))
- Internet connection for real-time data

## 🔧 Installation

### 1. Clone & Setup
```bash
git clone https://github.com/adammattis13/market-trend-tracker.git
cd market-trend-tracker
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. API Key Setup

**Copy the example secrets file:**
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

**Edit with your actual API keys:**
```bash
nano .streamlit/secrets.toml
```

**Get your API keys:**
- Finnhub: https://finnhub.io/

### 4. Run the Application

**Run the analyzer:**
```bash
python valid_ticker_filter.py
```

**Launch the dashboard:**
```bash
streamlit run dashboard.py
```

**Open your browser to:**
- Local: http://localhost:8501
- Network: http://YOUR_LOCAL_IP:8501

## 📚 Project Structure

```
market-trend-tracker/
├── dashboard.py              # Streamlit dashboard
├── valid_ticker_filter.py    # Core trend analyzer
├── sector_analyzer.py        # Sector-level aggregation logic
├── crypto_analyzer.py        # Crypto trend fetcher
├── requirements.txt          # Python dependencies
├── trend_df.csv             # Output: Current trend snapshot
├── trend_log.csv            # Output: Sector momentum over time
├── .streamlit/
│   ├── secrets.toml         # API key config (not in git)
│   └── secrets.toml.example # Example configuration
├── .gitignore              # Git ignore file
└── README.md               # This file
```

## ⚙️ Configuration

### Dashboard Settings
- **Auto-refresh interval**: Configurable in sidebar (30-300 seconds)
- **Data filters**: Toggle sectors and time ranges
- **Chart types**: Switch between different visualizations

### Modify Tickers
Edit `valid_ticker_filter.py` to customize which stocks to track:
```python
# Add or remove tickers from the analysis
TICKERS_TO_ANALYZE = ['AAPL', 'MSFT', 'GOOGL', ...]
```

## 🚀 Deployment

### Streamlit Community Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Add your Finnhub API key in Streamlit Cloud secrets:
   ```toml
   [api_keys]
   finnhub = "your_api_key_here"
   ```
5. Deploy!

### Local Network Access
To access from other devices on your network:
```bash
streamlit run dashboard.py --server.address 0.0.0.0
```

## 📈 Performance

- **Analysis speed**: ~2-3 seconds for 30 tickers
- **Dashboard refresh**: <1 second
- **API rate limits**: Respect Finnhub's free tier limits
- **Memory usage**: ~100-200MB typical

## 🔐 Security

- ✅ API keys stored in local secrets file (not in git)
- ✅ No user data collection
- ✅ All market data is public information
- ✅ Local data storage only

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🐛 Troubleshooting

### Common Issues

**"streamlit command not found"**
```bash
# Make sure you're in your virtual environment
source venv/bin/activate
pip install streamlit
```

**API key errors**
```bash
# Check your secrets file exists and has correct format
cat .streamlit/secrets.toml
```

**Import errors**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Finnhub](https://finnhub.io) for providing free market data API
- [Streamlit](https://streamlit.io) for the excellent dashboard framework
- [Plotly](https://plotly.com) for interactive visualizations

## 📞 Support

- 🐛 **Bug reports**: Create an issue with detailed description
- 💡 **Feature requests**: Use the "enhancement" label
- ❓ **Questions**: Check existing issues first

---

**Built by [Adam Mattis](https://github.com/adammattis13)**