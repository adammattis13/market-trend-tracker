# crypto_analyzer.py

import requests
import pandas as pd

TOP_CRYPTOS = ['bitcoin', 'ethereum', 'binancecoin', 'ripple', 'solana']

def fetch_crypto_momentum():
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": ",".join(TOP_CRYPTOS),
        "vs_currencies": "usd",
        "include_24hr_change": "true"
    }

    response = requests.get(url, params=params)
    data = response.json()

    records = []
    for name in TOP_CRYPTOS:
        if name in data:
            current = data[name]["usd"]
            change = data[name].get("usd_24h_change", 0)
            records.append({
                "crypto": name.capitalize(),
                "price_usd": current,
                "24h_change_pct": round(change, 2),
                "momentum_score": round(change / 100, 4)  # Normalize like equities
            })

    return pd.DataFrame(records)
