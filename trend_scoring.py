import numpy as np

# ----------------------------------------
# 📈 Trend Score Calculation
# ----------------------------------------

def calculate_trend(current_price, past_prices, min_length=5):
    """
    Calculates a z-score-like trend score:
    (current_price - mean of past_prices) / std_dev of past_prices

    This indicates how strongly the current price deviates
    from recent historical behavior.

    Args:
        current_price (float): Most recent price.
        past_prices (list of float): Previous closing prices.
        min_length (int): Minimum number of past prices required.

    Returns:
        float: Trend score. 
               > 0 = bullish momentum
               < 0 = bearish momentum
               0 = no trend or insufficient data
    """
    # 🔒 Require a minimum number of prices to calculate trend
    if len(past_prices) < min_length:
        return 0

    # 🧠 Calculate mean and standard deviation of the past prices
    mean = np.mean(past_prices)
    std = np.std(past_prices)

    # 🔐 Prevent division by zero (flat volatility case)
    if std == 0:
        return 0

    # 🔢 Return the normalized z-score
    return (current_price - mean) / std

# ----------------------------------------
# 🧪 Test Block (Run manually to validate)
# ----------------------------------------

if __name__ == "__main__":
    # Simulated past prices and a new price point
    past = [100, 102, 101, 99, 98]
    current = 105

    # Calculate trend score and print
    score = calculate_trend(current, past)
    print(f"Past Prices: {past}")
    print(f"Current Price: {current}")
    print(f"Trend Score: {score:.2f}")