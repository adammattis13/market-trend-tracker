"""
Technical Indicators Library
Professional-grade technical analysis calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """
    Comprehensive technical indicators calculation library
    """
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """
        Simple Moving Average
        """
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """
        Exponential Moving Average
        """
        return data.ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """
        Relative Strength Index
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence)
        Returns: {'macd': MACD line, 'signal': Signal line, 'histogram': MACD histogram}
        """
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """
        Bollinger Bands
        Returns: {'middle': SMA, 'upper': Upper band, 'lower': Lower band}
        """
        sma = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return {
            'middle': sma,
            'upper': upper_band,
            'lower': lower_band
        }
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """
        Stochastic Oscillator
        Returns: {'%K': %K line, '%D': %D line}
        """
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            '%K': k_percent,
            '%D': d_percent
        }
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Williams %R
        """
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        return williams_r
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Average True Range
        """
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume
        """
        obv = np.where(close > close.shift(), volume, 
               np.where(close < close.shift(), -volume, 0)).cumsum()
        return pd.Series(obv, index=close.index)
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Volume Weighted Average Price
        """
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> Dict[str, pd.Series]:
        """
        Average Directional Index
        Returns: {'ADX': ADX, '+DI': Positive DI, '-DI': Negative DI}
        """
        tr = TechnicalIndicators.atr(high, low, close, 1)
        
        up = high.diff()
        down = -low.diff()
        
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        
        plus_dm_smooth = pd.Series(plus_dm, index=close.index).rolling(window=window).mean()
        minus_dm_smooth = pd.Series(minus_dm, index=close.index).rolling(window=window).mean()
        tr_smooth = tr.rolling(window=window).mean()
        
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=window).mean()
        
        return {
            'ADX': adx,
            '+DI': plus_di,
            '-DI': minus_di
        }

class TechnicalAnalysis:
    """
    Higher-level technical analysis functions
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with OHLCV data
        Expected columns: ['open', 'high', 'low', 'close', 'volume']
        """
        self.data = data.copy()
        self.indicators = {}
        
    def add_all_indicators(self) -> pd.DataFrame:
        """
        Add all common technical indicators to the dataframe
        """
        df = self.data.copy()
        
        # Price-based indicators
        df['SMA_20'] = TechnicalIndicators.sma(df['close'], 20)
        df['SMA_50'] = TechnicalIndicators.sma(df['close'], 50)
        df['SMA_200'] = TechnicalIndicators.sma(df['close'], 200)
        
        df['EMA_12'] = TechnicalIndicators.ema(df['close'], 12)
        df['EMA_26'] = TechnicalIndicators.ema(df['close'], 26)
        
        df['RSI'] = TechnicalIndicators.rsi(df['close'])
        
        # MACD
        macd_data = TechnicalIndicators.macd(df['close'])
        df['MACD'] = macd_data['macd']
        df['MACD_Signal'] = macd_data['signal']
        df['MACD_Histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = TechnicalIndicators.bollinger_bands(df['close'])
        df['BB_Upper'] = bb_data['upper']
        df['BB_Middle'] = bb_data['middle']
        df['BB_Lower'] = bb_data['lower']
        
        # Stochastic
        if all(col in df.columns for col in ['high', 'low']):
            stoch_data = TechnicalIndicators.stochastic(df['high'], df['low'], df['close'])
            df['Stoch_K'] = stoch_data['%K']
            df['Stoch_D'] = stoch_data['%D']
            
            df['Williams_R'] = TechnicalIndicators.williams_r(df['high'], df['low'], df['close'])
            df['ATR'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'])
        
        # Volume-based indicators
        if 'volume' in df.columns:
            df['OBV'] = TechnicalIndicators.obv(df['close'], df['volume'])
            if all(col in df.columns for col in ['high', 'low']):
                df['VWAP'] = TechnicalIndicators.vwap(df['high'], df['low'], df['close'], df['volume'])
        
        # ADX
        if all(col in df.columns for col in ['high', 'low']):
            adx_data = TechnicalIndicators.adx(df['high'], df['low'], df['close'])
            df['ADX'] = adx_data['ADX']
            df['Plus_DI'] = adx_data['+DI']
            df['Minus_DI'] = adx_data['-DI']
        
        return df
    
    def get_signals(self) -> Dict[str, str]:
        """
        Generate trading signals based on technical indicators
        """
        df = self.add_all_indicators()
        latest = df.iloc[-1]
        signals = {}
        
        # RSI Signals
        if 'RSI' in df.columns and not pd.isna(latest['RSI']):
            if latest['RSI'] > 70:
                signals['RSI'] = 'Overbought (Sell Signal)'
            elif latest['RSI'] < 30:
                signals['RSI'] = 'Oversold (Buy Signal)'
            else:
                signals['RSI'] = 'Neutral'
        
        # MACD Signals
        if all(col in df.columns for col in ['MACD', 'MACD_Signal']):
            if not pd.isna(latest['MACD']) and not pd.isna(latest['MACD_Signal']):
                if latest['MACD'] > latest['MACD_Signal']:
                    signals['MACD'] = 'Bullish (Buy Signal)'
                else:
                    signals['MACD'] = 'Bearish (Sell Signal)'
        
        # Moving Average Signals
        if all(col in df.columns for col in ['close', 'SMA_20', 'SMA_50']):
            close_price = latest['close']
            if not pd.isna(latest['SMA_20']) and not pd.isna(latest['SMA_50']):
                if close_price > latest['SMA_20'] > latest['SMA_50']:
                    signals['Moving Averages'] = 'Strong Bullish'
                elif close_price > latest['SMA_20']:
                    signals['Moving Averages'] = 'Bullish'
                elif close_price < latest['SMA_20'] < latest['SMA_50']:
                    signals['Moving Averages'] = 'Strong Bearish'
                else:
                    signals['Moving Averages'] = 'Bearish'
        
        # Bollinger Bands Signals
        if all(col in df.columns for col in ['close', 'BB_Upper', 'BB_Lower']):
            close_price = latest['close']
            if not pd.isna(latest['BB_Upper']) and not pd.isna(latest['BB_Lower']):
                if close_price > latest['BB_Upper']:
                    signals['Bollinger Bands'] = 'Overbought'
                elif close_price < latest['BB_Lower']:
                    signals['Bollinger Bands'] = 'Oversold'
                else:
                    signals['Bollinger Bands'] = 'Normal Range'
        
        return signals
    
    def get_support_resistance(self, window: int = 20) -> Dict[str, float]:
        """
        Calculate support and resistance levels
        """
        df = self.data.copy()
        recent_data = df.tail(window)
        
        support = recent_data['low'].min()
        resistance = recent_data['high'].max()
        
        # More sophisticated S&R using pivot points
        pivot_high = recent_data['high'].rolling(window=5, center=True).max()
        pivot_low = recent_data['low'].rolling(window=5, center=True).min()
        
        resistance_levels = pivot_high.dropna().unique()
        support_levels = pivot_low.dropna().unique()
        
        return {
            'support': float(support),
            'resistance': float(resistance),
            'support_levels': sorted(support_levels)[-3:],  # Top 3 support levels
            'resistance_levels': sorted(resistance_levels, reverse=True)[:3]  # Top 3 resistance levels
        }

def create_sample_ohlcv_data(symbol: str = "AAPL", days: int = 100) -> pd.DataFrame:
    """
    Create sample OHLCV data for testing
    """
    from datetime import datetime, timedelta
    
    # Generate dates
    end_date = datetime.now()
    dates = [end_date - timedelta(days=x) for x in range(days, 0, -1)]
    
    # Generate realistic price data
    base_price = 150.0
    prices = [base_price]
    
    for i in range(1, days):
        # Random walk with slight upward bias
        change = np.random.normal(0.001, 0.02)  # 0.1% average daily return, 2% volatility
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1.0))  # Prevent negative prices
    
    # Generate OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate realistic open, high, low from close
        open_price = close * (1 + np.random.normal(0, 0.005))
        high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
        low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
        volume = np.random.uniform(20000000, 80000000)  # 20M to 80M shares
        
        data.append({
            'date': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    return df

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = create_sample_ohlcv_data("AAPL", 100)
    
    # Initialize technical analysis
    ta = TechnicalAnalysis(sample_data)
    
    # Add all indicators
    df_with_indicators = ta.add_all_indicators()
    
    # Get trading signals
    signals = ta.get_signals()
    
    # Get support/resistance
    levels = ta.get_support_resistance()
    
    print("Technical Analysis Results:")
    print("=" * 40)
    print("Latest Indicators:")
    print(df_with_indicators.tail(1)[['close', 'RSI', 'MACD', 'SMA_20', 'SMA_50']].to_string())
    print("\nTrading Signals:")
    for indicator, signal in signals.items():
        print(f"{indicator}: {signal}")
    print(f"\nSupport: ${levels['support']:.2f}")
    print(f"Resistance: ${levels['resistance']:.2f}")