import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass
from enum import Enum

class AlertType(Enum):
    """Types of market alerts"""
    MOMENTUM_SURGE = "momentum_surge"
    MOMENTUM_DROP = "momentum_drop"
    VOLUME_SPIKE = "volume_spike"
    SECTOR_ROTATION = "sector_rotation"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    TREND_REVERSAL = "trend_reversal"
    CRYPTO_MOVEMENT = "crypto_movement"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

@dataclass
class Alert:
    """Alert data structure"""
    alert_type: AlertType
    severity: AlertSeverity
    ticker: Optional[str]
    sector: Optional[str]
    message: str
    value: float
    threshold: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class MarketAlertSystem:
    """Intelligent alert system for market movements"""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.alerts = []
        
        # Alert thresholds (configurable)
        self.thresholds = {
            'momentum_surge': 5.0,      # 5% momentum increase
            'momentum_drop': -5.0,      # 5% momentum decrease
            'volume_spike': 2.0,        # 2x average volume
            'sector_rotation': 3.0,     # 3% sector momentum shift
            'breakout': 2.0,           # 2% above resistance
            'breakdown': -2.0,         # 2% below support
            'crypto_surge': 10.0,      # 10% crypto movement
        }
        
        # Store previous states for comparison
        self.previous_trends = {}
        self.previous_sector_trends = {}
        
    def check_ticker_alerts(self, current_data: pd.DataFrame) -> List[Alert]:
        """Check for ticker-specific alerts"""
        alerts = []
        
        for _, row in current_data.iterrows():
            ticker = row['ticker']
            
            # Momentum alerts
            if row['momentum'] >= self.thresholds['momentum_surge']:
                alerts.append(Alert(
                    alert_type=AlertType.MOMENTUM_SURGE,
                    severity=AlertSeverity.WARNING,
                    ticker=ticker,
                    sector=row.get('sector'),
                    message=f"{ticker} momentum surge: {row['momentum']:.2f}%",
                    value=row['momentum'],
                    threshold=self.thresholds['momentum_surge']
                ))
                
            elif row['momentum'] <= self.thresholds['momentum_drop']:
                alerts.append(Alert(
                    alert_type=AlertType.MOMENTUM_DROP,
                    severity=AlertSeverity.WARNING,
                    ticker=ticker,
                    sector=row.get('sector'),
                    message=f"{ticker} momentum drop: {row['momentum']:.2f}%",
                    value=row['momentum'],
                    threshold=self.thresholds['momentum_drop']
                ))
            
            # Volume spike alerts
            if 'volume_ratio' in row and row['volume_ratio'] >= self.thresholds['volume_spike']:
                alerts.append(Alert(
                    alert_type=AlertType.VOLUME_SPIKE,
                    severity=AlertSeverity.INFO,
                    ticker=ticker,
                    sector=row.get('sector'),
                    message=f"{ticker} volume spike: {row['volume_ratio']:.1f}x average",
                    value=row['volume_ratio'],
                    threshold=self.thresholds['volume_spike']
                ))
            
            # Check for breakouts/breakdowns using historical data
            if ticker in self.previous_trends:
                prev = self.previous_trends[ticker]
                price_change = ((row['price'] - prev['price']) / prev['price']) * 100
                
                if price_change >= self.thresholds['breakout']:
                    alerts.append(Alert(
                        alert_type=AlertType.BREAKOUT,
                        severity=AlertSeverity.WARNING,
                        ticker=ticker,
                        sector=row.get('sector'),
                        message=f"{ticker} potential breakout: +{price_change:.2f}%",
                        value=price_change,
                        threshold=self.thresholds['breakout']
                    ))
                elif price_change <= self.thresholds['breakdown']:
                    alerts.append(Alert(
                        alert_type=AlertType.BREAKDOWN,
                        severity=AlertSeverity.WARNING,
                        ticker=ticker,
                        sector=row.get('sector'),
                        message=f"{ticker} potential breakdown: {price_change:.2f}%",
                        value=price_change,
                        threshold=self.thresholds['breakdown']
                    ))
        
        # Update previous trends
        self.previous_trends = current_data.set_index('ticker').to_dict('index')
        
        return alerts
    
    def check_sector_alerts(self, sector_data: pd.DataFrame) -> List[Alert]:
        """Check for sector-level alerts"""
        alerts = []
        
        for _, row in sector_data.iterrows():
            sector = row['sector']
            
            # Check for sector rotation
            if sector in self.previous_sector_trends:
                prev = self.previous_sector_trends[sector]
                momentum_change = row['avg_momentum'] - prev['avg_momentum']
                
                if abs(momentum_change) >= self.thresholds['sector_rotation']:
                    severity = AlertSeverity.INFO if abs(momentum_change) < 5 else AlertSeverity.WARNING
                    alerts.append(Alert(
                        alert_type=AlertType.SECTOR_ROTATION,
                        severity=severity,
                        ticker=None,
                        sector=sector,
                        message=f"Sector rotation detected in {sector}: {momentum_change:+.2f}% momentum shift",
                        value=momentum_change,
                        threshold=self.thresholds['sector_rotation']
                    ))
        
        # Update previous sector trends
        self.previous_sector_trends = sector_data.set_index('sector').to_dict('index')
        
        return alerts
    
    def check_crypto_alerts(self, crypto_data: pd.DataFrame) -> List[Alert]:
        """Check for cryptocurrency alerts"""
        alerts = []
        
        for _, row in crypto_data.iterrows():
            symbol = row['symbol']
            
            if abs(row['momentum']) >= self.thresholds['crypto_surge']:
                severity = AlertSeverity.WARNING if abs(row['momentum']) >= 15 else AlertSeverity.INFO
                direction = "surge" if row['momentum'] > 0 else "crash"
                
                alerts.append(Alert(
                    alert_type=AlertType.CRYPTO_MOVEMENT,
                    severity=severity,
                    ticker=symbol,
                    sector='Crypto',
                    message=f"Crypto {direction}: {symbol} moved {row['momentum']:+.2f}%",
                    value=row['momentum'],
                    threshold=self.thresholds['crypto_surge']
                ))
        
        return alerts
    
    def detect_trend_reversals(self) -> List[Alert]:
        """Detect potential trend reversals using historical data"""
        alerts = []
        
        # Get recent historical data
        history = self.db.get_trends_history(hours=24)
        
        if not history.empty:
            for ticker in history['ticker'].unique():
                ticker_data = history[history['ticker'] == ticker].sort_values('timestamp')
                
                if len(ticker_data) >= 10:  # Need enough data points
                    # Simple trend reversal detection using momentum
                    recent_momentum = ticker_data['momentum'].tail(5).mean()
                    older_momentum = ticker_data['momentum'].head(5).mean()
                    
                    if older_momentum > 2 and recent_momentum < -2:
                        alerts.append(Alert(
                            alert_type=AlertType.TREND_REVERSAL,
                            severity=AlertSeverity.CRITICAL,
                            ticker=ticker,
                            sector=ticker_data.iloc[-1]['sector'],
                            message=f"Potential trend reversal: {ticker} shifting from bullish to bearish",
                            value=recent_momentum,
                            threshold=0
                        ))
                    elif older_momentum < -2 and recent_momentum > 2:
                        alerts.append(Alert(
                            alert_type=AlertType.TREND_REVERSAL,
                            severity=AlertSeverity.WARNING,
                            ticker=ticker,
                            sector=ticker_data.iloc[-1]['sector'],
                            message=f"Potential trend reversal: {ticker} shifting from bearish to bullish",
                            value=recent_momentum,
                            threshold=0
                        ))
        
        return alerts
    
    def process_all_alerts(self, ticker_data: pd.DataFrame, 
                          sector_data: pd.DataFrame, 
                          crypto_data: pd.DataFrame) -> List[Alert]:
        """Process all alert types and save to database"""
        all_alerts = []
        
        # Check different alert types
        all_alerts.extend(self.check_ticker_alerts(ticker_data))
        all_alerts.extend(self.check_sector_alerts(sector_data))
        all_alerts.extend(self.check_crypto_alerts(crypto_data))
        all_alerts.extend(self.detect_trend_reversals())
        
        # Save alerts to database
        for alert in all_alerts:
            self.db.add_alert(
                alert_type=alert.alert_type.value,
                ticker=alert.ticker,
                sector=alert.sector,
                message=alert.message,
                severity=alert.severity.value
            )
        
        self.alerts = all_alerts
        return all_alerts
    
    def get_critical_alerts(self) -> List[Alert]:
        """Get only critical alerts"""
        return [a for a in self.alerts if a.severity == AlertSeverity.CRITICAL]
    
    def get_alerts_summary(self) -> Dict:
        """Get summary of current alerts"""
        summary = {
            'total_alerts': len(self.alerts),
            'critical': len([a for a in self.alerts if a.severity == AlertSeverity.CRITICAL]),
            'warning': len([a for a in self.alerts if a.severity == AlertSeverity.WARNING]),
            'info': len([a for a in self.alerts if a.severity == AlertSeverity.INFO]),
            'by_type': {}
        }
        
        for alert_type in AlertType:
            count = len([a for a in self.alerts if a.alert_type == alert_type])
            if count > 0:
                summary['by_type'][alert_type.value] = count
        
        return summary
    
    def format_alerts_for_display(self) -> pd.DataFrame:
        """Format alerts for dashboard display"""
        if not self.alerts:
            return pd.DataFrame()
        
        alert_data = []
        for alert in self.alerts:
            alert_data.append({
                'Time': alert.timestamp.strftime('%H:%M:%S'),
                'Type': alert.alert_type.value.replace('_', ' ').title(),
                'Severity': alert.severity.value,
                'Ticker': alert.ticker or '-',
                'Sector': alert.sector or '-',
                'Message': alert.message,
                'Value': f"{alert.value:+.2f}%"
            })
        
        return pd.DataFrame(alert_data)