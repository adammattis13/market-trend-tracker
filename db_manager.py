#!/usr/bin/env python3
"""
Database Manager for Market Trend Tracker
=========================================

Handles all database operations using SQLite for persistent storage
of market trends, sector analysis, and alerts.

Author: Adam Mattis (Enhanced Version)
Date: 2024
"""

import sqlite3
import pandas as pd
from datetime import datetime
from contextlib import contextmanager
import logging
import os
from typing import Optional, Dict, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages SQLite database operations for Market Trend Tracker.
    
    This class provides:
    - Database initialization and schema management
    - Data persistence for trends, sectors, and alerts
    - Query methods for data retrieval
    - Database maintenance utilities
    """
    
    def __init__(self, db_path: str = 'market_trends.db'):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.init_database()
        logger.info(f"Database initialized at {os.path.abspath(db_path)}")
        
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        
        Yields:
            sqlite3.Connection: Database connection
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
            
    def init_database(self):
        """Initialize database tables and indices."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Trends table for current snapshot
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    sector TEXT,
                    trend_score REAL,
                    momentum REAL,
                    volume_ratio REAL,
                    price REAL,
                    price_change REAL,
                    volume INTEGER,
                    high REAL,
                    low REAL,
                    open REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, timestamp)
                )
            ''')
            
            # Sector trends table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sector_trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sector TEXT NOT NULL,
                    avg_momentum REAL,
                    momentum_std REAL,
                    sector_score REAL,
                    avg_volume_ratio REAL,
                    stock_count INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(sector, timestamp)
                )
            ''')
            
            # Crypto trends table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS crypto_trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    name TEXT,
                    price REAL,
                    momentum REAL,
                    volume_24h REAL,
                    market_cap REAL,
                    price_change_24h REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                )
            ''')
            
            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type TEXT NOT NULL,
                    ticker TEXT,
                    sector TEXT,
                    message TEXT,
                    severity TEXT,
                    value REAL,
                    threshold REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    acknowledged BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    duration_seconds REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indices for better performance
            indices = [
                'CREATE INDEX IF NOT EXISTS idx_trends_ticker ON trends(ticker)',
                'CREATE INDEX IF NOT EXISTS idx_trends_timestamp ON trends(timestamp)',
                'CREATE INDEX IF NOT EXISTS idx_trends_ticker_timestamp ON trends(ticker, timestamp)',
                'CREATE INDEX IF NOT EXISTS idx_sector_trends_timestamp ON sector_trends(timestamp)',
                'CREATE INDEX IF NOT EXISTS idx_sector_trends_sector ON sector_trends(sector)',
                'CREATE INDEX IF NOT EXISTS idx_crypto_trends_symbol ON crypto_trends(symbol)',
                'CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)',
                'CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)',
                'CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON alerts(acknowledged)'
            ]
            
            for index in indices:
                cursor.execute(index)
            
            conn.commit()
            logger.info("Database schema initialized successfully")
    
    def save_trends(self, df: pd.DataFrame) -> bool:
        """
        Save trend data to database.
        
        Args:
            df: DataFrame with trend data
            
        Returns:
            bool: Success status
        """
        try:
            with self.get_connection() as conn:
                # Ensure timestamp column exists
                if 'timestamp' not in df.columns:
                    df['timestamp'] = datetime.now()
                
                # Save to database
                df.to_sql('trends', conn, if_exists='append', index=False)
                
                logger.info(f"Saved {len(df)} trend records to database")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save trends: {str(e)}")
            return False
    
    def save_sector_trends(self, df: pd.DataFrame) -> bool:
        """
        Save sector trend data to database.
        
        Args:
            df: DataFrame with sector trend data
            
        Returns:
            bool: Success status
        """
        try:
            with self.get_connection() as conn:
                if 'timestamp' not in df.columns:
                    df['timestamp'] = datetime.now()
                
                df.to_sql('sector_trends', conn, if_exists='append', index=False)
                
                logger.info(f"Saved {len(df)} sector trend records")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save sector trends: {str(e)}")
            return False
    
    def save_crypto_trends(self, df: pd.DataFrame) -> bool:
        """
        Save cryptocurrency trend data to database.
        
        Args:
            df: DataFrame with crypto trend data
            
        Returns:
            bool: Success status
        """
        try:
            with self.get_connection() as conn:
                if 'timestamp' not in df.columns:
                    df['timestamp'] = datetime.now()
                
                df.to_sql('crypto_trends', conn, if_exists='append', index=False)
                
                logger.info(f"Saved {len(df)} crypto trend records")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save crypto trends: {str(e)}")
            return False
    
    def get_latest_trends(self, limit: int = 30) -> pd.DataFrame:
        """
        Get the latest trend snapshot.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with latest trends
        """
        query = '''
            SELECT * FROM trends 
            WHERE timestamp = (SELECT MAX(timestamp) FROM trends)
            ORDER BY trend_score DESC
            LIMIT ?
        '''
        
        try:
            with self.get_connection() as conn:
                return pd.read_sql_query(query, conn, params=(limit,))
        except Exception as e:
            logger.error(f"Failed to get latest trends: {str(e)}")
            return pd.DataFrame()
    
    def get_sector_trends_history(self, hours: int = 24) -> pd.DataFrame:
        """
        Get sector trends for the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            DataFrame with sector trend history
        """
        query = '''
            SELECT * FROM sector_trends 
            WHERE timestamp > datetime('now', '-{} hours')
            ORDER BY timestamp DESC, sector
        '''.format(hours)
        
        try:
            with self.get_connection() as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Failed to get sector history: {str(e)}")
            return pd.DataFrame()
    
    def get_ticker_history(self, ticker: str, days: int = 7) -> pd.DataFrame:
        """
        Get historical data for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back
            
        Returns:
            DataFrame with ticker history
        """
        query = '''
            SELECT * FROM trends 
            WHERE ticker = ? 
            AND timestamp > datetime('now', '-{} days')
            ORDER BY timestamp
        '''.format(days)
        
        try:
            with self.get_connection() as conn:
                return pd.read_sql_query(query, conn, params=(ticker,))
        except Exception as e:
            logger.error(f"Failed to get ticker history: {str(e)}")
            return pd.DataFrame()
    
    def get_trends_history(self, hours: int = 24) -> pd.DataFrame:
        """
        Get all trends for the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            DataFrame with trend history
        """
        query = '''
            SELECT * FROM trends 
            WHERE timestamp > datetime('now', '-{} hours')
            ORDER BY timestamp DESC
        '''.format(hours)
        
        try:
            with self.get_connection() as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Failed to get trends history: {str(e)}")
            return pd.DataFrame()
    
    def get_top_movers(self, direction: str = 'up', limit: int = 5) -> pd.DataFrame:
        """
        Get top gainers or losers.
        
        Args:
            direction: 'up' for gainers, 'down' for losers
            limit: Number of results to return
            
        Returns:
            DataFrame with top movers
        """
        order = 'DESC' if direction == 'up' else 'ASC'
        query = f'''
            SELECT ticker, sector, momentum, price_change, price, volume_ratio
            FROM trends 
            WHERE timestamp = (SELECT MAX(timestamp) FROM trends)
            ORDER BY momentum {order}
            LIMIT ?
        '''
        
        try:
            with self.get_connection() as conn:
                return pd.read_sql_query(query, conn, params=(limit,))
        except Exception as e:
            logger.error(f"Failed to get top movers: {str(e)}")
            return pd.DataFrame()
    
    def add_alert(self, alert_type: str, ticker: Optional[str] = None, 
                  sector: Optional[str] = None, message: str = '', 
                  severity: str = 'INFO', value: float = 0.0, 
                  threshold: float = 0.0) -> bool:
        """
        Add an alert to the database.
        
        Args:
            alert_type: Type of alert
            ticker: Stock ticker (optional)
            sector: Market sector (optional)
            message: Alert message
            severity: Alert severity level
            value: Metric value that triggered alert
            threshold: Threshold that was exceeded
            
        Returns:
            bool: Success status
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO alerts (alert_type, ticker, sector, message, 
                                      severity, value, threshold)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (alert_type, ticker, sector, message, severity, value, threshold))
                conn.commit()
                
                logger.info(f"Added {severity} alert: {message}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add alert: {str(e)}")
            return False
    
    def get_unacknowledged_alerts(self, limit: int = 50) -> pd.DataFrame:
        """
        Get all unacknowledged alerts.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            DataFrame with unacknowledged alerts
        """
        query = '''
            SELECT * FROM alerts 
            WHERE acknowledged = FALSE 
            ORDER BY timestamp DESC
            LIMIT ?
        '''
        
        try:
            with self.get_connection() as conn:
                return pd.read_sql_query(query, conn, params=(limit,))
        except Exception as e:
            logger.error(f"Failed to get alerts: {str(e)}")
            return pd.DataFrame()
    
    def acknowledge_alert(self, alert_id: int) -> bool:
        """
        Mark an alert as acknowledged.
        
        Args:
            alert_id: ID of the alert to acknowledge
            
        Returns:
            bool: Success status
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'UPDATE alerts SET acknowledged = TRUE WHERE id = ?',
                    (alert_id,)
                )
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {str(e)}")
            return False
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """
        Remove data older than specified days.
        
        Args:
            days_to_keep: Number of days of data to retain
            
        Returns:
            Dict with deletion counts by table
        """
        deletion_counts = {}
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                tables = ['trends', 'sector_trends', 'crypto_trends', 'alerts']
                
                for table in tables:
                    cursor.execute(f'''
                        DELETE FROM {table} 
                        WHERE timestamp < datetime('now', '-{days_to_keep} days')
                    ''')
                    deletion_counts[table] = cursor.rowcount
                
                conn.commit()
                
                total_deleted = sum(deletion_counts.values())
                logger.info(f"Cleaned up {total_deleted} old records")
                
                return deletion_counts
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {str(e)}")
            return {}
    
    def get_database_stats(self) -> Dict[str, Union[int, str]]:
        """
        Get database statistics.
        
        Returns:
            Dict with database statistics
        """
        stats = {}
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get record counts
                tables = ['trends', 'sector_trends', 'crypto_trends', 'alerts']
                for table in tables:
                    cursor.execute(f'SELECT COUNT(*) FROM {table}')
                    stats[f'{table}_count'] = cursor.fetchone()[0]
                
                # Get date range for trends
                cursor.execute('''
                    SELECT MIN(timestamp), MAX(timestamp) 
                    FROM trends
                ''')
                result = cursor.fetchone()
                if result and result[0]:
                    stats['date_range'] = f"{result[0]} to {result[1]}"
                else:
                    stats['date_range'] = "No data"
                
                # Get database file size
                stats['db_size_mb'] = round(os.path.getsize(self.db_path) / 1024 / 1024, 2)
                
                # Get alert summary
                cursor.execute('''
                    SELECT severity, COUNT(*) 
                    FROM alerts 
                    WHERE acknowledged = FALSE 
                    GROUP BY severity
                ''')
                alert_counts = cursor.fetchall()
                stats['unacknowledged_alerts'] = dict(alert_counts) if alert_counts else {}
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {str(e)}")
            return {'error': str(e)}
    
    def save_performance_metric(self, metric_name: str, metric_value: float, 
                               duration_seconds: float) -> bool:
        """
        Save a performance metric.
        
        Args:
            metric_name: Name of the metric
            metric_value: Value of the metric
            duration_seconds: Time taken
            
        Returns:
            bool: Success status
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_metrics (metric_name, metric_value, duration_seconds)
                    VALUES (?, ?, ?)
                ''', (metric_name, metric_value, duration_seconds))
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to save performance metric: {str(e)}")
            return False
    
    def vacuum_database(self):
        """Optimize database by running VACUUM command."""
        try:
            with self.get_connection() as conn:
                conn.execute('VACUUM')
                logger.info("Database optimized successfully")
        except Exception as e:
            logger.error(f"Failed to vacuum database: {str(e)}")


# Utility functions for testing
def test_database():
    """Test database functionality."""
    db = DatabaseManager('test_market_trends.db')
    
    # Test saving trends
    test_data = pd.DataFrame([
        {'ticker': 'AAPL', 'sector': 'Technology', 'trend_score': 85.5, 
         'momentum': 12.3, 'price': 150.25, 'volume_ratio': 1.2},
        {'ticker': 'MSFT', 'sector': 'Technology', 'trend_score': 82.1, 
         'momentum': 10.5, 'price': 300.50, 'volume_ratio': 1.1}
    ])
    
    print("Testing save_trends:", db.save_trends(test_data))
    print("\nLatest trends:")
    print(db.get_latest_trends())
    print("\nDatabase stats:")
    print(db.get_database_stats())
    
    # Clean up test database
    os.remove('test_market_trends.db')


if __name__ == "__main__":
    # Run test if executed directly
    test_database()