#!/usr/bin/env python3
"""
Fix Database Schema - Add missing columns
========================================

Run this once to update your database schema.
"""

import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_database_schema(db_path='market_trends.db'):
    """Add missing columns to existing database."""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if rank column exists
        cursor.execute("PRAGMA table_info(trends)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'rank' not in columns:
            logger.info("Adding 'rank' column to trends table...")
            cursor.execute("ALTER TABLE trends ADD COLUMN rank INTEGER")
            conn.commit()
            logger.info("Successfully added 'rank' column")
        else:
            logger.info("'rank' column already exists")
            
        # Verify the fix
        cursor.execute("PRAGMA table_info(trends)")
        columns = [column[1] for column in cursor.fetchall()]
        logger.info(f"Current columns in trends table: {columns}")
        
    except Exception as e:
        logger.error(f"Error fixing schema: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    fix_database_schema()
    print("\nDatabase schema fixed! You can now run the analyzer again.")