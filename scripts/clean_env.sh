#!/bin/bash

echo "🧼 Cleaning Python cache files..."
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.pyc" -delete

echo "🧽 Removing system junk files..."
find . -name ".DS_Store" -delete

echo "🗑️ Clearing old trend logs..."
if [ -f trend_log.csv ]; then
    mv trend_log.csv "trend_log_backup_$(date +%F).csv"
    echo "🔁 Backed up trend_log.csv"
fi

echo "✅ Environment cleanup complete!"
