# sector_analyzer.py

import pandas as pd

def analyze_sectors(file_path='trend_df.csv'):
    """
    Reads the trend_df CSV and calculates the average momentum score for each sector.
    Returns a DataFrame with sectors and their average momentum scores.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file at path: {file_path}")

    # Normalize column names for consistency
    df.columns = df.columns.str.lower()

    if 'sector' not in df.columns or 'momentum_score' not in df.columns:
        raise ValueError("CSV must contain 'sector' and 'momentum_score' columns")

    sector_summary = (
        df.groupby('sector')['momentum_score']
        .mean()
        .reset_index()
        .sort_values(by='momentum_score', ascending=False)
    )

    return sector_summary

