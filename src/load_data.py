"""Load raw CSV and save a cleaned parquet for the pipeline."""

import argparse
import pandas as pd

def load_raw(path: str) -> pd.DataFrame:
    """
    Load the UCI Beijing PM2.5 CSV and return a DataFrame with a datetime index.
    """
    # This loader assumes the UCI Beijing PM2.5 format with columns including 'year','month','day','hour'
    df = pd.read_csv(path)

    # If dataset has separate date columns, combine them
    if {"year", "month", "day", "hour"}.issubset(df.columns):
        df['date'] = pd.to_datetime(df[['year','month','day','hour']])
    else:
        # fallback: try common datetime column names
        for col in ['datetime', 'date', 'time']:
            if col in df.columns:
                df['date'] = pd.to_datetime(df[col])
                break

    # Sort by datetime and reset index
    df = df.sort_values('date').reset_index(drop=True)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    df = load_raw(args.input)
    df.to_parquet(args.out)
    print(f"Saved {len(df)} rows to {args.out}")
