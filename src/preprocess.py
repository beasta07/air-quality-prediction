"""Create features suitable for tree models and SHAP."""

import argparse
import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame, lags=(1,24,48), rolling_windows=(3,24)) -> pd.DataFrame:
    df = df.copy()

    # ensure date index
    if 'date' not in df.columns:
        # fallback: try common datetime column names
        for col in ['datetime','date','time']:
            if col in df.columns:
                df['date'] = pd.to_datetime(df[col])
                break
        else:
            raise ValueError("No datetime column found")
    else:
        df['date'] = pd.to_datetime(df['date'])

    df = df.set_index('date')

    # Target: assume column named 'pm2.5' or 'pm25'
    if 'pm2.5' in df.columns:
        df['pm25'] = df['pm2.5']
    elif 'pm25' in df.columns:
        df['pm25'] = df['pm25']
    else:
        raise ValueError("No pm2.5 or pm25 column found in data")

    # Keep numeric columns if present
    keep_cols = []
    for c in ['TEMP','temp','DEWP','dewpoint','PRES','pres','Iws','wind_speed','cbwd','wd']:
        if c in df.columns:
            keep_cols.append(c)

    # Encode categorical columns
    for c in keep_cols:
        if df[c].dtype.name == 'object' or str(df[c].dtype).startswith('category'):
            df[c] = df[c].astype('category').cat.codes

    # Time features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    time_cols = ['hour','dayofweek','month','is_weekend']

    # Lag features on pm25
    for lag in lags:
        df[f'pm25_lag_{lag}'] = df['pm25'].shift(lag)

    # Rolling means
    for w in rolling_windows:
        df[f'pm25_roll_mean_{w}'] = df['pm25'].shift(1).rolling(w, min_periods=1).mean()

    # Interaction feature example
    if 'TEMP' in df.columns or 'temp' in df.columns:
        temp_col = 'TEMP' if 'TEMP' in df.columns else 'temp'
        df['temp_pm25_inter'] = df[temp_col] * df['pm25'].shift(1)

    # Assemble feature columns
    feature_cols = time_cols + keep_cols + \
                   [c for c in df.columns if c.startswith('pm25_lag_') or
                    c.startswith('pm25_roll_mean_') or c.endswith('_inter')]

    # Drop rows with NaNs introduced by shifts
    df = df.dropna(subset=feature_cols + ['pm25'])
    
    return df[feature_cols + ['pm25']]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    features = create_features(df)
    features.to_parquet(args.out)
    print(f"Saved features with shape {features.shape} to {args.out}")
