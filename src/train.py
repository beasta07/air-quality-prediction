"""Train RandomForest and XGBoost regressors and save models + metrics."""

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import os

def time_train_test_split(df: pd.DataFrame, test_frac=0.2):
    n = len(df)
    split = int(n * (1 - test_frac))
    train = df.iloc[:split]
    test = df.iloc[split:]
    return train, test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    X = df.drop(columns=['pm25'])
    y = df['pm25']

    train_df, test_df = time_train_test_split(df)
    X_train, y_train = train_df.drop(columns=['pm25']), train_df['pm25']
    X_test, y_test = test_df.drop(columns=['pm25']), test_df['pm25']

    os.makedirs(args.out, exist_ok=True)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X_test)

    metrics = {
        'rf_rmse': float(np.sqrt(mean_squared_error(y_test, preds_rf))),
        'rf_mae': float(mean_absolute_error(y_test, preds_rf)),
        'rf_r2': float(r2_score(y_test, preds_rf))
    }

    joblib.dump(rf, os.path.join(args.out, 'rf_model.pkl'))

    # XGBoost
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, tree_method='hist', verbosity=0)
        xgb.fit(X_train, y_train)
        preds_xgb = xgb.predict(X_test)
        metrics.update({
            'xgb_rmse': float(np.sqrt(mean_squared_error(y_test, preds_xgb))),
            'xgb_mae': float(mean_absolute_error(y_test, preds_xgb)),
            'xgb_r2': float(r2_score(y_test, preds_xgb))
        })
        joblib.dump(xgb, os.path.join(args.out, 'xgb_model.pkl'))
    except Exception as e:
        print("XGBoost training skipped (xgboost not installed or failed):", e)

    with open(os.path.join(args.out, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("Training finished. Metrics:", metrics)
