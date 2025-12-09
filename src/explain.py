"""Run SHAP explanations and save summary, dependence, and local plots with readable labels."""

import argparse
import joblib
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os

def friendly_name(col):
    """Convert raw column names to more readable labels."""
    mapping = {
        'TEMP': 'Temperature (°C)',
        'temp': 'Temperature (°C)',
        'DEWP': 'Dew Point (°C)',
        'dewpoint': 'Dew Point (°C)',
        'PRES': 'Pressure (hPa)',
        'pres': 'Pressure (hPa)',
        'Iws': 'Wind Speed (m/s)',
        'wind_speed': 'Wind Speed (m/s)',
        'cbwd': 'Wind Direction',
        'cbwd_code': 'Wind Direction Code',
        'hour': 'Hour of Day',
        'dayofweek': 'Day of Week',
        'month': 'Month',
        'is_weekend': 'Is Weekend',
    }
    return mapping.get(col, col)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load model and data
    model = joblib.load(args.model)
    df = pd.read_parquet(args.input)
    X = df.drop(columns=['pm25'])
    y = df['pm25']

    # SHAP explainer
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Friendly column names
    X_friendly = X.rename(columns=friendly_name)

    # Summary plot
    plt.figure(figsize=(10,7))
    shap.summary_plot(shap_values, X_friendly, show=False)
    plt.title("Impact of Features on PM2.5 Concentration", fontsize=16)
    plt.xlabel("SHAP Value (Impact on PM2.5)", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "shap_summary.png"))
    plt.close()

    # Dependence plot for top feature
    top_feat_idx = np.argsort(np.abs(shap_values.values).mean(axis=0))[-1]
    top_feat = X.columns[top_feat_idx]
    plt.figure(figsize=(10,7))
    shap.dependence_plot(top_feat, shap_values.values, X_friendly, show=False)
    plt.title(f"Effect of {friendly_name(top_feat)} on PM2.5 Concentration", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, f"shap_dependence_{top_feat}.png"))
    plt.close()

    # Local explanation for last row
    idx = -1
    plt.figure(figsize=(8,4))
    shap.plots.waterfall(shap_values[idx], max_display=12, show=False)
    plt.title("Feature Contributions for a Single Observation", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "shap_local_sample.png"))
    plt.close()

    print("Readable SHAP plots saved to", args.out)
