# AQI Forecast + SHAP Explorer (Beijing)

This project predicts PM2.5 (air quality) in Beijing using machine learning and explains the predictions using SHAP.

## Project Structure

- `data/raw/` - raw CSV data (`PRSA_data.csv`)
- `data/processed/` - processed parquet files for features and models
- `models/` - trained models (`rf_model.pkl`, `xgb_model.pkl`) and metrics
- `src/` - Python scripts
  - `load_data.py` - load raw CSV and save as parquet
  - `preprocess.py` - create features for model and SHAP
  - `train.py` - train models (Random Forest and XGBoost)
  - `explain.py` - generate SHAP visualizations
- `explain/` - output folder for SHAP plots

## How to Run

1. **Load raw data**
```bash
python src/load_data.py --input data/raw/PRSA_data.csv --out data/processed/pm25_raw.parquet
