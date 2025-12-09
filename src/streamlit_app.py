"""Streamlit app to explore time series, model predictions, and SHAP images."""

import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(layout='wide', page_title='AQI Explainable ML')

@st.cache_data
def load_features(path):
    return pd.read_parquet(path)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

st.title('AQI Forecast + SHAP Explorer (Beijing)')

col1, col2 = st.columns([2,1])

with col1:
    st.header('Time Series')
    feats_path = st.text_input('Features parquet path', 'data/processed/pm25_features.parquet')
    if os.path.exists(feats_path):
        df = load_features(feats_path)
        st.line_chart(df['pm25'].rename('pm25'))
    else:
        st.info('Place features parquet at data/processed/pm25_features.parquet or change the path')

with col2:
    st.header('Model')
    model_path = st.text_input('Model path', 'models/xgb_model.pkl')
    if os.path.exists(model_path):
        st.write('Model found:', model_path)
    else:
        st.warning('Model not found; train and save models to models/')

st.markdown('---')

st.header('SHAP Visualizations')
shap_dir = st.text_input('SHAP images folder', 'explain')
if os.path.exists(shap_dir):
    files = [f for f in os.listdir(shap_dir) if f.endswith('.png')]
    if files:
        for f in files:
            st.image(os.path.join(shap_dir, f), caption=f)
    else:
        st.info('No PNGs found in explain/')
else:
    st.info('Run `python src/explain.py` to generate SHAP plots')
