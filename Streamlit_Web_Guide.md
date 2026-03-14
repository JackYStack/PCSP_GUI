# Streamlit Web Guide

## 1) Install
In project root:

- `pip install -r requirements.txt`

## 2) Run Web App

- `streamlit run scripts/streamlit_web_app.py`

## 3) Features

- Manual input prediction
- CSV batch prediction
- Multi-model comparison
- Custom model upload (`.pkl`)
- Output table with model average
- Export prediction CSV

## 4) Notes

- Built-in model list excludes `composite_model.pkl`.
- CSV must contain required feature columns shown in the app.
- If a model lacks feature metadata, app falls back to config features.
