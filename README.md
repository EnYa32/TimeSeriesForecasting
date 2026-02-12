---
title: TimeSeriesForecasting
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: LightGBM-based time series forecasting app that predicts...
license: mit
---

# 📈 Sales Forecast (LightGBM)

This Streamlit app predicts **`num_sold`** using a trained **LightGBM** model with time-based features and lag features.

## What it does
- Takes calendar features (year/month/week/dayofweek/dayofyear, weekend)
- Uses lag features (lag_364, lag_365, lag_371)
- Uses categorical inputs (country/store/product) via saved encoders
- Outputs a `num_sold` prediction

## Files required (put in the repo root)
- `app.py`
- `lgbm_model.pkl`
- `feature_names.pkl` (list of feature names in correct order)
- `encoders.pkl` (dict of LabelEncoders for `country`, `store`, `product`)
- `fill_map.pkl` (optional: medians for numeric feature filling)

## How to save artifacts in your notebook (training side)

```python
import joblib

joblib.dump(model_lgb, 'lgbm_model.pkl')
joblib.dump(FEATURES, 'feature_names.pkl')
joblib.dump(encoders, 'encoders.pkl')

# optional numeric medians for filling missing
num_cols = [c for c in FEATURES if c not in ['country', 'store', 'product']]
fill_map = train_fe[num_cols].median().to_dict()
joblib.dump(fill_map, 'fill_map.pkl')