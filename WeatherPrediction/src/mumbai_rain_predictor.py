import os
import joblib
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from flask import Flask, jsonify, Response
import json
import time

# Setup logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s'
    )

# Configuration
LAT, LON = 19.0760, 72.8777  # Mumbai
epochs_back = 3  # include past 3 hours
MODEL_VERSION = 'v2'
MODEL_PATH = f'models/rain_model_{MODEL_VERSION}.joblib'

# 1. Fetch historical data
def fetch_historical_data(start_date: str, end_date: str) -> pd.DataFrame:
    url = 'https://archive-api.open-meteo.com/v1/era5'
    params = {
        'latitude': LAT,
        'longitude': LON,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': 'temperature_2m,relativehumidity_2m,pressure_msl,windspeed_10m,precipitation'
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json().get('hourly', {})
    df = pd.DataFrame({
        'dt': pd.to_datetime(data['time']),
        'temp': data['temperature_2m'],
        'humidity': data['relativehumidity_2m'],
        'pressure': data['pressure_msl'],
        'wind_speed': data['windspeed_10m'],
        'rain_1h': data['precipitation']
    })
    if df.empty:
        raise ValueError("Empty historical data")
    return df

# Synthetic fallback
def generate_synthetic(start_date: str, end_date: str) -> pd.DataFrame:
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    dates = pd.date_range(start, end, freq='H')
    n = len(dates)
    return pd.DataFrame({
        'dt': dates,
        'temp': np.random.normal(30, 5, n),
        'humidity': np.random.uniform(60, 100, n),
        'pressure': np.random.normal(1010, 5, n),
        'wind_speed': np.random.exponential(2, n),
        'rain_1h': np.random.binomial(1, 0.2, n) * np.random.exponential(2, n)
    })

# Label creation
def label_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values('dt').reset_index(drop=True)
    df['rain_next'] = (df['rain_1h'].shift(-1) > 0).astype(int)
    return df.dropna(subset=['rain_next'])

# Feature engineering with lag & rolling stats
def create_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    for lag in range(1, epochs_back+1):
        df[f'rain_lag_{lag}'] = df['rain_1h'].shift(lag)
        df[f'temp_lag_{lag}'] = df['temp'].shift(lag)
        df[f'humidity_lag_{lag}'] = df['humidity'].shift(lag)
    df['precip_3h_sum'] = df['rain_1h'].rolling(epochs_back).sum()
    df['temp_3h_avg'] = df['temp'].rolling(epochs_back).mean()
    df = df.dropna().reset_index(drop=True)
    feature_cols = [col for col in df.columns if 'lag' in col or 'avg' in col or 'sum' in col] + ['hour', 'dayofweek']
    df['hour'] = df['dt'].dt.hour
    df['dayofweek'] = df['dt'].dt.dayofweek
    X = df[feature_cols]
    y = df['rain_next']
    return X, y

# Train & save model with hyperparameter tuning
def train_and_save(start_date: str, end_date: str):
    try:
        df = fetch_historical_data(start_date, end_date)
    except Exception:
        logging.info("Using synthetic data fallback.")
        df = generate_synthetic(start_date, end_date)
    df = label_data(df)
    X, y = create_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    param_grid = {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='roc_auc')
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    cv_scores = cross_val_score(best, X_train, y_train, cv=3, scoring='roc_auc')
    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, best.predict_proba(X_test)[:,1])
    logging.info(f"Best params: {grid.best_params_}")
    logging.info(f"Train CV AUC: {cv_scores.mean():.3f}")
    logging.info(f"Test Accuracy: {acc:.3f}, Test AUC: {auc:.3f}")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    logging.info(f"Model v2 saved at {MODEL_PATH}")

# Prediction for next hour using rolling features
def fetch_forecast_features() -> dict:
    url = 'https://api.open-meteo.com/v1/forecast'
    params = {'latitude': LAT, 'longitude': LON, 'hourly': 'temperature_2m,relativehumidity_2m,precipitation'}
    data = requests.get(url, params=params).json()['hourly']
    times = pd.to_datetime(data['time'])
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    idx = list(times).index(now)
    # assemble past epochs_back hours
    feats = {}
    for lag in range(epochs_back):
        feats[f'temp_lag_{lag+1}'] = data['temperature_2m'][idx-lag]
        feats[f'humidity_lag_{lag+1}'] = data['relativehumidity_2m'][idx-lag]
        feats[f'rain_lag_{lag+1}'] = data['precipitation'][idx-lag]
    feats['precip_3h_sum'] = sum(data['precipitation'][idx-epochs_back+1:idx+1])
    feats['temp_3h_avg'] = np.mean(data['temperature_2m'][idx-epochs_back+1:idx+1])
    next_hour = now + timedelta(hours=1)
    feats['hour'] = next_hour.hour
    feats['dayofweek'] = next_hour.weekday()
    feats['forecast_time'] = next_hour.isoformat() + 'Z'
    return feats

# Initialize Flask app & model
app = Flask(__name__)
model = None
load_model = lambda: None
if not os.path.exists(MODEL_PATH):
    train_and_save((datetime.utcnow()-timedelta(days=30)).strftime('%Y-%m-%d'), datetime.utcnow().strftime('%Y-%m-%d'))
model = joblib.load(MODEL_PATH)

# Prediction logic
def get_prediction() -> dict:
    feats = fetch_forecast_features()
    X = pd.DataFrame([feats])[model.feature_names_in_]

    prob = model.predict_proba(X)[:,1][0]
    pred = model.predict(X)[0]
    return {'location':'Mumbai','forecast_for':feats['forecast_time'],'will_rain_next_hour':bool(pred),'probability':round(float(prob),3),'features':feats}

@app.route('/predict', methods=['GET'])
def predict_route():
    return jsonify(get_prediction())

@app.route('/stream')
def stream():
    def event_stream():
        while True:
            data = get_prediction()
            data['server_time'] = datetime.utcnow().isoformat()+'Z'
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(60)
    return Response(event_stream(), mimetype='text/event-stream')

if __name__ == '__main__':
    import argparse
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port)