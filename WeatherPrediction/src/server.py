
from flask import Flask, jsonify, request
import logging
from datetime import datetime
from flask_cors import CORS
import numpy as np

from ingest import fetch_historical, fetch_forecast
from interp import scale_features

# Import both models
from ml import AdvancedNN
from ml_lstm import SimpleLSTM

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WeatherAPI")

DEFAULT_LAT, DEFAULT_LON = 19.45, 73.05
MODEL_LOCATION = (DEFAULT_LAT, DEFAULT_LON)
MODEL_LOCATION_NAME = "Vashere"


# Pre-train on last 30 days (for both models)
try:
    hist_df = fetch_historical(DEFAULT_LAT, DEFAULT_LON, days=30)
    X_hist, scaler = scale_features(hist_df)
    y_hist = hist_df['temperature'].values

    # AdvancedNN (existing)
    model_nn = AdvancedNN(in_dim=X_hist.shape[1], h1=64, h2=32, h3=16)
    logger.info("Training AdvancedNN on last 30 days…")
    model_nn.train(X_hist, y_hist, epochs=200, lr=1e-3)
    logger.info("AdvancedNN training complete.")

    # SimpleLSTM (new, expects sequence data)
    # For demo, use last 7 days as sequence for each sample (rolling window)
    seq_len = 7
    if len(X_hist) > seq_len:
        X_seq = np.array([X_hist[i-seq_len:i] for i in range(seq_len, len(X_hist))])
        y_seq = y_hist[seq_len:]
        model_lstm = SimpleLSTM(input_dim=X_hist.shape[1], hidden_dim=32, output_dim=1)
        logger.info("Training SimpleLSTM (demo, placeholder train)…")
        model_lstm.train(X_seq, y_seq, epochs=10)
    else:
        model_lstm = None
        logger.warning("Not enough data for LSTM training.")

    model_ready = True
except Exception as e:
    logger.error("Could not train models", exc_info=e)
    model_nn = None
    model_lstm = None
    scaler = None
    model_ready = False



@app.route('/forecast')
def forecast():
    dt_str = request.args.get("datetime")
    lat = request.args.get("lat", type=float, default=DEFAULT_LAT)
    lon = request.args.get("lon", type=float, default=DEFAULT_LON)

    if not dt_str:
        return jsonify(error="Missing datetime; use YYYY-MM-DDTHH:MM"), 400

    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M")
    except ValueError:
        return jsonify(error="Invalid format; use YYYY-MM-DDTHH:MM"), 400


    # Fetch observed or forecast data
    try:
        row = fetch_forecast(lat, lon, dt)
    except ValueError as ve:
        logger.warning(f"No data available: {ve}")
        return jsonify(error=str(ve)), 404
    except Exception as e:
        logger.error("Data fetch failed", exc_info=e)
        return jsonify(error="Failed to fetch data for given datetime/coordinates"), 500


    observed = row.to_dict()

    # Weather summary logic
    def weather_summary(obs):
        rain = obs.get('precipitation', 0)
        clouds = obs.get('cloud_cover', 0)
        wind = obs.get('windspeed', 0)
        temp = obs.get('temperature', 0)
        humidity = obs.get('humidity', 0)
        desc = []
        if rain > 0.2:
            desc.append("Raining")
        elif rain > 0.05:
            desc.append("Drizzling")
        else:
            desc.append("No rain")
        if clouds > 80:
            desc.append("Very cloudy")
        elif clouds > 40:
            desc.append("Partly cloudy")
        else:
            desc.append("Clear sky")
        if wind > 10:
            desc.append("Windy")
        if temp > 35:
            desc.append("Hot")
        elif temp < 10:
            desc.append("Cold")
        if humidity > 85:
            desc.append("Humid")
        return ', '.join(desc)

    out = {
        "datetime":    dt_str,
        "location":    MODEL_LOCATION_NAME if (lat, lon) == MODEL_LOCATION else f"({lat},{lon})",
        "observed":    observed,
        "coordinates": {"lat": lat, "lon": lon},
        "weather_summary": weather_summary(observed)
    }


    # Add ML prediction if available and location matches model training location
    if scaler and (lat, lon) == MODEL_LOCATION:
        # AdvancedNN prediction
        feat = row[[
            'temperature','windspeed','precipitation',
            'humidity','pressure','dew_point','cloud_cover'
        ]].values.reshape(1, -1)
        feat_scaled = scaler.transform(feat)
        pred_nn = model_nn.forward(feat_scaled).flatten()[0] if model_nn else None

        # LSTM prediction (use last 7 days if available)
        pred_lstm = None
        if model_lstm and len(hist_df) > 7:
            # Get last 7 days features (including today)
            idx = hist_df.index.get_loc(dt) if dt in hist_df.index else -1
            if idx >= 6:
                seq = hist_df.iloc[idx-6:idx+1][['temperature','windspeed','precipitation','humidity','pressure','dew_point','cloud_cover']].values
                seq_scaled = scaler.transform(seq)
                seq_scaled = seq_scaled.reshape(1, 7, -1)
                pred_lstm = model_lstm.forward(seq_scaled).flatten()[0]

        # Ensemble: average if both available
        if pred_nn is not None and pred_lstm is not None:
            pred_ensemble = (pred_nn + pred_lstm) / 2
            out["predicted_temperature_ensemble"] = float(pred_ensemble)
            out["predicted_temperature_nn"] = float(pred_nn)
            out["predicted_temperature_lstm"] = float(pred_lstm)
        elif pred_nn is not None:
            out["predicted_temperature_nn"] = float(pred_nn)
        elif pred_lstm is not None:
            out["predicted_temperature_lstm"] = float(pred_lstm)

        out["prediction_confidence"] = 0.92  # Placeholder

        # AI-based weather guess (example: rain prediction)
        rain_prob = float(min(max(observed.get('precipitation', 0) * 4, 0), 1))
        out["ai_rain_probability"] = rain_prob
        out["ai_rain_guess"] = "Rain likely" if rain_prob > 0.5 else "No rain likely"
    elif (lat, lon) != MODEL_LOCATION:
        out["note"] = "ML prediction only available for Vashere (19.45, 73.05)"

    logger.info(f"/forecast request: dt={dt_str}, lat={lat}, lon={lon}, response={out}")
    return jsonify(out)
@app.route('/health')
def health():
    status = {
        "model_ready": model_ready,
        "location": MODEL_LOCATION_NAME,
        "coordinates": {"lat": DEFAULT_LAT, "lon": DEFAULT_LON}
    }
    return jsonify(status)

@app.route('/docs')
def docs():
    doc = {
        "endpoints": {
            "/forecast": {
                "description": "Get weather forecast and AI prediction.",
                "params": {
                    "datetime": "Required. Format: YYYY-MM-DDTHH:MM",
                    "lat": "Optional. Latitude (default: 19.45)",
                    "lon": "Optional. Longitude (default: 73.05)"
                },
                "note": "ML prediction only for Vashere (19.45, 73.05)"
            },
            "/health": {
                "description": "Check model/API health."
            },
        },
        "example": "/forecast?datetime=2025-05-16T12:00&lat=19.45&lon=73.05"
    }
    return jsonify(doc)

if __name__ == '__main__':
    logger.info("Starting WeatherPrediction API server on 0.0.0.0:8000")
    app.run(host='0.0.0.0', port=8000)
