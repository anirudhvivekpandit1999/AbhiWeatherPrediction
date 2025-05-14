from flask import Flask, jsonify, request
import logging
from datetime import datetime

from ingest import fetch_historical, fetch_forecast
from interp import scale_features
from ml import AdvancedNN

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WeatherAPI")

V_LAT, V_LON = 19.45, 73.05

# Pre-train on last 30 days
try:
    hist_df = fetch_historical(V_LAT, V_LON, days=30)
    X_hist, scaler = scale_features(hist_df)
    y_hist = hist_df['temperature'].values

    model = AdvancedNN(in_dim=X_hist.shape[1], h1=64, h2=32, h3=16)
    logger.info("Training model on last 30 days…")
    model.train(X_hist, y_hist, epochs=200, lr=1e-3)
    logger.info("Model training complete.")
except Exception as e:
    logger.error("Could not train model", exc_info=e)
    model = None
    scaler = None

@app.route('/forecast')
def forecast():
    dt_str = request.args.get("datetime")
    if not dt_str:
        return jsonify(error="Missing datetime; use YYYY-MM-DDTHH:MM"), 400

    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M")
    except ValueError:
        return jsonify(error="Invalid format; use YYYY-MM-DDTHH:MM"), 400

    # Fetch observed or forecast data
    try:
        row = fetch_forecast(V_LAT, V_LON, dt)
    except Exception as e:
        logger.error("Data fetch failed", exc_info=e)
        return jsonify(error="Failed to fetch data for given datetime"), 500

    out = {
        "datetime":    dt_str,
        "location":    "Vashere",
        "observed":    row.to_dict(),
        "coordinates": {"lat": V_LAT, "lon": V_LON}
    }

    # Add ML prediction if available
    if model and scaler:
        feat = row[[
            'temperature','windspeed','precipitation',
            'humidity','pressure','dew_point','cloud_cover'
        ]].values.reshape(1, -1)
        feat_scaled = scaler.transform(feat)
        pred = model.forward(feat_scaled).flatten()[0]
        out["predicted_temperature"] = float(pred)

    return jsonify(out)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
