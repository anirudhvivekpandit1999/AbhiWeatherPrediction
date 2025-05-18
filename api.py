from fastapi import FastAPI, Query
from pydantic import BaseModel
import requests
import pickle
from datetime import datetime, timedelta, timezone
from timezonefinder import TimezoneFinder
from zoneinfo import ZoneInfo  

app = FastAPI()
with open("rain_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/predict_next_rain")
def predict_next_rain(lat: float = Query(None), lon: float = Query(None)):
    if lat is None or lon is None:
        resp = requests.get("http://ip-api.com/json/").json()
        if resp.get("status") != "success":
            return {"error": "Could not determine location from IP."}
        lat = resp["lat"]
        lon = resp["lon"]
        tz_name = resp["timezone"]  
    else:
        tf = TimezoneFinder()
        tz_name = tf.timezone_at(lat=lat, lng=lon) or "UTC"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relativehumidity_2m,pressure_msl,precipitation",
        "timezone": "auto",
        "past_hours": 1  
    }
    forecast = requests.get("https://api.open-meteo.com/v1/forecast", params=params).json()
    times = forecast["hourly"]["time"]
    temps = forecast["hourly"]["temperature_2m"]
    hums = forecast["hourly"]["relativehumidity_2m"]
    press = forecast["hourly"]["pressure_msl"]
    precs = forecast["hourly"]["precipitation"]

    now_local = datetime.now(ZoneInfo(tz_name))
    now_floor = now_local.replace(minute=0, second=0, microsecond=0)
    time_str = now_floor.isoformat(sep='T', timespec='hours')
    if time_str in times:
        idx = times.index(time_str)
    else:
        idx = 0

    feat = [
        temps[idx],
        hums[idx],
        press[idx],
        now_floor.hour,
        now_floor.timetuple().tm_yday
    ]
    hours_pred = model.predict([feat])[0]
    hours_pred = max(0, float(hours_pred))
    predicted_local = now_floor + timedelta(hours=hours_pred)
    predicted_str = predicted_local.replace(tzinfo=ZoneInfo(tz_name)).isoformat()

    return {"next_rain_time": predicted_str}

