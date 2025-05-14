import os
import requests
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_historical(lat, lon, days=30, timezone='Asia/Kolkata'):
    """
    Fetch last `days` days of hourly data from Open‑Meteo Archive API.
    """
    end = datetime.now().date()
    start = end - timedelta(days=days)
    url = 'https://archive-api.open-meteo.com/v1/archive'
    params = {
        'latitude': lat,
        'longitude': lon,
        # corrected parameter names:
        'hourly': 'temperature_2m,windspeed_10m,precipitation,relativehumidity_2m,pressure_msl,dew_point_2m,cloudcover',
        'start_date': start.isoformat(),
        'end_date':   end.isoformat(),
        'timezone':   timezone
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()['hourly']

    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['time'])
    df.set_index('datetime', inplace=True)
    df = df[[
        'temperature_2m',
        'windspeed_10m',
        'precipitation',
        'relativehumidity_2m',
        'pressure_msl',
        'dew_point_2m',
        'cloudcover'
    ]]
    df.columns = [
        'temperature',
        'windspeed',
        'precipitation',
        'humidity',
        'pressure',
        'dew_point',
        'cloud_cover'
    ]
    return df

def fetch_forecast(lat, lon, target_dt, timezone='Asia/Kolkata'):
    """
    Fetch one hour of data (future via forecast API, past/present via archive API).
    """
    is_future = target_dt > datetime.now()
    base_url = 'https://api.open-meteo.com/v1/forecast' if is_future else 'https://archive-api.open-meteo.com/v1/archive'
    date = target_dt.date().isoformat()

    params = {
        'latitude': lat,
        'longitude': lon,
        # same corrected names here:
        'hourly': 'temperature_2m,windspeed_10m,precipitation,relativehumidity_2m,pressure_msl,dew_point_2m,cloudcover',
        'start_date': date,
        'end_date':   date,
        'timezone':   timezone
    }
    r = requests.get(base_url, params=params)
    r.raise_for_status()
    data = r.json()['hourly']

    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['time'])
    df.set_index('datetime', inplace=True)
    df = df[[
        'temperature_2m',
        'windspeed_10m',
        'precipitation',
        'relativehumidity_2m',
        'pressure_msl',
        'dew_point_2m',
        'cloudcover'
    ]]
    df.columns = [
        'temperature',
        'windspeed',
        'precipitation',
        'humidity',
        'pressure',
        'dew_point',
        'cloud_cover'
    ]

    if target_dt not in df.index:
        raise ValueError(f"No data available for {target_dt.isoformat()}")
    return df.loc[target_dt]
