import argparse
import requests
import pandas as pd

def fetch_weather(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relativehumidity_2m,pressure_msl,precipitation",
        "timezone": "auto"  
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json().get("hourly", {})
    if not data:
        raise ValueError("No data returned from Open-Meteo. Check parameters or date range.")
    df = pd.DataFrame(data)
    return df

def main():
    parser = argparse.ArgumentParser(description="Collect historical weather data (Open-Meteo Archive API).")
    parser.add_argument("--lat", type=float, required=True, help="Latitude of location")
    parser.add_argument("--lon", type=float, required=True, help="Longitude of location")
    parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--output", type=str, default="weather_data.csv", help="Output CSV file")
    args = parser.parse_args()

    df = fetch_weather(args.lat, args.lon, args.start, args.end)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} records to {args.output}")

if __name__ == "__main__":
    main()
