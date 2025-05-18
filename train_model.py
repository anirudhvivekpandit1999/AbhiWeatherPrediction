import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

def prepare_training_data(csv_file):
    df = pd.read_csv(csv_file, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df["hour"] = df["time"].dt.hour
    df["day_of_year"] = df["time"].dt.dayofyear
    
    rain_idx = np.where(df["precipitation"] > 0)[0]
    X_list, y_list = [], []
    for i in range(len(df)):
        if df.loc[i, "precipitation"] > 0:
            continue
        pos = np.searchsorted(rain_idx, i)
        if pos >= len(rain_idx):
            continue
        next_i = rain_idx[pos]
        hours_until = (df.loc[next_i, "time"] - df.loc[i, "time"]).total_seconds() / 3600.0
        feat = [
            df.loc[i, "temperature_2m"],
            df.loc[i, "relativehumidity_2m"],
            df.loc[i, "pressure_msl"],
            df.loc[i, "hour"],
            df.loc[i, "day_of_year"]
        ]
        X_list.append(feat)
        y_list.append(hours_until)
    X = np.array(X_list)
    y = np.array(y_list)
    return X, y

def main():
    csv_file = "history.csv"
    X, y = prepare_training_data(csv_file)
    print(f"Training samples: {X.shape[0]}")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    with open("rain_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model trained and saved to rain_model.pkl")

if __name__ == "__main__":
    main()
