import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_historical_data(lat, lon):
    """
    Mock function to generate synthetic weather data.
    Replace this with actual API or CSV data loading.
    """
    date_range = pd.date_range(end=pd.Timestamp.today(), periods=30, freq='D')
    data = {
        "temperature": np.random.normal(30, 5, size=30),
        "windspeed": np.random.normal(5, 2, size=30),
        "precipitation": np.random.uniform(0, 0.5, size=30),
        "humidity": np.random.normal(60, 10, size=30),
        "pressure": np.random.normal(1013, 5, size=30),
        "dew_point": np.random.normal(20, 3, size=30),
        "cloud_cover": np.random.uniform(0, 100, size=30)
    }
    df = pd.DataFrame(data, index=date_range)
    return df

def preprocess_data(df):
    """
    Extracts features and targets and fits a scaler.
    """
    features = df[[
        "temperature",
        "windspeed",
        "precipitation",
        "humidity",
        "pressure",
        "dew_point",
        "cloud_cover"
    ]]
    target = df["temperature"].shift(-1).dropna()  # Next-day prediction

    X = features.iloc[:-1].values
    y = target.values

    return X, y
