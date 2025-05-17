import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as XGBRegressor
import requests
import joblib
import argparse
from datetime import timedelta, datetime
import warnings

warnings.filterwarnings('ignore')

class WeatherDataCollector:
    
    def __init__(self, api_keys=None):
        self.api_keys = api_keys or {}
        
    def get_openweathermap_historical(self, city, start_date, end_date, api_key=None):
        
        api_key = api_key or self.api_keys.get('openweathermap')
        if not api_key:
            print("Warning: OpenWeatherMap API key not provided")
            return pd.DataFrame()
            
        base_url = "https://api.openweathermap.org/data/2.5/onecall/timemachine"
        
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_data = []
        current_date = start_date
        
        while current_date <= end_date:
            timestamp = int(current_date.timestamp())
            params = {
                'lat': city['lat'],
                'lon': city['lon'],
                'dt': timestamp,
                'appid': api_key,
                'units': 'metric'
            }
            
            try:
                response = requests.get(base_url, params=params)
                if response.status_code == 200:
                    all_data.append(response.json())
                else:
                    print(f"Error fetching data for {current_date.date()}: {response.status_code}")
            except Exception as e:
                print(f"Exception when fetching data: {e}")
                
            current_date += timedelta(days=1)
            
        if all_data:
            return self._process_openweathermap_data(all_data)
        return pd.DataFrame()
    
    def get_meteostat_historical(self, station_id, start_date, end_date):
       
        base_url = f"https://meteostat.p.rapidapi.com/stations/hourly"
        
        headers = {
            "X-RapidAPI-Host": "meteostat.p.rapidapi.com",
            "X-RapidAPI-Key": self.api_keys.get('rapidapi', '')
        }
        
        params = {
            "station": station_id,
            "start": start_date,
            "end": end_date
        }
        
        try:
            response = requests.get(base_url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                return pd.DataFrame(data['data'])
            else:
                print(f"Error fetching data: {response.status_code}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Exception when fetching data: {e}")
            return pd.DataFrame()
    
    def download_noaa_data(self, station, year, month, output_dir='data'):
        
        base_url = "https://www.ncei.noaa.gov/data/global-hourly/access"
        url = f"{base_url}/{year}/{station}.csv"
        
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{station}_{year}_{month}.csv")
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(output_file, 'w') as f:
                    f.write(response.text)
                print(f"Downloaded data to {output_file}")
                return output_file
            else:
                print(f"Error downloading data: {response.status_code}")
                return None
        except Exception as e:
            print(f"Exception when downloading data: {e}")
            return None
    
    def load_csv_data(self, filepath):
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return pd.DataFrame()
            
    def _process_openweathermap_data(self, data_list):
        processed_data = []
        
        for day_data in data_list:
            date = datetime.fromtimestamp(day_data['current']['dt'])
            
            daily_record = {
                'date': date,
                'temp': day_data['current']['temp'],
                'feels_like': day_data['current']['feels_like'],
                'pressure': day_data['current']['pressure'],
                'humidity': day_data['current']['humidity'],
                'wind_speed': day_data['current']['wind_speed'],
                'wind_deg': day_data['current']['wind_deg'],
                'weather_main': day_data['current']['weather'][0]['main'],
                'weather_description': day_data['current']['weather'][0]['description']
            }
            
            for field in ['clouds', 'visibility', 'rain', 'snow']:
                if field in day_data['current']:
                    daily_record[field] = day_data['current'][field]
                else:
                    daily_record[field] = 0
                    
            processed_data.append(daily_record)
            
        return pd.DataFrame(processed_data)

    def generate_sample_data(self, city_name, start_date, end_date, hourly=True):
        
        print(f"Generating synthetic weather data for {city_name} from {start_date} to {end_date}")
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        if hourly:
            date_range = pd.date_range(start=start, end=end, freq='H')
        else:
            date_range = pd.date_range(start=start, end=end, freq='D')
        
        n = len(date_range)
        
        city_params = {
            'New York': {'temp_mean': 15, 'temp_std': 10, 'humid_mean': 65, 'pressure_mean': 1013},
            'London': {'temp_mean': 12, 'temp_std': 6, 'humid_mean': 80, 'pressure_mean': 1011},
            'Tokyo': {'temp_mean': 18, 'temp_std': 8, 'humid_mean': 70, 'pressure_mean': 1012},
            'Sydney': {'temp_mean': 22, 'temp_std': 5, 'humid_mean': 60, 'pressure_mean': 1015},
            'default': {'temp_mean': 17, 'temp_std': 8, 'humid_mean': 65, 'pressure_mean': 1013}
        }
        
        params = city_params.get(city_name, city_params['default'])
        
        day_of_year = np.array([d.dayofyear for d in date_range])
        season_factor = np.sin(2 * np.pi * day_of_year / 365)
        
        hour_of_day = np.array([d.hour for d in date_range])
        daily_factor = np.sin(2 * np.pi * (hour_of_day - 4) / 24)  
        
        temp = params['temp_mean'] + params['temp_std'] * 3 * season_factor
        if hourly:
            temp += params['temp_std'] * daily_factor
        temp += np.random.normal(0, params['temp_std'] * 0.2, n)  
        
        humidity = params['humid_mean'] - 10 * season_factor + np.random.normal(0, 5, n)
        humidity = np.clip(humidity, 0, 100)  
        
        pressure = params['pressure_mean'] + 2 * season_factor + np.random.normal(0, 1, n)
        
        wind_speed = 5 + 2 * np.sin(np.gradient(pressure)) + np.random.exponential(2, n)
        wind_speed = np.clip(wind_speed, 0, 30)  
        
        wind_dir = np.random.uniform(0, 360, n)
        
        precip_prob = 0.2 + 0.1 * season_factor  
        precipitation = np.random.exponential(1, n) * (np.random.random(n) < precip_prob)
        
        conditions = []
        for i in range(n):
            if precipitation[i] > 2:
                conditions.append('Heavy Rain')
            elif precipitation[i] > 0:
                conditions.append('Light Rain')
            elif humidity[i] > 85:
                conditions.append('Cloudy')
            elif humidity[i] > 60:
                conditions.append('Partly Cloudy')
            else:
                conditions.append('Clear')
        
        weather_data = pd.DataFrame({
            'datetime': date_range,
            'temperature': temp,
            'humidity': humidity,
            'pressure': pressure,
            'wind_speed': wind_speed,
            'wind_direction': wind_dir,
            'precipitation': precipitation,
            'condition': conditions
        })
        
        return weather_data


class WeatherDataProcessor:
    
    def __init__(self):
        pass
    
    def clean_data(self, df):
        df_clean = df.copy()
        
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        for col in df_clean.select_dtypes(include=['object']).columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            if col in ['datetime', 'date', 'id']:
                continue
                
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        
        return df_clean
    
    def extract_datetime_features(self, df, datetime_col='datetime'):
        df = df.copy()
        
        if datetime_col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
                df[datetime_col] = pd.to_datetime(df[datetime_col])
                
            df['year'] = df[datetime_col].dt.year
            df['month'] = df[datetime_col].dt.month
            df['day'] = df[datetime_col].dt.day
            df['hour'] = df[datetime_col].dt.hour
            df['dayofweek'] = df[datetime_col].dt.dayofweek
            df['quarter'] = df[datetime_col].dt.quarter
            df['dayofyear'] = df[datetime_col].dt.dayofyear
            
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
            
        return df
    
    def create_lag_features(self, df, target_cols, lag_hours=[1, 3, 6, 12, 24]):
        df = df.copy()
        
        if 'datetime' in df.columns:
            df = df.sort_values('datetime')
        
        for col in target_cols:
            for lag in lag_hours:
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
        
        df = df.dropna()
        
        return df
    
    def create_rolling_features(self, df, target_cols, windows=[3, 6, 12, 24]):
        df = df.copy()
        
        if 'datetime' in df.columns:
            df = df.sort_values('datetime')
        
        for col in target_cols:
            for window in windows:
                df[f'{col}_rolling_mean_{window}h'] = df[col].rolling(window=window).mean()
                df[f'{col}_rolling_std_{window}h'] = df[col].rolling(window=window).std()
                df[f'{col}_rolling_min_{window}h'] = df[col].rolling(window=window).min()
                df[f'{col}_rolling_max_{window}h'] = df[col].rolling(window=window).max()
        
        df = df.dropna()
        
        return df
    
    def encode_categorical_features(self, df, cat_cols):
        df = df.copy()
        
        for col in cat_cols:
            if col in df.columns:
                one_hot = pd.get_dummies(df[col], prefix=col)
                df = df.drop(col, axis=1)
                df = df.join(one_hot)
        
        return df
    
    def prepare_data_for_model(self, df, target_col, drop_cols=None):
        df = df.copy()
        
        if drop_cols:
            df = df.drop(columns=drop_cols, errors='ignore')
        
        if target_col in df.columns:
            y = df[target_col].values
            X = df.drop(columns=[target_col])
            
            return X, y
        else:
            print(f"Target column '{target_col}' not found in dataframe")
            return None, None


class WeatherModel:
    
    def __init__(self, model_type='random_forest', params=None):
        self.model_type = model_type
        self.params = params or {}
        self.model = None
        self.scaler = StandardScaler()
        
    def get_model(self):
        if self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=self.params.get('n_estimators', 100),
                max_depth=self.params.get('max_depth', None),
                min_samples_split=self.params.get('min_samples_split', 2),
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=self.params.get('n_estimators', 100),
                learning_rate=self.params.get('learning_rate', 0.1),
                max_depth=self.params.get('max_depth', 3),
                random_state=42
            )
        elif self.model_type == 'xgboost':
            return XGBRegressor(
                n_estimators=self.params.get('n_estimators', 100),
                learning_rate=self.params.get('learning_rate', 0.1),
                max_depth=self.params.get('max_depth', 3),
                random_state=42
            )
        elif self.model_type == 'linear':
            return LinearRegression()
        else:
            print(f"Unknown model type: {self.model_type}. Using RandomForest as default.")
            return RandomForestRegressor(random_state=42)
    
    def train(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model = self.get_model()
        self.model.fit(X_train_scaled, y_train)
        
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
    
    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        X_test_scaled = self.scaler.transform(X_test)
        
        y_pred = self.model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred
        }
    
    def tune_hyperparameters(self, X_train, y_train, param_grid=None):
        if param_grid is None:
            if self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
            elif self.model_type in ['gradient_boosting', 'xgboost']:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            else:
                print("No parameter grid defined for this model type.")
                return self
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        model = self.get_model()
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        self.params = grid_search.best_params_
        print(f"Best parameters: {self.params}")
        
        self.model = self.get_model()
        self.model.fit(X_train_scaled, y_train)
        
        return self
    
    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("No trained model to save.")
            
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'params': self.params
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        model_data = joblib.load(filepath)
        
        instance = cls(model_type=model_data['model_type'], params=model_data['params'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        
        return instance
    
    def get_feature_importance(self, feature_names):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            return feature_importance_df
        else:
            print("This model doesn't provide feature importances.")
            return None


class WeatherVisualizer:
    
    def __init__(self, output_dir='visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = {
            'actual': '#1f77b4',  # blue
            'predicted': '#ff7f0e',  # orange
            'error': '#d62728'  # red
        }
    
    def plot_time_series(self, df, date_col, value_col, title, filename=None):
        plt.figure(figsize=(12, 6))
        plt.plot(df[date_col], df[value_col], color=self.colors['actual'], linewidth=2)
        plt.title(title, fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel(value_col, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            print(f"Plot saved to {filepath}")
            
        plt.close()
    
    def plot_actual_vs_predicted(self, dates, y_true, y_pred, title, y_label, filename=None):
        plt.figure(figsize=(14, 7))
        
        plt.plot(dates, y_true, color=self.colors['actual'], label='Actual', linewidth=2)
        plt.plot(dates, y_pred, color=self.colors['predicted'], label='Predicted', linewidth=2, linestyle='--')
        
        plt.title(title, fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        plt.figtext(0.5, 0.01, f'RMSE: {rmse:.2f}', ha='center', fontsize=12)
        
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            print(f"Plot saved to {filepath}")
            
        plt.close()
    
    def plot_error_distribution(self, y_true, y_pred, title, filename=None):
        errors = y_true - y_pred
        
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True, color=self.colors['error'])
        
        plt.title(title, fontsize=16)
        plt.xlabel('Prediction Error', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.figtext(0.7, 0.8, f'Mean Error: {np.mean(errors):.2f}\nStd Dev: {np.std(errors):.2f}', 
                   backgroundcolor='white', fontsize=10)
        
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            print(f"Plot saved to {filepath}")
            
        plt.close()
    
    def plot_feature_importance(self, feature_importance_df, title, filename=None):
        plt.figure(figsize=(12, 8))
        
        top_features = feature_importance_df.head(20)
        sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
        
        plt.title(title, fontsize=16)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            print(f"Plot saved to {filepath}")
            
        plt.close()
    
    def plot_correlation_matrix(self, df, title, filename=None):
        numeric_df = df.select_dtypes(include=[np.number])
        
        corr_matrix = numeric_df.corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', 
                   square=True, linewidths=0.5)
        
        plt.title(title, fontsize=16)
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            print(f"Plot saved to {filepath}")
            
        plt.close()
    
    def plot_seasonal_patterns(self, df, date_col, value_col, title, filename=None):
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col])
        
        df = df.copy()
        df['month'] = df[date_col].dt.month
        df['hour'] = df[date_col].dt.hour
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        monthly_avg = df.groupby('month')[value_col].mean()
        ax1.plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2, color=self.colors['actual'])
        ax1.set_title(f'Monthly {title}', fontsize=14)
        ax1.set_xlabel('Month', fontsize=12)
        ax1.set_ylabel(value_col, fontsize=12)
        ax1.set_xticks(range(1, 13))
        ax1.grid(True, alpha=0.3)
        
        hourly_avg = df.groupby('hour')[value_col].mean()
        ax2.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, color=self.colors['actual'])
        ax2.set_title(f'Hourly {title}', fontsize=14)
        ax2.set_xlabel('Hour', fontsize=12)
        ax2.set_ylabel(value_col, fontsize=12)
        ax2.set_xticks(range(0, 24, 3))
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            print(f"Plot saved to {filepath}")
            
        plt.close()
        
    def plot_forecast(self, historical_dates, historical_values, forecast_dates, forecast_values, 
                      title, y_label, filename=None):
        plt.figure(figsize=(14, 7))
        
        plt.plot(historical_dates, historical_values, color=self.colors['actual'], 
                label='Historical', linewidth=2)
        
        plt.plot(forecast_dates, forecast_values, color=self.colors['predicted'], 
                label='Forecast', linewidth=2, linestyle='--')
        
        plt.axvline(x=historical_dates.iloc[-1], color='gray', linestyle='-', linewidth=1)
        
        plt.title(title, fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            print(f"Plot saved to {filepath}")
            
        plt.close()


class WeatherForecaster:
    
    def __init__(self, models=None, data_processor=None):
        self.models = models or {}  # Dictionary of {target_variable: model}
        self.data_processor = data_processor or WeatherDataProcessor()
        
    def add_model(self, target_variable, model):
        self.models[target_variable] = model
        
    def load_models(self, model_dir):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
            
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
        
        for model_file in model_files:
            target_variable = model_file.replace('_model.joblib', '')
            
            model_path = os.path.join(model_dir, model_file)
            model = WeatherModel.load_model(model_path)
            
            self.models[target_variable] = model
            print(f"Loaded model for {target_variable}")
            
    def prepare_forecast_features(self, latest_data, horizon=24):
      """Prepare features for forecasting next 'horizon' hours"""
      forecast_features = []
      predictions = {}  
    
      data = latest_data.copy()
    
      if 'datetime' in data.columns:
        latest_datetime = data['datetime'].max()
      else:
        latest_datetime = datetime.now()
        
      for i in range(1, horizon + 1):
        forecast_datetime = latest_datetime + timedelta(hours=i)
        forecast_row = pd.DataFrame([data.iloc[-1]]).copy() 
        
        if 'datetime' in forecast_row.columns:
            forecast_row['datetime'] = forecast_datetime
            
            forecast_row = self.data_processor.extract_datetime_features(forecast_row)
        
        if i > 1:
            for j in range(1, i):
                for var in self.models.keys():
                    
                    lag_col = f'{var}_lag_{j}h'
                    pred_key = f'{var}_{i-j}'  
                    
                    if lag_col in forecast_row.columns and pred_key in predictions:
                        forecast_row[lag_col] = predictions[pred_key]
        
        forecast_features.append(forecast_row)
    
      forecast_df = pd.concat(forecast_features, ignore_index=True)
    
      return forecast_df
    
    def generate_forecast(self, latest_data, horizon=24):
      if not self.models:
        raise ValueError("No trained models available for forecasting.")
        
      forecast_df = self.prepare_forecast_features(latest_data, horizon)
    
      predictions = {}
    
      for target_var, model in self.models.items():
        X_forecast = forecast_df.drop(columns=[target_var], errors='ignore')
        
        drop_cols = ['datetime', 'date', 'id']
        X_forecast = X_forecast.drop(columns=[col for col in drop_cols if col in X_forecast.columns])
        
        try:
            model_predictions = model.predict(X_forecast)
            forecast_df[f'{target_var}_pred'] = model_predictions
            
            for i, pred in enumerate(model_predictions):
                predictions[f'{target_var}_{i+1}'] = pred
                
        except Exception as e:
            print(f"Error predicting {target_var}: {e}")
    
      if 'datetime' in latest_data.columns:
        latest_datetime = latest_data['datetime'].max()
        forecast_dates = [latest_datetime + timedelta(hours=i) for i in range(1, horizon + 1)]
        forecast_df['forecast_datetime'] = forecast_dates
    
      return forecast_df
    
    def evaluate_forecast(self, test_data, horizon=24):
        """Evaluate forecast accuracy on test data"""
        if not self.models:
            raise ValueError("No trained models available for forecasting.")
            
        test_data = test_data.copy()
        
        test_chunks = [test_data.iloc[i:i+horizon] for i in range(0, len(test_data), horizon)]
        
        all_actuals = {}
        all_predictions = {}
        
        for target_var in self.models.keys():
            all_actuals[target_var] = []
            all_predictions[target_var] = []
        
        for chunk in test_chunks:
            if len(chunk) < horizon:
                continue  
                
            latest_data = chunk.iloc[[0]]
            
            forecast = self.generate_forecast(latest_data, horizon=len(chunk)-1)
            
            for target_var in self.models.keys():
                actuals = chunk.iloc[1:][target_var].values
                predictions = forecast[f'{target_var}_pred'].values
                
                all_actuals[target_var].extend(actuals)
                all_predictions[target_var].extend(predictions)
        
        results = {}
        for target_var in self.models.keys():
            actuals = np.array(all_actuals[target_var])
            predictions = np.array(all_predictions[target_var])
            
            results[target_var] = {
                'mse': mean_squared_error(actuals, predictions),
                'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
                'mae': mean_absolute_error(actuals, predictions),
                'r2': r2_score(actuals, predictions)
            }
        
        return results


class WeatherPredictionSystem:
    
    def __init__(self, config=None):
        self.config = config or {}
        self.data_collector = WeatherDataCollector(api_keys=self.config.get('api_keys', {}))
        self.data_processor = WeatherDataProcessor()
        self.models = {}  
        self.visualizer = WeatherVisualizer(output_dir=self.config.get('output_dir', 'output'))
        self.forecaster = WeatherForecaster(data_processor=self.data_processor)
        
    def collect_data(self, method='sample', **kwargs):
        if method == 'openweathermap':
            return self.data_collector.get_openweathermap_historical(**kwargs)
        elif method == 'meteostat':
            return self.data_collector.get_meteostat_historical(**kwargs)
        elif method == 'noaa':
            return self.data_collector.download_noaa_data(**kwargs)
        elif method == 'load_csv':
            return self.data_collector.load_csv_data(**kwargs)
        elif method == 'sample':
            return self.data_collector.generate_sample_data(**kwargs)
        else:
            raise ValueError(f"Unknown data collection method: {method}")
            
    def process_data(self, data, target_cols, clean=True, extract_datetime=True, 
                    create_lag=True, create_rolling=True, encode_categorical=True):
        if clean:
            data = self.data_processor.clean_data(data)
            
        if extract_datetime and 'datetime' in data.columns:
            data = self.data_processor.extract_datetime_features(data)
            
        if create_lag:
            data = self.data_processor.create_lag_features(data, target_cols)
            
        if create_rolling:
            data = self.data_processor.create_rolling_features(data, target_cols)
            
        if encode_categorical:
            cat_cols = data.select_dtypes(include=['object']).columns.tolist()
            data = self.data_processor.encode_categorical_features(data, cat_cols)
            
        return data
            
    def train_model(self, data, target_col, model_type='random_forest', params=None, 
                   test_size=0.2, tune_hyperparams=False, save_model=True):
        drop_cols = ['datetime', 'date', 'id']
        X, y = self.data_processor.prepare_data_for_model(
            data, target_col, drop_cols=[col for col in drop_cols if col in data.columns]
        )
        
        if X is None or y is None:
            return None
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        model = WeatherModel(model_type=model_type, params=params or {})
        
        if tune_hyperparams:
            model.tune_hyperparameters(X_train, y_train)
        else:
            model.train(X_train, y_train)
        
        evaluation = model.evaluate(X_test, y_test)
        print(f"\nModel evaluation for {target_col}:")
        print(f"RMSE: {evaluation['rmse']:.4f}")
        print(f"MAE: {evaluation['mae']:.4f}")
        print(f"R²: {evaluation['r2']:.4f}")
        
        self.models[target_col] = model
        
        if save_model:
            os.makedirs('models', exist_ok=True)
            model_path = f"models/{target_col}_model.joblib"
            model.save_model(model_path)
        
        self.visualize_model_results(X_test, y_test, evaluation['predictions'], target_col)
        
        if hasattr(model.model, 'feature_importances_'):
            feature_importance = model.get_feature_importance(X.columns)
            self.visualizer.plot_feature_importance(
                feature_importance, 
                title=f'Feature Importance for {target_col}',
                filename=f'{target_col}_feature_importance.png'
            )
        
        return model
    
    def train_all_models(self, data, target_cols, **kwargs):
        for target_col in target_cols:
            print(f"\nTraining model for {target_col}...")
            self.train_model(data, target_col, **kwargs)
            
        self.forecaster = WeatherForecaster(models=self.models, data_processor=self.data_processor)
    
    def generate_forecast(self, latest_data, horizon=24):
        if not self.forecaster.models and self.models:
            self.forecaster.models = self.models
            
        return self.forecaster.generate_forecast(latest_data, horizon)
    
    def visualize_model_results(self, X_test, y_test, y_pred, target_col):
        test_df = pd.DataFrame(X_test)
        test_df['actual'] = y_test
        test_df['predicted'] = y_pred
        
        if 'datetime' in test_df.columns:
            dates = test_df['datetime']
        else:
            dates = pd.date_range(start='2023-01-01', periods=len(test_df), freq='H')
        
        self.visualizer.plot_actual_vs_predicted(
            dates, y_test, y_pred,
            title=f'Actual vs Predicted {target_col}',
            y_label=target_col,
            filename=f'{target_col}_actual_vs_predicted.png'
        )
        
        self.visualizer.plot_error_distribution(
            y_test, y_pred,
            title=f'Error Distribution for {target_col} Predictions',
            filename=f'{target_col}_error_distribution.png'
        )
    
    def visualize_forecast(self, historical_data, forecast_data, target_col):
        if 'datetime' in historical_data.columns:
            historical_dates = historical_data['datetime']
        else:
            historical_dates = pd.date_range(
                end=datetime.now(), periods=len(historical_data), freq='H'
            )
        
        historical_values = historical_data[target_col]
        
        if 'forecast_datetime' in forecast_data.columns:
            forecast_dates = forecast_data['forecast_datetime']
        else:
            forecast_dates = pd.date_range(
                start=historical_dates.iloc[-1] + timedelta(hours=1),
                periods=len(forecast_data),
                freq='H'
            )
        
        forecast_values = forecast_data[f'{target_col}_pred']
        
        self.visualizer.plot_forecast(
            historical_dates, historical_values,
            forecast_dates, forecast_values,
            title=f'{target_col} Forecast',
            y_label=target_col,
            filename=f'{target_col}_forecast.png'
        )
    
    def run_full_pipeline(self, data_params, target_cols, model_params=None, horizon=24):
        print("Collecting weather data...")
        data = self.collect_data(**data_params)
        print(f"Collected {len(data)} records")
        
        print("\nProcessing data...")
        processed_data = self.process_data(data, target_cols)
        print(f"Processed data shape: {processed_data.shape}")
        
        print("\nTraining models...")
        self.train_all_models(processed_data, target_cols, **(model_params or {}))
        
        print("\nGenerating forecast...")
        latest_data = processed_data.tail(max(horizon, 24))  # Use at least 24 hours of data
        forecast = self.generate_forecast(latest_data, horizon)
        print(f"Generated forecast for next {horizon} hours")
        
        print("\nVisualizing results...")
        for target_col in target_cols:
            self.visualize_forecast(data, forecast, target_col)
            
        return forecast


def main():
    parser = argparse.ArgumentParser(description='Weather Prediction System')
    parser.add_argument('--city', type=str, default='New York', help='City name for weather data')
    parser.add_argument('--start_date', type=str, default='2023-01-01', help='Start date for data (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2023-12-31', help='End date for data (YYYY-MM-DD)')
    parser.add_argument('--target_cols', type=str, nargs='+', default=['temperature'], help='Target columns to predict')
    parser.add_argument('--model_type', type=str, default='random_forest', choices=['random_forest', 'gradient_boosting', 'xgboost', 'linear'], help='Model type')
    parser.add_argument('--horizon', type=int, default=24, help='Forecast horizon in hours')
    parser.add_argument('--tune', action='store_true', help='Tune hyperparameters')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory for results')
    
    args = parser.parse_args()
    
    config = {
        'api_keys': {
            'openweathermap': os.environ.get('OPENWEATHERMAP_API_KEY', ''),
            'rapidapi': os.environ.get('RAPIDAPI_KEY', '')
        },
        'output_dir': args.output_dir
    }
    
    weather_system = WeatherPredictionSystem(config)
    
    data_params = {
        'method': 'sample',
        'city_name': args.city,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'hourly': True
    }
    
    model_params = {
        'model_type': args.model_type,
        'tune_hyperparams': args.tune,
        'save_model': True
    }
    
    forecast = weather_system.run_full_pipeline(data_params, args.target_cols, model_params, args.horizon)
    
    print("\nForecast Summary:")
    for target in args.target_cols:
        mean_value = forecast[f'{target}_pred'].mean()
        min_value = forecast[f'{target}_pred'].min()
        max_value = forecast[f'{target}_pred'].max()
        print(f"{target.capitalize()}: Mean = {mean_value:.2f}, Min = {min_value:.2f}, Max = {max_value:.2f}")
    
    print(f"\nForecast and visualization outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()