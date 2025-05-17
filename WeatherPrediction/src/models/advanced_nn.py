from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

class AdvancedNN:
    def __init__(self, input_dim):
        self.model = MLPRegressor(hidden_layer_sizes=(64, 32, 16), max_iter=200)
        self.scaler = StandardScaler()
        self.input_dim = input_dim

    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
