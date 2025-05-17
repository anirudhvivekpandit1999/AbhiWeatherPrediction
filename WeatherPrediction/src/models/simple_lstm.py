import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class SimpleLSTM:
    def __init__(self, input_shape):
        self.model = Sequential([
            LSTM(32, input_shape=input_shape),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self, X, y, epochs=10):
        self.model.fit(X, y, epochs=epochs)

    def predict(self, X):
        return self.model.predict(X)
