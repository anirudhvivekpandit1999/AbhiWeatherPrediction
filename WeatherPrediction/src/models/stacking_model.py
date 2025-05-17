from sklearn.linear_model import LinearRegression
import numpy as np

class StackingModel:
    def __init__(self, base_models):
        self.base_models = base_models
        self.meta_model = LinearRegression()

    def fit(self, X, y):
        base_predictions = np.column_stack([model.predict(X) for model in self.base_models])
        self.meta_model.fit(base_predictions, y)

    def predict(self, X):
        base_predictions = np.column_stack([model.predict(X) for model in self.base_models])
        return self.meta_model.predict(base_predictions)
