import numpy as np
from sklearn.ensemble import RandomForestRegressor
from ..config import RF_PARAMS, RANDOM_SEED


class RFModel:
    def __init__(self, params=None):
        self.params = params or RF_PARAMS
        self.model = None

    def fit(self, X_train, y_train):
        self.model = RandomForestRegressor(
            **self.params,
            random_state=RANDOM_SEED,
        )
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(np.atleast_2d(X_test)).ravel()
