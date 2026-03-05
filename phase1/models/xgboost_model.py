import numpy as np
from xgboost import XGBRegressor
from ..config import XGBOOST_PARAMS, RANDOM_SEED


class XGBoostModel:
    def __init__(self, params=None):
        self.params = params or XGBOOST_PARAMS
        self.model = None

    def fit(self, X_train, y_train):
        self.model = XGBRegressor(
            **self.params,
            random_state=RANDOM_SEED,
            verbosity=0,
        )
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(np.atleast_2d(X_test)).ravel()
