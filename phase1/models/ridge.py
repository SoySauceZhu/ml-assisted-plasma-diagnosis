import numpy as np
from sklearn.linear_model import RidgeCV
from ..config import RIDGE_ALPHAS


class RidgeModel:
    def __init__(self, alphas=None):
        self.alphas = alphas or RIDGE_ALPHAS
        self.model = None

    def fit(self, X_train, y_train):
        self.model = RidgeCV(alphas=self.alphas, cv=None)  # cv=None uses efficient LOOCV
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(np.atleast_2d(X_test)).ravel()
