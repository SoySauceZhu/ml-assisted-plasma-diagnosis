import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from ..config import PLS_MAX_COMPONENTS


class PLSModel:
    def __init__(self, max_components=None):
        self.max_components = max_components or PLS_MAX_COMPONENTS
        self.model = None

    def fit(self, X_train, y_train):
        max_k = min(self.max_components, X_train.shape[0] - 1, X_train.shape[1])
        best_k, best_mse = 1, np.inf
        loo = LeaveOneOut()

        for k in range(1, max_k + 1):
            preds = []
            for train_idx, val_idx in loo.split(X_train):
                pls = PLSRegression(n_components=k)
                pls.fit(X_train[train_idx], y_train[train_idx])
                preds.append(pls.predict(X_train[val_idx]).ravel()[0])
            mse = mean_squared_error(y_train, preds)
            if mse < best_mse:
                best_mse = mse
                best_k = k

        self.model = PLSRegression(n_components=best_k)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(np.atleast_2d(X_test)).ravel()
