import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from ..config import SVR_PARAM_GRID


class SVRModel:
    def __init__(self, param_grid=None):
        self.param_grid = param_grid or SVR_PARAM_GRID
        self.model = None

    def fit(self, X_train, y_train):
        search = GridSearchCV(
            SVR(kernel="rbf"),
            self.param_grid,
            cv=LeaveOneOut(),
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        self.model = search.best_estimator_

    def predict(self, X_test):
        return self.model.predict(np.atleast_2d(X_test)).ravel()
