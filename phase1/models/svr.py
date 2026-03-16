"""
Support Vector Regression (SVR) Model
=======================================
SVR with RBF (Radial Basis Function) kernel, tuned via GridSearchCV with LOOCV.
The RBF kernel maps input features into a high-dimensional space where a linear
epsilon-tube regression is performed.

Grid search over:
  - C: regularisation parameter (controls bias-variance trade-off)
  - epsilon: width of the insensitive tube (errors within epsilon are ignored)
  - gamma: RBF kernel width (controls how far the influence of a single sample reaches)
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from ..config import SVR_PARAM_GRID


class SVRModel:
    """SVR with automatic hyperparameter selection via grid search + LOOCV."""

    def __init__(self, param_grid=None):
        """Initialise with hyperparameter grid.

        Args:
            param_grid: Dict of {param_name: [values]} for grid search
                       (default: SVR_PARAM_GRID with C, epsilon, gamma).
        """
        self.param_grid = param_grid or SVR_PARAM_GRID
        self.model = None

    def fit(self, X_train, y_train):
        """Grid search over all hyperparameter combinations using LOOCV.

        Evaluates every combination of C, epsilon, and gamma using negative MSE
        as the scoring metric. Uses n_jobs=-1 for parallel computation across
        all CPU cores. Stores the best estimator.

        Args:
            X_train: Training features (n_train, n_features).
            y_train: Training targets (n_train,).
        """
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
        """Predict using the best SVR from grid search.

        Args:
            X_test: Test features (n_test, n_features) or (n_features,).

        Returns:
            np.ndarray: Predicted values (n_test,).
        """
        return self.model.predict(np.atleast_2d(X_test)).ravel()
