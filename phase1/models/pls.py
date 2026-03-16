"""
Partial Least Squares (PLS) Regression Model
==============================================
PLS is the standard chemometric method for spectral prediction. Unlike PCA
followed by regression (two separate steps), PLS simultaneously finds latent
components that maximise covariance between X and y. This means PLS components
are optimised for prediction, not just variance explanation.

The number of PLS components k is selected via inner LOOCV: for each candidate
k from 1 to max_k, a full LOOCV is run on the training data, and the k with
lowest MSE is chosen.
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from ..config import PLS_MAX_COMPONENTS


class PLSModel:
    """PLS regression with adaptive component selection via inner LOOCV."""

    def __init__(self, max_components=None):
        """Initialise with maximum number of latent components to try.

        Args:
            max_components: Upper limit on k (default: 10). Actual max_k is
                           min(max_components, n_train - 1, n_features).
        """
        self.max_components = max_components or PLS_MAX_COMPONENTS
        self.model = None

    def fit(self, X_train, y_train):
        """Select optimal number of PLS components via inner LOOCV, then fit.

        For each k from 1 to max_k, runs a full LOOCV on the training data
        to compute MSE. The k with lowest MSE is selected. Then the final PLS
        model is fitted on all training data with the best k.

        Note: This inner LOOCV is nested inside the outer LOOCV in evaluation.py,
        creating a properly nested cross-validation procedure.

        Args:
            X_train: Training features (n_train, n_features).
            y_train: Training targets (n_train,).
        """
        max_k = min(self.max_components, X_train.shape[0] - 1, X_train.shape[1])
        best_k, best_mse = 1, np.inf
        loo = LeaveOneOut()

        # Inner LOOCV to select optimal k
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

        # Fit final model with best k on all training data
        self.model = PLSRegression(n_components=best_k)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Predict H2O2 yield rate using the fitted PLS model.

        Args:
            X_test: Test features (n_test, n_features) or (n_features,).

        Returns:
            np.ndarray: Predicted values (n_test,).
        """
        return self.model.predict(np.atleast_2d(X_test)).ravel()
