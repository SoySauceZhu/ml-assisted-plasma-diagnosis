"""
Ridge Regression Model
=======================
L2-regularised linear regression using sklearn's RidgeCV. The regularisation
strength alpha is automatically selected via efficient LOOCV (cv=None uses
the analytical LOOCV formula, no explicit loop needed). This makes Ridge
both fast and well-regularised for our small dataset.

Ridge is the most consistent model across all phases, achieving R2=0.90 in
Config B (Phase 1) and R2=0.92 in the final pruned model (Phase 4).
"""

import numpy as np
from sklearn.linear_model import RidgeCV
from ..config import RIDGE_ALPHAS


class RidgeModel:
    """Wrapper around sklearn RidgeCV with a consistent fit/predict interface."""

    def __init__(self, alphas=None):
        """Initialise with candidate regularisation strengths.

        Args:
            alphas: List of alpha values to try (default: [0.01, 0.1, 1, 10, 100]).
                    RidgeCV will select the best alpha via internal LOOCV.
        """
        self.alphas = alphas or RIDGE_ALPHAS
        self.model = None

    def fit(self, X_train, y_train):
        """Train Ridge regression with automatic alpha selection.

        cv=None triggers sklearn's efficient analytical LOOCV formula for alpha
        selection, which is O(n) rather than O(n^2) like explicit LOOCV.

        Args:
            X_train: Training features (n_train, n_features).
            y_train: Training targets (n_train,).
        """
        self.model = RidgeCV(alphas=self.alphas, cv=None)  # cv=None uses efficient LOOCV
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Predict H2O2 yield rate for test samples.

        Args:
            X_test: Test features (n_test, n_features) or (n_features,) for single sample.

        Returns:
            np.ndarray: Predicted values (n_test,).
        """
        return self.model.predict(np.atleast_2d(X_test)).ravel()
