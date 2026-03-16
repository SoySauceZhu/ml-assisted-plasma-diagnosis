"""
XGBoost Model
==============
Gradient boosting via XGBoost with conservative hyperparameters to mitigate
overfitting on the small dataset (n=20). Uses very shallow trees (max_depth=2),
few estimators (50), slow learning rate (0.05), and heavy L1+L2 regularisation
(reg_alpha=10, reg_lambda=10).

Note: XGBoost collapsed to near-trivial predictions across all configs in Phase 1,
suggesting gradient boosting is not well-suited for this extremely small sample size.
"""

import numpy as np
from xgboost import XGBRegressor
from ..config import XGBOOST_PARAMS, RANDOM_SEED


class XGBoostModel:
    """Wrapper around XGBRegressor with fixed conservative hyperparameters."""

    def __init__(self, params=None):
        """Initialise with XGBoost parameters.

        Args:
            params: Dict of XGBoost hyperparameters (default: XGBOOST_PARAMS).
        """
        self.params = params or XGBOOST_PARAMS
        self.model = None

    def fit(self, X_train, y_train):
        """Train XGBoost regressor with fixed parameters.

        Args:
            X_train: Training features (n_train, n_features).
            y_train: Training targets (n_train,).
        """
        self.model = XGBRegressor(
            **self.params,
            random_state=RANDOM_SEED,
            verbosity=0,  # suppress XGBoost output
        )
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Predict using the trained XGBoost model.

        Args:
            X_test: Test features (n_test, n_features) or (n_features,).

        Returns:
            np.ndarray: Predicted values (n_test,).
        """
        return self.model.predict(np.atleast_2d(X_test)).ravel()
