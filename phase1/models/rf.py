"""
Random Forest Model
====================
Ensemble of decision trees, each trained on a bootstrap sample of the data
with random feature subsets. The final prediction is the average of all trees.
Uses moderate complexity (max_depth=3, min_samples_split=5) to prevent
overfitting on n=20 samples.

RF is later tuned in Phase 2 via Optuna, improving Config B from R2=0.38 to 0.75.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from ..config import RF_PARAMS, RANDOM_SEED


class RFModel:
    """Wrapper around sklearn RandomForestRegressor."""

    def __init__(self, params=None):
        """Initialise with RF hyperparameters.

        Args:
            params: Dict of RF parameters (default: RF_PARAMS with
                    n_estimators=100, max_depth=3, etc.). Can be overridden
                    with tuned params from Phase 2/3.
        """
        self.params = params or RF_PARAMS
        self.model = None

    def fit(self, X_train, y_train):
        """Train Random Forest with fixed parameters and random seed.

        Args:
            X_train: Training features (n_train, n_features).
            y_train: Training targets (n_train,).
        """
        self.model = RandomForestRegressor(
            **self.params,
            random_state=RANDOM_SEED,
        )
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Predict using the trained Random Forest (average of all tree predictions).

        Args:
            X_test: Test features (n_test, n_features) or (n_features,).

        Returns:
            np.ndarray: Predicted values (n_test,).
        """
        return self.model.predict(np.atleast_2d(X_test)).ravel()
