"""
Multi-Layer Perceptron (MLP) Model
====================================
Fully-connected neural network implemented in PyTorch. Architecture:
    Input -> [Linear -> ReLU -> Dropout] x n_hidden_layers -> Linear(1) -> Output

Default: 2 hidden layers [32, 16] with 40% dropout. Trained with Adam optimiser
and MSE loss, with early stopping based on training loss patience.

Note: MLP severely overfitted in Phase 1 (Config C R2 = -1.13), which was
partially resolved by Phase 2 tuning (R2 = 0.37) and fully resolved by
Phase 3 feature engineering (R2 = 0.81).
"""

import numpy as np
import torch
import torch.nn as nn
from ..config import MLP_CONFIG, RANDOM_SEED


class MLPNet(nn.Module):
    """PyTorch neural network architecture for MLP regression.

    Builds a sequential network: for each hidden size, adds Linear -> ReLU -> Dropout.
    The final layer is Linear(last_hidden, 1) for scalar regression output.
    """

    def __init__(self, input_dim, hidden_sizes, dropout):
        """Build the MLP architecture.

        Args:
            input_dim: Number of input features (varies by config: 11 for PCA, 4 for discharge, 15 for combined).
            hidden_sizes: List of hidden layer sizes (e.g., [32, 16] = two hidden layers).
            dropout: Dropout probability applied after each hidden layer for regularisation.
        """
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))  # output layer: single scalar prediction
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x: Input tensor (batch_size, input_dim).

        Returns:
            Tensor of shape (batch_size,) — predicted H2O2 yield rates.
        """
        return self.net(x).squeeze(-1)  # squeeze removes the trailing dimension from (batch, 1)


class MLPModel:
    """Training wrapper for MLPNet with early stopping."""

    def __init__(self, config=None):
        """Initialise with training configuration.

        Args:
            config: Dict with keys: hidden_sizes, dropout, weight_decay, lr,
                    max_epochs, patience (default: MLP_CONFIG).
        """
        self.config = config or MLP_CONFIG
        self.model = None

    def fit(self, X_train, y_train):
        """Train the MLP with Adam optimiser and early stopping.

        Training loop:
          1. Forward pass: compute predictions
          2. Compute MSE loss
          3. Backward pass: compute gradients
          4. Update weights via Adam (with L2 weight_decay regularisation)
          5. Early stopping: if loss doesn't improve for 'patience' epochs,
             stop and restore the best model state

        Args:
            X_train: Training features as numpy array (n_train, n_features).
            y_train: Training targets as numpy array (n_train,).
        """
        torch.manual_seed(RANDOM_SEED)
        cfg = self.config
        input_dim = X_train.shape[1]
        self.model = MLPNet(input_dim, cfg["hidden_sizes"], cfg["dropout"])

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        criterion = nn.MSELoss()

        # Convert numpy arrays to PyTorch tensors
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)

        best_loss = np.inf
        patience_counter = 0
        best_state = None

        self.model.train()  # enable dropout during training
        for epoch in range(cfg["max_epochs"]):
            optimizer.zero_grad()
            pred = self.model(X_t)
            loss = criterion(pred, y_t)
            loss.backward()
            optimizer.step()

            # Early stopping: track best loss and restore best state
            current_loss = loss.item()
            if current_loss < best_loss - 1e-6:
                best_loss = current_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    break  # stop training, no improvement for 'patience' epochs

        # Restore best model state (the one with lowest training loss)
        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self, X_test):
        """Predict H2O2 yield rate in inference mode (dropout disabled).

        Args:
            X_test: Test features as numpy array (n_test, n_features) or (n_features,).

        Returns:
            np.ndarray: Predicted values (n_test,).
        """
        self.model.eval()  # disable dropout for inference
        with torch.no_grad():  # disable gradient computation for efficiency
            X_t = torch.tensor(np.atleast_2d(X_test), dtype=torch.float32)
            return self.model(X_t).numpy().ravel()
