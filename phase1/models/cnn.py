"""
1D Convolutional Neural Network (CNN) Model
=============================================
Processes the raw 701-point OES spectrum directly using 1D convolutions,
without requiring PCA dimensionality reduction. The convolution kernels learn
local spectral patterns (e.g., emission line shapes, band structures).

Architecture:
    OES input (1, 701) -> Conv1d layers -> MaxPool -> AdaptiveAvgPool(1) -> Dropout -> Linear(1)

For Config C, the CNN supports a two-input architecture:
    - OES goes through conv layers -> pooled feature vector
    - Discharge params are concatenated to the pooled features before the output layer
    This allows the CNN to process spectral data with convolution while also
    using the scalar discharge parameters.

CNN Config C achieved R2 = 0.69 in Phase 1 (the only model where OES helped)
and R2 = 0.77 after Phase 2 tuning.
"""

import numpy as np
import torch
import torch.nn as nn
from ..config import CNN_CONFIG, RANDOM_SEED


class CNN1D(nn.Module):
    """1D CNN architecture for spectral regression with optional extra features.

    The conv layers process the OES spectrum (1D signal), then AdaptiveAvgPool
    compresses the spatial dimension to 1, producing a fixed-size feature vector
    regardless of input length. Optional extra features (discharge params for
    Config C) are concatenated before the final linear output layer.
    """

    def __init__(self, input_length, conv_channels, kernel_size, dropout, n_extra_features=0):
        """Build the 1D CNN architecture.

        Args:
            input_length: Length of OES spectrum (701 for our dataset).
            conv_channels: List of output channels per conv layer (e.g., [16, 32]).
            kernel_size: Convolution kernel width (e.g., 7 = 7 nm spectral window).
            dropout: Dropout probability before the output layer.
            n_extra_features: Number of extra scalar features to concatenate
                             (0 for Config A, 4 for Config C discharge params).
        """
        super().__init__()
        self.n_extra = n_extra_features

        # Build convolutional layers
        layers = []
        in_ch = 1  # single channel input (OES spectrum)
        for i, out_ch in enumerate(conv_channels):
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
            ]
            if i == 0:
                layers.append(nn.MaxPool1d(2))  # halve spatial dimension after first conv
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool1d(1))  # compress to single value per channel
        self.conv = nn.Sequential(*layers)

        # Output head: takes conv features (+ optional discharge params) -> scalar prediction
        fc_input = conv_channels[-1] + n_extra_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fc_input, 1),
        )

    def forward(self, x_oes, x_extra=None):
        """Forward pass through the CNN.

        Args:
            x_oes: OES spectrum tensor of shape (batch, 1, 701).
            x_extra: Optional discharge parameters tensor (batch, 4) for Config C.
                     None for Config A.

        Returns:
            Tensor of shape (batch,) — predicted H2O2 yield rates.
        """
        out = self.conv(x_oes)          # (batch, last_channel, 1) after adaptive pool
        out = out.squeeze(-1)           # (batch, last_channel)
        # Concatenate discharge features if provided (Config C two-input architecture)
        if x_extra is not None and self.n_extra > 0:
            out = torch.cat([out, x_extra], dim=-1)  # (batch, last_channel + 4)
        return self.head(out).squeeze(-1)  # (batch,)


class CNNModel:
    """Training wrapper for CNN1D with early stopping."""

    def __init__(self, config=None):
        """Initialise with CNN training configuration.

        Args:
            config: Dict with keys: conv_channels, kernel_size, dropout,
                    weight_decay, lr, max_epochs, patience (default: CNN_CONFIG).
        """
        self.config = config or CNN_CONFIG
        self.model = None

    def fit(self, X_train_oes, y_train, X_train_extra=None):
        """Train the CNN with Adam optimiser and early stopping.

        Reshapes the OES input from (N, 701) to (N, 1, 701) for Conv1d
        (which expects batch x channels x length). Uses the same early
        stopping mechanism as MLPModel.

        Args:
            X_train_oes: Training OES spectra as numpy array (n_train, 701).
            y_train: Training targets as numpy array (n_train,).
            X_train_extra: Optional discharge parameters (n_train, 4) for Config C.
                          None for Config A.
        """
        torch.manual_seed(RANDOM_SEED)
        cfg = self.config
        n_extra = X_train_extra.shape[1] if X_train_extra is not None else 0

        self.model = CNN1D(
            input_length=X_train_oes.shape[1],
            conv_channels=cfg["conv_channels"],
            kernel_size=cfg["kernel_size"],
            dropout=cfg["dropout"],
            n_extra_features=n_extra,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        criterion = nn.MSELoss()

        # Reshape OES: (N, 701) -> (N, 1, 701) for Conv1d input format
        X_oes_t = torch.tensor(X_train_oes, dtype=torch.float32).unsqueeze(1)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        X_ext_t = torch.tensor(X_train_extra, dtype=torch.float32) if X_train_extra is not None else None

        best_loss = np.inf
        patience_counter = 0
        best_state = None

        self.model.train()  # enable dropout during training
        for epoch in range(cfg["max_epochs"]):
            optimizer.zero_grad()
            pred = self.model(X_oes_t, X_ext_t)
            loss = criterion(pred, y_t)
            loss.backward()
            optimizer.step()

            # Early stopping with patience
            current_loss = loss.item()
            if current_loss < best_loss - 1e-6:
                best_loss = current_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    break

        # Restore best model state
        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self, X_test_oes, X_test_extra=None):
        """Predict in inference mode (dropout disabled).

        Args:
            X_test_oes: Test OES spectra (n_test, 701) or (701,).
            X_test_extra: Optional test discharge params (n_test, 4).

        Returns:
            np.ndarray: Predicted values (n_test,).
        """
        self.model.eval()  # disable dropout for inference
        with torch.no_grad():
            X_oes_t = torch.tensor(np.atleast_2d(X_test_oes), dtype=torch.float32).unsqueeze(1)
            X_ext_t = torch.tensor(np.atleast_2d(X_test_extra), dtype=torch.float32) if X_test_extra is not None else None
            return self.model(X_oes_t, X_ext_t).numpy().ravel()
