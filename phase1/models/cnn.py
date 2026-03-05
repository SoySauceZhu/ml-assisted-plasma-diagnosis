import numpy as np
import torch
import torch.nn as nn
from ..config import CNN_CONFIG, RANDOM_SEED


class CNN1D(nn.Module):
    def __init__(self, input_length, conv_channels, kernel_size, dropout, n_extra_features=0):
        super().__init__()
        self.n_extra = n_extra_features

        layers = []
        in_ch = 1
        for i, out_ch in enumerate(conv_channels):
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
            ]
            if i == 0:
                layers.append(nn.MaxPool1d(2))
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.conv = nn.Sequential(*layers)

        fc_input = conv_channels[-1] + n_extra_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fc_input, 1),
        )

    def forward(self, x_oes, x_extra=None):
        # x_oes: (batch, 1, seq_len)
        out = self.conv(x_oes)          # (batch, C, 1)
        out = out.squeeze(-1)           # (batch, C)
        if x_extra is not None and self.n_extra > 0:
            out = torch.cat([out, x_extra], dim=-1)
        return self.head(out).squeeze(-1)


class CNNModel:
    def __init__(self, config=None):
        self.config = config or CNN_CONFIG
        self.model = None

    def fit(self, X_train_oes, y_train, X_train_extra=None):
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

        X_oes_t = torch.tensor(X_train_oes, dtype=torch.float32).unsqueeze(1)  # (N, 1, 701)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        X_ext_t = torch.tensor(X_train_extra, dtype=torch.float32) if X_train_extra is not None else None

        best_loss = np.inf
        patience_counter = 0
        best_state = None

        self.model.train()
        for epoch in range(cfg["max_epochs"]):
            optimizer.zero_grad()
            pred = self.model(X_oes_t, X_ext_t)
            loss = criterion(pred, y_t)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            if current_loss < best_loss - 1e-6:
                best_loss = current_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self, X_test_oes, X_test_extra=None):
        self.model.eval()
        with torch.no_grad():
            X_oes_t = torch.tensor(np.atleast_2d(X_test_oes), dtype=torch.float32).unsqueeze(1)
            X_ext_t = torch.tensor(np.atleast_2d(X_test_extra), dtype=torch.float32) if X_test_extra is not None else None
            return self.model(X_oes_t, X_ext_t).numpy().ravel()
