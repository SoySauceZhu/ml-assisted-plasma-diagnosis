import numpy as np
import torch
import torch.nn as nn
from ..config import MLP_CONFIG, RANDOM_SEED


class MLPNet(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPModel:
    def __init__(self, config=None):
        self.config = config or MLP_CONFIG
        self.model = None

    def fit(self, X_train, y_train):
        torch.manual_seed(RANDOM_SEED)
        cfg = self.config
        input_dim = X_train.shape[1]
        self.model = MLPNet(input_dim, cfg["hidden_sizes"], cfg["dropout"])

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        criterion = nn.MSELoss()

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)

        best_loss = np.inf
        patience_counter = 0
        best_state = None

        self.model.train()
        for epoch in range(cfg["max_epochs"]):
            optimizer.zero_grad()
            pred = self.model(X_t)
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

    def predict(self, X_test):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(np.atleast_2d(X_test), dtype=torch.float32)
            return self.model(X_t).numpy().ravel()
