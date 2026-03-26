"""
Ultra-lightweight 1D CNN model (scikit-learn + numpy fallback).
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from utils.logger import get_logger

log = get_logger("cnn_model")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class Conv1DNet(nn.Module):
        def __init__(self, n_features: int, window: int):
            super().__init__()
            self.conv1 = nn.Conv1d(n_features, 32, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
            self.pool  = nn.AdaptiveAvgPool1d(8)
            self.fc    = nn.Linear(16 * 8, 3)
            self.relu  = nn.ReLU()

        def forward(self, x):
            # x: (B, W, F) → (B, F, W)
            x = x.permute(0, 2, 1)
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = x.reshape(x.shape[0], -1)
            return torch.softmax(self.fc(x), dim=-1)

    class CNNModel:
        def __init__(self, n_features: int = 15, window: int = 60, device: str = "cpu"):
            self.device = torch.device(device)
            self.net = Conv1DNet(n_features, window).to(self.device)
            self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
            self.crit = nn.CrossEntropyLoss()
            self._trained = False

        def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, batch: int = 256):
            n = len(X)
            for ep in range(epochs):
                idx = np.random.permutation(n)
                for s in range(0, n, batch):
                    bi = idx[s:s+batch]
                    t_x = torch.from_numpy(X[bi]).float().to(self.device)
                    t_y = torch.from_numpy(y[bi]).long().to(self.device)
                    self.opt.zero_grad()
                    self.crit(self.net(t_x), t_y).backward()
                    self.opt.step()
            self._trained = True

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            self.net.eval()
            with torch.no_grad():
                t = torch.from_numpy(X).float().to(self.device)
                return self.net(t).cpu().numpy()

        def save(self, path: str):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.net.state_dict(), path)

        def load(self, path: str):
            self.net.load_state_dict(torch.load(path, map_location=self.device))
            self._trained = True

        @property
        def is_trained(self): return self._trained

else:
    # Fallback to MLP when torch is unavailable
    from models.mlp import MLPModel as CNNModel
