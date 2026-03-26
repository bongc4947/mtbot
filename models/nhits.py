"""
N-HiTS (Neural Hierarchical Interpolation for Time Series).
Lightweight CPU-friendly implementation using pure PyTorch or numpy fallback.
"""
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ------------------------------------------------------------------
# Numpy-only fallback (ultra-lightweight)
# ------------------------------------------------------------------

class NHiTSNumpy:
    """Single-layer N-HiTS approximation using MLP (numpy only)."""

    def __init__(self, input_size: int, n_features: int, hidden: int = 64):
        self.input_size = input_size
        self.n_features = n_features
        flat = input_size * n_features
        self.W1 = np.random.randn(flat, hidden).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = np.random.randn(hidden, 3).astype(np.float32) * 0.1  # [up, flat, down]
        self.b2 = np.zeros(3, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (batch, window, features)
        b = x.shape[0]
        flat = x.reshape(b, -1)
        h = np.maximum(0, flat @ self.W1 + self.b1)  # ReLU
        logits = h @ self.W2 + self.b2
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)  # softmax

    def get_weights(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def set_weights(self, d: dict):
        self.W1 = d["W1"]
        self.b1 = d["b1"]
        self.W2 = d["W2"]
        self.b2 = d["b2"]


# ------------------------------------------------------------------
# PyTorch N-HiTS block
# ------------------------------------------------------------------

if TORCH_AVAILABLE:
    class NHiTSBlock(nn.Module):
        def __init__(self, input_size: int, n_features: int,
                     hidden: int = 64, n_stacks: int = 2):
            super().__init__()
            flat = input_size * n_features
            layers = []
            in_dim = flat
            for _ in range(n_stacks):
                layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
                in_dim = hidden
            layers.append(nn.Linear(hidden, 3))  # BUY, HOLD, SELL
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            # x: (batch, window, features)
            b = x.shape[0]
            out = self.net(x.reshape(b, -1))
            return torch.softmax(out, dim=-1)

    class NHiTSTorch:
        def __init__(self, input_size: int, n_features: int,
                     hidden: int = 64, n_stacks: int = 2, device: str = "cpu"):
            self.device = torch.device(device)
            self.model = NHiTSBlock(input_size, n_features, hidden, n_stacks).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            self.criterion = nn.CrossEntropyLoss()

        def predict(self, x: np.ndarray) -> np.ndarray:
            self.model.eval()
            with torch.no_grad():
                t = torch.from_numpy(x).float().to(self.device)
                probs = self.model(t)
                return probs.cpu().numpy()

        def train_batch(self, x: np.ndarray, y: np.ndarray) -> float:
            self.model.train()
            t_x = torch.from_numpy(x).float().to(self.device)
            t_y = torch.from_numpy(y).long().to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(t_x)
            loss = self.criterion(logits, t_y)
            loss.backward()
            self.optimizer.step()
            return float(loss.item())

        def save(self, path: str):
            torch.save(self.model.state_dict(), path)

        def load(self, path: str):
            self.model.load_state_dict(torch.load(path, map_location=self.device))


def build_nhits(input_size: int, n_features: int, hidden: int = 64,
                device: str = "cpu"):
    """Factory — returns PyTorch model if available, else numpy fallback."""
    if TORCH_AVAILABLE:
        return NHiTSTorch(input_size, n_features, hidden, device=device)
    return NHiTSNumpy(input_size, n_features, hidden)
