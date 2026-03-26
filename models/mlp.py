"""
Lightweight MLP fallback model. Scikit-learn based — no GPU required.
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from utils.logger import get_logger

log = get_logger("mlp_model")


class MLPModel:
    def __init__(self, input_size: int = 60, n_features: int = 15,
                 hidden_layer_sizes=(64, 32)):
        self.input_size = input_size
        self.n_features = n_features
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
        )
        self._trained = False

    def _flatten(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(len(x), -1)

    def fit(self, x: np.ndarray, y: np.ndarray):
        flat = self._flatten(x)
        flat_scaled = self.scaler.fit_transform(flat)
        self.model.fit(flat_scaled, y)
        self._trained = True
        log.info(f"MLP trained — classes: {self.model.classes_}, iters: {self.model.n_iter_}")

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        flat = self._flatten(x)
        flat_scaled = self.scaler.transform(flat)
        return self.model.predict_proba(flat_scaled)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)
        log.info(f"MLP saved to {path}")

    def load(self, path: str):
        d = joblib.load(path)
        self.model = d["model"]
        self.scaler = d["scaler"]
        self._trained = True
        log.info(f"MLP loaded from {path}")

    @property
    def is_trained(self) -> bool:
        return self._trained
