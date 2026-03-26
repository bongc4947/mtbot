"""
LightGBM model wrapper — tabular microstructure edge detection.
Flattens sequence features into a tabular representation and trains
a gradient-boosted tree for multiclass classification.
Designed for low-spec hardware (AMD A8): fast inference, low RAM.
"""
import numpy as np
import os
import joblib
from utils.logger import get_logger

log = get_logger("lgbm_model")

SELL, HOLD, BUY = 0, 1, 2


def _seq_to_tabular(x: np.ndarray) -> np.ndarray:
    """
    Convert sequence (N, W, F) → tabular features (N, 4*F).
    Uses last tick, 20-bar mean, 20-bar std, and trend delta.
    Keeps RAM and compute minimal for low-spec hardware.
    """
    # Last tick features
    last = x[:, -1, :]                    # (N, F)
    # Recent 20-bar stats
    recent = x[:, -20:, :]               # (N, 20, F)
    mu    = recent.mean(axis=1)           # (N, F)
    sigma = recent.std(axis=1) + 1e-8    # (N, F)
    # Short-term trend: last 5 bars vs previous 5 bars
    delta = x[:, -5:, :].mean(axis=1) - x[:, -10:-5, :].mean(axis=1)  # (N, F)
    return np.concatenate([last, mu, sigma, delta], axis=1).astype(np.float32)


class LGBMModel:
    def __init__(self, n_features: int = 15, window: int = 60):
        self.n_features = n_features
        self.window = window
        self.model = None
        self._trained = False
        self.classes_ = np.array([SELL, HOLD, BUY])

    def fit(self, x: np.ndarray, y: np.ndarray):
        try:
            import lightgbm as lgb
        except ImportError:
            log.warning("LightGBM not installed — using sklearn GBM fallback")
            self._fit_fallback(x, y)
            return

        flat = _seq_to_tabular(x)
        self.model = lgb.LGBMClassifier(
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=200,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=2,          # limit cores for low-spec hardware
        )
        self.model.fit(flat, y)
        self.classes_ = self.model.classes_
        # Strip auto-generated feature names so sklearn validation does not
        # warn on every predict_proba call (both fit and predict use numpy).
        if hasattr(self.model, "feature_names_in_"):
            del self.model.feature_names_in_
        self._trained = True
        log.info(f"LightGBM trained — classes: {self.classes_} "
                 f"feature_imp_top3: {sorted(self.model.feature_importances_, reverse=True)[:3]}")

    def _fit_fallback(self, x: np.ndarray, y: np.ndarray):
        from sklearn.ensemble import GradientBoostingClassifier
        flat = _seq_to_tabular(x)
        self.model = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        self.model.fit(flat, y)
        self.classes_ = self.model.classes_
        if hasattr(self.model, "feature_names_in_"):
            del self.model.feature_names_in_
        self._trained = True
        log.info("Sklearn GBM fallback trained")

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        flat = _seq_to_tabular(x)
        # Ensure no stale feature names on models loaded from disk
        if hasattr(self.model, "feature_names_in_"):
            del self.model.feature_names_in_
        raw = self.model.predict_proba(flat)   # (N, n_classes)
        # Ensure shape (N, 3) even if a class was absent during training
        if raw.shape[1] == 3:
            return raw
        out = np.ones((len(x), 3)) / 3.0
        for ci, cls in enumerate(self.classes_):
            if 0 <= cls <= 2:
                out[:, cls] = raw[:, ci]
        out /= out.sum(axis=1, keepdims=True)
        return out

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({"model": self.model, "classes_": self.classes_}, path)
        log.info(f"LGBM saved to {path}")

    def load(self, path: str):
        d = joblib.load(path)
        self.model = d["model"]
        self.classes_ = d["classes_"]
        self._trained = True
        log.info(f"LGBM loaded from {path}")

    @property
    def is_trained(self) -> bool:
        return self._trained
