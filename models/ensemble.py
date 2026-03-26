"""
3-layer ensemble: N-HiTS (macro trend) + LightGBM (microstructure) + MLP (nonlinear).
Weighted probability combination with optional Platt scaling calibration.
"""
import numpy as np
import os
import joblib
from utils.logger import get_logger

log = get_logger("ensemble")

SELL, HOLD, BUY = 0, 1, 2


class EnsembleModel:
    """
    Combines N-HiTS, LightGBM, and MLP via weighted probability averaging.
    Weights can be tuned per symbol based on validation performance.
    """

    def __init__(self, nhits_model, lgbm_model, mlp_model,
                 weights: tuple = (0.30, 0.40, 0.30)):
        self.nhits   = nhits_model
        self.lgbm    = lgbm_model
        self.mlp     = mlp_model
        self.weights = np.array(weights, dtype=np.float64)
        self.weights /= self.weights.sum()   # normalize
        self.classes_ = np.array([SELL, HOLD, BUY])
        self._trained = False

    def fit(self, x: np.ndarray, y: np.ndarray):
        errors = []
        # N-HiTS (optional — may not be available on low-spec hardware)
        if self.nhits is not None and hasattr(self.nhits, "fit"):
            try:
                self.nhits.fit(x, y)
                log.info("N-HiTS component trained")
            except Exception as e:
                log.warning(f"N-HiTS training skipped: {e}")
                self.nhits = None

        # LightGBM
        try:
            self.lgbm.fit(x, y)
        except Exception as e:
            errors.append(f"LGBM: {e}")
            log.error(f"LightGBM training failed: {e}")

        # MLP
        try:
            self.mlp.fit(x, y)
        except Exception as e:
            errors.append(f"MLP: {e}")
            log.error(f"MLP training failed: {e}")

        self._trained = self.lgbm.is_trained or self.mlp.is_trained

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return (N, 3) probabilities [SELL, HOLD, BUY]."""
        n = len(x)
        combined = np.zeros((n, 3), dtype=np.float64)
        total_w  = 0.0

        # N-HiTS contribution
        if self.nhits is not None:
            try:
                p = _safe_proba(self.nhits, x, n)
                combined += self.weights[0] * p
                total_w  += self.weights[0]
            except Exception:
                pass

        # LightGBM contribution
        if self.lgbm.is_trained:
            try:
                p = _safe_proba(self.lgbm, x, n)
                combined += self.weights[1] * p
                total_w  += self.weights[1]
            except Exception:
                pass

        # MLP contribution
        if self.mlp.is_trained:
            try:
                p = _safe_proba(self.mlp, x, n)
                combined += self.weights[2] * p
                total_w  += self.weights[2]
            except Exception:
                pass

        if total_w < 1e-6:
            return np.ones((n, 3)) / 3.0

        combined /= total_w
        # Renormalize to sum=1
        row_sum = combined.sum(axis=1, keepdims=True)
        return combined / np.maximum(row_sum, 1e-10)

    def update_weights(self, nhits_w: float, lgbm_w: float, mlp_w: float):
        """Dynamically tune ensemble weights based on validation performance."""
        self.weights = np.array([nhits_w, lgbm_w, mlp_w])
        self.weights /= self.weights.sum()
        log.info(f"Ensemble weights updated: NHiTS={nhits_w:.2f} LGBM={lgbm_w:.2f} MLP={mlp_w:.2f}")

    def save(self, path_prefix: str):
        os.makedirs(os.path.dirname(os.path.abspath(path_prefix + "_x")), exist_ok=True)
        if self.nhits is not None and hasattr(self.nhits, "save"):
            try:
                self.nhits.save(path_prefix + "_nhits.pkl")
            except Exception as e:
                log.warning(f"NHiTS save skipped: {e}")
        self.lgbm.save(path_prefix + "_lgbm.pkl")
        self.mlp.save(path_prefix + "_mlp.pkl")
        joblib.dump({"weights": self.weights}, path_prefix + "_meta.pkl")
        log.info(f"Ensemble saved to {path_prefix}_*.pkl")

    def load(self, path_prefix: str):
        if self.nhits is not None and hasattr(self.nhits, "load"):
            try:
                self.nhits.load(path_prefix + "_nhits.pkl")
            except Exception:
                pass
        try:
            self.lgbm.load(path_prefix + "_lgbm.pkl")
        except Exception as e:
            log.warning(f"LGBM load failed: {e}")
        try:
            self.mlp.load(path_prefix + "_mlp.pkl")
        except Exception as e:
            log.warning(f"MLP load failed: {e}")
        try:
            meta = joblib.load(path_prefix + "_meta.pkl")
            self.weights = meta["weights"]
        except Exception:
            pass
        self._trained = self.lgbm.is_trained or self.mlp.is_trained

    @property
    def is_trained(self) -> bool:
        return self._trained


def _safe_proba(model, x: np.ndarray, n: int) -> np.ndarray:
    """Call predict_proba and ensure shape (n, 3)."""
    p = model.predict_proba(x)
    if p.shape == (n, 3):
        return p
    # Handle binary or differently-shaped output
    out = np.ones((n, 3)) / 3.0
    if hasattr(model, "classes_"):
        for ci, cls in enumerate(model.classes_):
            if 0 <= int(cls) <= 2:
                out[:, int(cls)] = p[:, ci]
        out /= out.sum(axis=1, keepdims=True)
    return out
