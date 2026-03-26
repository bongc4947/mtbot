"""
Model manager: load, save, auto-select ensemble models per symbol.
Uses EnsembleModel (N-HiTS + LightGBM + MLP) as the primary model type.
"""
import os
import time
import numpy as np
from utils.logger import get_logger

log = get_logger("model_manager")

SELL, HOLD, BUY = 0, 1, 2


class ModelManager:
    def __init__(self, cfg: dict, accel=None):
        self.cfg    = cfg
        self.accel  = accel
        mcfg = cfg.get("model", {})
        self.primary_type      = mcfg.get("primary", "ensemble")
        self.input_window      = mcfg.get("input_window", 60)
        self.model_dir         = mcfg.get("model_dir", "data/cache/models")
        self.retrain_interval  = mcfg.get("retrain_interval_hours", 4) * 3600
        self.n_features        = 15   # from encoder.N_FEATURES

        icfg = cfg.get("ensemble", {})
        self.ens_weights = (
            icfg.get("nhits_weight", 0.30),
            icfg.get("lgbm_weight",  0.40),
            icfg.get("mlp_weight",   0.30),
        )

        self._models: dict       = {}   # symbol → EnsembleModel
        self._last_trained: dict = {}
        os.makedirs(self.model_dir, exist_ok=True)

    def _ensemble_prefix(self, symbol: str) -> str:
        return os.path.join(self.model_dir, f"{symbol}_ensemble")

    def _build_ensemble(self):
        from models.nhits    import build_nhits, TORCH_AVAILABLE
        from models.lgbm_model import LGBMModel
        from models.mlp      import MLPModel
        from models.ensemble import EnsembleModel

        # N-HiTS (optional — skip if PyTorch unavailable)
        nhits = None
        if TORCH_AVAILABLE:
            device = "cpu"
            if self.accel and self.accel.use_gpu:
                device = "dml"
            try:
                nhits = build_nhits(self.input_window, self.n_features,
                                    hidden=64, device=device)
            except Exception as e:
                log.warning(f"N-HiTS unavailable: {e}")

        lgbm = LGBMModel(self.n_features, self.input_window)
        mlp  = MLPModel(self.input_window, self.n_features)
        return EnsembleModel(nhits, lgbm, mlp, weights=self.ens_weights)

    def load_or_create(self, symbol: str):
        prefix = self._ensemble_prefix(symbol)
        model  = self._build_ensemble()
        # Try loading existing components
        meta_path = prefix + "_meta.pkl"
        if os.path.exists(meta_path):
            try:
                model.load(prefix)
                log.info(f"[{symbol}] Ensemble loaded from {prefix}_*.pkl")
                self._models[symbol] = model
                self._last_trained[symbol] = os.path.getmtime(meta_path)
                return model
            except Exception as e:
                log.warning(f"[{symbol}] Failed to load ensemble: {e} — creating fresh")
        self._models[symbol] = model
        return model

    def save(self, symbol: str):
        model = self._models.get(symbol)
        if model is None:
            return
        prefix = self._ensemble_prefix(symbol)
        try:
            model.save(prefix)
        except Exception as e:
            log.error(f"[{symbol}] Save failed: {e}")

    def get(self, symbol: str):
        return self._models.get(symbol)

    def needs_retrain(self, symbol: str) -> bool:
        last = self._last_trained.get(symbol, 0)
        return (time.time() - last) > self.retrain_interval

    def mark_trained(self, symbol: str):
        self._last_trained[symbol] = time.time()

    def predict_proba(self, symbol: str, x: np.ndarray) -> np.ndarray:
        """Return (batch, 3) probabilities [SELL, HOLD, BUY]."""
        model = self._models.get(symbol)
        if model is None or not model.is_trained:
            return np.ones((len(x), 3)) / 3.0
        try:
            return model.predict_proba(x)
        except Exception as e:
            log.error(f"[{symbol}] Predict error: {e}")
            return np.ones((len(x), 3)) / 3.0
