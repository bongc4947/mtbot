"""
Dynamic position sizing: lot = base_lot * f(confidence, volatility, drawdown).
"""
import numpy as np
from utils.logger import get_logger

log = get_logger("position_sizer")


class PositionSizer:
    def __init__(self, cfg: dict):
        tcfg = cfg.get("trading", {})
        self.base_lot = tcfg.get("base_lot", 0.01)
        self.max_lot = tcfg.get("max_lot", 1.0)
        self.per_trade_risk = cfg.get("risk", {}).get("per_trade_risk_pct", 1.0) / 100.0

    def compute(self, confidence: float, volatility: float,
                current_drawdown_pct: float, balance: float = 10000.0) -> float:
        """
        confidence: [0..1]
        volatility: rolling std of log returns
        current_drawdown_pct: 0..100
        """
        # Scale up with confidence, down with drawdown and volatility
        conf_scale = max(0.5, min(2.0, (confidence - 0.5) * 4.0))

        vol_scale = 1.0
        if volatility > 0:
            vol_scale = max(0.3, min(1.5, 0.001 / (volatility + 1e-8)))

        dd_scale = max(0.2, 1.0 - (current_drawdown_pct / 10.0))

        lot = self.base_lot * conf_scale * vol_scale * dd_scale
        lot = round(max(self.base_lot, min(self.max_lot, lot)), 2)
        return lot
