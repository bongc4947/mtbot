"""
Trade outcome labeling: simulates BUY/SELL trades with fixed TP/SL.
Produces 3-class labels (SELL=0, HOLD=1, BUY=2) based on which direction
would hit TP before SL within a forward horizon of bars.

This replaces the direction-threshold labeling and captures realistic
trade profitability including multi-bar trajectories.
"""
import numpy as np
import pandas as pd
from utils.logger import get_logger

log = get_logger("labeler")

SELL, HOLD, BUY = 0, 1, 2


def _detect_point_size(closes: np.ndarray, spreads: np.ndarray) -> float:
    """Auto-detect pip/point size from close price magnitude."""
    if spreads.mean() <= 1.0:
        return 1.0  # already in price units
    close_med = float(np.median(closes))
    if close_med > 500:
        return 0.01      # XAUUSD
    elif close_med > 10:
        return 0.001     # USDJPY
    else:
        return 0.00001   # EURUSD etc.


def build_outcome_labels(
    df: pd.DataFrame,
    sl_pips:  float = 10.0,
    tp_pips:  float = 20.0,
    horizon:  int   = 20,
) -> np.ndarray:
    """
    For each bar i, simulate a BUY and a SELL trade:
    - BUY wins if high[j] >= entry + tp first (before low[j] <= entry - sl)
    - SELL wins if low[j] <= entry - tp first (before high[j] >= entry + sl)
    - HOLD if neither hits within `horizon` bars

    Requires columns: open, high, low, close, spread (standard OHLCV).
    Falls back gracefully if high/low missing (uses close as proxy).
    """
    n = len(df)
    targets = np.ones(n, dtype=np.int8) * HOLD

    closes  = df["close"].values.astype(np.float64)
    spreads = df["spread"].values.astype(np.float64)

    # Use high/low if available, else proxy with close +/- small buffer
    if "high" in df.columns and "low" in df.columns:
        highs = df["high"].values.astype(np.float64)
        lows  = df["low"].values.astype(np.float64)
    else:
        # Estimate HL from close with rolling window
        c = closes
        highs = np.maximum(c, np.roll(c, -1))   # very rough proxy
        lows  = np.minimum(c, np.roll(c, -1))
        log.warning("No H/L columns found — using close-based proxy (less accurate labels)")

    pip = _detect_point_size(closes, spreads)
    tp  = tp_pips * pip
    sl  = sl_pips * pip

    buy_won  = np.zeros(n, dtype=bool)
    sell_won = np.zeros(n, dtype=bool)
    buy_first  = np.full(n, horizon + 1, dtype=np.int32)
    sell_first = np.full(n, horizon + 1, dtype=np.int32)

    for i in range(n - horizon):
        entry = closes[i]
        buy_tp_price  = entry + tp
        buy_sl_price  = entry - sl
        sell_tp_price = entry - tp
        sell_sl_price = entry + sl

        fw_h = highs[i + 1: i + horizon + 1]
        fw_l = lows[i + 1:  i + horizon + 1]

        # BUY: TP = high >= buy_tp_price, SL = low <= buy_sl_price
        b_tp_mask = fw_h >= buy_tp_price
        b_sl_mask = fw_l <= buy_sl_price
        b_tp_hit  = np.argmax(b_tp_mask) if b_tp_mask.any() else horizon
        b_sl_hit  = np.argmax(b_sl_mask) if b_sl_mask.any() else horizon

        if b_tp_mask.any() and (not b_sl_mask.any() or b_tp_hit <= b_sl_hit):
            buy_won[i]  = True
            buy_first[i] = b_tp_hit

        # SELL: TP = low <= sell_tp_price, SL = high >= sell_sl_price
        s_tp_mask = fw_l <= sell_tp_price
        s_sl_mask = fw_h >= sell_sl_price
        s_tp_hit  = np.argmax(s_tp_mask) if s_tp_mask.any() else horizon
        s_sl_hit  = np.argmax(s_sl_mask) if s_sl_mask.any() else horizon

        if s_tp_mask.any() and (not s_sl_mask.any() or s_tp_hit <= s_sl_hit):
            sell_won[i]  = True
            sell_first[i] = s_tp_hit

    # Assign labels: both won → use earlier TP hit
    both  = buy_won & sell_won
    targets[buy_won  & ~sell_won] = BUY
    targets[sell_won & ~buy_won]  = SELL
    targets[both & (buy_first <= sell_first)] = BUY
    targets[both & (buy_first >  sell_first)] = SELL
    # HOLD is the default (already set)

    b = int(np.sum(targets == BUY))
    s = int(np.sum(targets == SELL))
    h = int(np.sum(targets == HOLD))
    log.info(f"Outcome labels: BUY={b} ({b/n:.1%}) SELL={s} ({s/n:.1%}) HOLD={h} ({h/n:.1%})")
    return targets
