"""
Vectorized feature encoder. No TA-Lib or indicator libraries.
All features derived from raw OHLCV + spread.

Fixes applied:
  - 5-bar return (feature 2): was incorrectly a growing cumsum — now a proper
    rolling 5-bar sum using cumsum differences.
  - Session gap handling: gap returns are zeroed before rolling computations
    to prevent weekend/overnight spikes from inflating volatility features.
    Bars within invalid_window=22 bars after a gap are marked NaN so the
    trainer's existing NaN filter removes them from training sequences.
"""
import numpy as np
import pandas as pd
from utils.logger import get_logger

log = get_logger("encoder")

FEATURE_NAMES = [
    "log_return",        # 0
    "log_return_2",      # 1  (lag 1)
    "log_return_5",      # 2  (5-bar rolling sum — FIXED)
    "rolling_vol_10",    # 3
    "rolling_vol_20",    # 4
    "tick_imbalance",    # 5
    "price_velocity",    # 6
    "spread_zscore",     # 7
    "spread_norm",       # 8
    "volume_norm",       # 9
    "high_low_range",    # 10
    "close_position",    # 11 (close-low)/(high-low)
    "body_ratio",        # 12 |close-open|/(high-low)
    "upper_shadow",      # 13
    "lower_shadow",      # 14
]

N_FEATURES = len(FEATURE_NAMES)

# Bars after a session gap whose rolling features are invalid and must be NaN
_GAP_INVALID_WINDOW = 22   # covers rolling_vol_20 (window=20)


def encode_ohlcv(df: pd.DataFrame, spread_window: int = 100) -> np.ndarray:
    """
    Encode a DataFrame of OHLCV+spread into a float32 feature matrix.
    Returns shape (N, N_FEATURES).

    Bars within _GAP_INVALID_WINDOW positions after a session gap are returned
    as NaN rows so the training pipeline can filter them out.
    """
    close  = df["close"].values.astype(np.float64)
    open_  = df["open"].values.astype(np.float64)
    high   = df["high"].values.astype(np.float64)
    low    = df["low"].values.astype(np.float64)
    volume = df["tick_volume"].values.astype(np.float64)
    spread = df["spread"].values.astype(np.float64)

    n   = len(close)
    out = np.zeros((n, N_FEATURES), dtype=np.float64)

    # ── Detect session gaps ─────────────────────────────────────────────────
    from utils.data_utils import detect_session_gaps
    is_gap = detect_session_gaps(df, gap_minutes=2.0)  # True = starts new session

    # ── Log returns ─────────────────────────────────────────────────────────
    log_ret = np.zeros(n)
    log_ret[1:] = np.log(close[1:] / np.maximum(close[:-1], 1e-10))

    # Zero gap returns BEFORE computing any rolling statistic so that
    # weekend/overnight spikes don't inflate rolling vol or cumulative sums.
    log_ret[is_gap] = 0.0

    out[:, 0] = log_ret

    # Lag-1 log return
    out[1:, 1] = log_ret[:-1]

    # 5-bar rolling sum (FIXED: was growing cumsum, now correct rolling window)
    cs = np.cumsum(log_ret)
    out[5:, 2] = cs[5:] - cs[:-5]

    # ── Rolling volatility ──────────────────────────────────────────────────
    out[:, 3] = _rolling_std(log_ret, 10)
    out[:, 4] = _rolling_std(log_ret, 20)

    # ── Candle-structure features (no rolling window — safe across gaps) ────
    hl_range = high - low + 1e-10
    out[:, 5]  = (close - open_) / hl_range                        # tick imbalance
    out[1:, 6] = np.diff(log_ret)                                   # price velocity
    out[:, 10] = hl_range / np.maximum(close, 1e-10)               # hl range / price
    out[:, 11] = (close - low)  / hl_range                         # close position
    out[:, 12] = np.abs(close - open_) / hl_range                  # body ratio
    out[:, 13] = (high - np.maximum(close, open_)) / hl_range      # upper shadow
    out[:, 14] = (np.minimum(close, open_) - low) / hl_range       # lower shadow

    # ── Spread stats (rolling) ───────────────────────────────────────────────
    sp_mean = _rolling_mean(spread, spread_window)
    sp_std  = _rolling_std(spread, spread_window) + 1e-10
    out[:, 7] = (spread - sp_mean) / sp_std
    out[:, 8] = spread / np.maximum(close, 1e-10)

    # ── Volume normalised ────────────────────────────────────────────────────
    vol_mean = _rolling_mean(volume, 20) + 1e-10
    out[:, 9] = volume / vol_mean

    # ── Clip and clean numeric artefacts ────────────────────────────────────
    np.clip(out, -10.0, 10.0, out=out)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Mark post-gap window as NaN for trainer to filter ───────────────────
    # Rolling features within _GAP_INVALID_WINDOW bars after a gap are
    # unreliable even after zeroing gap returns; mark them NaN so the trainer's
    # existing `~np.isnan(X).any(axis=(1,2))` filter removes those sequences.
    gap_indices = np.where(is_gap)[0]
    for gi in gap_indices:
        end = min(n, gi + _GAP_INVALID_WINDOW)
        out[gi:end] = np.nan

    return out.astype(np.float32)


def encode_tick_window(bids: np.ndarray, asks: np.ndarray,
                       spreads: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """
    Encode a window of raw ticks into features.  All arrays shape (W,).
    Returns shape (N_FEATURES,) for the latest tick.
    """
    mid = (bids + asks) / 2.0
    n   = len(mid)
    if n < 2:
        return np.zeros(N_FEATURES, dtype=np.float32)

    log_ret = np.log(mid[1:] / np.maximum(mid[:-1], 1e-10))

    lr    = log_ret[-1]              if len(log_ret) > 0 else 0.0
    lr2   = log_ret[-2]              if len(log_ret) > 1 else 0.0
    lr5   = log_ret[-5:].sum()       if len(log_ret) >= 5 else log_ret.sum()
    v10   = log_ret[-10:].std()      if len(log_ret) >= 10 else log_ret.std()
    v20   = log_ret[-20:].std()      if len(log_ret) >= 20 else log_ret.std()
    rng   = mid.max() - mid.min() + 1e-10
    t_imb = (mid[-1] - mid[0]) / rng
    vel   = lr - lr2
    sp    = spreads[-1]
    sp_z  = (sp - spreads.mean()) / (spreads.std() + 1e-10)
    sp_n  = sp / max(mid[-1], 1e-10)
    vol_n = volumes[-1] / (volumes.mean() + 1e-10)

    feat = np.array([lr, lr2, lr5, v10, v20, t_imb, vel,
                     sp_z, sp_n, vol_n, 0, 0, 0, 0, 0], dtype=np.float32)
    return np.clip(feat, -10.0, 10.0)


# ── Rolling helpers ───────────────────────────────────────────────────────────

def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(arr)
    for i in range(window, len(arr)):
        out[i] = arr[i - window:i].std()
    return out


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(arr)
    cs  = np.cumsum(arr)
    out[window:] = (cs[window:] - cs[:-window]) / window
    out[:window] = arr[:window].mean() if window > 0 else 0
    return out


# ── Sequence builder ──────────────────────────────────────────────────────────

def build_sequences(features: np.ndarray, targets: np.ndarray,
                    window: int = 60) -> tuple:
    """
    Convert flat feature matrix to (N, window, F) sequences.
    Sequences that contain any NaN (from gap windows) are implicitly
    filtered by the trainer's existing NaN check.
    """
    n = len(features) - window
    if n <= 0:
        return np.empty((0, window, features.shape[1])), np.empty(0)
    X = np.lib.stride_tricks.sliding_window_view(
        features[:-1], (window, features.shape[1])
    )
    X = X[:, 0, :, :]          # (N, window, F)
    y = targets[window:]
    return X.astype(np.float32), y
