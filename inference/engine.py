"""
Batched inference engine. Processes all symbols in a single pass per cycle.
Target latency: < 100ms for 30 symbols.
"""
import time
import numpy as np
from utils.logger import get_logger
from features.encoder import N_FEATURES
from utils.symbol_info import effective_spread
from interface.state_manager import get_state_manager

log = get_logger("inference_engine")

SELL, HOLD, BUY = 0, 1, 2
ACTION_NAMES = {SELL: "SELL", HOLD: "HOLD", BUY: "BUY"}

# Tick buffer column indices (matches tick_buffer.py TICK_FIELDS)
_BID, _ASK, _SPR, _VOL, _TS = 0, 1, 2, 3, 4


class Signal:
    __slots__ = ["symbol", "action", "confidence", "probs", "timestamp"]

    def __init__(self, symbol: str, action: int, confidence: float,
                 probs: np.ndarray):
        self.symbol = symbol
        self.action = action
        self.confidence = confidence
        self.probs = probs
        self.timestamp = time.time()

    @property
    def action_name(self) -> str:
        return ACTION_NAMES.get(self.action, "HOLD")

    def __repr__(self):
        return (f"Signal({self.symbol} {self.action_name} "
                f"conf={self.confidence:.3f})")


def _encode_tick_sequence(raw: np.ndarray, window: int) -> np.ndarray | None:
    """
    Convert raw tick array (N, 5) → feature matrix (window, N_FEATURES).

    Each row of the output is one tick's feature vector, matching the feature
    layout produced by encode_ohlcv() for OHLCV bars.  Features 10-14
    (candle structure) stay zero — tick data has no H/L.

    Returns None if not enough ticks.
    """
    n = len(raw)
    needed = window + 21   # extra for rolling_vol_20 warm-up
    if n < needed:
        return None

    mid     = (raw[:, _BID] + raw[:, _ASK]) / 2.0
    spreads = raw[:, _SPR]
    volumes = raw[:, _VOL]

    # ── Log returns ──────────────────────────────────────────────────────────
    log_ret = np.zeros(n)
    log_ret[1:] = np.log(np.maximum(mid[1:] / mid[:-1], 1e-10))

    feats = np.zeros((n, N_FEATURES), dtype=np.float64)

    # ── Features 0-4: returns and volatility ─────────────────────────────────
    # 0: log_return
    feats[:, 0] = log_ret
    # 1: lag-1 log_return
    feats[1:, 1] = log_ret[:-1]
    # 2: 5-tick rolling sum of log returns (momentum)
    cs = np.cumsum(log_ret)
    feats[5:, 2] = cs[5:] - cs[:-5]

    def _rstd(x, w):
        cs_  = np.cumsum(x)
        cs2_ = np.cumsum(x ** 2)
        out  = np.zeros_like(x)
        ex   = (cs_[w:] - cs_[:n - w]) / w
        ex2  = (cs2_[w:] - cs2_[:n - w]) / w
        out[w:] = np.sqrt(np.maximum(0.0, ex2 - ex ** 2))
        return out

    feats[:, 3] = _rstd(log_ret, 10)
    feats[:, 4] = _rstd(log_ret, 20)

    # ── Feature 5: candle body direction (matches encode_ohlcv training feature) ─
    # Training: (close - open) / hl_range — momentum direction of each bar.
    # Tick approximation: price move over a 5-tick window / 5-tick range.
    # This aligns with the training signal instead of the spread-position proxy.
    for i in range(5, n):
        lo = mid[i - 5:i + 1].min()
        hi = mid[i - 5:i + 1].max()
        rng = hi - lo + 1e-10
        feats[i, 5] = (mid[i] - mid[i - 5]) / rng   # direction over 5 ticks

    # ── Feature 6: price velocity (acceleration of log return) ───────────────
    feats[1:, 6] = np.diff(log_ret)

    # ── Features 7-8: spread ─────────────────────────────────────────────────
    sp_mean = spreads.mean()
    sp_std  = spreads.std() + 1e-10
    feats[:, 7] = (spreads - sp_mean) / sp_std
    feats[:, 8] = spreads / np.maximum(mid, 1e-10)

    # ── Feature 9: volume normalised ─────────────────────────────────────────
    vol_mean = volumes.mean() + 1e-10
    feats[:, 9] = volumes / vol_mean

    # ── Features 10-14: approximate candle structure from ticks ──────────────
    # These match the training features: hl_range, close_position, body_ratio,
    # upper_shadow, lower_shadow — computed over a 5-tick rolling window.
    for i in range(5, n):
        lo  = mid[i - 5:i + 1].min()
        hi  = mid[i - 5:i + 1].max()
        rng = hi - lo + 1e-10
        feats[i, 10] = rng / max(mid[i], 1e-10)                          # hl_range/price
        feats[i, 11] = (mid[i] - lo) / rng                               # close_position
        feats[i, 12] = abs(mid[i] - mid[i - 1]) / rng                    # body_ratio
        feats[i, 13] = (hi - max(mid[i], mid[i - 1])) / rng              # upper_shadow
        feats[i, 14] = (min(mid[i], mid[i - 1]) - lo) / rng              # lower_shadow

    np.clip(feats, -10.0, 10.0, out=feats)

    # Return last `window` rows
    return feats[-window:].astype(np.float32)


class InferenceEngine:
    def __init__(self, model_manager, tick_buffers, cfg: dict):
        self.models = model_manager
        self.ticks  = tick_buffers
        icfg  = cfg.get("inference", {})
        mcfg  = cfg.get("model", {})
        self.window               = mcfg.get("input_window", 60)
        self.confidence_threshold = icfg.get("confidence_threshold", 0.60)
        self.stack_threshold      = icfg.get("stack_confidence_threshold", 0.80)
        self._latency_log         = []
        self._cycle               = 0

    def run_batch(self, symbols: list) -> list[Signal]:
        """Run inference on all symbols, return list of Signal objects."""
        t0 = time.perf_counter()
        signals = []
        for symbol in symbols:
            sig = self._infer_symbol(symbol)
            if sig is not None:
                signals.append(sig)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._latency_log.append(elapsed_ms)
        self._cycle += 1

        # Log signal activity every 60 cycles (~30 s at 0.5 s cycle)
        if self._cycle % 60 == 0:
            non_hold = [s for s in signals if s.action != HOLD]
            log.info(
                f"[inference] cycle={self._cycle} symbols={len(signals)} "
                f"non-HOLD={len(non_hold)} latency={elapsed_ms:.1f}ms "
                f"conf_thr={self.confidence_threshold:.2f}"
            )
            if non_hold:
                for s in non_hold[:5]:   # show up to 5
                    log.info(f"  {s}")

        return signals

    def _infer_symbol(self, symbol: str) -> Signal | None:
        buf = self.ticks.get(symbol)
        needed = self.window + 21
        if buf is None or buf.size < needed:
            return None

        raw          = buf.get_last(needed)
        feat_window  = _encode_tick_sequence(raw, self.window)
        if feat_window is None:
            return None

        x     = feat_window[np.newaxis]               # (1, window, N_FEATURES)
        probs = self.models.predict_proba(symbol, x)[0]  # (3,)
        action     = int(np.argmax(probs))
        confidence = float(probs[action])

        # Boost confidence using directional margin:
        # When model clearly prefers BUY over SELL (or vice versa), the margin
        # between them reflects genuine directional conviction — reward it.
        if action != HOLD:
            opp = SELL if action == BUY else BUY
            directional_margin = float(probs[action] - probs[opp])
            # Also consider the margin over HOLD: model is confident AND directional
            hold_margin = float(probs[action] - probs[HOLD])
            # Weighted boost: directional edge matters more than beating HOLD
            boost = 0.35 * directional_margin + 0.15 * max(0.0, hold_margin)
            confidence = min(1.0, confidence + boost)

        # Spread guard for trade signals — use max(live, reference) to prevent
        # false "wide spread" passes when live feed momentarily reads zero.
        if action != HOLD:
            current_spread = effective_spread(symbol, float(buf.latest()[_SPR]))
            if buf.spread_too_wide(current_spread):
                return Signal(symbol, HOLD, float(probs[HOLD]), probs)

        return Signal(symbol, action, confidence, probs)

    def _live_confidence(self) -> float:
        """Read confidence threshold from runtime config (hot-reloadable)."""
        rc = get_state_manager().runtime_config
        return rc.get("confidence", self.confidence_threshold)

    def _live_mode(self) -> str:
        return get_state_manager().runtime_config.get("mode", "balanced")

    def filter_tradeable(self, signals: list[Signal]) -> list[Signal]:
        """Return non-HOLD signals above live confidence threshold."""
        thr = self._live_confidence()
        return [s for s in signals
                if s.action != HOLD and s.confidence >= thr]

    def filter_stackable(self, signals: list[Signal]) -> list[Signal]:
        """Return signals eligible for trade stacking (very high confidence)."""
        thr = self._live_confidence()
        return [s for s in signals
                if s.action != HOLD and s.confidence >= max(thr, self.stack_threshold)]

    def should_early_exit(self, ticket: int, signal_action: int,
                          confidence: float) -> bool:
        """True when model flips direction with high confidence."""
        return confidence >= self.stack_threshold

    @property
    def avg_latency_ms(self) -> float:
        if not self._latency_log:
            return 0.0
        return float(np.mean(self._latency_log[-50:]))
