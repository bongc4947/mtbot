"""
Training pipeline: load CSV → encode features → build targets → train → evaluate → save.
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
from utils.logger import get_logger
from utils.data_utils import load_csv, validate_dataset, detect_session_gaps
from features.encoder import encode_ohlcv, build_sequences
from training.labeler import build_outcome_labels
from interface.state_manager import get_state_manager, SystemState

log = get_logger("trainer")

# Target: next-bar direction
# 2=BUY if next close > current close + spread*1.5
# 0=SELL if next close < current close - spread*1.5
# 1=HOLD otherwise
SELL, HOLD, BUY = 0, 1, 2


def build_targets(df: pd.DataFrame, threshold_multiplier: float = 1.5) -> np.ndarray:
    """Legacy direction-threshold labeling — kept for backward compatibility."""
    close = df["close"].values.astype(np.float64)
    spread = df["spread"].values.astype(np.float64)

    # MT5 exports spread as integer points (e.g. 10 = 1 pip on 5-decimal broker).
    # close is in price units (e.g. 1.0850 for EURUSD, 150.0 for USDJPY).
    # Convert spread to the same price units as (close[i+1] - close[i]).
    if spread.mean() > 1.0:  # integer points — needs conversion
        close_med = np.median(close)
        if close_med > 500:       # XAUUSD, XAGUSD
            point_size = 0.01
        elif close_med > 10:      # USDJPY, CADJPY, GBPJPY ...
            point_size = 0.001
        else:                     # EURUSD, GBPUSD ... (5-decimal)
            point_size = 0.00001
        spread_price = spread * point_size
    else:
        spread_price = spread     # already in price units

    threshold = spread_price * threshold_multiplier
    targets = np.ones(len(close), dtype=np.int8) * HOLD
    next_ret = np.zeros(len(close))
    next_ret[:-1] = close[1:] - close[:-1]
    targets[next_ret > threshold] = BUY
    targets[next_ret < -threshold] = SELL

    # Zero out targets that span a session gap — the bar before a gap maps to an
    # overnight/weekend price move that we cannot trade, so force it to HOLD.
    is_gap = detect_session_gaps(df)
    pre_gap = np.zeros(len(df), dtype=bool)
    pre_gap[:-1] = is_gap[1:]   # bar i is pre-gap when bar i+1 starts a new session
    targets[pre_gap] = HOLD
    targets[is_gap]  = HOLD     # gap-open bar itself: previous close is stale

    return targets


def _balance_classes(X: np.ndarray, y: np.ndarray) -> tuple:
    """Oversample minority classes so BUY/SELL match HOLD count."""
    from sklearn.utils import resample
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X, y
    target_count = int(np.median(counts) * 2)
    parts_X, parts_y = [], []
    for cls in classes:
        mask = y == cls
        Xc, yc = X[mask], y[mask]
        if len(Xc) < target_count:
            Xc, yc = resample(Xc, yc, n_samples=target_count,
                              replace=True, random_state=42)
        parts_X.append(Xc)
        parts_y.append(yc)
    return np.concatenate(parts_X), np.concatenate(parts_y)


def train_symbol(symbol: str, raw_dir: str, model_manager,
                 min_samples: int = 5000, window: int = 60,
                 cfg: dict = None) -> dict:
    cfg = cfg or {}
    path = os.path.join(raw_dir, f"{symbol}.csv")
    if not os.path.exists(path):
        log.warning(f"[{symbol}] No CSV found at {path}")
        return {"symbol": symbol, "status": "no_data"}

    df = load_csv(path)
    if not validate_dataset(df, symbol, min_samples):
        return {"symbol": symbol, "status": "invalid_data"}

    log.info(f"[{symbol}] Encoding features ({len(df)} bars)...")

    # Update agent status
    sm = get_state_manager()
    sm.update_agent_status("Feature Encoder", "running", symbol)
    sm.update_agent_status("Labeler", "running", symbol)

    features = encode_ohlcv(df)

    # Use TP/SL outcome labels
    tcfg = cfg.get("trading", {})
    sl_pips  = tcfg.get("label_sl_pips",  10.0)
    tp_pips  = tcfg.get("label_tp_pips",  20.0)
    horizon  = tcfg.get("label_horizon",  20)
    targets  = build_outcome_labels(df, sl_pips=sl_pips, tp_pips=tp_pips, horizon=horizon)

    sm.update_agent_status("Feature Encoder", "completed", symbol)
    sm.update_agent_status("Labeler", "completed", symbol)

    X, y = build_sequences(features, targets, window=window)
    if len(X) < min_samples:
        log.warning(f"[{symbol}] Not enough sequences: {len(X)}")
        return {"symbol": symbol, "status": "insufficient_sequences"}

    valid = ~np.isnan(X).any(axis=(1, 2))
    X, y = X[valid], y[valid]

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    X_train_b, y_train_b = _balance_classes(X_train, y_train)
    classes, counts = np.unique(y_train_b, return_counts=True)
    dist = " ".join(f"{c}:{n}" for c, n in zip(classes, counts))
    log.info(f"[{symbol}] Training — {len(X_train_b)} train (balanced: {dist}), {len(X_val)} val")

    model = model_manager.get(symbol)
    if model is None:
        model = model_manager.load_or_create(symbol)

    sm.update_agent_status("N-HiTS",   "training", symbol)
    sm.update_agent_status("LightGBM", "training", symbol)
    sm.update_agent_status("MLP",      "training", symbol)

    try:
        if hasattr(model, "fit"):
            model.fit(X_train_b, y_train_b)
        else:
            _train_torch(model, X_train_b, y_train_b)
    except Exception as e:
        log.error(f"[{symbol}] Training failed: {e}")
        return {"symbol": symbol, "status": "training_error", "error": str(e)}

    sm.update_agent_status("N-HiTS",   "completed", symbol)
    sm.update_agent_status("LightGBM", "completed", symbol)
    sm.update_agent_status("MLP",      "completed", symbol)
    sm.update_agent_status("Ensemble", "running",   symbol)

    metrics = evaluate(model, X_val, y_val, symbol)
    model_manager.mark_trained(symbol)
    model_manager.save(symbol)

    sm.update_agent_status("Ensemble", "completed", symbol)

    wr  = metrics.get("win_rate", 0.0)
    exp = metrics.get("expectancy", 0.0)
    pf  = metrics.get("profit_factor", 1.0)
    log.info(f"[{symbol}] Done — win_rate={wr:.3f} expectancy={exp:+.3f} profit_factor={pf:.2f}")

    sm.model_info[symbol] = {
        "type":          "ensemble",
        "last_train":    datetime.now().strftime("%H:%M"),
        "win_rate":      wr,
        "expectancy":    exp,
        "profit_factor": pf,
    }

    # Set current report
    sm.set_current_report(
        f"Symbol: {symbol}\n"
        f"Win Rate: {wr:.1%}  Expectancy: {exp:+.3f}  Profit Factor: {pf:.2f}\n"
        f"Trade Count: {metrics.get('trade_count', 0)}  Confidence: {metrics.get('avg_confidence', 0):.3f}"
    )

    return {"symbol": symbol, "status": "ok", **metrics}


def _train_torch(model, X: np.ndarray, y: np.ndarray,
                 batch_size: int = 256, epochs: int = 10):
    n = len(X)
    for epoch in range(epochs):
        idx = np.random.permutation(n)
        total_loss = 0.0
        batches = 0
        for start in range(0, n, batch_size):
            bi = idx[start:start + batch_size]
            loss = model.train_batch(X[bi], y[bi])
            total_loss += loss
            batches += 1
        log.debug(f"  Epoch {epoch+1}/{epochs} loss={total_loss/max(batches,1):.4f}")


def evaluate(model, X_val: np.ndarray, y_val: np.ndarray, symbol: str) -> dict:
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_val)
        elif hasattr(model, "predict"):
            probs = model.predict(X_val)
        else:
            return {}

        col_idx = np.argmax(probs, axis=1)
        confidence = probs.max(axis=1)

        # Map column index → actual class label using sklearn's classes_ if available
        inner = getattr(model, "model", model)  # unwrap MLPModel wrapper
        if hasattr(inner, "classes_"):
            preds = inner.classes_[col_idx]
        else:
            preds = col_idx  # assume classes are 0,1,2

        # Log class distribution for diagnostics
        buy_pct = (preds == BUY).mean()
        sell_pct = (preds == SELL).mean()
        hold_pct = (preds == HOLD).mean()
        log.info(f"[{symbol}] Val preds — BUY={buy_pct:.1%} SELL={sell_pct:.1%} HOLD={hold_pct:.1%}")

        # Only score non-HOLD predictions
        mask = preds != HOLD
        if mask.sum() == 0:
            log.warning(f"[{symbol}] Model predicts HOLD for all validation samples")
            return {"win_rate": 0.0, "trade_count": 0, "avg_confidence": 0.0,
                    "expectancy": 0.0, "profit_factor": 0.0}

        p_trade  = preds[mask]
        y_trade  = y_val[mask]
        correct  = (p_trade == y_trade).sum()
        win_rate = correct / mask.sum()

        # ── Profitability simulation ─────────────────────────────────────────
        # Payoff per trade based on label agreement:
        #   +1.0 : correct direction and price cleared the threshold (BUY→BUY, SELL→SELL)
        #   -2.0 : reversed (BUY→SELL or SELL→BUY) — ate spread going both ways
        #   -0.5 : predicted directional but label was HOLD — price didn't move enough
        correct_dir  = (p_trade == y_trade) & (y_trade != HOLD)
        reversed_dir = (
            ((p_trade == BUY)  & (y_trade == SELL)) |
            ((p_trade == SELL) & (y_trade == BUY))
        )
        missed_move  = ~correct_dir & ~reversed_dir   # spread cost, no gain

        gross_win  = float(correct_dir.sum())   * 1.0
        gross_loss = float(reversed_dir.sum())  * 2.0 + float(missed_move.sum()) * 0.5
        n_trades   = mask.sum()
        expectancy    = (gross_win - gross_loss) / max(n_trades, 1)
        profit_factor = gross_win / max(gross_loss, 1e-10)

        log.info(
            f"[{symbol}] Profitability — expectancy={expectancy:+.3f} "
            f"profit_factor={profit_factor:.2f}  "
            f"win={correct_dir.sum()} rev={reversed_dir.sum()} missed={missed_move.sum()}"
        )

        return {
            "win_rate":      float(win_rate),
            "trade_count":   int(n_trades),
            "avg_confidence": float(confidence[mask].mean()),
            "hold_rate":     float(hold_pct),
            "expectancy":    float(expectancy),
            "profit_factor": float(profit_factor),
        }
    except Exception as e:
        log.error(f"[{symbol}] Evaluation error: {e}")
        return {}


def train_all_symbols(symbols: list, raw_dir: str, model_manager,
                      cfg: dict) -> list:
    min_samples = cfg.get("model", {}).get("min_train_samples", 5000)
    window = cfg.get("model", {}).get("input_window", 60)
    results = []
    for symbol in symbols:
        result = train_symbol(symbol, raw_dir, model_manager, min_samples, window, cfg=cfg)
        results.append(result)
    ok = sum(1 for r in results if r.get("status") == "ok")
    log.info(f"Training complete: {ok}/{len(symbols)} symbols trained successfully")
    return results
