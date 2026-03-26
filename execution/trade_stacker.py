"""
Trade Stacker: opens multiple simultaneous trades per symbol
when confidence >= stack_threshold (default 0.97).
"""
import math
import time
from collections import deque
from utils.logger import get_logger
from utils.symbol_info import effective_spread, effective_pip
from execution.trailing_manager import TradeState
from interface.state_manager import get_state_manager, TradeInfo

log = get_logger("trade_stacker")


class TradeStacker:
    def __init__(self, zmq_ctrl, risk_manager, trailing_manager, position_sizer, cfg: dict):
        self.zmq = zmq_ctrl
        self.risk = risk_manager
        self.trail = trailing_manager
        self.sizer = position_sizer
        tcfg = cfg.get("trading", {})
        self.max_per_symbol   = tcfg.get("max_trades_per_symbol", 5)
        self.stack_enabled    = tcfg.get("trade_stacking_enabled", True)
        self.stack_threshold  = cfg.get("inference", {}).get("stack_confidence_threshold", 0.97)
        self._magic_counter   = 20250001
        self._trade_ts: deque = deque(maxlen=500)  # timestamps for 10-min gate

    def _next_magic(self) -> int:
        self._magic_counter += 1
        return self._magic_counter

    def execute_signal(self, signal, tick_buf, balance: float,
                       drawdown_pct: float) -> int:
        """
        Execute a single trade. Returns ticket placeholder (-1 if skipped).
        stack_mode: if True, allows multiple trades for the same symbol.
        """
        symbol = signal.symbol
        action = signal.action_name  # "BUY" or "SELL"
        confidence = signal.confidence

        ok, reason = self.risk.can_trade(symbol)
        if not ok:
            log.debug(f"[{symbol}] Trade blocked: {reason}")
            get_state_manager().register_trade_reject(symbol, action, reason)
            return -1

        # ── Runtime config gates ─────────────────────────────────────────────
        sm = get_state_manager()
        rc = sm.runtime_config
        mode = rc.get("mode", "balanced")
        max_10min = rc.get("max_trades_10min", 30)

        now_ts = time.time()
        trades_10min = sum(1 for t in self._trade_ts if t > now_ts - 600)
        if trades_10min >= max_10min:
            log.debug(f"[{symbol}] Blocked: trades_10min={trades_10min}/{max_10min}")
            sm.register_trade_reject(symbol, action, f"10min_limit({trades_10min})")
            return -1

        open_count = self.risk.open_count(symbol)

        # Mode-based stacking:
        # scalp    → max 1 trade, no stacking (fast in/out)
        # balanced → confidence-based stacking (default)
        # swing    → stacking enabled, hold longer (trailing manager governs hold)
        if mode == "scalp":
            n_trades = 1 if open_count < self.max_per_symbol else 0
        elif self.stack_enabled and confidence >= self.stack_threshold:
            if confidence >= 0.99:
                n_stack = 6
            elif confidence >= 0.98:
                n_stack = 4
            elif confidence >= 0.975:
                n_stack = 3
            else:
                n_stack = 2
            n_trades = min(n_stack, self.max_per_symbol - open_count)
        else:
            n_trades = 1 if open_count < self.max_per_symbol else 0

        if n_trades <= 0:
            return -1

        latest = tick_buf.latest()
        bid   = float(latest[0])
        ask   = float(latest[1])
        spread_pips = float(latest[2])
        current_price = (bid + ask) / 2.0
        price_spread  = ask - bid

        # Volatility for position sizing
        raw = tick_buf.get_last(20)
        if len(raw) > 1:
            mids = (raw[:, 0] + raw[:, 1]) / 2.0
            vol = float(mids.std() / mids.mean()) if mids.mean() > 0 else 0.001
        else:
            vol = 0.001

        lot = self.sizer.compute(confidence, vol, drawdown_pct, balance)

        # ── Pip size: live tick data, with reference fallback ─────────────────────
        # Primary: pip = price_spread / ea_spread_pips  (works for any symbol)
        # Fallback: reference pip from symbol_info (Market Watch 20260325)
        if spread_pips > 0 and price_spread > 0:
            pip = price_spread / spread_pips
        else:
            pip = effective_pip(symbol, 0.0)  # reference pip from symbol_info

        # Effective spread = max(live, reference) — prevents SL below broker minimum
        # when live spread is temporarily zero (tick gap) or abnormally low.
        spread_in_pips = effective_spread(symbol, spread_pips)

        # SL/TP floor in pips — whichever is largest:
        #   • spread × multiplier  (broker minimum is typically 3-5× spread)
        #   • pip-count floor
        #   • 0.20% of current price as price-level floor (universal for crypto/indices)
        price_pct_pips = (current_price * 0.0020) / pip   # 0.20% in pip units
        sl_pips = max(35.0, spread_in_pips * 5, price_pct_pips)
        tp_pips = max(55.0, spread_in_pips * 8, price_pct_pips * 1.5)

        # Round to tick precision (1 sub-pip decimal beyond the pip):
        #   pip=0.0001 → precision=5  (EURUSD)
        #   pip=0.01   → precision=3  (USDJPY)
        #   pip=0.10   → precision=2  (GOLD, CrudeOIL, US_500)
        #   pip=10.0   → precision=0  (US_30)
        precision = max(0, -int(math.floor(math.log10(pip))) + 1) if pip > 0 else 5

        for i in range(n_trades):
            magic = self._next_magic()
            # Anchor SL/TP to bid/ask so MT5 validation always passes:
            # BUY  → SL below bid, TP above ask
            # SELL → SL above ask, TP below bid
            if action == "BUY":
                sl = bid - sl_pips * pip
                tp = ask + tp_pips * pip
                direction = 1
            else:
                sl = ask + sl_pips * pip
                tp = bid - tp_pips * pip
                direction = -1
            self.zmq.send_trade(symbol, action, lot,
                                round(sl, precision), round(tp, precision),
                                comment=f"mk2_{'stack' if i>0 else 'base'}",
                                magic=magic)

            # Register in risk + trailing (ticket=magic placeholder until confirmed)
            self.risk.register_open(symbol, magic)
            ts = TradeState(symbol, magic, direction, current_price, lot, sl, tp, pip_mult=pip)
            self.trail.register(ts)
            get_state_manager().register_trade_open(TradeInfo(
                ticket=magic, symbol=symbol, direction=action,
                open_price=current_price, lot=lot, sl=sl, tp=tp,
                open_time=time.time(),
            ))
            log.info(f"[{symbol}] {action} lot={lot} conf={confidence:.3f} "
                     f"sl={sl:.5f} tp={tp:.5f} magic={magic} mode={mode}")
            get_state_manager().signal_count += 1
            self._trade_ts.append(time.time())

        return n_trades
