"""
AI Trailing Stop Loss and Take Profit.
- SL tightens as profit increases
- TP expands in strong trends, tightens on reversal
- Early exit on confidence drop or opposite signal
"""
import time
import numpy as np
from utils.logger import get_logger
from utils.symbol_info import effective_spread
from interface.state_manager import get_state_manager

log = get_logger("trailing_manager")


class TradeState:
    __slots__ = ["symbol", "ticket", "direction", "open_price", "lot",
                 "sl", "tp", "open_time", "last_confidence", "peak_profit", "pip_mult"]

    def __init__(self, symbol, ticket, direction, open_price, lot, sl, tp, pip_mult=0.0001):
        self.symbol = symbol
        self.ticket = ticket
        self.direction = direction  # 1=BUY, -1=SELL
        self.open_price = open_price
        self.lot = lot
        self.sl = sl
        self.tp = tp
        self.open_time = time.time()
        self.last_confidence = 0.0
        self.peak_profit = 0.0
        self.pip_mult = pip_mult  # price units per pip — derived from live tick data


class TrailingManager:
    def __init__(self, zmq_ctrl, cfg: dict):
        self.zmq = zmq_ctrl
        tcfg = cfg.get("trading", {})
        self._trades: dict[int, TradeState] = {}  # ticket → state
        self._lock_free = True  # single-threaded update

    def register(self, trade: TradeState):
        self._trades[trade.ticket] = trade

    def unregister(self, ticket: int):
        self._trades.pop(ticket, None)

    def replace_ticket(self, old_ticket: int, new_ticket: int):
        """Replace magic placeholder with real MT5 ticket."""
        state = self._trades.pop(old_ticket, None)
        if state is not None:
            state.ticket = new_ticket
            self._trades[new_ticket] = state

    def sync_positions(self, live_tickets: set):
        """Remove stale trades (magic placeholders < 10000 or not in MT5 positions)."""
        for ticket in list(self._trades.keys()):
            if ticket >= 10000 and ticket not in live_tickets:
                self._trades.pop(ticket, None)
                log.debug(f"Trailing: removed stale ticket {ticket}")

    def update_all(self, tick_buffers, inference_engine):
        """Called each cycle. Adjusts SL/TP and triggers early exits."""
        for ticket, state in list(self._trades.items()):
            buf = tick_buffers.get(state.symbol)
            if buf is None or buf.size < 5:
                continue
            latest = buf.latest()
            current_price = float((latest[0] + latest[1]) / 2.0)  # mid
            self._update_trade(state, current_price, buf, inference_engine)

    def _update_trade(self, state: TradeState, current_price: float,
                      buf, inference_engine):
        get_state_manager().update_trade_price(state.ticket, current_price)
        direction = state.direction
        open_p = state.open_price

        # Update pip_mult from latest tick if available (bid, ask, ea_spread_pips)
        latest = buf.latest()
        bid, ask, ea_pips = float(latest[0]), float(latest[1]), float(latest[2])
        price_spread = ask - bid
        if ea_pips > 0 and price_spread > 0:
            state.pip_mult = price_spread / ea_pips  # live-accurate for any symbol
        pip_mult = state.pip_mult

        profit_pips = (current_price - open_p) * direction / pip_mult

        # Update peak profit
        if profit_pips > state.peak_profit:
            state.peak_profit = profit_pips

        # AI trailing SL: tighten as profit grows
        new_sl = self._compute_trailing_sl(state, current_price, buf, bid, ask)
        new_tp = self._compute_trailing_tp(state, current_price, buf, bid, ask)

        min_move = pip_mult * 0.5  # half a pip — avoids spam, works for all symbol types
        sl_moved = abs(new_sl - state.sl) > min_move
        tp_moved = abs(new_tp - state.tp) > min_move

        if sl_moved or tp_moved:
            state.sl = new_sl
            state.tp = new_tp
            import math as _math
            prec = max(0, -int(_math.floor(_math.log10(pip_mult))) + 1) if pip_mult > 0 else 5
            self.zmq.send_trade(
                symbol=state.symbol,
                action="MODIFY",
                lot=state.lot,
                sl=round(new_sl, prec),
                tp=round(new_tp, prec),
                comment=f"trail_t{state.ticket}",
                magic=state.ticket,
                ticket=state.ticket,
            )

    def _compute_trailing_sl(self, state: TradeState, current_price: float,
                              buf, bid: float, ask: float) -> float:
        from interface.state_manager import get_state_manager
        raw = buf.get_last(20)
        if len(raw) < 5:
            return state.sl

        mids    = (raw[:, 0] + raw[:, 1]) / 2.0
        pip_mult = state.pip_mult
        vol = np.std(np.diff(mids)) / pip_mult  # volatility in pips

        # Minimum SL distance = max(mode floor, 2× current spread, vol-based)
        # The spread floor prevents crypto/index SL below broker's minimum stop level
        ea_pips_live = float(raw[-1, 2])            # spread in pips from EA
        ea_pips      = effective_spread(state.symbol, ea_pips_live)  # max(live, ref)
        spread_floor = max(5.0, ea_pips * 2.0)     # 2× spread, at least 5 pips

        mode = get_state_manager().runtime_config.get("mode", "balanced")
        mode_floor = {"scalp": 3.0, "balanced": 8.0, "swing": 15.0}.get(mode, 8.0)

        base_offset = max(mode_floor, spread_floor, vol * 1.5)

        # Tighten as profit grows (only in scalp/balanced; swing stays wide)
        if mode == "swing":
            tight_factor = 1.0
        else:
            profit_ratio = min(1.0, state.peak_profit / max(base_offset * 3, 1.0))
            tight_factor = 1.0 - profit_ratio * 0.4   # tighten up to 40%

        sl_offset = (base_offset * tight_factor + ea_pips) * pip_mult

        if state.direction == 1:  # BUY
            new_sl = bid - sl_offset
            return max(new_sl, state.sl)  # only move up
        else:  # SELL
            new_sl = ask + sl_offset
            return min(new_sl, state.sl)  # only move down

    def _compute_trailing_tp(self, state: TradeState, current_price: float,
                              buf, bid: float, ask: float) -> float:
        from interface.state_manager import get_state_manager
        raw = buf.get_last(20)
        if len(raw) < 5:
            return state.tp

        mids    = (raw[:, 0] + raw[:, 1]) / 2.0
        pip_mult = state.pip_mult
        rets    = np.diff(mids)
        trend_strength = abs(rets.sum()) / (np.abs(rets).sum() + 1e-10)
        vol     = np.std(rets) / pip_mult  # in pips
        ea_pips = effective_spread(state.symbol, float(raw[-1, 2]))

        mode = get_state_manager().runtime_config.get("mode", "balanced")
        # TP base: mode determines how far we let profits run
        if mode == "scalp":
            base_tp = max(3.0, ea_pips * 1.5)     # tight TP, fast cycle
        elif mode == "swing":
            base_tp = max(30.0, vol * 6.0)         # wide TP, let trends run
        else:  # balanced
            base_tp = max(10.0, vol * 3.0)

        expand_factor = 1.0 + trend_strength * 1.5  # up to 2.5× in strong trends
        tp_offset = base_tp * expand_factor * pip_mult

        if state.direction == 1:
            return ask + tp_offset
        else:
            return bid - tp_offset

    def should_early_exit(self, ticket: int, current_signal_action: int,
                          current_confidence: float) -> bool:
        """Return True if trade should be closed early."""
        from interface.state_manager import get_state_manager
        state = self._trades.get(ticket)
        if state is None:
            return False

        mode = get_state_manager().runtime_config.get("mode", "balanced")

        # Opposite signal with meaningful confidence — mode-gated
        opp_conf_gate = {"scalp": 0.55, "balanced": 0.65, "swing": 0.75}.get(mode, 0.65)
        if state.direction == 1 and current_signal_action == 0:
            if current_confidence >= opp_conf_gate:
                log.info(f"[{state.symbol}] Early exit: opposite signal "
                         f"conf={current_confidence:.2f} mode={mode}")
                state.last_confidence = current_confidence
                return True
        if state.direction == -1 and current_signal_action == 2:
            if current_confidence >= opp_conf_gate:
                log.info(f"[{state.symbol}] Early exit: opposite signal "
                         f"conf={current_confidence:.2f} mode={mode}")
                state.last_confidence = current_confidence
                return True

        # Confidence collapse: only exit if it drops >35% relative to entry peak
        # Never exit in swing mode on confidence alone — let trailing SL protect
        if mode != "swing" and state.last_confidence > 0:
            collapse_thr = state.last_confidence * 0.65   # 35% relative drop
            if current_confidence < collapse_thr and current_confidence < 0.50:
                log.info(f"[{state.symbol}] Early exit: confidence collapse "
                         f"{state.last_confidence:.2f}→{current_confidence:.2f}")
                state.last_confidence = current_confidence
                return True

        # Track peak confidence so collapse threshold self-adjusts upward
        if current_confidence > state.last_confidence:
            state.last_confidence = current_confidence
        return False
