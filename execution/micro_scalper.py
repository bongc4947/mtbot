"""
Micro-scalper: close trades at small profit, immediately re-enter if signal persists.
High-frequency compounding loop. Prioritized over swing holding.
"""
import time
from collections import deque
from utils.logger import get_logger
from utils.symbol_info import effective_spread
from interface.state_manager import get_state_manager

log = get_logger("micro_scalper")


class MicroScalper:
    def __init__(self, zmq_ctrl, trailing_manager, cfg: dict):
        self.zmq   = zmq_ctrl
        self.trail = trailing_manager
        tcfg = cfg.get("trading", {})
        self.target_pips   = tcfg.get("micro_scalp_profit_pips", 1.5)
        self.enabled       = tcfg.get("micro_scalp_enabled", True)
        self.max_hold_secs = tcfg.get("micro_scalp_max_hold_secs", 300)
        self._scalp_count   = 0
        self._scalp_ts:   deque = deque(maxlen=1000)
        self._timeout_ts: deque = deque(maxlen=200)

    def check_and_scalp(self, ticket: int, state, current_price: float,
                        ea_spread_pips: float, current_signal_action: int,
                        reentry_fn=None):
        if not self.enabled:
            return

        pip = state.pip_mult if state.pip_mult > 0 else 0.0001
        profit_pips = (current_price - state.open_price) * state.direction / pip
        hold_secs   = time.time() - state.open_time

        rc   = get_state_manager().runtime_config
        mode = rc.get("mode", "balanced")

        if mode == "scalp":
            base_target = max(self.target_pips * 0.7, 1.5)
            max_hold    = min(self.max_hold_secs, 120)
        elif mode == "swing":
            base_target = self.target_pips * 3.0
            max_hold    = self.max_hold_secs * 4
        else:
            base_target = self.target_pips
            max_hold    = self.max_hold_secs

        # Target must exceed the effective spread (max of live and reference).
        # Require at least spread + 1 pip margin so the trade is net positive after costs.
        eff_spread   = effective_spread(state.symbol, ea_spread_pips)
        spread_floor = eff_spread + 1.0
        target = max(base_target, spread_floor)

        if profit_pips >= target:
            self._do_close(ticket, state, reason="scalp", profit_pips=profit_pips)
            if reentry_fn and current_signal_action == (2 if state.direction == 1 else 0):
                reentry_fn()
            return

        if hold_secs >= max_hold:
            self._do_close(ticket, state, reason="timeout", profit_pips=profit_pips)
            return

    def _do_close(self, ticket: int, state, reason: str, profit_pips: float):
        self.zmq.send_trade(
            symbol=state.symbol,
            action="CLOSE",
            lot=state.lot,
            sl=0.0,
            tp=0.0,
            comment=f"{reason}_t{ticket}",
            magic=state.ticket,
            ticket=ticket,
        )
        self.trail.unregister(ticket)
        get_state_manager().register_trade_close(ticket, 0.0, reason=reason)

        if reason == "scalp":
            self._scalp_count += 1
            self._scalp_ts.append(time.time())
            t10 = sum(1 for t in self._scalp_ts if t > time.time() - 600)
            log.info(f"[{state.symbol}] Scalp close +{profit_pips:.1f}pip "
                     f"(total:{self._scalp_count} last-10min:{t10})")
        else:
            self._timeout_ts.append(time.time())
            log.info(f"[{state.symbol}] Timeout close after "
                     f"{(time.time() - state.open_time):.0f}s pips={profit_pips:.1f}")

    def run_cycle(self, tick_buffers, trailing_manager, signal_map: dict,
                  reentry_fns: dict):
        for ticket, state in list(trailing_manager._trades.items()):
            buf = tick_buffers.get(state.symbol)
            if buf is None or buf.size == 0:
                continue
            latest = buf.latest()
            bid, ask        = float(latest[0]), float(latest[1])
            ea_spread_pips  = float(latest[2])   # already in pip units from EA
            current_price   = (bid + ask) / 2.0
            sig             = signal_map.get(state.symbol)
            current_action  = sig.action if sig else 1
            reentry         = reentry_fns.get(state.symbol)
            self.check_and_scalp(ticket, state, current_price,
                                 ea_spread_pips, current_action, reentry)

    @property
    def scalp_count(self) -> int:
        return self._scalp_count
