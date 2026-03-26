"""
Risk manager: daily drawdown gate, exposure caps, frequency control.
Auto-shuts down trading if limits breached.
"""
import time
import threading
from utils.logger import get_logger

log = get_logger("risk_manager")


class RiskManager:
    def __init__(self, cfg: dict):
        rcfg = cfg.get("risk", {})
        tcfg = cfg.get("trading", {})
        self.daily_loss_limit_pct = rcfg.get("daily_loss_limit_pct", 5.0)
        self.shutdown_dd_pct = rcfg.get("max_drawdown_shutdown_pct", 10.0)
        self.max_symbol_exposure = tcfg.get("max_symbol_exposure_pct", 3.0) / 100.0
        self.max_total_trades = tcfg.get("max_total_trades", 50)
        self.cooldown = rcfg.get("cooldown_after_loss_seconds", 60)

        self._balance_start = 0.0
        self._balance_now = 0.0
        self._daily_start = 0.0
        self._open_trades: dict[str, list] = {}  # symbol → list of ticket_ids
        self._last_loss_time: dict[str, float] = {}
        self._shutdown = False
        self._lock = threading.Lock()

    def initialize(self, balance: float):
        with self._lock:
            self._balance_start = balance
            self._balance_now = balance
            self._daily_start = balance

    def update_balance(self, balance: float):
        with self._lock:
            self._balance_now = balance
            dd = (self._balance_start - balance) / max(self._balance_start, 1.0) * 100
            if dd >= self.shutdown_dd_pct:
                log.critical(f"SHUTDOWN: drawdown {dd:.2f}% >= {self.shutdown_dd_pct}%")
                self._shutdown = True

    def reset_daily(self):
        with self._lock:
            self._daily_start = self._balance_now

    @property
    def is_shutdown(self) -> bool:
        return self._shutdown

    @property
    def daily_drawdown_pct(self) -> float:
        if self._daily_start == 0:
            return 0.0
        return (self._daily_start - self._balance_now) / self._daily_start * 100

    @property
    def total_drawdown_pct(self) -> float:
        if self._balance_start == 0:
            return 0.0
        return (self._balance_start - self._balance_now) / self._balance_start * 100

    def can_trade(self, symbol: str) -> tuple[bool, str]:
        with self._lock:
            if self._shutdown:
                return False, "shutdown"
            if self.daily_drawdown_pct >= self.daily_loss_limit_pct:
                return False, f"daily_loss_limit ({self.daily_drawdown_pct:.1f}%)"
            total_open = sum(len(v) for v in self._open_trades.values())
            if total_open >= self.max_total_trades:
                return False, f"max_trades ({total_open})"
            # Cooldown check
            last_loss = self._last_loss_time.get(symbol, 0)
            if time.time() - last_loss < self.cooldown:
                return False, f"cooldown ({symbol})"
        return True, ""

    def register_open(self, symbol: str, ticket: int):
        with self._lock:
            if symbol not in self._open_trades:
                self._open_trades[symbol] = []
            if ticket not in self._open_trades[symbol]:
                self._open_trades[symbol].append(ticket)

    def replace_ticket(self, symbol: str, magic: int, real_ticket: int):
        """Replace the magic-number placeholder with the confirmed MT5 ticket."""
        with self._lock:
            trades = self._open_trades.get(symbol, [])
            if magic in trades:
                idx = trades.index(magic)
                trades[idx] = real_ticket
            elif real_ticket not in trades:
                # Fallback: just register the real ticket if magic was never found
                trades.append(real_ticket)

    def sync_positions(self, open_positions: list):
        """
        Reconcile open trades against the live MT5 position list.
        Removes any tickets Python thinks are open but MT5 does not have.
        Adds any tickets MT5 has that Python doesn't know about.
        open_positions: list of dicts with keys: symbol, ticket, magic
        """
        with self._lock:
            live_by_symbol: dict[str, set] = {}
            for p in open_positions:
                sym = p.get("symbol", "")
                tkt = int(p.get("ticket", 0))
                if sym and tkt:
                    live_by_symbol.setdefault(sym, set()).add(tkt)

            # Remove stale entries (closed in MT5 but Python doesn't know yet)
            for sym, tickets in list(self._open_trades.items()):
                live = live_by_symbol.get(sym, set())
                stale = [t for t in tickets if t not in live and t >= 10000]
                for t in stale:
                    tickets.remove(t)
                    log.info(f"[{sym}] Reconciled: removed stale ticket {t}")

            # Add any positions MT5 has that Python missed
            for sym, live_tickets in live_by_symbol.items():
                known = set(self._open_trades.get(sym, []))
                for tkt in live_tickets:
                    if tkt not in known:
                        self._open_trades.setdefault(sym, []).append(tkt)
                        log.info(f"[{sym}] Reconciled: added missing ticket {tkt}")

    def register_close(self, symbol: str, ticket: int, profit: float):
        with self._lock:
            trades = self._open_trades.get(symbol, [])
            if ticket in trades:
                trades.remove(ticket)
            if profit < 0:
                self._last_loss_time[symbol] = time.time()

    def open_count(self, symbol: str = None) -> int:
        with self._lock:
            if symbol:
                return len(self._open_trades.get(symbol, []))
            return sum(len(v) for v in self._open_trades.values())
