"""
Central state store for MT5_Bot_mk2 dashboard.
Singleton — all modules share the same instance via get_state_manager().
"""
import threading
import queue
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────

class SystemState(Enum):
    IDLE     = "IDLE"
    TRAINING = "TRAINING"
    RUNNING  = "RUNNING"
    PAUSED   = "PAUSED"
    ERROR    = "ERROR"
    SHUTDOWN = "SHUTDOWN"


class Command(Enum):
    PAUSE             = "PAUSE"
    RESUME            = "RESUME"
    SHUTDOWN          = "SHUTDOWN"
    RETRAIN           = "RETRAIN"
    REFRESH_DATA      = "REFRESH_DATA"
    CLOSE_PROFITABLE  = "CLOSE_PROFITABLE"
    CLOSE_LOSING      = "CLOSE_LOSING"
    CLOSE_ALL         = "CLOSE_ALL"


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeInfo:
    ticket:        int
    symbol:        str
    direction:     str    # "BUY" / "SELL"
    open_price:    float
    lot:           float
    sl:            float
    tp:            float
    open_time:     float
    current_price: float = 0.0
    profit_pips:   float = 0.0


@dataclass
class ClosedTrade:
    symbol:     str
    direction:  str
    profit:     float
    pips:       float
    duration_s: float
    close_time: float
    close_reason: str = ""   # "tp", "sl", "manual", "scalp", "timeout", "early_exit"


@dataclass
class RejectedTrade:
    symbol:     str
    direction:  str
    reason:     str
    timestamp:  float


@dataclass
class LogEntry:
    timestamp: float
    level:     str
    name:      str
    message:   str


# ──────────────────────────────────────────────────────────────────────────────
# Singleton state manager
# ──────────────────────────────────────────────────────────────────────────────

class StateManager:
    _instance: "Optional[StateManager]" = None
    _class_lock = threading.Lock()

    def __new__(cls) -> "StateManager":
        with cls._class_lock:
            if cls._instance is None:
                obj = super().__new__(cls)
                obj._init_fields()
                cls._instance = obj
        return cls._instance

    def _init_fields(self):
        self._lock = threading.Lock()

        # System state
        self.state: SystemState = SystemState.IDLE
        self.error_message: str = ""

        # Connectivity
        self.zmq_connected:       bool            = False
        self.mt5_connected:       bool            = False
        self.last_tick_time:      Optional[float] = None
        self.last_heartbeat:      Optional[float] = None
        self.inference_latency_ms: float          = 0.0
        self.symbols_active:      int             = 0

        # Account / Performance
        self.balance:            float = 0.0
        self.equity:             float = 0.0
        self.daily_pnl:          float = 0.0
        self._session_start_bal: float = 0.0
        self.win_count:          int   = 0
        self.loss_count:         int   = 0
        self.drawdown_pct:       float = 0.0
        self._trade_ts:          deque = deque(maxlen=120)

        # Trades
        self.active_trades:   dict  = {}               # ticket → TradeInfo
        self.recent_closed:   deque = deque(maxlen=50) # ClosedTrade
        self.rejected_trades: deque = deque(maxlen=30) # RejectedTrade

        # Model info
        self.model_info:        dict = {}  # symbol → {type, last_train, win_rate}
        self.training_symbol:   str  = ""
        self.training_progress: int  = 0   # 0-100

        # Log stream
        self.log_buffer: deque = deque(maxlen=500)  # LogEntry

        # Control
        self.command_queue: queue.Queue  = queue.Queue()
        self.shutdown_flag: threading.Event = threading.Event()

        # TradingAgents-style agent status tracking
        self.agent_status: dict = {}   # name → {team, status, info, updated}
        self.current_report: str = ""  # Latest analysis/decision text
        self.signal_count: int = 0     # Total signals generated this session
        self.ensemble_weights: dict = {"nhits": 0.30, "lgbm": 0.40, "mlp": 0.30}
        self.trade_count_10min: int = 0  # trades in last 10 min

        # ── Runtime-editable config (hot-reload without restart) ──────────────
        self.runtime_config: dict = {
            "confidence":      0.90,   # inference confidence threshold
            "max_trades_10min": 30,    # max trades per 10-minute window
            "mode":            "balanced",  # "scalp" | "balanced" | "swing"
        }

        # Initialize default agent statuses
        self._init_agent_statuses()

    def _init_agent_statuses(self):
        defaults = [
            ("ZMQ Bridge",      "Data Team"),
            ("Tick Buffer",     "Data Team"),
            ("Data Exporter",   "Data Team"),
            ("Feature Encoder", "Analysis Team"),
            ("Labeler",         "Analysis Team"),
            ("Trainer",         "Analysis Team"),
            ("N-HiTS",          "Inference Team"),
            ("LightGBM",        "Inference Team"),
            ("MLP",             "Inference Team"),
            ("Ensemble",        "Inference Team"),
            ("Trade Stacker",   "Execution Team"),
            ("Micro Scalper",   "Execution Team"),
            ("Trailing Mgr",    "Execution Team"),
            ("Risk Manager",    "Risk Management"),
            ("Position Sizer",  "Risk Management"),
            ("Portfolio Mgr",   "Portfolio"),
        ]
        for name, team in defaults:
            self.agent_status[name] = {"team": team, "status": "idle", "info": ""}

    def update_agent_status(self, name: str, status: str, info: str = ""):
        if name in self.agent_status:
            self.agent_status[name]["status"] = status
            self.agent_status[name]["info"] = info

    def set_current_report(self, text: str):
        self.current_report = text

    # ── State ─────────────────────────────────────────────────────────────────

    def set_state(self, state: SystemState, error: str = ""):
        with self._lock:
            self.state = state
            self.error_message = error

    # ── Logging ───────────────────────────────────────────────────────────────

    def add_log(self, level: str, name: str, message: str):
        self.log_buffer.append(LogEntry(time.time(), level, name, message))

    # ── Account ───────────────────────────────────────────────────────────────

    def update_account(self, balance: float, equity: float):
        with self._lock:
            if self._session_start_bal == 0 and balance > 0:
                self._session_start_bal = balance
            self.balance = balance
            self.equity  = equity
            if self._session_start_bal > 0:
                self.daily_pnl = balance - self._session_start_bal

    # ── Trades ────────────────────────────────────────────────────────────────

    def register_trade_open(self, trade: TradeInfo):
        with self._lock:
            self.active_trades[trade.ticket] = trade
            self._trade_ts.append(time.time())

    def replace_trade_ticket(self, old_ticket: int, new_ticket: int):
        """Replace a magic-number placeholder with the confirmed MT5 ticket."""
        with self._lock:
            trade = self.active_trades.pop(old_ticket, None)
            if trade is not None:
                trade.ticket = new_ticket
                self.active_trades[new_ticket] = trade

    def register_trade_close(self, ticket: int, profit: float, pips: float = 0.0,
                              reason: str = ""):
        with self._lock:
            trade = self.active_trades.pop(ticket, None)
            if trade and pips == 0.0 and trade.current_price > 0 and trade.open_price > 0:
                d = 1 if trade.direction == "BUY" else -1
                price_move = (trade.current_price - trade.open_price) * d
                # Use profit sign when price unavailable; otherwise derive pips from price
                pips = price_move * 10000  # rough; trailing_manager uses pip_mult
            dur = (time.time() - trade.open_time) if trade else 0.0
            sym = trade.symbol if trade else "?"
            dirn = trade.direction if trade else "?"
            self.recent_closed.appendleft(
                ClosedTrade(sym, dirn, profit, pips, dur, time.time(), reason)
            )
            if profit >= 0:
                self.win_count += 1
            else:
                self.loss_count += 1

    def update_runtime_param(self, key: str, value):
        """Hot-update a runtime config value (thread-safe)."""
        _MODES = ("scalp", "balanced", "swing")
        with self._lock:
            if key == "confidence":
                self.runtime_config["confidence"] = max(0.50, min(1.0, float(value)))
            elif key == "max_trades_10min":
                self.runtime_config["max_trades_10min"] = max(1, min(200, int(value)))
            elif key == "mode":
                if value in _MODES:
                    self.runtime_config["mode"] = value
                else:
                    # cycle through modes if passed as index
                    try:
                        self.runtime_config["mode"] = _MODES[int(value) % len(_MODES)]
                    except (ValueError, TypeError):
                        pass

    def register_trade_reject(self, symbol: str, direction: str, reason: str):
        with self._lock:
            self.rejected_trades.appendleft(
                RejectedTrade(symbol, direction, reason, time.time())
            )

    def update_trade_price(self, ticket: int, current_price: float):
        trade = self.active_trades.get(ticket)
        if trade and current_price > 0:
            trade.current_price = current_price
            d = 1 if trade.direction == "BUY" else -1
            trade.profit_pips = (current_price - trade.open_price) * d * 10000

    # ── Computed ──────────────────────────────────────────────────────────────

    @property
    def win_rate(self) -> float:
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0.0

    @property
    def total_trades(self) -> int:
        return self.win_count + self.loss_count

    @property
    def trades_per_minute(self) -> float:
        cutoff = time.time() - 60
        return float(sum(1 for t in self._trade_ts if t > cutoff))

    @property
    def trades_10min(self) -> int:
        cutoff = time.time() - 600
        return sum(1 for t in self._trade_ts if t > cutoff)

    @property
    def profit_factor(self) -> float:
        closed = list(self.recent_closed)
        wins   = sum(abs(t.profit) for t in closed if t.profit > 0)
        losses = sum(abs(t.profit) for t in closed if t.profit < 0)
        return wins / losses if losses > 0 else (wins if wins > 0 else 1.0)


def get_state_manager() -> StateManager:
    """Returns the singleton StateManager instance."""
    return StateManager()
