"""
File-based IPC between the engine process and the dashboard window process.

Engine  →  writes state.json   →  Dashboard window reads & renders
Dashboard window  →  writes cmds.json  →  Engine reads & processes
"""
import json
import os
import time
import tempfile
from typing import Optional

from interface.state_manager import LogEntry, SystemState


# ── Path helpers ──────────────────────────────────────────────────────────────

def get_ipc_paths() -> tuple:
    base = os.path.join(tempfile.gettempdir(), "mt5_bot_mk2_ipc")
    os.makedirs(base, exist_ok=True)
    return (
        os.path.join(base, "state.json"),
        os.path.join(base, "cmds.json"),
    )


# ── State serialiser (engine → file) ─────────────────────────────────────────

def write_state(sm, path: str):
    """Atomically serialise StateManager to JSON."""
    try:
        logs = []
        for e in list(sm.log_buffer)[-150:]:
            logs.append({"ts": e.timestamp, "lvl": e.level,
                         "name": e.name, "msg": e.message})

        trades = []
        for ticket, t in list(sm.active_trades.items()):
            trades.append({
                "ticket": t.ticket, "symbol": t.symbol,
                "direction": t.direction, "open_price": t.open_price,
                "lot": t.lot, "sl": t.sl, "tp": t.tp,
                "open_time": t.open_time, "current_price": t.current_price,
                "profit_pips": t.profit_pips,
            })

        closed = []
        for t in list(sm.recent_closed)[:50]:
            closed.append({
                "symbol": t.symbol, "direction": t.direction,
                "profit": t.profit, "pips": t.pips,
                "duration_s": t.duration_s, "close_time": t.close_time,
                "close_reason": getattr(t, "close_reason", ""),
            })

        rejected = []
        for r in list(sm.rejected_trades)[:30]:
            rejected.append({
                "symbol": r.symbol, "direction": r.direction,
                "reason": r.reason, "ts": r.timestamp,
            })

        data = {
            "ts":               time.time(),
            "state":            sm.state.value,
            "error":            sm.error_message,
            "zmq":              sm.zmq_connected,
            "mt5":              sm.mt5_connected,
            "last_tick":        sm.last_tick_time,
            "latency_ms":       sm.inference_latency_ms,
            "symbols":          sm.symbols_active,
            "balance":          sm.balance,
            "equity":           sm.equity,
            "daily_pnl":        sm.daily_pnl,
            "win_count":        sm.win_count,
            "loss_count":       sm.loss_count,
            "drawdown_pct":     sm.drawdown_pct,
            "trades_per_min":   sm.trades_per_minute,
            "trades_10min":     sm.trades_10min,
            "profit_factor":    sm.profit_factor,
            "active_trades":    trades,
            "recent_closed":    closed,
            "rejected_trades":  rejected,
            "model_info":       dict(sm.model_info),
            "training_symbol":  sm.training_symbol,
            "training_progress": sm.training_progress,
            "logs":             logs,
            # TradingAgents-style fields
            "agent_status":     dict(sm.agent_status),
            "current_report":   sm.current_report,
            "signal_count":     sm.signal_count,
            "ensemble_weights": dict(sm.ensemble_weights),
            "runtime_config":   dict(sm.runtime_config),
        }

        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)  # atomic on Windows
    except Exception:
        pass


# ── State deserialiser (file → dashboard window) ──────────────────────────────

class StateSnapshot:
    """
    Duck-typed equivalent of StateManager built from a JSON dict.
    All fields accessed by metrics_view.py are present.
    """

    def __init__(self, d: dict):
        try:
            self.state = SystemState(d.get("state", "IDLE"))
        except ValueError:
            self.state = SystemState.IDLE

        self.error_message        = d.get("error", "")
        self.zmq_connected        = d.get("zmq", False)
        self.mt5_connected        = d.get("mt5", False)
        self.last_tick_time       = d.get("last_tick")
        self.inference_latency_ms = d.get("latency_ms", 0.0)
        self.symbols_active       = d.get("symbols", 0)
        self.balance              = d.get("balance", 0.0)
        self.equity               = d.get("equity", 0.0)
        self.daily_pnl            = d.get("daily_pnl", 0.0)
        self.win_count            = d.get("win_count", 0)
        self.loss_count           = d.get("loss_count", 0)
        self.drawdown_pct         = d.get("drawdown_pct", 0.0)
        self.training_symbol      = d.get("training_symbol", "")
        self.training_progress    = d.get("training_progress", 0)
        self.model_info           = d.get("model_info", {})
        self._tpm                 = d.get("trades_per_min", 0.0)
        self._pf                  = d.get("profit_factor", 1.0)
        self._trades_10min        = d.get("trades_10min", 0)

        # TradingAgents-style fields
        self.agent_status    = d.get("agent_status", {})
        self.current_report  = d.get("current_report", "")
        self.signal_count    = d.get("signal_count", 0)
        self.ensemble_weights = d.get("ensemble_weights",
                                      {"nhits": 0.30, "lgbm": 0.40, "mlp": 0.30})
        self.runtime_config   = d.get("runtime_config",
                                      {"confidence": 0.90, "max_trades_10min": 30,
                                       "mode": "balanced"})

        # Reconstruct trade/closed lists (duck-typed via SimpleNamespace)
        self.active_trades = {}
        for t in d.get("active_trades", []):
            obj = _Obj(t)
            self.active_trades[obj.ticket] = obj

        self.recent_closed   = [_Obj(c) for c in d.get("recent_closed", [])]
        self.rejected_trades = [_Obj(r) for r in d.get("rejected_trades", [])]

        self.log_buffer = [
            LogEntry(e["ts"], e["lvl"], e["name"], e["msg"])
            for e in d.get("logs", [])
        ]

    @property
    def win_rate(self) -> float:
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0.0

    @property
    def total_trades(self) -> int:
        return self.win_count + self.loss_count

    @property
    def trades_per_minute(self) -> float:
        return self._tpm

    @property
    def trades_10min(self) -> int:
        return self._trades_10min

    @property
    def profit_factor(self) -> float:
        return self._pf


class _Obj:
    """Simple attribute container from a dict."""
    def __init__(self, d: dict):
        self.__dict__.update(d)


# ── Command bridge (dashboard → engine) ──────────────────────────────────────

def write_command(cmd_str: str, path: str):
    """Append a command string to the commands file."""
    try:
        cmds = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cmds = json.load(f)
        cmds.append(cmd_str)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cmds, f)
    except Exception:
        pass


def read_commands(path: str) -> list:
    """Read and delete all pending commands. Returns list of strings."""
    try:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            cmds = json.load(f)
        try:
            os.remove(path)
        except Exception:
            pass
        return cmds
    except Exception:
        return []


# ── State reader (dashboard window) ──────────────────────────────────────────

def read_state(path: str) -> Optional[dict]:
    """Read state JSON. Returns None if missing or stale (>5 s old)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if time.time() - data.get("ts", 0) > 5:
            return None
        return data
    except Exception:
        return None
