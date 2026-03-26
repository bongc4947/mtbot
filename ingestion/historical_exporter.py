"""
Manages historical data requests from MT5 EA and waits for export completion.
Python issues EXPORT_HISTORICAL → MT5 exports CSV → signals EXPORT_DONE.
"""
import os
import time
import threading
from utils.logger import get_logger
from utils.data_utils import load_csv, validate_dataset

log = get_logger("historical_exporter")


class HistoricalExporter:
    def __init__(self, zmq_ctrl, cfg: dict):
        self.zmq = zmq_ctrl
        self.raw_dir = cfg.get("data", {}).get("raw_dir", "data/raw")
        self.min_bars = cfg.get("data", {}).get("min_bars_required", 10000)
        self.history_years = cfg.get("data", {}).get("history_years", 5)
        self.timeframe = cfg.get("data", {}).get("primary_timeframe", "M1")
        self._export_events: dict[str, threading.Event] = {}
        self._export_results: dict[str, bool] = {}
        self.ea_online: bool = False  # set True by run.py when MARKETWATCH_LIST is received

    def on_export_done(self, symbol: str, success: bool):
        """Called by message router when MT5 sends EXPORT_DONE."""
        self._export_results[symbol] = success
        ev = self._export_events.get(symbol)
        if ev:
            ev.set()

    def symbol_has_data(self, symbol: str) -> bool:
        # Check MT5 shared path first, then fall back to local data/raw.
        # This allows synthetic CSVs (tools/generate_synthetic_data.py) to satisfy
        # the data requirement without needing a live MT5 connection.
        candidates = [
            os.path.join(self.raw_dir, f"{symbol}.csv"),           # MT5 export path
            os.path.join("data", "raw", f"{symbol}.csv"),          # local project path
            os.path.join("data/raw", f"{symbol}.csv"),             # forward-slash variant
        ]
        for path in dict.fromkeys(candidates):  # dedup while preserving order
            if not os.path.exists(path):
                continue
            try:
                df = load_csv(path)
                if validate_dataset(df, symbol, self.min_bars):
                    log.debug(f"[{symbol}] Found data at {path}")
                    return True
            except Exception:
                continue
        return False

    def export_symbol(self, symbol: str, timeout: int = 60) -> bool:
        """Request export for one symbol, wait for completion (60s max)."""
        if self.symbol_has_data(symbol):
            log.info(f"[{symbol}] Historical data already present — skipping export")
            return True

        if not self.ea_online:
            log.warning(f"[{symbol}] EA offline — skipping MT5 export request")
            return False

        log.info(f"[{symbol}] Requesting historical export ({self.history_years}yr {self.timeframe})")
        ev = threading.Event()
        self._export_events[symbol] = ev
        self._export_results[symbol] = False

        self.zmq.request_historical_export(symbol, self.timeframe, self.history_years)

        done = ev.wait(timeout=timeout)
        del self._export_events[symbol]

        if not done:
            log.error(f"[{symbol}] Export timed out after {timeout}s")
            return False

        success = self._export_results.get(symbol, False)
        if success:
            log.info(f"[{symbol}] Export complete")
        else:
            log.warning(f"[{symbol}] Export reported failure")
        return success

    def ensure_all_symbols(self, symbols: list) -> list:
        """
        Export missing symbols sequentially. Returns list of symbols
        that have valid data (either pre-existing or newly exported).
        """
        os.makedirs(self.raw_dir, exist_ok=True)
        ready = []
        for symbol in symbols:
            if self.export_symbol(symbol):
                ready.append(symbol)
            else:
                log.warning(f"[{symbol}] Skipped — no historical data available")
        log.info(f"Historical data ready for {len(ready)}/{len(symbols)} symbols")
        return ready
