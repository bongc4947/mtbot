"""
Per-symbol tick buffer and spread tracker.
Uses fixed-size circular arrays — no dynamic allocation.
"""
import numpy as np
import threading
from utils.logger import get_logger

log = get_logger("tick_buffer")

# Tick fields: bid, ask, spread, volume, timestamp_epoch
TICK_FIELDS = 5
BID_IDX, ASK_IDX, SPREAD_IDX, VOL_IDX, TS_IDX = range(5)


class SymbolBuffer:
    def __init__(self, symbol: str, capacity: int = 1000, spread_window: int = 100):
        self.symbol = symbol
        self.capacity = capacity
        self.spread_window = spread_window

        self._buf = np.zeros((capacity, TICK_FIELDS), dtype=np.float64)
        self._head = 0
        self._size = 0
        self._lock = threading.Lock()

        # spread stats (online update)
        self._spread_sum = 0.0
        self._spread_sq_sum = 0.0
        self._spread_count = 0
        self._spread_ring = np.zeros(spread_window, dtype=np.float64)
        self._spread_ring_head = 0
        self._spread_ring_size = 0

    def push(self, bid: float, ask: float, spread: float, volume: float, ts: float):
        with self._lock:
            self._buf[self._head] = [bid, ask, spread, volume, ts]
            self._head = (self._head + 1) % self.capacity
            if self._size < self.capacity:
                self._size += 1
            self._update_spread_stats(spread)

    def _update_spread_stats(self, spread: float):
        if self._spread_ring_size == self.spread_window:
            old = self._spread_ring[self._spread_ring_head]
            self._spread_sum -= old
            self._spread_sq_sum -= old * old
        else:
            self._spread_ring_size += 1

        self._spread_ring[self._spread_ring_head] = spread
        self._spread_ring_head = (self._spread_ring_head + 1) % self.spread_window
        self._spread_sum += spread
        self._spread_sq_sum += spread * spread
        self._spread_count = self._spread_ring_size

    @property
    def mean_spread(self) -> float:
        if self._spread_count == 0:
            return 0.0
        return self._spread_sum / self._spread_count

    @property
    def std_spread(self) -> float:
        if self._spread_count < 2:
            return 0.0
        mean = self.mean_spread
        variance = (self._spread_sq_sum / self._spread_count) - mean * mean
        return float(np.sqrt(max(0.0, variance)))

    def spread_too_wide(self, current_spread: float, multiplier: float = 2.5) -> bool:
        threshold = self.mean_spread + multiplier * self.std_spread
        return current_spread > threshold

    def get_last(self, n: int) -> np.ndarray:
        with self._lock:
            n = min(n, self._size)
            end = self._head
            start = (end - n) % self.capacity
            if start < end:
                return self._buf[start:end].copy()
            return np.concatenate([self._buf[start:], self._buf[:end]], axis=0)

    @property
    def size(self) -> int:
        return self._size

    def latest(self) -> np.ndarray:
        with self._lock:
            if self._size == 0:
                return np.zeros(TICK_FIELDS)
            idx = (self._head - 1) % self.capacity
            return self._buf[idx].copy()


class TickBufferManager:
    def __init__(self, symbols: list, cfg: dict):
        cap = cfg.get("performance", {}).get("tick_buffer_size", 1000)
        sw = cfg.get("performance", {}).get("spread_window", 100)
        self._buffers: dict[str, SymbolBuffer] = {
            s: SymbolBuffer(s, capacity=cap, spread_window=sw) for s in symbols
        }

    def push_tick(self, symbol: str, bid: float, ask: float,
                  spread: float, volume: float, ts: float):
        buf = self._buffers.get(symbol)
        if buf:
            buf.push(bid, ask, spread, volume, ts)

    def get(self, symbol: str) -> SymbolBuffer:
        return self._buffers.get(symbol)

    def add_symbol(self, symbol: str, capacity: int = 1000, spread_window: int = 100):
        if symbol not in self._buffers:
            self._buffers[symbol] = SymbolBuffer(symbol, capacity, spread_window)

    @property
    def symbols(self) -> list:
        return list(self._buffers.keys())
